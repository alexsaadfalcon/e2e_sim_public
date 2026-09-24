"""`Simulation` -- ONE serial spine, one product fan-out.

The owner's 2026-09-24 directive ("the block diagram shown is still very confusing
from the fact that there are two pipelines ... Needs to be fixed immediately from the
ground up") retires the structure this module used to have: two feed-forward loops
(`feed_forward` for a CFR start, `_feed_forward_from` for any other domain) and a
default stage builder that produced an EMPTY stage list for a non-CFR source, so a
replayed ADC cube ran a different pipeline from a live frame.

What replaces it, per notes/ONE_CHAIN_CONTRACT_2026-09-24.md section 1.2:

* ONE stage list, in the contract's order, and ONE loop that walks it. The DEFAULT
  order is the FULL contract's (owner 2026-09-24, ballot answer 1B):

      [TxPowerStage] -> [InterconnectStage] -> DechirpBlock -> [FrontEndBlock]
        -> [ThermalNoiseBlock(mode="once")] -> RangeTransformBlock -> [MeasurementStage]

  i.e. the front end acts on the SAMPLED BEAT RECORD, after the dechirp, and thermal
  noise is injected exactly once. `composition="legacy_impulse"` reaches the v1.0
  order (front end on `ifft(CFR)`, before the dechirp, no link-budget stages) and
  exists ONLY for the stored corpora's bit-parity gate -- see `COMPOSITIONS`.
* Replay is a START INDEX into that same list: a source advertising
  `signal_domain == DOMAIN_RX_TIME` enters at the first stage that consumes RX time,
  and the stages it skipped are recorded BY NAME in `self.skipped_stages` /
  `outputs['skipped_stages']` rather than silently not existing.
* Every product reads the one spine's state. The range products consume the cube the
  `RangeTransformBlock` emits; nothing computes a second range transform of its own.
"""

import warnings

import torch
from tqdm import tqdm
from collections import defaultdict

from e2e import frames
from e2e.blocks import CircuitStage, InterconnectStage, MeasurementStage
from e2e.chain.dechirp import ANTENNA_INDEX_REVERSED, DechirpBlock
from e2e.chain.frontend import FrontEndBlock
from e2e.chain.link_budget import ThermalNoiseBlock, TxPowerStage
from e2e.chain.receive import RangeTransformBlock
from e2e.chain.waveform import fmcw_plan_from_freq_plan


# Relative threshold (fraction of the top singular value) below which a singular value
# is treated as noise floor when computing effective rank -- see rank_diagnostic.
_RANK_RTOL = 1e-2

#: Canonical pipeline state a downstream product must NOT clobber. One definition,
#: because there used to be two copies (one per feed-forward loop) and they had
#: already drifted apart by the time the loops were merged.
_RESERVED_STATE_KEYS = frozenset({
    'U', 'U_true', 's_pars', 'PRX', 'frame_layout', 'signal_domain',
    'signal_dimension', 'sensing_matrix', 'aperture_shape', 'tx_wave', 'adc',
    'cube', 'cube_axes',
})


def _check_frame_contract(component, state_dict):
    """Validate the pipeline's current frame against `component`'s declared
    `frames.FrameCapabilities`, BEFORE handing it the state dict.

    Raises `frames.FrameContractError` naming the component (stages name the block they
    wrap) and the offending domain or axis. Components that declare nothing get the
    historical contract (frequency domain, no MIMO, single chirp) -- see
    frames.DEFAULT_CAPABILITIES.

    The DOMAIN check always runs: it is what catches a mis-ordered chain (an impairment
    before the dechirp) and it does not depend on the payload's rank. The AXIS checks
    additionally need a 4-D S-parameter frame; a non-4-D payload skips them, because a
    custom serial stage may park something the frame contract can't speak to in
    's_pars' and rejecting it would break flows that worked before.
    """
    domain = state_dict.get('signal_domain', frames.DOMAIN_CFR)
    dimension = state_dict.get('signal_dimension', frames.DIMENSION_FULL)
    frames.require_domain(domain, component)
    frames.require_dimension(dimension, component)
    payload = state_dict.get(frames.DOMAIN_PAYLOAD_KEY.get(domain, 's_pars'))
    if not torch.is_tensor(payload) or payload.ndim != 4:
        return
    frames.check_capabilities(
        payload, component, layout=state_dict.get('frame_layout', frames.LAYOUT_RAW),
        domain=domain, dimension=dimension,
    )


def _advance_domain(component, state_dict, before):
    """Enforce what a component promised about the chain's domain, AFTER it ran.

    Two jobs, both of which exist because an adversarial review found them missing:

    1. A block declaring `emits_domain` must actually deliver it. Without this, a bridge
       that forgets to set `signal_domain` produces a misleading error at the NEXT block
       ("insert a DechirpBlock") even though the dechirp already ran and its output is
       sitting right there.
    2. When the domain changes, the previous domain's payload is DROPPED from state.
       Otherwise `s_pars` outlives the crossing, and a block that declares the RX-time
       domain but reads `state['s_pars']` by mistake computes happily on stale
       pre-dechirp data with no error anywhere -- a silent wrong answer, which is the
       worst failure mode a contract can have.
    """
    caps = frames.capabilities_of(component)
    after = state_dict.get('signal_domain', before)
    if caps.emits_domain is not None and after != caps.emits_domain:
        raise frames.FrameContractError(
            f"{frames.component_name(component)} declares it emits the "
            f"{caps.emits_domain} domain but left the chain in {after!r}; a bridge block "
            f"must set state['signal_domain'] to the domain it hands downstream."
        )
    if after != before:
        stale_key = frames.DOMAIN_PAYLOAD_KEY.get(before)
        if stale_key and stale_key in state_dict:
            del state_dict[stale_key]
    return after


def _advance_dimension(component, state_dict, before):
    """Enforce what a component promised about full-vs-reduced dimension, AFTER it ran.

    The dimension counterpart of `_advance_domain`'s first job: a block declaring
    `emits_dimension` must actually deliver it, so a compress/decompress step that
    forgets to set `signal_dimension` fails where the mistake is rather than at the next
    block. There is no stale-payload half here -- both dimensions live in `s_pars`, and
    a compressed frame REPLACES the full one rather than sitting beside it.
    """
    caps = frames.capabilities_of(component)
    after = state_dict.get('signal_dimension', before)
    if caps.emits_dimension is not None and after != caps.emits_dimension:
        raise frames.FrameContractError(
            f"{frames.component_name(component)} declares it emits {caps.emits_dimension} "
            f"dimension but left the chain in {after!r}; a compress/decompress block must "
            f"set state['signal_dimension'] to what it hands downstream."
        )
    return after


def _svd_frame(s_pars):
    """SVD of a single frame's flattened S-parameter matrix, computed once.

    Returns (U, S): the full left-singular-vector matrix and the singular values
    (descending). Shared by get_U_true and rank_diagnostic so callers that need both
    the top-k basis and the singular-value spectrum don't pay for two decompositions.
    Multi-chirp frames are summarized by their FIRST chirp (the subspace path is
    single-chirp by declaration; see MeasurementStage.frame_capabilities).
    """
    assert len(s_pars.shape) == 4
    s_pars_0 = s_pars[:, :, 0, :]
    s_pars_0 = s_pars_0.view(-1, s_pars_0.shape[-1])
    U, S, _ = torch.linalg.svd(s_pars_0)
    return U, S


def get_U_true(s_pars, k):
    # "Ground truth" here means the top-k left singular vectors of a single frame's
    # S-parameter matrix -- a meaningful reference for subspace tracking only when the
    # frame's effective rank (number of singular values well above the noise floor) is
    # >= k. Below that, the trailing directions returned are noise-dominated, and any
    # subspace_err computed against them partly measures how well the tracker follows
    # noise rather than signal structure. See rank_diagnostic (and Simulation.feed_forward,
    # which records it per frame) for the diagnostic that makes this failure mode visible.
    U, _ = _svd_frame(s_pars)
    return U[:, :k]


def rank_diagnostic(S, k, rtol=_RANK_RTOL):
    """Rank / singular-value-gap diagnostic for a frame's singular-value spectrum `S`
    (descending, as returned by `_svd_frame`), against a requested subspace rank `k`.

    - `effective_rank`: count of singular values above `rtol * S[0]` (default rtol
      _RANK_RTOL) -- i.e. singular values still well above the noise floor.
    - `sv_gap_at_k`: ratio S[k-1] / S[k], the singular-value gap right at the requested
      cutoff (large gap = a clean signal/noise boundary at k; near 1 = no real
      boundary there). NaN if k >= len(S) (no S[k] to compare against).
    - `sv_gap_norm`: (S[k-1] - S[k]) / S[0], the absolute gap at the cutoff normalized
      by the dominant singular value. This is the quantity that actually conditions the
      top-k subspace's identity (Davis-Kahan: a perturbation E rotates the subspace by
      ~||E|| / (S[k-1] - S[k])), made scale-free by S[0]. Small under BOTH failure
      modes: a near-degenerate cluster straddling the cutoff (S[k-1] ~= S[k], the
      ratio's territory) AND a tail that has sunk into insignificance (S[k-1], S[k]
      both << S[0] -- where the ratio of two noise-floor values can still look
      "healthy"). NaN if k >= len(S).
    - `rank_ok`: True iff `k <= effective_rank`, i.e. the requested subspace rank is
      supported by the frame's actual signal content.

    Honesty note: this diagnostic reads the FULL spectrum of the frame -- available
    here only because Simulation computes a full SVD anyway to construct the scoring
    ground truth. A deployed receiver stores a rank-k basis and never holds the full
    aperture frame, so it cannot know S[k] this way. The boundary singular values ARE
    estimable from what a receiver genuinely has -- the m >> k measurements X = A V,
    whose sensing matrix already contains rows orthogonal to the tracked subspace
    (gen_A_ada), so an eigendecomposition of the m x m covariance X X^H (or k+p guard
    directions with Rayleigh quotients) recovers sigma_k, sigma_{k+1} estimates in the
    measurement domain -- but that estimator is NOT implemented; anything gating on
    these values (AdaOjaBlock.gap_response) is consuming simulator-side
    instrumentation, spectrum-only but oracle-sourced.
    """
    # k < 1 would silently wrap S[k-1] to S[-1] (the SMALLEST singular value) via
    # Python negative indexing and log nonsense diagnostics; a rank request below 1
    # is a caller bug, so fail loudly.
    if k < 1:
        raise ValueError(f"rank_diagnostic requires k >= 1, got k={k}")
    threshold = rtol * S[0]
    effective_rank = int((S > threshold).sum().item())
    if k < len(S):
        denom = S[k]
        sv_gap_at_k = float((S[k - 1] / denom).item()) if denom > 0 else float('inf')
        sv_gap_norm = (
            float(((S[k - 1] - S[k]) / S[0]).item()) if S[0] > 0 else float('nan')
        )
    else:
        sv_gap_at_k = float('nan')
        sv_gap_norm = float('nan')
    rank_ok = k <= effective_rank
    return {
        'effective_rank': effective_rank,
        'sv_gap_at_k': sv_gap_at_k,
        'sv_gap_norm': sv_gap_norm,
        'rank_ok': rank_ok,
    }


def to_beat_basis(U):
    """Carry an element-axis basis from the CFR domain into the BEAT domain.

    The dechirp is `conj` plus a reversal of the RX antenna index
    (`e2e.chain.dechirp.beat_from_cfr`). On the one spine the subspace tracker's input
    is the cube, so the tracker estimates the column space of `flip(conj(U))`, not of
    `U`. The oracle has to be expressed in the same basis or `subspace_err` compares
    two unrelated subspaces -- which is what a naive "keep the SVD where it was" move
    would have done, silently, with plausible-looking numbers.

    This is exact, not an approximation, and it is why NO published subspace number
    changes: for a matrix `M`, `svd(conj(flip(M)))` has left factor `flip(conj(U))`;
    the range DFT is scaled-unitary on the OTHER axis and so leaves the left column
    space alone; and `subspace_dist_frob` is invariant under applying the same
    permutation-and-conjugation to both arguments (`|<conj a, conj b>| = |<a, b>|`).
    Measured 2026-09-24 against the v1.0 tracker output pinned in
    tests/test_one_chain_spine.py.
    """
    out = U.conj()
    if ANTENNA_INDEX_REVERSED:
        out = torch.flip(out, dims=(0,))
    return out


def perturb_basis(U):
    U = U + 1e-3 * (torch.randn_like(U) + 1j * torch.randn_like(U))
    return torch.linalg.qr(U)[0]

class Simulation:
    def __init__(self,
        environment_block,
        downstream_blocks,
        k,
        circuit_block=None,
        interconnect_block=None,
        afe_block=None,
        subspace_block=None,
        array_shape=None,
        serial_stages=None,
        warm_start=True,
        radar_cfg=None,
        range_transform=None,
        composition="full",
        front_end=None,
        link_budget="auto",
    ):
        self.environment_block = environment_block
        self.downstream_blocks = downstream_blocks
        self.k = k
        self.circuit_block = circuit_block
        self.interconnect_block = interconnect_block
        self.afe_block = afe_block
        self.subspace_block = subspace_block
        # Whether the one-time tracker init (see feed_forward) warm-starts from a
        # perturbed ground truth (True, default -- preserves pre-existing numbers) or
        # leaves Oja's own random cold-start basis untouched (False -- an honest
        # cold-start run, where subspace_err reflects tracking from scratch with no
        # peek at ground truth).
        self.warm_start = warm_start
        if subspace_block is None and afe_block is not None:
            raise ValueError('Need subspace block to pair with AFE block')
        # Receive-array geometry (n_rx_x, n_rx_y). Explicit arg wins; otherwise take it
        # from the environment block if it advertises one; otherwise default to 32x32
        # (the historical hardcoded size) for backward compatibility.
        if array_shape is None:
            array_shape = getattr(environment_block, 'array_shape', (32, 32))
        self.n_rx_x, self.n_rx_y = array_shape
        # THE serial spine. One list, the contract's order (section 1.2), built the
        # SAME way whatever domain the source starts in -- a replayed ADC cube enters
        # it at a start index (see `_start_index`) instead of getting a pipeline of
        # its own. `serial_stages=` still replaces the whole list for callers that
        # compose their own chain (the ML generator, the webapp), but there is no
        # longer a branch that silently builds a DIFFERENT default.
        if composition not in self.COMPOSITIONS:
            raise ValueError(
                f"unknown composition {composition!r}; expected one of "
                f"{self.COMPOSITIONS}"
            )
        self.composition = composition
        self.radar_cfg = radar_cfg
        self.link_budget = link_budget
        self.link_budget_active = False
        if serial_stages is not None:
            self.serial_stages = list(serial_stages)
        else:
            self.serial_stages = self._build_spine(
                circuit_block, interconnect_block, afe_block, subspace_block,
                range_transform, front_end,
            )
        self._check_single_source()
        self.outputs = defaultdict(list)
        # The online subspace tracker is initialized once (from the first frame's
        # subspace) and then tracks the evolving scene; this flag guards that
        # one-time warm start. See feed_forward.
        self._subspace_started = False
        # Throttles the rank-diagnostic warning to once per run (see feed_forward).
        self._rank_warned = False
        # Names of the spine stages the LAST frame entered past (replay start index).
        # Empty for a live CFR source; that is the same rule with nothing skipped.
        self.skipped_stages = []

    # --------------------------------------------------------------- the one spine
    #: Minimal `cfg` for the imaging spine's dechirp: one TX, no multiplexing to undo.
    #: `DechirpBlock` reads only `cfg.mimo` (see its docstring), so a full
    #: `RadarConfig` is needed only when a caller wants TDM/DDMA combining or a
    #: cfg-derived range calibration -- pass `radar_cfg=` for that.
    class _SingleTxCfg:
        mimo = "single"
        n_tx = 1

    #: Which block ORDER the spine is built in.
    #:   "full"  -- THE DEFAULT (owner 2026-09-24, ballot answer 1B). The front end
    #:              acts on the SAMPLED BEAT RECORD, after the dechirp, and thermal
    #:              noise is injected exactly once. This is the physics; see
    #:              `e2e/chain/frontend.py` for the commutation argument that licenses
    #:              it and for the drive level at which that licence stops holding.
    #:   "legacy_impulse" -- the v1.0/v1.1 order: the front end on `ifft(CFR)`, BEFORE
    #:              the dechirp, no `TxPowerStage`, no `ThermalNoiseBlock`. It exists
    #:              for ONE reason -- the stored corpora were generated that way and
    #:              the live-vs-stored gates read max |diff| = 0 CODES, which a
    #:              differently-ordered RNG consumption fails at any floor. It is a
    #:              recorded fact about files on disk, not an alternative physics, and
    #:              nothing new should select it.
    COMPOSITIONS = ("full", "legacy_impulse")

    def _resolve_radar_cfg(self):
        """The `RadarConfig` the spine's dechirp, front end and floor are built from.

        Explicit `radar_cfg=` wins. Otherwise it is DERIVED from the source's own
        `freq_plan`, because the stored grid is what decides the chirp that can
        consume it (`fmcw_plan_from_freq_plan`: `S/fs` must equal the grid's own
        endpoint-inclusive spacing). A source with no plan -- a legacy pkl, a
        synthetic test fixture -- yields None, and the stages that genuinely need a
        sample rate are then left off rather than built on an invented one.
        """
        if self.radar_cfg is not None:
            return self.radar_cfg
        plan = getattr(self.environment_block, "freq_plan", None)
        if not plan:
            return None
        try:
            return fmcw_plan_from_freq_plan(plan, n_rx=self.n_rx_x * self.n_rx_y)
        except ValueError:
            return None

    def _resolve_link_budget(self, cfg):
        """Whether the spine carries `TxPowerStage` + `ThermalNoiseBlock`.

        `link_budget="auto"` (the default) enables them iff BOTH a `RadarConfig` is
        available AND the source advertises an absolute amplitude scale
        (`physical_scale`). That is not a hedge, it is F63 and contract section 1.2
        row 7: the munich frames are normalised (`physical_scale=False`), so there is
        no absolute reference for a `k*T*B*F` floor to sit beneath, and
        `ThermalNoiseBlock` would refuse them BY NAME if it were built. Corpus sources
        carry volts and get both stages.

        True forces them on (and `ThermalNoiseBlock`'s own F63 guard will refuse a
        normalised frame, loudly, which is the right failure); False forces them off.
        """
        if self.link_budget is not True and self.link_budget is not False:
            return bool(cfg is not None
                        and getattr(self.environment_block, "physical_scale", False))
        return bool(self.link_budget)

    def _build_spine(self, circuit_block, interconnect_block, afe_block,
                     subspace_block, range_transform, front_end):
        """The contract's stage list (section 1.2), in order, for every source.

        `composition="full"` (the default) builds:

            [TxPowerStage] -> [InterconnectStage] -> DechirpBlock -> [FrontEndBlock]
              -> [ThermalNoiseBlock(mode="once")] -> RangeTransformBlock
              -> [MeasurementStage]

        `composition="legacy_impulse"` builds the v1.0 order instead, with
        `CircuitStage` BEFORE the dechirp and no link-budget stages.

        Square brackets are PHYSICAL OPTIONS, present only when configured; the
        dechirp and the range transform are always present, because they are what make
        the chain one chain -- deleting them is what used to leave the imaging products
        computing their own range FFTs off a second path.

        The interconnect stays in the CFR domain, before the dechirp, in BOTH
        compositions: a linear RF filter is a per-sample gain after the dechirp and
        commutes with the unit-modulus dechirp, so `interconnect -> dechirp -> LNA` is
        `interconnect -> LNA -> mixer` (contract section 1.2 row 4).

        A legacy `circuit_block` (an `RFFEBlock`) is TRANSLATED into a `FrontEndBlock`
        under `composition="full"` via `FrontEndBlock.from_rffe`, so existing callers
        move onto the new placement without restating their knobs. Pass `front_end=`
        to supply one directly.

        The range transform is built at the IMAGING identity point -- `window="none"`,
        `dc_removal=False` -- not at `RangeTransformBlock`'s own (ML-protocol)
        defaults. `window="none"` because it is the point at which the cube's column
        space is exactly the v1.0 frame's (contract 5.4, and T2/T3's refine gate reads
        a singular-value gap a window would move); `dc_removal=False` because the
        munich trace is generated with `normalize_delays=True`, putting the
        line-of-sight path AT bin 0, which the fast-time mean subtraction would zero.
        Pass `range_transform=` to override.
        """
        cfg = self._resolve_radar_cfg()
        self.radar_cfg = cfg
        legacy = self.composition == "legacy_impulse"
        use_link_budget = (not legacy) and self._resolve_link_budget(cfg)
        self.link_budget_active = use_link_budget

        stages = []
        if use_link_budget:
            # sqrt(P_tx) at the SOURCE, so receiver noise cannot scale with transmit
            # power -- the coupling F81 is actually about (contract section 1.4).
            stages.append(TxPowerStage(cfg))
        if legacy and circuit_block is not None:
            stages.append(CircuitStage(circuit_block))
        if interconnect_block is not None:
            stages.append(InterconnectStage(interconnect_block))
        stages.append(DechirpBlock(cfg or self._SingleTxCfg()))
        if not legacy:
            fe = front_end
            if fe is None and circuit_block is not None:
                if cfg is None:
                    raise ValueError(
                        "the 'full' composition puts the front end on the beat record, "
                        "which needs the beat SAMPLE RATE to reference its noise "
                        "bandwidth to -- and this chain has no RadarConfig and no "
                        "source freq_plan to derive one from. Pass radar_cfg=, or "
                        "front_end=FrontEndBlock(..., fs_hz=...), or "
                        "composition='legacy_impulse' if you are reproducing a stored "
                        "corpus."
                    )
                fe = FrontEndBlock.from_rffe(circuit_block, cfg)
            if fe is not None:
                stages.append(fe)
            if use_link_budget:
                # mode="once": adds NOTHING when the front end already injected, and
                # is the one injection when it did not. Never a second floor.
                stages.append(ThermalNoiseBlock(cfg, mode="once"))
        stages.append(range_transform or RangeTransformBlock(
            cfg, window="none", dc_removal=False))
        # No subspace block -> no measurement stage: the spine then ends at the range
        # transform, which is all the image/profile products need. (AFE without a
        # subspace block was already rejected above.)
        if subspace_block is not None:
            stages.append(MeasurementStage(afe_block, subspace_block))
        return stages

    def _check_single_source(self):
        """One chain, one source. A stage that is itself a frame source (it advertises
        `get_S_pars`, the environment-block protocol) sitting in the serial list means
        two things are trying to originate the chain, and whichever runs second
        silently wins. Refuse, naming both."""
        sources = [s for s in self.serial_stages if hasattr(s, "get_S_pars")]
        if sources:
            names = ", ".join(frames.component_name(s) for s in sources)
            raise ValueError(
                f"two sources on one chain: the environment block is "
                f"{frames.component_name(self.environment_block)}, but the serial "
                f"stage list also contains source block(s) {names}. The spine has one "
                f"origin; a stored artifact is replayed by passing it as the "
                f"environment block, which enters the spine at a start index."
            )

    def _check_source_frame(self, payload, domain):
        """Refuse a source frame the spine's dechirp would silently mangle.

        `DechirpBlock` accepts MIMO -- combining the TX axis is its job -- but it reads
        the scheme from its `cfg`, and the imaging spine's default cfg is single-TX.
        Handing that a 2-TX frame is not an error inside `mimo_combine`: for
        `mimo="single"` it selects TX `c % n_tx` and quietly KEEPS ONLY TX 0. So the
        guard has to live here, where the cfg and the frame are both in view.

        This replaces the old `GridStage: MIMO not supported yet` error, which came
        from a stage that is no longer on the spine (the aperture view moved into the
        products). Same refusal, named for the thing that can actually fix it.
        """
        if domain != frames.DOMAIN_CFR or not torch.is_tensor(payload) or payload.ndim != 4:
            return
        n_tx = payload.shape[1]
        if n_tx == 1:
            return
        # The cfg that MATTERS is the one the spine's dechirp will actually read --
        # a caller composing its own `serial_stages` supplies a full RadarConfig there
        # and never touches `self.radar_cfg`. Find the bridge, ask it.
        cfg = self.radar_cfg
        for stage in self.serial_stages:
            caps = frames.capabilities_of(stage)
            if caps.emits_domain == frames.DOMAIN_RX_TIME and hasattr(stage, "cfg"):
                cfg = stage.cfg
                break
        cfg_tx = int(getattr(cfg, "n_tx", 1)) if cfg is not None else 1
        scheme = str(getattr(cfg, "mimo", "single")).lower() if cfg is not None else "single"
        if cfg_tx == n_tx and scheme in ("tdm", "ddma"):
            return
        raise frames.FrameContractError(
            f"Simulation: MIMO not supported yet by this spine -- the frame has "
            f"n_tx={n_tx} but the chain's radar_cfg declares n_tx={cfg_tx}, "
            f"mimo={scheme!r}. The dechirp would keep only TX 0 and say nothing. "
            f"Pass radar_cfg=RadarConfig(..., n_tx={n_tx}, mimo='tdm'|'ddma') so the "
            f"TX axis is actually combined."
        )

    def _start_index(self, domain):
        """Where a source in `domain` ENTERS the one spine.

        The replay capability, kept, without a second loop: the first stage that
        consumes `domain` is where the chain begins, and everything before it is
        skipped -- and NAMED, in `self.skipped_stages`, because "this run did not
        apply the front end" is exactly the fact a stored-vs-live comparison turns on.
        A CFR source starts at 0, so the live path is the no-skip case of one rule.
        """
        if domain == frames.DOMAIN_CFR:
            return 0
        for i, stage in enumerate(self.serial_stages):
            if frames.capabilities_of(stage).domain == domain:
                return i
        # Nothing on this spine consumes the source's domain: the products may still
        # be able to (a stored cube feeding an image). Start at the top and let the
        # per-stage contract check name the first mismatch.
        return len(self.serial_stages)

    def step(self):
        self.environment_block.step()

    def reset(self):
        self.environment_block.reset()
        # Re-arm the one-time tracker warm start and rank-diagnostic warning so each
        # run() starts fresh.
        self._subspace_started = False
        self._rank_warned = False
        # Blocks with per-run state (e.g. ModemBlock's frame-indexed noise counter, a
        # SinkBlock's frame counter, an ImpairmentBlock's per-frame seed) expose
        # reset(); rewind them so repeated run() calls on the same Simulation are
        # reproducible. Serial stages need this as much as downstream blocks do -- a
        # sink placed mid-chain would otherwise keep counting across runs and write a
        # second run's frames under the first run's numbering.
        for block in list(self.serial_stages) + list(self.downstream_blocks):
            if hasattr(block, "reset"):
                block.reset()

    def feed_forward(self):
        payload = self.environment_block.get_S_pars()

        # Which signal domain does the chain START in? Normally the frequency domain --
        # an environment block hands over an S-parameter frame. A SourceBlock replaying
        # a stored artifact starts further down the SAME spine (see `_start_index`),
        # in which case the payload is an ADC cube and the frequency-domain machinery
        # below (the SVD, the subspace ground truth) has nothing to say about it.
        # Blocks that advertise nothing get the historical frequency-domain start.
        domain = getattr(self.environment_block, 'signal_domain', frames.DOMAIN_CFR)
        self._check_source_frame(payload, domain)
        start = self._start_index(domain)
        self.skipped_stages = [frames.component_name(s)
                               for s in self.serial_stages[:start]]

        state_dict = {
            frames.DOMAIN_PAYLOAD_KEY.get(domain, 's_pars'): payload,
            'PRX': None,
            'signal_domain': domain,
            # Seeded explicitly, like signal_domain above: a replayed payload is stored
            # post-digitization and therefore full-dimension, and leaving it to a
            # .get() default elsewhere would mean the two entry points disagreed about
            # what the contract's starting state even is.
            'signal_dimension': frames.DIMENSION_FULL,
            # The receive-array geometry the aperture products factor the cube's
            # element axis with. It travels in state because the cube stays flat
            # through the compressor (see e2e/blocks.py:_aperture_shape_for).
            'aperture_shape': (self.n_rx_x, self.n_rx_y),
        }
        if self.skipped_stages:
            # Not a warning and not silence: a recorded fact, per frame, that a
            # stored-vs-live comparison can read back.
            state_dict['skipped_stages'] = list(self.skipped_stages)
        # The stored frame's own frequency plan, seeded at the source so the range
        # transform calibrates the ONE range axis from the grid the frames were
        # actually traced on -- `(stop - start) / (num_freqs - 1)`, endpoint-inclusive
        # -- instead of a display helper re-deriving `B / num_freqs` downstream and
        # landing 0.02% (half a metre at 125 m on munich Ka) away from it.
        plan = getattr(self.environment_block, 'freq_plan', None)
        if plan:
            state_dict['freq_plan'] = plan

        if domain == frames.DOMAIN_CFR:
            self._seed_subspace_oracle(payload, state_dict)
        state_dict.update(self._environment_state_updates())

        for stage in self.serial_stages[start:]:
            _check_frame_contract(stage, state_dict)
            before = state_dict.get('signal_domain', domain)
            before_dim = state_dict.get('signal_dimension', frames.DIMENSION_FULL)
            state_dict.update(stage.apply(state_dict))
            _advance_domain(stage, state_dict, before)
            _advance_dimension(stage, state_dict, before_dim)

        # Serial stages that touch the subspace tracker (MeasurementStage) refresh
        # 'U' via their return dict; re-read it from the tracker here too so 'U' is
        # always current. Guarded: a serial_stages override may legitimately run
        # without a subspace block, in which case 'U' is whatever the stages set.
        if self.subspace_block is not None:
            state_dict['U'] = self.subspace_block.oja.U

        # Per-frame refine-pass count MeasurementStage actually ran (constant unless a
        # gap_response is enabled). Logged next to 'sv_gap_at_k' so the adaptive
        # compute cost of a gap response is visible in the same outputs record that
        # holds the diagnostic that triggered it.
        if 'n_refine_used' in state_dict:
            self.outputs['n_refine_used'].append(state_dict['n_refine_used'])
        if self.skipped_stages:
            self.outputs['skipped_stages'].append(list(self.skipped_stages))

        for downstream_block in self.downstream_blocks:
            _check_frame_contract(downstream_block, state_dict)
            outputs = downstream_block.apply(state_dict)
            for output_name, output in outputs.items():
                self.outputs[output_name].append(output)
            # Make a block's outputs visible to subsequent downstream blocks, so they
            # can compose (e.g. a comms BERBlock consumes a ModemBlock's tx/rx bits).
            # Existing product blocks emit disjoint keys, so this is a no-op for them.
            # Guard the reserved pipeline keys: a block must not clobber the canonical
            # state the orchestrator feeds every block, so the contract is explicit.
            for key in outputs:
                if key in _RESERVED_STATE_KEYS:
                    raise ValueError(
                        f"downstream block {downstream_block} emitted reserved key {key!r}"
                    )
            state_dict.update(outputs)

    def _seed_subspace_oracle(self, s_pars, state_dict):
        """The subspace ground truth + rank diagnostic, for a chain that starts at the
        CFR. Unchanged arithmetic; lifted out of `feed_forward` so the one loop reads
        as one loop. See `to_beat_basis` for why `U_true` is expressed in the beat
        basis and why that moves no published number."""
        U, S = _svd_frame(s_pars)
        U_true = to_beat_basis(U[:, :self.k])

        # Rank / singular-value-gap diagnostic on the frame get_U_true saw: flags when
        # the requested subspace rank self.k exceeds the frame's effective rank, in
        # which case the trailing "ground truth" directions are noise and subspace_err
        # partly measures noise-tracking rather than signal-subspace tracking. Reuses
        # the SVD above -- no second decomposition.
        rank_diag = rank_diagnostic(S, self.k)
        self.outputs['effective_rank'].append(rank_diag['effective_rank'])
        self.outputs['sv_gap_at_k'].append(rank_diag['sv_gap_at_k'])
        self.outputs['sv_gap_norm'].append(rank_diag['sv_gap_norm'])
        self.outputs['rank_ok'].append(rank_diag['rank_ok'])
        if not rank_diag['rank_ok'] and not self._rank_warned:
            warnings.warn(
                f"requested subspace rank k={self.k} exceeds frame effective rank "
                f"{rank_diag['effective_rank']}; subspace_err partly reflects noise "
                f"tracking."
            )
            self._rank_warned = True

        # Initialize the online (Oja) tracker ONCE, then let it actually track the
        # evolving scene across frames -- never reset it every frame (that used to make
        # online tracking a no-op: subspace_err reflected the injected perturbation,
        # not the tracker). Note: because each frame is a fresh moving-platform channel
        # snapshot, the per-frame subspace can change faster than a one-step tracker
        # follows, so subspace_err reflects that tracking lag.
        #
        # warm_start=True (default): warm-start from a perturbed ground truth (the
        # historical behavior; subspace_err then measures tracking lag from a
        # near-truth starting point).
        # warm_start=False: leave Oja's own random cold-start basis (rand_orth_complex)
        # untouched -- an honest cold start with no peek at ground truth; subspace_err
        # then also reflects the tracker converging from scratch.
        if self.subspace_block is not None and not self._subspace_started:
            if self.warm_start:
                self.subspace_block.oja.U = perturb_basis(U_true)
            self._subspace_started = True

        state_dict['U_true'] = U_true
        # Spectrum-only diagnostic ((S[k-1]-S[k])/S[0], from rank_diagnostic above) --
        # NOT derived from U_true. Threaded through so a stage/block downstream (e.g.
        # MeasurementStage/AdaOjaBlock's opt-in gap_response) can react to an
        # ill-conditioned subspace identity at the k cutoff -- a degenerate cluster
        # there OR an insignificant tail -- without peeking at the ground-truth basis.
        state_dict['sv_gap_norm'] = rank_diag['sv_gap_norm']

    def _environment_state_updates(self):
        """Extra state an environment block wants to seed the chain with.

        A ray-traced environment carries ground-truth labels alongside the frame, and a
        SourceBlock carries whatever metadata the stored artifact held. Attaching them
        HERE, at the source, is what lets labels travel with their frame through the
        chain instead of being recomputed at the far end against a scene that may since
        have moved.
        """
        getter = getattr(self.environment_block, 'get_state_updates', None)
        return dict(getter() or {}) if callable(getter) else {}


    def get_outputs(self):
        return self.outputs

    def run(self, n_steps=10, should_stop=None):
        """Run `n_steps` frames; `should_stop()` is polled BEFORE each frame and ends the
        run early when it returns True (the GUI's Cancel button). `self.n_steps_run`
        records how many frames actually ran, and `self.cancelled` whether the run was
        cut short, so a caller can label partial outputs honestly instead of presenting
        three frames as ten."""
        self.reset()
        self.n_steps_run = 0
        self.cancelled = False
        for i in tqdm(range(n_steps), desc='RUNNING ARRAY SIMULATION'):
            if should_stop is not None and should_stop():
                self.cancelled = True
                break
            self.feed_forward()
            self.step()
            self.n_steps_run += 1
        return self.get_outputs()

