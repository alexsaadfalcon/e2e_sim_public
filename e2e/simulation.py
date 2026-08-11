import warnings

import torch
from tqdm import tqdm
from collections import defaultdict

from e2e import frames
from e2e.blocks import CircuitStage, GridStage, InterconnectStage, MeasurementStage


# Relative threshold (fraction of the top singular value) below which a singular value
# is treated as noise floor when computing effective rank -- see rank_diagnostic.
_RANK_RTOL = 1e-2


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
    - `rank_ok`: True iff `k <= effective_rank`, i.e. the requested subspace rank is
      supported by the frame's actual signal content.
    """
    threshold = rtol * S[0]
    effective_rank = int((S > threshold).sum().item())
    if k < len(S):
        denom = S[k]
        sv_gap_at_k = float((S[k - 1] / denom).item()) if denom > 0 else float('inf')
    else:
        sv_gap_at_k = float('nan')
    rank_ok = k <= effective_rank
    return {
        'effective_rank': effective_rank,
        'sv_gap_at_k': sv_gap_at_k,
        'rank_ok': rank_ok,
    }


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
        # The serial stages form the pipeline proper (each may rewrite 's_pars'), run
        # in order inside feed_forward. By default they're built from the legacy
        # block args above; pass `serial_stages` explicitly to replace the whole list
        # (composability hook -- e.g. inserting a custom stage).
        if serial_stages is not None:
            self.serial_stages = serial_stages
        elif getattr(environment_block, 'signal_domain', frames.DOMAIN_CFR) != frames.DOMAIN_CFR:
            # The default stages below (circuit, aperture grid, measurement) are the
            # frequency-domain imaging path. An environment that starts the chain in
            # another domain -- a SourceBlock replaying a stored ADC cube -- has already
            # passed the point where they apply, so building them would only produce a
            # contract error naming a stage the caller never asked for. Such a chain
            # supplies its own stages, or none.
            self.serial_stages = []
        else:
            self.serial_stages = []
            if circuit_block is not None:
                self.serial_stages.append(CircuitStage(circuit_block))
            self.serial_stages.append(GridStage((self.n_rx_x, self.n_rx_y)))
            if interconnect_block is not None:
                self.serial_stages.append(InterconnectStage(interconnect_block))
            # No subspace block -> no measurement stage: the pipeline then ends at the
            # aperture grid, which is all FFT/range-map products need. (AFE without a
            # subspace block was already rejected above.)
            if subspace_block is not None:
                self.serial_stages.append(MeasurementStage(afe_block, subspace_block))
        self.outputs = defaultdict(list)
        # The online subspace tracker is initialized once (from the first frame's
        # subspace) and then tracks the evolving scene; this flag guards that
        # one-time warm start. See feed_forward.
        self._subspace_started = False
        # Throttles the rank-diagnostic warning to once per run (see feed_forward).
        self._rank_warned = False

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
        # an environment block hands over an S-parameter frame. But a SourceBlock
        # replaying a stored artifact may start the chain mid-way, already past the
        # dechirp, in which case the payload is an ADC cube and the frequency-domain
        # machinery below (the SVD, the subspace ground truth) has nothing to say about
        # it. Blocks that advertise nothing get the historical frequency-domain start.
        domain = getattr(self.environment_block, 'signal_domain', frames.DOMAIN_CFR)
        if domain != frames.DOMAIN_CFR:
            return self._feed_forward_from(domain, payload)

        s_pars = payload
        U, S = _svd_frame(s_pars)
        U_true = U[:, :self.k]

        # Rank / singular-value-gap diagnostic on the frame get_U_true saw: flags when
        # the requested subspace rank self.k exceeds the frame's effective rank, in
        # which case the trailing "ground truth" directions are noise and subspace_err
        # partly measures noise-tracking rather than signal-subspace tracking. Reuses
        # the SVD above -- no second decomposition.
        rank_diag = rank_diagnostic(S, self.k)
        self.outputs['effective_rank'].append(rank_diag['effective_rank'])
        self.outputs['sv_gap_at_k'].append(rank_diag['sv_gap_at_k'])
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

        state_dict = {
            's_pars': s_pars,
            'U_true': U_true,
            'PRX': None,
            'signal_domain': frames.DOMAIN_CFR,
            'signal_dimension': frames.DIMENSION_FULL,
        }
        state_dict.update(self._environment_state_updates())
        for stage in self.serial_stages:
            _check_frame_contract(stage, state_dict)
            before = state_dict.get('signal_domain', frames.DOMAIN_CFR)
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

        reserved_keys = {'U', 'U_true', 's_pars', 'PRX', 'frame_layout',
                         'signal_domain', 'signal_dimension', 'sensing_matrix',
                         'aperture_shape', 'tx_wave', 'adc'}

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
            for k in outputs:
                if k in reserved_keys:
                    raise ValueError(
                        f"downstream block {downstream_block} emitted reserved key {k!r}"
                    )
            state_dict.update(outputs)
        
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

    def _feed_forward_from(self, domain, payload):
        """Run a chain that starts in a domain other than the frequency domain.

        This is the replay path: a stored ADC cube injected part-way down the chain, so
        impairments or products can be re-derived without paying for ray tracing again.
        The subspace ground truth is deliberately absent -- it is a property of an
        S-parameter frame, and inventing one here would be a fiction downstream blocks
        could not distinguish from the real thing.
        """
        state_dict = {
            frames.DOMAIN_PAYLOAD_KEY.get(domain, 's_pars'): payload,
            'PRX': None,
            'signal_domain': domain,
        }
        state_dict.update(self._environment_state_updates())

        for stage in self.serial_stages:
            _check_frame_contract(stage, state_dict)
            before = state_dict.get('signal_domain', domain)
            state_dict.update(stage.apply(state_dict))
            _advance_domain(stage, state_dict, before)

        reserved_keys = {'U', 'U_true', 's_pars', 'PRX', 'frame_layout',
                         'signal_domain', 'signal_dimension', 'sensing_matrix',
                         'aperture_shape', 'tx_wave', 'adc'}
        for downstream_block in self.downstream_blocks:
            _check_frame_contract(downstream_block, state_dict)
            outputs = downstream_block.apply(state_dict)
            for output_name, output in outputs.items():
                self.outputs[output_name].append(output)
            for key in outputs:
                if key in reserved_keys:
                    raise ValueError(
                        f"downstream block {downstream_block} emitted reserved key {key!r}"
                    )
            state_dict.update(outputs)

    def get_outputs(self):
        return self.outputs

    def run(self, n_steps=10):
        self.reset()
        for i in tqdm(range(n_steps), desc='RUNNING ARRAY SIMULATION'):
            self.feed_forward()
            self.step()
        return self.get_outputs()

