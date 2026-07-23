import warnings

import torch
from tqdm import tqdm
from collections import defaultdict

from e2e import frames
from e2e.blocks import CircuitStage, GridStage, InterconnectStage, MeasurementStage


# Relative threshold (fraction of the top singular value) below which a singular value
# is treated as noise floor when computing effective rank -- see rank_diagnostic.
_RANK_RTOL = 1e-2


def _svd_frame(s_pars):
    """SVD of a single frame's flattened S-parameter matrix, computed once.

    Returns (U, S): the full left-singular-vector matrix and the singular values
    (descending). Shared by get_U_true and rank_diagnostic so callers that need both
    the top-d basis and the singular-value spectrum don't pay for two decompositions.
    """
    assert len(s_pars.shape) == 4
    s_pars_0 = s_pars[:, :, 0, :]
    s_pars_0 = s_pars_0.view(-1, s_pars_0.shape[-1])
    U, S, _ = torch.linalg.svd(s_pars_0)
    return U, S


def get_U_true(s_pars, d):
    # "Ground truth" here means the top-d left singular vectors of a single frame's
    # S-parameter matrix -- a meaningful reference for subspace tracking only when the
    # frame's effective rank (number of singular values well above the noise floor) is
    # >= d. Below that, the trailing directions returned are noise-dominated, and any
    # subspace_err computed against them partly measures how well the tracker follows
    # noise rather than signal structure. See rank_diagnostic (and Simulation.feed_forward,
    # which records it per frame) for the diagnostic that makes this failure mode visible.
    U, _ = _svd_frame(s_pars)
    return U[:, :d]


def rank_diagnostic(S, d, rtol=_RANK_RTOL):
    """Rank / singular-value-gap diagnostic for a frame's singular-value spectrum `S`
    (descending, as returned by `_svd_frame`), against a requested subspace dim `d`.

    - `effective_rank`: count of singular values above `rtol * S[0]` (default rtol
      _RANK_RTOL) -- i.e. singular values still well above the noise floor.
    - `sv_gap_at_d`: ratio S[d-1] / S[d], the singular-value gap right at the requested
      cutoff (large gap = a clean signal/noise boundary at d; near 1 = no real
      boundary there). NaN if d >= len(S) (no S[d] to compare against).
    - `rank_ok`: True iff `d <= effective_rank`, i.e. the requested subspace dim is
      supported by the frame's actual signal content.
    """
    threshold = rtol * S[0]
    effective_rank = int((S > threshold).sum().item())
    if d < len(S):
        denom = S[d]
        sv_gap_at_d = float((S[d - 1] / denom).item()) if denom > 0 else float('inf')
    else:
        sv_gap_at_d = float('nan')
    rank_ok = d <= effective_rank
    return {
        'effective_rank': effective_rank,
        'sv_gap_at_d': sv_gap_at_d,
        'rank_ok': rank_ok,
    }


def perturb_basis(U):
    U = U + 1e-3 * (torch.randn_like(U) + 1j * torch.randn_like(U))
    return torch.linalg.qr(U)[0]

class Simulation:
    def __init__(self,
        environment_block,
        downstream_blocks,
        d,
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
        self.d = d
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
        # Downstream blocks with per-run state (e.g. ModemBlock's frame-indexed
        # noise counter) expose reset(); rewind them so repeated run() calls on
        # the same Simulation are reproducible.
        for block in self.downstream_blocks:
            if hasattr(block, "reset"):
                block.reset()

    def feed_forward(self):
        s_pars = self.environment_block.get_S_pars()
        U, S = _svd_frame(s_pars)
        U_true = U[:, :self.d]

        # Rank / singular-value-gap diagnostic on the frame get_U_true saw: flags when
        # the requested subspace dim self.d exceeds the frame's effective rank, in
        # which case the trailing "ground truth" directions are noise and subspace_err
        # partly measures noise-tracking rather than signal-subspace tracking. Reuses
        # the SVD above -- no second decomposition.
        rank_diag = rank_diagnostic(S, self.d)
        self.outputs['effective_rank'].append(rank_diag['effective_rank'])
        self.outputs['sv_gap_at_d'].append(rank_diag['sv_gap_at_d'])
        self.outputs['rank_ok'].append(rank_diag['rank_ok'])
        if not rank_diag['rank_ok'] and not self._rank_warned:
            warnings.warn(
                f"requested subspace dim d={self.d} exceeds frame effective rank "
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
        }
        for stage in self.serial_stages:
            state_dict.update(stage.apply(state_dict))
        # Serial stages that touch the subspace tracker (MeasurementStage) refresh
        # 'U' via their return dict; re-read it from the tracker here too so 'U' is
        # always current. Guarded: a serial_stages override may legitimately run
        # without a subspace block, in which case 'U' is whatever the stages set.
        if self.subspace_block is not None:
            state_dict['U'] = self.subspace_block.oja.U

        reserved_keys = {'U', 'U_true', 's_pars', 'PRX'}

        for downstream_block in self.downstream_blocks:
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
        
    def get_outputs(self):
        return self.outputs

    def run(self, n_steps=10):
        self.reset()
        for i in tqdm(range(n_steps), desc='RUNNING ARRAY SIMULATION'):
            self.feed_forward()
            self.step()
        return self.get_outputs()

