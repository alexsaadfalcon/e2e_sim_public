"""
Analog aperture compression and its inverse, as standalone chain blocks.

THE ARCHITECTURE THIS MODELS
-----------------------------
A large analog receive aperture -- hundreds to a thousand elements; this package's array
is 32x32 = 1024 -- feeding a far smaller number of digitized channels, typically 16 to
64. The combining happens in the ANALOG domain and the compressed channels are what get
digitized, so compression and digitization are one operation on the hardware: you never
build 1024 converters. That `N/M` of 16x to 64x is the whole point, and it is why
DIGITAL compression is not modelled here -- if every element has already been digitized,
the converters have been paid for and compressing afterwards saves only data rate.

Whether that compression happens is an architectural choice independent of how many bits
each converter has, which is why compression and quantization are two blocks rather than
one: `CompressBlock` feeding a quantizer digitizes `M` combined channels, a quantizer
alone digitizes all `N` elements.

The consequence propagates. After compression the chain carries `M` measurements in an
arbitrary basis, and `frames.DIMENSION_REDUCED` says so. Some algorithms are content
there -- a subspace tracker estimates the same subspace from random projections, which
is the entire premise of the AFE -- while anything that indexes physical elements (an
angle FFT, a beamformer, an aperture image) is meaningless applied to projections and
must be preceded by `DecompressBlock`. Before this contract existed the pipeline
reconstructed unconditionally and immediately, so reduced-dimension processing could
not be expressed at all and the reconstruction cost was paid even when nothing needed it.

ANALOG, NOT DIGITAL -- AND WHAT THAT DOES NOT YET BUY
------------------------------------------------------
The compression modelled here happens BEFORE digitization. That is why the sensing
matrix itself is quantized (`weight_bits`): its entries are physical analog combining
weights -- attenuator/phase-shifter settings -- realizable only to finite precision, and
that imprecision is part of the measurement, not a rounding of the data.

CONTROL x COMPUTE: TWO ORTHOGONAL AXES
---------------------------------------
Compression FIDELITY (how `y = A x` is realized) and CONTROL (how `A` is chosen) are
independent design choices, and this module keeps them that way: `CONTROL_STATIC` /
`CONTROL_ADAPTIVE` cross with `COMPUTE_IDEAL` / `COMPUTE_QUANTIZED` / `COMPUTE_NOISY` (see
the constants below `CompressBlock` -- an ideal full-VMM `y = A x`, a quantized-weight
combining network at a chosen bit depth/model, or a noisy compute-in-memory array, each
with either a fixed matrix or one redrawn from a subspace tracker). `CompressBlock` is the
one primitive exposing both axes; `e2e.blocks.AFEBlock` is a thin preset of it (adaptive
control -- `MeasurementStage` supplies a fresh `A` on every call -- with quantized compute
at the float/compute-datapath weight model), sharing `realize_weights`/`combine_realized`
rather than a second implementation.

**ORDERING CONSTRAINT, and it is a physical one, not an implementation gap.** Two steps
in the receive chain index PHYSICAL ANTENNAS and are therefore meaningless once dim 0
counts linear combinations of them:

* `chain.dechirp.beat_from_cfr` reverses the RX/TX antenna axes to fix array handedness
  -- a property of the ARRAY, so it belongs to the aperture and must be settled before
  the aperture is combined away;
* `chain.dechirp.mimo_combine` applies a per-TX code down dim 1, which after compression
  is not a TX axis at all.

`DechirpBlock` therefore declares `DIMENSION_FULL`, and `CompressBlock -> DechirpBlock`
raises a `FrameContractError` rather than silently producing a plausible wrong cube --
the contract catching its own author. The fix is ordering, not a missing feature:
resolve array handedness and TX de-multiplexing while the antenna axes still mean
something, and compress after. Compression then feeds consumers that are basis-agnostic,
above all the subspace tracker -- which is exactly what the AFE does, via
`MeasurementStage(reconstruct=False)`.

STATUS, stated plainly rather than implied: `AFEBlock`/`MeasurementStage` (the adaptive
variant) IS reachable and exercised end to end. `CompressBlock`/`DecompressBlock` (the
static-matrix variant) are NOT yet constructed by the webapp or by any script under
`e2e/main/` -- they are infrastructure ahead of use, kept because the static combining
network is the architecture the AFE's adaptive draw is compared against. Do not read
their presence as evidence that a static-compression pipeline has been run.

RECONSTRUCTION IS LOSSY AND SAYS SO
------------------------------------
`DecompressBlock` applies the Moore-Penrose pseudo-inverse of the (quantized) sensing
matrix. For `M < N` that is a minimum-norm least-squares solution, NOT an inverse: it
recovers the component of the aperture lying in the sensing matrix's row space and sets
the rest to zero. It is exact only when `M >= N` and the matrix is well conditioned.
Compressed-sensing reconstruction under a sparsity prior would do better on sparse
scenes; that is deliberately not attempted here, because the honest default is the
linear solution whose failure mode is easy to state.
"""

from __future__ import annotations

import torch

from e2e import frames
from e2e.frames import (
    CHIRP_NATIVE,
    DIMENSION_FULL,
    DIMENSION_REDUCED,
    DOMAIN_CFR,
    FrameCapabilities,
    FrameContractError,
)


#: Weight-quantization models. These describe DIFFERENT HARDWARE and are not
#: interchangeable, which is why the choice is explicit rather than a default.
WEIGHT_UNIFORM = "uniform"   # analog control settings: constant ABSOLUTE step
WEIGHT_FLOAT = "float"       # digital compute datapath: constant RELATIVE error


#: CONTROL axis: how the sensing matrix `A` is CHOSEN -- orthogonal to how `y = A x` is
#: REALIZED once chosen (the COMPUTE axis, below). `CompressBlock` composes one of each.
#: `CONTROL_STATIC` (default) draws once (or takes `generator`'s draw once) and caches
#: it, modelling a fixed analog combining network. `CONTROL_ADAPTIVE` redraws on every
#: call instead of caching. This module does not itself know how to steer a draw onto a
#: signal subspace -- that lives in `e2e.subspace`, deliberately not imported here --
#: so adaptive control needs a `generator` that does, e.g.
#: `lambda m, n: subspace_block.gen_A_ada(m)`; `e2e.blocks.AFEBlock`/`MeasurementStage`
#: are exactly that (a fresh `A` supplied externally on every call), just not expressed
#: as a `CompressBlock` instance.
CONTROL_STATIC = "static"
CONTROL_ADAPTIVE = "adaptive"

#: COMPUTE axis: how `y = A x` is REALIZED once `A` is chosen -- the fidelity of the
#: analog combining hardware, independent of how `A` was picked. Three non-interchangeable
#: hardware models:
#: `COMPUTE_IDEAL` -- full-precision weights, exact matmul, no noise. The reference
#: `y = A x` every other mode is a non-ideal model OF (the "full VMM" case).
#: `COMPUTE_QUANTIZED` (default, matches this module's historical behaviour) -- weights
#: realized at finite precision (`quantize_weights`; `weight_bits`/`weight_model` select
#: uniform-analog vs float-datapath), then an exact matmul: one non-ideal AFE
#: architecture (a quantized combining network).
#: `COMPUTE_NOISY` -- full-precision weights, but the matmul itself is corrupted by
#: multiplicative output noise (`e2e.afe.afe_utils.approx_matmul`): a DIFFERENT non-ideal
#: architecture (a noisy compute-in-memory array rather than a quantized-weight network).
#: The two non-ideal modes are not combined by default -- they model different hardware,
#: not different severities of the same hardware.
COMPUTE_IDEAL = "ideal"
COMPUTE_QUANTIZED = "quantized"
COMPUTE_NOISY = "noisy"

_COMPUTE_MODES = (COMPUTE_IDEAL, COMPUTE_QUANTIZED, COMPUTE_NOISY)


def combine(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply a `[M, N]` sensing matrix to a flattened `[N, ...]` aperture.

    Trivial on its own -- it exists so that the analog-combining step has ONE
    implementation shared by `CompressBlock` (static matrix) and `e2e.blocks.AFEBlock`
    (matrix redrawn per frame from the subspace tracker). Those two differ in how the
    matrix is CHOSEN and how its weights are quantized, not in what combining means.
    """
    return a @ v


def reconstruct_aperture(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Minimum-norm least-squares aperture from measurements `x` and weights `a`.

    Exact only when `M >= N` and `a` is well conditioned; for `M < N` it recovers the
    component of the aperture in `a`'s row space and zeroes the rest. Shared by
    `DecompressBlock` and `AFEBlock.reconstruct` so there is one reconstruction.
    """
    return torch.linalg.pinv(a) @ x


def quantize_weights(a: torch.Tensor, bits: int = None, *, model: str = WEIGHT_UNIFORM,
                     exp: int = 5, mantissa: int = 6) -> torch.Tensor:
    """Quantize complex combining weights under one of two hardware models.

    `model=WEIGHT_UNIFORM` (default, `bits`) is the analog case: the weights are
    attenuator/phase-shifter settings with a constant step, so the error floor is
    constant in ABSOLUTE terms and small weights are hit proportionally hardest. That is
    what a physical combining network does, and it is the model for `CompressBlock`.

    `model=WEIGHT_FLOAT` (`exp`/`mantissa`) is the digital case: a low-precision
    floating-point format whose error is roughly constant in RELATIVE terms. That is
    right for a compute datapath and wrong for an analog control, and it is what
    `e2e.blocks.AFEBlock` has always used. The two are kept distinct deliberately --
    swapping them would flatter or penalise weak measurements by construction.

    Below, the uniform case.

    Real and imaginary parts are quantized INDEPENDENTLY against a shared full scale
    (the max absolute part over the whole matrix), matching how an IQ combining network
    is actually set: two real controls per weight, sharing one hardware range. Uniform,
    not floating-point, for the same reason `chain.receive.QuantizerBlock` is: a physical
    control has a constant step size, so its error floor is constant in ABSOLUTE terms.

    NOTE the deliberate difference from that ADC quantizer: its step is
    `full_scale / 2^(bits-1)`, which leaves `+full_scale` unrepresentable and CLIPS it --
    correct there, because an ADC really does clip a signal that exceeds its range. Here
    there is no signal to clip. These are control settings we are free to scale, so the
    step is `full_scale / (2^(bits-1) - 1)`, placing the largest weight exactly on the
    top code. Every weight is then within half an LSB, with no code wasted and no
    spurious clamp on the extreme.
    """
    if model == WEIGHT_FLOAT:
        # Imported lazily: e2e.afe is the AFE's own package and pulling it in at module
        # scope would tie this chain module to it for a branch most callers never take.
        from e2e.afe.afe_utils import quantizer_fp

        return torch.complex(quantizer_fp(a.real, exp, mantissa),
                             quantizer_fp(a.imag, exp, mantissa))
    if model != WEIGHT_UNIFORM:
        raise ValueError(
            f"model must be {WEIGHT_UNIFORM!r} or {WEIGHT_FLOAT!r}, got {model!r}")
    if bits is None:
        return a
    if bits < 2:
        raise ValueError(f"weight_bits must be >= 2 or None, got {bits}")
    full_scale = torch.max(torch.abs(torch.view_as_real(a)))
    if full_scale == 0:
        return a
    levels = 2 ** (bits - 1) - 1
    lsb = full_scale / levels
    re = torch.round(a.real / lsb).clamp_(-levels, levels) * lsb
    im = torch.round(a.imag / lsb).clamp_(-levels, levels) * lsb
    return torch.complex(re, im)


def realize_weights(a: torch.Tensor, compute: str = COMPUTE_QUANTIZED, *,
                    weight_bits: int = 8, weight_model: str = WEIGHT_UNIFORM,
                    exp: int = 5, mantissa: int = 6) -> torch.Tensor:
    """Apply the COMPUTE axis's weight realization to `a`.

    Identity for `COMPUTE_IDEAL` and `COMPUTE_NOISY` -- the former is full precision by
    definition, the latter's non-ideality lives in the matmul (`combine_realized`), not
    in the stored weight setting. `COMPUTE_QUANTIZED` delegates to `quantize_weights`
    (`weight_bits=8`/`weight_model=WEIGHT_UNIFORM` reproduces this module's historical
    default exactly).
    """
    if compute == COMPUTE_QUANTIZED:
        return quantize_weights(a, bits=weight_bits, model=weight_model, exp=exp,
                                mantissa=mantissa)
    if compute in (COMPUTE_IDEAL, COMPUTE_NOISY):
        return a
    raise ValueError(f"compute must be one of {_COMPUTE_MODES}, got {compute!r}")


def combine_realized(a: torch.Tensor, v: torch.Tensor, compute: str = COMPUTE_QUANTIZED,
                     *, noise_settings=None) -> torch.Tensor:
    """Apply the COMPUTE axis's combine step.

    Exact `a @ v` (`combine`) for `COMPUTE_IDEAL`/`COMPUTE_QUANTIZED` -- their
    non-ideality, if any, already lives in `a` via `realize_weights`. `COMPUTE_NOISY`
    instead runs `e2e.afe.afe_utils.approx_matmul` (multiplicative output noise,
    `noise_settings` defaulting to that module's `matmul_noise_settings`) on IDEAL
    weights -- a compute-in-memory architecture, not a quantized one.
    """
    if compute == COMPUTE_NOISY:
        # Imported lazily for the same reason `quantize_weights`(WEIGHT_FLOAT) imports
        # e2e.afe lazily: most callers never take this branch.
        from e2e.afe.afe_utils import approx_matmul

        return approx_matmul(a, v, noise_settings)
    if compute in (COMPUTE_IDEAL, COMPUTE_QUANTIZED):
        return combine(a, v)
    raise ValueError(f"compute must be one of {_COMPUTE_MODES}, got {compute!r}")


class CompressBlock:
    """Analog aperture compression: `M < N` quantized-weight linear measurements.

    Serial stage. Consumes a full-dimension frequency-domain frame and rewrites
    `s_pars` as `[M, 1, n_chirp, n_freqs]`, where dim 0 now indexes MEASUREMENTS rather
    than antennas. Publishes the realized sensing matrix as `state['sensing_matrix']`
    so `DecompressBlock` (or a subspace tracker) can use the weights that were actually
    applied, quantization included, rather than the ideal ones.

    `n_measurements` is `M`. Two orthogonal axes, both above:

    * `control` (`CONTROL_STATIC` default / `CONTROL_ADAPTIVE`) -- how `A` is CHOSEN.
      Static caches the first draw; adaptive redraws every call from `generator`
      (required in that mode), e.g. `lambda m, n: subspace_block.gen_A_ada(m)` to
      reproduce what `AFEBlock`/`MeasurementStage` do.
    * `compute` (`COMPUTE_IDEAL` / `COMPUTE_QUANTIZED` default / `COMPUTE_NOISY`) -- how
      `y = A x` is REALIZED. Quantized takes `weight_bits`/`weight_model` (default 8-bit
      uniform, this class's historical-and-only behaviour before this axis existed) or
      `exp`/`mantissa` for the float model; noisy takes `noise_settings`
      (`e2e.afe.afe_utils.matmul_noise_settings` if omitted).

    `weight_bits`/`generator`/`seed` are the pre-existing constructor surface and keep
    their historical meaning and defaults; `weight_bits=8` + `compute=COMPUTE_QUANTIZED`
    (the defaults) reproduce this class's behaviour before `compute` existed, bit-exactly.
    Publishes the realized sensing matrix as `state['sensing_matrix']` so `DecompressBlock`
    (or a subspace tracker) can use the weights that were actually applied, non-ideality
    included, rather than the ideal ones.
    """

    frame_capabilities = FrameCapabilities(
        accepts_mimo=True,
        chirps=CHIRP_NATIVE,
        domain=DOMAIN_CFR,
        dimension=DIMENSION_FULL,
        emits_dimension=DIMENSION_REDUCED,
    )

    def __init__(self, n_measurements: int, *, weight_bits: int = 8, generator=None,
                 seed: int = 0, control: str = CONTROL_STATIC,
                 compute: str = COMPUTE_QUANTIZED, weight_model: str = WEIGHT_UNIFORM,
                 exp: int = 5, mantissa: int = 6, noise_settings=None):
        if n_measurements < 1:
            raise ValueError(f"n_measurements must be >= 1, got {n_measurements}")
        if control not in (CONTROL_STATIC, CONTROL_ADAPTIVE):
            raise ValueError(
                f"control must be {CONTROL_STATIC!r} or {CONTROL_ADAPTIVE!r}, got {control!r}"
            )
        if control == CONTROL_ADAPTIVE and generator is None:
            raise ValueError(
                "control='adaptive' redraws A on every call and this module does not "
                "itself know how to steer that draw (see e2e.subspace) -- pass a "
                "`generator(m, n)` that does, e.g. a subspace tracker's gen_A_ada."
            )
        self.n_measurements = int(n_measurements)
        self.weight_bits = weight_bits
        self.generator = generator
        self.seed = int(seed)
        self.control = control
        self.compute = compute
        self.weight_model = weight_model
        self.exp = exp
        self.mantissa = mantissa
        self.noise_settings = noise_settings
        self._a = None          # cached realized sensing matrix (CONTROL_STATIC only)

    def sensing_matrix(self, n_elements: int, device, dtype) -> torch.Tensor:
        """The realized `[M, N]` combining matrix.

        `control=CONTROL_STATIC` (default): drawn once and cached, because a static
        analog network has ONE set of weights -- redrawing per frame would model a
        different (and much more capable) architecture, and would also make the
        subspace tracker's job artificially easy. `control=CONTROL_ADAPTIVE`: redrawn
        (from `generator`) and realized fresh on every call, never cached.
        """
        if self.control == CONTROL_STATIC and self._a is not None \
                and self._a.shape[1] == n_elements:
            return self._a.to(device=device, dtype=dtype)
        m = self.n_measurements
        if self.generator is not None:
            a = self.generator(m, n_elements)
        else:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            # Unit-variance complex Gaussian, scaled so E[||A x||^2] == ||x||^2: each
            # entry has variance 1/M across the N summed terms.
            real = torch.randn(m, n_elements, generator=g)
            imag = torch.randn(m, n_elements, generator=g)
            a = torch.complex(real, imag) / (2.0 * m) ** 0.5
        a = realize_weights(a.to(torch.complex64), self.compute, weight_bits=self.weight_bits,
                            weight_model=self.weight_model, exp=self.exp,
                            mantissa=self.mantissa)
        if self.control == CONTROL_STATIC:
            self._a = a
        return a.to(device=device, dtype=dtype)

    def apply(self, state):
        s_pars = state["s_pars"]
        n_rx, n_tx, n_chirp, n_freqs = s_pars.shape
        n_elements = n_rx * n_tx
        if self.n_measurements > n_elements:
            raise FrameContractError(
                f"CompressBlock: n_measurements={self.n_measurements} exceeds the "
                f"aperture's {n_elements} elements ({n_rx} rx x {n_tx} tx) -- that is an "
                f"expansion, not a compression; reduce n_measurements."
            )
        a = self.sensing_matrix(n_elements, s_pars.device, s_pars.dtype)
        # Flatten the aperture, combine, and restore the frame's [*, 1, chirp, freq]
        # rank so downstream shape handling is unchanged apart from dim 0's meaning.
        v = s_pars.reshape(n_elements, n_chirp * n_freqs)
        x = combine_realized(a, v, self.compute, noise_settings=self.noise_settings)
        x = x.reshape(self.n_measurements, 1, n_chirp, n_freqs)
        return {
            "s_pars": x,
            "sensing_matrix": a,
            "signal_dimension": DIMENSION_REDUCED,
            "aperture_shape": (n_rx, n_tx),
        }


class DecompressBlock:
    """Least-squares reconstruction back to the full aperture. The dimension bridge.

    Serial stage. Consumes reduced-dimension `s_pars` plus `state['sensing_matrix']`
    and rewrites `s_pars` at the original `[n_rx, n_tx, n_chirp, n_freqs]` shape.

    See the module docstring: for `M < N` this is minimum-norm least squares, not an
    inverse, and it recovers only the row-space component of the aperture. Placing this
    block is therefore a modelling decision with a cost, which is exactly why it is
    explicit rather than automatic.
    """

    frame_capabilities = FrameCapabilities(
        accepts_mimo=True,
        chirps=CHIRP_NATIVE,
        domain=DOMAIN_CFR,
        dimension=DIMENSION_REDUCED,
        emits_dimension=DIMENSION_FULL,
    )

    def apply(self, state):
        x = state["s_pars"]
        a = state.get("sensing_matrix")
        if a is None:
            raise FrameContractError(
                "DecompressBlock: state has no 'sensing_matrix' -- it can only follow a "
                "CompressBlock, which publishes the weights that were actually applied."
            )
        shape = state.get("aperture_shape")
        if shape is None:
            raise FrameContractError(
                "DecompressBlock: state has no 'aperture_shape' -- CompressBlock records "
                "the aperture it collapsed so this block can restore it."
            )
        n_rx, n_tx = shape
        m, _, n_chirp, n_freqs = x.shape
        v = torch.linalg.pinv(a) @ x.reshape(m, n_chirp * n_freqs)
        return {
            "s_pars": v.reshape(n_rx, n_tx, n_chirp, n_freqs),
            "signal_dimension": DIMENSION_FULL,
        }
