"""
Analog aperture compression and its inverse, as standalone chain blocks.

WHY THESE ARE SEPARATE BLOCKS
------------------------------
Digitizing an N-element aperture needs N converters. The alternative this package
models is to combine the aperture in the ANALOG domain first -- `M < N` weighted sums,
each digitized by one converter -- so the ADC count, and every downstream data rate,
drops by `N/M`. Whether that compression happens is an architectural choice independent
of how many bits the converter has, which is why compression and quantization are two
blocks rather than one: `CompressBlock` -> `QuantizerBlock` digitizes `M` compressed
measurements, `QuantizerBlock` alone digitizes all `N` elements.

The consequence propagates. After compression the chain carries `M` measurements in an
arbitrary basis, and `frames.DIMENSION_REDUCED` says so. Some algorithms are content
there -- a subspace tracker estimates the same subspace from random projections, which
is the entire premise of the AFE -- while anything that indexes physical elements (an
angle FFT, a beamformer, an aperture image) is meaningless applied to projections and
must be preceded by `DecompressBlock`. Before this contract existed the pipeline
reconstructed unconditionally and immediately, so reduced-dimension processing could
not be expressed at all and the reconstruction cost was paid even when nothing needed it.

ANALOG, NOT DIGITAL
--------------------
The compression modelled here happens BEFORE digitization. That is why the sensing
matrix itself is quantized (`weight_bits`): its entries are physical analog combining
weights -- attenuator/phase-shifter settings -- realizable only to finite precision, and
that imprecision is part of the measurement, not a rounding of the data. The signal
being combined is still analog and unquantized at this point; `QuantizerBlock` digitizes
what comes out. (If you instead want digital dimensionality reduction after the ADC,
place this block after the quantizer and set `weight_bits=None` -- but then it is a data
compression scheme, not a converter-count saving, and it cannot reduce ADC count.)

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


def quantize_weights(a: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform mid-tread quantization of complex combining weights to `bits` bits.

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


class CompressBlock:
    """Analog aperture compression: `M < N` quantized-weight linear measurements.

    Serial stage. Consumes a full-dimension frequency-domain frame and rewrites
    `s_pars` as `[M, 1, n_chirp, n_freqs]`, where dim 0 now indexes MEASUREMENTS rather
    than antennas. Publishes the realized sensing matrix as `state['sensing_matrix']`
    so `DecompressBlock` (or a subspace tracker) can use the weights that were actually
    applied, quantization included, rather than the ideal ones.

    `n_measurements` is `M`. `weight_bits` is the analog combining-weight resolution
    (None = ideal weights). `generator` optionally supplies the matrix for a given
    `(M, N)` -- pass the AFE's adaptive draw here to reproduce the adaptive behaviour;
    the default is a fixed random Gaussian ensemble drawn once and reused across frames,
    which is what a static analog combining network does.
    """

    frame_capabilities = FrameCapabilities(
        accepts_mimo=True,
        chirps=CHIRP_NATIVE,
        domain=DOMAIN_CFR,
        dimension=DIMENSION_FULL,
        emits_dimension=DIMENSION_REDUCED,
    )

    def __init__(self, n_measurements: int, *, weight_bits: int = 8, generator=None,
                 seed: int = 0):
        if n_measurements < 1:
            raise ValueError(f"n_measurements must be >= 1, got {n_measurements}")
        self.n_measurements = int(n_measurements)
        self.weight_bits = weight_bits
        self.generator = generator
        self.seed = int(seed)
        self._a = None          # cached realized (quantized) sensing matrix

    def sensing_matrix(self, n_elements: int, device, dtype) -> torch.Tensor:
        """The realized `[M, N]` combining matrix, drawn once and cached.

        Cached because a static analog network has ONE set of weights: redrawing per
        frame would model a different (and much more capable) architecture, and would
        also make the subspace tracker's job artificially easy.
        """
        if self._a is not None and self._a.shape[1] == n_elements:
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
        a = quantize_weights(a.to(torch.complex64), self.weight_bits)
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
        x = (a @ v).reshape(self.n_measurements, 1, n_chirp, n_freqs)
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
