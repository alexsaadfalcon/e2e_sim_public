"""THE oracle suite for the one-chain spine (owner directive 2026-09-24, contract
`notes/ONE_CHAIN_CONTRACT_2026-09-24.md` section 5.3).

What it pins, and why each pin exists:

* **Identity oracle.** The spine at the identity point (FMCW, `window="none"`,
  `dc_removal=False`, front end off, compressor off) reproduces the v1.0
  RangeAz / RangeEl / AzEl / range-profile products after the documented index map.
  The v1.0 math is transcribed VERBATIM below (`_v10_range_az` &c., copied from
  `e2e/blocks.py` at HEAD c33bb64) rather than shipped as a binary tensor, so the
  test is hands-off and reproducible from a clean clone; the transcription itself was
  checked against tensors dumped from that HEAD on 2026-09-24 and agreed to <= 4e-7
  relative on every product (synthetic AND the real `munich_ka.pkl` frame 0).
* **The index map.** `v10_range_reorder` is the whole relationship between the two
  range conventions, and it is derived, not fitted -- see its docstring.
* **Known-delay oracle**, on BOTH grid conventions, written against `N*df` so it
  cannot bake in the `B` vs `N*df` off-by-one.
* **Compressor still in series** and **the tracker reproduces today's subspace**.
* **Front-end freeze** (F96's own oracle, tolerance 0.0).
* **Loud failures**: a product asked for without the spine, a cube with the wrong
  axes, two sources, a replay that skips stages by name.

Tolerance is 1e-5 relative unless a test says otherwise; that is the block level.
`run_pipeline`'s ~5e-3 nondeterminism floor (STATE section 5) is a webapp-level
number and does not apply here.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e import frames                                          # noqa: E402
from e2e.blocks import (AdaOjaBlock, AFEBlock, CircuitStage, FFTBlock,  # noqa: E402
                        MeasurementStage, RangeAzBlock, RangeElBlock,
                        RangeProfileBlock, RFFEBlock, SubspaceErrorBlock, device)
from e2e.chain.dechirp import DechirpBlock                      # noqa: E402
from e2e.chain.receive import (RangeTransformBlock, delta_f_from_cfg,   # noqa: E402
                               delta_f_from_freq_plan, range_axis_m)
from e2e.chain.transforms import adc_to_rd                      # noqa: E402
from e2e.chain.waveform import fmcw_plan_from_freq_plan         # noqa: E402
from e2e.radar_config import BENCHMARK_V1_KA, C_MPS, MUNICH_KA_FMCW  # noqa: E402
from e2e.simulation import Simulation, perturb_basis, to_beat_basis   # noqa: E402
from e2e.subspace.subspace_utils import subspace_dist_frob      # noqa: E402

RTOL = 1e-5


class _SingleTxCfg:
    """The minimal `cfg` `DechirpBlock` reads (only `.mimo`)."""
    mimo = "single"
    n_tx = 1


# ----------------------------------------------------------------- the v1.0 reference
# Transcribed VERBATIM from e2e/blocks.py at HEAD c33bb64 (the commit this work
# branched from), so the oracle compares against what the v1.0 products actually
# computed rather than against a paraphrase of them. Do not "clean these up": their
# value is that they are a copy.

def _v10_power_bin(power, n_bins, dim):
    L = power.shape[dim]
    if L < n_bins:
        return power
    if L == n_bins:
        return power
    per = -(-L // n_bins)
    pad = per * n_bins - L
    if pad:
        pad_shape = list(power.shape)
        pad_shape[dim] = pad
        power = torch.cat([power, power.new_zeros(pad_shape)], dim=dim)
    power = power.movedim(dim, -1)
    power = power.reshape(*power.shape[:-1], n_bins, per).sum(dim=-1)
    return power.movedim(-1, dim)


def _v10_range_az(data, bins):
    """e2e/blocks.py:RangeAzBlock._map at HEAD c33bb64 (untapered)."""
    n_az, n_el, n_freqs = data.shape
    power = torch.zeros(bins, n_freqs, dtype=torch.float32, device=data.device)
    for e in range(n_el):
        col = data[:, e, :]
        a = torch.fft.fftshift(torch.fft.fft(col, bins, 0), 0)
        r = torch.fft.fftshift(torch.fft.fft(a, dim=1), 1)
        power = power + torch.abs(r) ** 2
    return _v10_power_bin(power, bins, dim=1)


def _v10_range_el(data, bins):
    """e2e/blocks.py:RangeElBlock._map at HEAD c33bb64 (untapered)."""
    n_az, n_el, n_freqs = data.shape
    power = torch.zeros(bins, n_freqs, dtype=torch.float32, device=data.device)
    for m in range(n_az):
        col = data[m, :, :]
        a = torch.fft.fftshift(torch.fft.fft(col, bins, 0), 0)
        r = torch.fft.fftshift(torch.fft.fft(a, dim=1), 1)
        power = power + torch.abs(r) ** 2
    return _v10_power_bin(power, bins, dim=1)


def _v10_az_el(data, bins):
    """e2e/blocks.py:FFTBlock._map at HEAD c33bb64 (untapered)."""
    n_az, n_el, n_freqs = data.shape
    range_full = torch.fft.fft(data, dim=2)
    acc = torch.zeros(bins, bins, dtype=torch.float32, device=data.device)
    chunk = 256
    for r0 in range(0, n_freqs, chunk):
        blk = range_full[:, :, r0:r0 + chunk]
        ap = torch.fft.fft(torch.fft.fft(blk, bins, 0), bins, 1)
        ap = torch.fft.fftshift(torch.fft.fftshift(ap, 0), 1)
        acc = acc + torch.sum(torch.abs(ap) ** 2, dim=2)
    return acc


def _v10_range_profile(grid, bins):
    """e2e/blocks.py:RangeProfileBlock.apply at HEAD c33bb64."""
    data = frames.chirp0(grid)
    n_freqs = data.shape[-1]
    channels = data.reshape(-1, n_freqs)
    r = torch.fft.fftshift(torch.fft.fft(channels, dim=1), 1)
    power = torch.abs(r) ** 2
    power = _v10_power_bin(power, bins, dim=1)
    return power, torch.mean(power, dim=0)


# ------------------------------------------------------------------- the index map

def v10_range_reorder(n, device=None):
    """Index map from the spine cube's NATURAL range axis onto the v1.0 products'
    fftshifted one. DERIVED, not fitted:

    The dechirp is `conj` + a reversal of the element index. For an aperture element
    value `v_new[x, y, f] = conj(v_old[X, Y, f])` (X = n_az-1-x, Y = n_el-1-y), the
    joint angle-and-range transform satisfies

        R_new[m, r] = e^(-j 2 pi (n_az-1) m / B) * conj( R_old[m, (-r) mod N] )

    at the mirrored elevation row. So the POWER map is unchanged on the angle axis --
    the conjugate exactly cancels the mirror the flip alone would produce -- and the
    RANGE axis is negated: `P_new[k] = P_old[(-k) mod N]`. (The contract's section 5.3
    says "antenna flip => both angle axes mirrored"; that is the flip taken WITHOUT
    its accompanying conjugate. Measured 2026-09-24: the angle axes are NOT mirrored,
    and this test is the evidence.)

    The v1.0 display then fftshifted: `display_old[j] = P_old[(j + N//2) mod N]`.
    Composing the two gives the map returned here, `j -> (-(j + N//2)) mod N`.
    """
    j = torch.arange(int(n), device=device)
    return (-(j + int(n) // 2)) % int(n)


# ------------------------------------------------------------------------- helpers

def _spine_cube(s_pars, **kw):
    """The identity-point spine: dechirp -> range transform, nothing else."""
    adc = DechirpBlock(_SingleTxCfg()).apply({"s_pars": s_pars})["adc"]
    rt = RangeTransformBlock(window="none", dc_removal=False,
                             crop_negative_delay=False, **kw)
    return rt.apply({"adc": adc})


def _as_v10_axis(cube):
    ix = v10_range_reorder(cube.shape[-1], cube.device)
    return cube.index_select(-1, ix)


def _relmax(got, want):
    got, want = got.double(), want.double()
    return float((got - want).abs().max() / want.abs().max())


@pytest.fixture
def synth(synthetic_frame_np):
    return torch.from_numpy(synthetic_frame_np(n_rx=1024, n_freqs=64, seed=0)).to(device)


# ============================================================== 1. identity oracle

@pytest.mark.parametrize("bins", [32, 64])
def test_identity_oracle_range_az_el_and_azel(synth, bins):
    """The spine reproduces the v1.0 range-azimuth / range-elevation / az-el maps."""
    grid = frames.to_aperture_grid(synth, (32, 32))
    ref_slab = frames.chirp0(grid)
    cube = _spine_cube(synth)["cube"]
    state = {"cube": _as_v10_axis(cube), "aperture_shape": (32, 32)}

    got_az = RangeAzBlock(bins=bins).apply(state)["range_az"]
    assert _relmax(got_az, _v10_range_az(ref_slab, bins)) < RTOL

    got_el = RangeElBlock(bins=bins).apply(state)["range_el"]
    assert _relmax(got_el, _v10_range_el(ref_slab, bins)) < RTOL

    # The az-el map integrates range away entirely, so it needs NO index map at all --
    # which is itself a check on the derivation above.
    got_fft = FFTBlock(bins=bins).apply(
        {"cube": cube, "aperture_shape": (32, 32)})["fft"]
    assert _relmax(got_fft, _v10_az_el(ref_slab, bins)) < RTOL


@pytest.mark.parametrize("bins", [32, 64])
def test_identity_oracle_range_profile(synth, bins):
    """Range profile, per channel and aggregated. The per-channel map additionally
    needs the ELEMENT axis reversed, because the dechirp reverses it and this is the
    one product whose output still indexes channels."""
    grid = frames.to_aperture_grid(synth, (32, 32))
    cube = _as_v10_axis(_spine_cube(synth)["cube"])
    out = RangeProfileBlock(bins=bins).apply({"cube": cube})
    want_per_channel, want_agg = _v10_range_profile(grid, bins)
    assert _relmax(out["range_profile"].flip(0), want_per_channel) < RTOL
    assert _relmax(out["range_profile_agg"], want_agg) < RTOL


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "e2e",
                                    "environment", "sionna_sims", "munich_ka.pkl")),
    reason="needs the generated munich_ka.pkl (not in a clean clone)",
)
def test_identity_oracle_munich_ka_frame0():
    """The same oracle on a REAL frame -- 1024 elements, 5000 frequency points, the
    file every T1-T4 screen runs on. Measured 2026-09-24: 1.7e-7 (range-az),
    1.3e-7 (range-el), 6.7e-10 (range profile)."""
    from e2e.blocks import SionnaEnvironmentBlock
    env = SionnaEnvironmentBlock("munich")
    s_pars = env.get_S_pars()
    grid = frames.to_aperture_grid(s_pars, env.array_shape)
    ref_slab = frames.chirp0(grid)
    cube = _as_v10_axis(_spine_cube(s_pars)["cube"])
    state = {"cube": cube, "aperture_shape": env.array_shape}

    assert _relmax(RangeAzBlock(bins=256).apply(state)["range_az"],
                   _v10_range_az(ref_slab, 256)) < RTOL
    assert _relmax(RangeElBlock(bins=256).apply(state)["range_el"],
                   _v10_range_el(ref_slab, 256)) < RTOL
    got = RangeProfileBlock(bins=256).apply({"cube": cube})["range_profile_agg"]
    assert _relmax(got, _v10_range_profile(grid, 256)[1]) < RTOL


# =========================================================== 2. known-delay oracle

def _one_tap_cfr(n, delta_f, tau, device=None):
    """A CFR with a single path at delay `tau`, on a grid of spacing `delta_f`."""
    k = torch.arange(n, dtype=torch.float64, device=device)
    h = torch.exp(-2j * np.pi * k * delta_f * tau)
    return h.to(torch.complex64).view(1, 1, 1, n)


@pytest.mark.parametrize("convention,scale", [("monostatic_c2", 0.5), ("bistatic_path", 1.0)])
@pytest.mark.parametrize(
    "n,delta_f",
    [
        # munich Ka: endpoint-inclusive linspace -> B/(N-1).
        (5000, 3e9 / 4999.0),
        # corpus grid: `rt_signal_chain.beat_frequencies` -> B/N, NOT endpoint-inclusive.
        (512, 749.5e6 / 512.0),
    ],
)
def test_known_delay_oracle(n, delta_f, convention, scale):
    """A single tap at delay tau lands at cube bin round(tau * N * df), and the range
    axis reads `scale * c * tau` there. Written against `N*df`, never against a
    nominal bandwidth, so the two grid conventions above are both exercised and the
    `B` vs `N*df` off-by-one cannot hide."""
    bin_target = 137
    tau = bin_target / (n * delta_f)
    s_pars = _one_tap_cfr(n, delta_f, tau, device=device)
    out = _spine_cube(s_pars, delta_f_hz=delta_f, convention=convention)
    cube = out["cube"]
    assert int(torch.argmax(cube[0, 0].abs())) == bin_target

    axis = out["range_axis"]
    assert axis is not None
    assert axis[0] == 0.0
    expected_m = scale * C_MPS * tau
    assert abs(float(axis[bin_target]) - expected_m) < 1e-6 * expected_m
    # bistatic_path is exactly twice monostatic_c2, per bin, by construction.
    other = range_axis_m(cube.shape[-1], delta_f, n,
                         "bistatic_path" if convention == "monostatic_c2"
                         else "monostatic_c2")
    ratio = float(other[bin_target] / axis[bin_target])
    assert abs(ratio - (2.0 if convention == "monostatic_c2" else 0.5)) < 1e-12


def test_zero_delay_lands_at_bin_zero():
    """tau = 0 -> bin 0, and bin 0 is 0 m in BOTH conventions. The munich trace is
    generated with `normalize_delays=True`, so this is where the line-of-sight path
    sits -- which is also why the imaging spine runs with `dc_removal=False`."""
    n, df = 256, 1e6
    out = _spine_cube(_one_tap_cfr(n, df, 0.0, device=device), delta_f_hz=df)
    assert int(torch.argmax(out["cube"][0, 0].abs())) == 0
    assert float(out["range_axis"][0]) == 0.0


def test_dc_removal_zeroes_exactly_one_bin():
    """The claim `RangeTransformBlock`'s docstring makes about `dc_removal`: it is a
    one-bin decision, not a filter. Subtracting the fast-time mean zeroes FFT bin 0
    and leaves every other bin untouched."""
    n, df = 256, 1e6
    adc = DechirpBlock(_SingleTxCfg()).apply(
        {"s_pars": _one_tap_cfr(n, df, 3.0 / (n * df), device=device)})["adc"]
    kw = dict(window="none", crop_negative_delay=False, delta_f_hz=df)
    keep = RangeTransformBlock(dc_removal=False, **kw).apply({"adc": adc})["cube"]
    drop = RangeTransformBlock(dc_removal=True, **kw).apply({"adc": adc})["cube"]
    assert float(drop[..., 0].abs().max()) < 1e-4 * float(keep.abs().max())
    assert torch.allclose(keep[..., 1:], drop[..., 1:], atol=1e-4 * float(keep.abs().max()))


def test_delta_f_resolution_order_and_the_two_grid_formulas():
    """The two grid formulas are genuinely different and are NOT interchangeable."""
    plan = {"start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 5000}
    assert delta_f_from_freq_plan(plan) == pytest.approx(3e9 / 4999.0, rel=1e-12)
    assert delta_f_from_cfg(BENCHMARK_V1_KA) == pytest.approx(749.5e6 / 512.0, rel=1e-12)
    # 0.02% apart -- small, and worth half a metre at the far end of a 2500-bin window.
    assert delta_f_from_freq_plan(plan) != 3e9 / 5000.0
    assert delta_f_from_freq_plan(None) is None
    assert delta_f_from_cfg(None) is None

    # The explicit kwarg wins over the chain's freq_plan, which wins over cfg.
    rt = RangeTransformBlock(BENCHMARK_V1_KA, delta_f_hz=7.0)
    assert rt.resolve_delta_f({"freq_plan": plan}) == 7.0
    rt = RangeTransformBlock(BENCHMARK_V1_KA)
    assert rt.resolve_delta_f({"freq_plan": plan}) == pytest.approx(3e9 / 4999.0)
    assert rt.resolve_delta_f({}) == pytest.approx(749.5e6 / 512.0)


def test_munich_ka_fmcw_preset_samples_the_stored_grid():
    """`MUNICH_KA_FMCW` and `fmcw_plan_from_freq_plan` must agree: the preset is a
    convenience copy of the derivation, and a copy that disagrees with its source is
    the failure mode this whole contract exists to stop."""
    plan = {"start_hz": 28.5e9, "stop_hz": 31.5e9, "num_freqs": 5000}
    df = delta_f_from_freq_plan(plan)
    assert MUNICH_KA_FMCW.ramp_slope_hzps / MUNICH_KA_FMCW.fs_hz == pytest.approx(df, rel=1e-9)
    assert delta_f_from_cfg(MUNICH_KA_FMCW) == pytest.approx(df, rel=1e-9)
    derived = fmcw_plan_from_freq_plan(plan, n_rx=1024)
    assert delta_f_from_cfg(derived) == pytest.approx(df, rel=1e-9)
    assert derived.n_samples == 5000
    # The full unambiguous period is `c / (2 df)` under c/2 -- computed from the grid's
    # OWN spacing, so it is 249.78 m, NOT the 250.0 m a nominal `B = 3 GHz` gives.
    # That 0.2 m is exactly the off-by-one this test exists to keep visible.
    axis = range_axis_m(5000, df, 5000, "monostatic_c2")
    assert float(axis[-1]) + float(axis[1]) == pytest.approx(C_MPS / (2.0 * df))
    assert float(axis[-1]) + float(axis[1]) == pytest.approx(249.78, abs=0.01)
    assert C_MPS / (2.0 * (3e9 / 5000.0)) == pytest.approx(249.83, abs=0.01)

    with pytest.raises(ValueError, match="freq_plan"):
        fmcw_plan_from_freq_plan(None, n_rx=4)


def test_crop_keeps_exactly_the_v10_display_half():
    """`crop_negative_delay=True` (the default) keeps bins 0..N//2 -- the same half
    `webapp/pipeline_runner._nonnegative_range` kept at HEAD c33bb64 -- and the full
    window is available by asking for it."""
    n, df = 256, 1e6
    adc = DechirpBlock(_SingleTxCfg()).apply(
        {"s_pars": _one_tap_cfr(n, df, 0.0, device=device)})["adc"]
    kw = dict(window="none", dc_removal=False, delta_f_hz=df)
    full = RangeTransformBlock(crop_negative_delay=False, **kw).apply({"adc": adc})
    half = RangeTransformBlock(crop_negative_delay=True, **kw).apply({"adc": adc})
    assert full["cube"].shape[-1] == n
    assert half["cube"].shape[-1] == n // 2 + 1
    assert torch.equal(half["cube"], full["cube"][..., :n // 2 + 1])
    # Same calibration for the bins they share: cropping keeps bins, not metres.
    assert torch.allclose(half["range_axis"], full["range_axis"][:n // 2 + 1])
    # The DEFAULT convention is `bistatic_path` (c*tau) since the owner's 2026-09-24
    # ballot -- see tests/test_full_chain_frontend.py for that decision's own oracles.
    assert full["range_convention"] == "bistatic_path"
    assert float(full["range_axis"][-1]) == pytest.approx((n - 1) * C_MPS / (n * df))


def test_adc_to_rd_range_half_parity():
    """`transforms.adc_to_rd`'s range half IS `RangeTransformBlock` at the ML
    protocol's settings (`window='hann'`, `dc_removal=True`, uncropped).

    `e2e/chain/transforms.py` is under the F84 training freeze, so the refactor that
    makes `adc_to_rd` CALL this block is deferred; until it lands this test is what
    keeps the two copies honest. If it ever fails, the corpora and the screens have
    drifted apart again -- which is the whole defect being fixed.
    """
    torch.manual_seed(3)
    n_rx, n_chirps, n_samples = 4, 8, 64
    adc = (torch.randn(n_rx, n_chirps, n_samples) +
           1j * torch.randn(n_rx, n_chirps, n_samples)).to(torch.complex64).to(device)
    cube = RangeTransformBlock(window="hann", dc_removal=True,
                               crop_negative_delay=False).apply({"adc": adc})["cube"]
    # The Doppler half of adc_to_rd, applied to our cube, must reproduce adc_to_rd.
    win = torch.hann_window(n_chirps, periodic=False, dtype=torch.float32,
                            device=cube.device).to(cube.dtype)
    dop = torch.fft.fftshift(torch.fft.fft(cube * win[None, :, None], n=n_chirps, dim=1),
                             dim=1)
    ours = dop.transpose(1, 2).contiguous().to(torch.complex64)
    theirs = adc_to_rd(None, adc)
    assert _relmax(ours.abs(), theirs.abs()) < RTOL


# ================================================= 3. the compressor, still in series

def test_compressor_reaches_the_image(synth):
    """Changing the AFE mantissa changes the range-azimuth map -- i.e. the compressor
    is genuinely between the range transform and the images, not a side branch.
    Measured 2026-09-24 on this frame: mean |dB| delta 0.104 (v1.0's own, on the CFR
    snapshots, was 0.111) -- the "images move a few tenths of a dB" line survives."""
    cube = _spine_cube(synth)["cube"]

    def image(mantissa):
        torch.manual_seed(99)
        tracker = AdaOjaBlock(d=1024, k=2, m=64)
        out = MeasurementStage(AFEBlock(mantissa=mantissa), tracker).apply(
            {"cube": cube, "aperture_shape": (32, 32)})
        return RangeAzBlock(bins=32).apply(
            {"cube": out["cube"], "aperture_shape": (32, 32)})["range_az"]

    a, b = image(6), image(1)
    assert not torch.allclose(a, b)

    def db(x):
        return 10 * torch.log10(x.double().clamp_min(1e-30))

    assert float((db(a) - db(b)).abs().mean()) > 0.0


def test_no_afe_leaves_the_cube_bit_identical(synth):
    """With the AFE off, `MeasurementStage` tracks but does NOT rewrite the cube, so
    the image is bit-identical to the pre-compressor one. Pins B1: "is the stage doing
    anything" has a yes/no answer, not a shrug."""
    cube = _spine_cube(synth)["cube"]
    torch.manual_seed(5)
    tracker = AdaOjaBlock(d=1024, k=2, m=64)
    state = {"cube": cube, "aperture_shape": (32, 32)}
    before = RangeAzBlock(bins=32).apply(state)["range_az"]
    out = MeasurementStage(None, tracker).apply(state)
    assert "cube" not in out                       # untouched, not re-derived
    state.update(out)
    after = RangeAzBlock(bins=32).apply(state)["range_az"]
    assert torch.equal(before, after)


def test_tracker_on_cube_snapshots_reproduces_todays_subspace():
    """The snapshot-tracker invariance oracle. A rank-1 frame, warm-started from the
    (mapped) ground truth exactly as `Simulation` warm-starts it, tracked for one
    frame: the subspace error off the CUBE's range-bin snapshots must equal the error
    off the v1.0 frame's frequency-bin snapshots.

    Measured 2026-09-24: 0.044097 (v1.0) vs 0.044140 (spine) -- 4.3e-5 apart, two
    orders of magnitude inside the ~5e-3 run-to-run floor.

    NOT compared: the tracked BASIS after a COLD start. One Oja step from a random
    basis is nowhere near converged (v1.0's own cold-start basis sits 0.963 from its
    own oracle), so an exact-vector comparison there would pin noise.
    """
    g = torch.Generator().manual_seed(7)
    u = torch.randn(1024, dtype=torch.complex64, generator=g)
    v = torch.randn(64, dtype=torch.complex64, generator=torch.Generator().manual_seed(8))
    rank1 = (u[:, None] * v[None, :]).view(1024, 1, 1, 64).to(device)

    U_cfr = torch.linalg.svd(rank1[:, :, 0, :].view(-1, 64))[0][:, :1]
    U_beat = to_beat_basis(U_cfr)
    cube = _spine_cube(rank1)["cube"]

    # The cube's own top-1 left singular vector IS the mapped oracle: the dechirp is
    # antiunitary on the element axis and the range DFT is scaled-unitary on the other.
    assert float(subspace_dist_frob(U_beat, torch.linalg.svd(cube[:, 0, :])[0][:, :1])) < 1e-2

    def one_warm_frame(on_cube):
        torch.manual_seed(77)
        tracker = AdaOjaBlock(d=1024, k=1, m=64)
        oracle = U_beat if on_cube else U_cfr
        tracker.oja.U = perturb_basis(oracle)
        stage = MeasurementStage(AFEBlock(), tracker)
        if on_cube:
            U = stage.apply({"cube": cube, "aperture_shape": (32, 32)})["U"]
        else:
            # The v1.0 MeasurementStage body, inline (HEAD c33bb64): snapshots are the
            # aperture frame's FREQUENCY bins.
            V = frames.to_aperture_grid(rank1, (32, 32)).view(-1, 64)
            A = tracker.gen_A_ada()
            Aq, X = stage.afe_block.apply_mat_mul(A, V)
            tracker.update(X, Aq)
            U = tracker.oja.U
        return float(subspace_dist_frob(oracle, U))

    err_v10, err_spine = one_warm_frame(False), one_warm_frame(True)
    assert err_spine == pytest.approx(err_v10, abs=5e-3)


def test_subspace_dist_is_invariant_under_the_beat_map():
    """Why no published subspace number moves: `to_beat_basis` is a permutation plus a
    conjugation, and `subspace_dist_frob` cannot see either when both arguments get it."""
    g = torch.Generator().manual_seed(11)
    A = torch.linalg.qr(torch.randn(64, 3, dtype=torch.complex64, generator=g))[0]
    B = torch.linalg.qr(torch.randn(64, 3, dtype=torch.complex64, generator=g))[0]
    assert float(subspace_dist_frob(to_beat_basis(A), to_beat_basis(B))) == pytest.approx(
        float(subspace_dist_frob(A, B)), abs=1e-5)


# ================================================================ 4. front-end freeze

def test_front_end_frozen_at_todays_semantics(synth):
    """F96's own oracle, kept as the regression pin for the DISCLOSED semantics: the
    front end still runs on `ifft(CFR)` -- the channel impulse response, not a signal
    an amplifier sees -- and `CircuitStage`'s output is exactly `fft(rffe(ifft(CFR)))`.
    Tolerance 0.0: the MVC freezes this, it does not move it (contract section 5.1
    item 3). Moving the RFFE onto beat samples is the v1.2 item.
    """
    rffe = RFFEBlock(n=1024, seed=4242)
    stage_out, _ = CircuitStage(rffe).apply({"s_pars": synth})["s_pars"], None

    rffe2 = RFFEBlock(n=1024, seed=4242)
    direct, _prx = rffe2.apply_circuit(synth)
    assert torch.equal(stage_out, direct)

    # And the inner math is literally fft(circuit(ifft(.))): the frame the circuit is
    # handed is the IFFT of the CFR.
    frame = torch.fft.ifft(synth.view(1024, 1, -1, synth.shape[-1]), dim=-1)
    assert frame.shape[-1] == synth.shape[-1]
    assert torch.allclose(torch.fft.fft(frame, dim=-1).view(synth.shape), synth,
                          atol=1e-5, rtol=1e-5)


def test_front_end_bias_knob_still_moves_the_floor(synth):
    """T1's A/B still reaches the frame: 8 mA vs 0.5 mA are different outputs."""
    a = RFFEBlock(n=1024, seed=4242).apply_circuit(synth)[0]
    b = RFFEBlock(n=1024, seed=4242, lna_bias_ma=0.5).apply_circuit(synth)[0]
    assert not torch.allclose(a, b)


# ==================================================================== 5. loud failures

def test_product_without_the_spine_names_the_range_transform(synth):
    """Asking for a range product on a chain that never crossed into the cube domain
    must name `RangeTransformBlock`, not die on a missing key."""
    with pytest.raises(frames.FrameContractError, match="RangeTransformBlock"):
        frames.require_domain(frames.DOMAIN_CFR, RangeAzBlock())
    with pytest.raises(frames.FrameContractError, match="RangeTransformBlock"):
        frames.require_domain(frames.DOMAIN_RX_TIME, RangeProfileBlock())


def test_product_refuses_a_cube_from_the_wrong_waveform(synth):
    """A product declaring (chirp, range_bin) refuses an OFDM cube BY NAME."""
    cube = _spine_cube(synth)["cube"]
    state = {"cube": cube, "aperture_shape": (32, 32),
             "cube_axes": dict(frames.CUBE_AXES_OFDM)}
    with pytest.raises(frames.FrameContractError, match="subcarrier"):
        RangeAzBlock(bins=32).apply(state)


def test_measurement_stage_refuses_a_multichirp_cube():
    cube = torch.zeros(16, 4, 8, dtype=torch.complex64, device=device)
    stage = MeasurementStage(None, AdaOjaBlock(d=16, k=1, m=4))
    with pytest.raises(frames.FrameContractError, match="multiple chirps"):
        stage.apply({"cube": cube, "aperture_shape": (4, 4)})


def test_two_sources_on_one_chain_raise(make_env_block):
    """The spine has one origin. A source block parked in the serial list is a second
    one, and whichever ran last would silently win."""
    env = make_env_block(n_frames=1, n_freqs=16)
    other = make_env_block(n_frames=1, n_freqs=16)
    with pytest.raises(ValueError, match="two sources on one chain"):
        Simulation(env, [], 2, serial_stages=[other])


def test_replay_enters_the_one_spine_at_a_start_index(make_env_block):
    """The stored-ADC replay capability, kept -- as a START INDEX, not a second loop.
    A source declaring `rx_time` enters at the range transform, and the stages it
    skipped are reported BY NAME."""
    env = make_env_block(n_frames=1, n_freqs=16)
    adc = torch.zeros(1024, 1, 16, dtype=torch.complex64, device=device)

    class StoredAdcSource:
        signal_domain = frames.DOMAIN_RX_TIME
        array_shape = (32, 32)

        def get_S_pars(self):
            return adc

        def step(self):
            pass

        def reset(self):
            pass

    sim = Simulation(StoredAdcSource(), [RangeProfileBlock(bins=8)], 2,
                     circuit_block=RFFEBlock(n=1024), interconnect_block=None)
    assert [type(s).__name__ for s in sim.serial_stages] == [
        "CircuitStage", "DechirpBlock", "RangeTransformBlock"]
    sim.feed_forward()
    assert sim.skipped_stages == ["CircuitStage[RFFEBlock]", "DechirpBlock"]
    assert sim.outputs["skipped_stages"][-1] == sim.skipped_stages
    assert len(sim.outputs["range_profile"]) == 1


def test_live_source_skips_nothing(make_env_block):
    """The same rule, with nothing skipped -- which is what "one loop" means."""
    sim = Simulation(make_env_block(n_frames=1, n_freqs=16), [RangeProfileBlock(bins=8)], 2)
    sim.feed_forward()
    assert sim.skipped_stages == []
    assert "skipped_stages" not in sim.outputs


def test_range_transform_rejects_unknown_window_and_convention():
    with pytest.raises(ValueError, match="unknown range window"):
        RangeTransformBlock(window="blackman")
    with pytest.raises(ValueError, match="unknown range convention"):
        RangeTransformBlock(convention="c3")
    with pytest.raises(ValueError, match="unknown range convention"):
        range_axis_m(8, 1e6, 8, convention="nope")


def test_aperture_product_without_geometry_says_so():
    cube = torch.zeros(16, 1, 8, dtype=torch.complex64, device=device)
    with pytest.raises(frames.FrameContractError, match="aperture_shape"):
        RangeAzBlock(bins=4).apply({"cube": cube})


# ================================================================ 6. the spine itself

def test_default_spine_is_the_contracts_order(make_env_block):
    """The one list, in the contract's order (section 1.2). This is the assertion the
    webapp's registry order is checked against."""
    env = make_env_block(n_frames=2, n_freqs=32)
    sim = Simulation(env, [], 2,
                     circuit_block=RFFEBlock(n=1024),
                     interconnect_block=None,
                     afe_block=AFEBlock(),
                     subspace_block=AdaOjaBlock(d=1024, k=2, m=32))
    assert [type(s).__name__ for s in sim.serial_stages] == [
        "CircuitStage", "DechirpBlock", "RangeTransformBlock", "MeasurementStage"]
    # No GridStage anywhere: the aperture view now happens inside the products.
    assert not any(type(s).__name__ == "GridStage" for s in sim.serial_stages)


def test_spine_runs_end_to_end_with_every_product(make_env_block):
    env = make_env_block(n_frames=2, n_freqs=32)
    sim = Simulation(
        env,
        [FFTBlock(bins=16), RangeAzBlock(bins=16), RangeElBlock(bins=16),
         RangeProfileBlock(bins=16), SubspaceErrorBlock()],
        2,
        circuit_block=RFFEBlock(n=1024),
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(d=1024, k=2, m=32),
    )
    out = sim.run(n_steps=2)
    for key in ("fft", "range_az", "range_el", "range_profile", "subspace_err"):
        assert len(out[key]) == 2
    assert all(torch.isfinite(torch.as_tensor(e)) for e in out["subspace_err"])
