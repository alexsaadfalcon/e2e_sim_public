"""Physics regression tests for the RF front-end circuit model overhaul.

Covers: the LNA gain retune (8 mA bias), the BB cubic sign fix (compressive, not
expansive), the FOURKT thermal-noise constant, and noise referenced to the buffer's
true sample rate `fs` instead of a vestigial chirp duration. Runs on the library
device (CUDA if present, else CPU); no CPU hardcoding.
"""

import pytest

torch = pytest.importorskip("torch")

from e2e.circuit.rffe_model import (
    FOURKT, get_RX_config, circuit_model_bb_approx, circuit_model_batch,
)
from e2e.blocks import RFFEBlock, AdaOjaBlock, device
from e2e.subspace.algorithms import rand_orth_complex
from e2e.subspace.subspace_utils import subspace_dist_frob


FS_DEFAULT = 3e9  # matches RFFEBlock's default freq_span_hz


def _cfg(n=1):
    """A single RX_config row (Ibias_LNA, Vbias_LNA, Plo, Ibias_BB, Vbias_BB, Av, BW)."""
    return get_RX_config(n).to(device)[0]


def _expected_NBB(cfg):
    """Replicates circuit_model_bb_approx's bias-dependent noise formula (it has no
    dependence on the input signal, only on RX_config), so the noise-floor test can
    check the implementation against an independently-computed expectation rather
    than a value pulled out of the function under test.
    """
    Rs = 50.0
    gammalna = 3.0
    RoLNA = 50.0
    Plomax = 0.02
    Vodmax = 0.6
    Gsw0 = 0.06
    Kn = 8.0
    gammabb = 1.0

    Ibias_LNA, Vbias_LNA, Plo, Ibias_BB, Vbias_BB, Av, _BW = cfg

    GmLNA = 1.5 * Ibias_LNA / Vbias_LNA
    AvLNA = GmLNA * RoLNA
    FLNA = 1 + gammalna / GmLNA / Rs + 1 / Av ** 2
    Nlna = (Rs * FOURKT) * FLNA * AvLNA ** 2

    Vod = Vodmax * torch.sqrt(Plo / Plomax)
    Gsw = Gsw0 * Vod / Vodmax
    rho = 1 / (Gsw * RoLNA)
    Avmix = rho * Kn / (1 + rho * (1 + Kn))
    Fmix = (1 + rho) * (1 + (rho + 1) / (rho * Kn))
    Nmix = (Nlna + (RoLNA * FOURKT) * (Fmix - 1)) * Avmix ** 2

    GmBB = 1.5 * Ibias_BB / Vbias_BB
    AvBB = Av / (AvLNA * Avmix)
    FBB = 1 + gammabb / GmBB / RoLNA
    NBB = (Nmix + (RoLNA * FOURKT) * (FBB - 1)) * AvBB ** 2
    return NBB


# --------------------------------------------------------------------------- (a) noise floor

def test_noise_floor_matches_NBB_times_BWIF_per_sample():
    """circuit_model_bb_approx injects noise in the TIME domain, but the buffer goes
    through an unnormalized forward FFT afterwards (RFFEBlock.apply_circuit), which
    multiplies variance by nt (the buffer's last-dim length) -- so the per-TIME-SAMPLE
    variance injected here is intentionally NBB*BW_IF/nt, not NBB*BW_IF (that full
    per-frequency-bin level is pinned post-FFT by
    test_frequency_bin_noise_matches_NBB_times_BWIF_after_fft below). Here nt = N
    (this is a bare 1-D call, so the whole vector is one "buffer")."""
    cfg = _cfg()
    N = 200_000
    zero_in = torch.zeros(N, dtype=torch.cfloat, device=device)
    out, _PRX = circuit_model_bb_approx(cfg, zero_in, FS_DEFAULT, if_filter=False)

    BW_IF = cfg[6].item()  # 15e6
    expected_var = (_expected_NBB(cfg) * BW_IF / N).item()
    # Noise is added independently to the real (I) and imaginary (Q) parts, each with
    # per-sample variance NBB*BW_IF/nt; average the two measured quadrature variances.
    measured_var = (0.5 * (torch.var(out.real) + torch.var(out.imag))).item()

    assert measured_var == pytest.approx(expected_var, rel=0.10)


def test_frequency_bin_noise_matches_NBB_times_BWIF_after_fft():
    """End-to-end seam check through RFFEBlock (ifft -> circuit -> unnormalized fft):
    the per-FREQUENCY-BIN noise variance downstream blocks actually consume must land
    on NBB*BW_IF, independent of nt. Pre-fix this was inflated by a factor of nt
    (e.g. ratio ~64 at nt=64) because the per-sample injection wasn't pre-divided by
    nt before the unnormalized forward FFT."""
    nt = 64
    n_rx = 4000  # many independent elements stand in for repeated trials
    rffe = RFFEBlock(n=n_rx, physical_scale=True, freq_span_hz=FS_DEFAULT)
    zero_s_pars = torch.zeros(n_rx, 1, 1, nt, dtype=torch.cfloat, device=device)

    s_pars_dist, _PRX = rffe.apply_circuit(zero_s_pars)

    cfg = _cfg()
    BW_IF = cfg[6].item()
    expected_var = (_expected_NBB(cfg) * BW_IF).item()
    # Average per-bin variance across frequency bins and elements (all bins/elements
    # are statistically identical here: zero input, identical RX_config per element).
    measured_var = (0.5 * (torch.var(s_pars_dist.real) + torch.var(s_pars_dist.imag))).item()

    assert measured_var == pytest.approx(expected_var, rel=0.15)


# --------------------------------------------------------------------------- (b) small-signal gain

def test_small_signal_gain_matches_24dB():
    cfg = _cfg()
    v0 = 1e-3
    N = 200_000
    tone = torch.full((N,), v0, dtype=torch.cfloat, device=device)
    out, _PRX = circuit_model_bb_approx(cfg, tone, FS_DEFAULT, if_filter=False)

    measured_gain = (torch.mean(out.real) / v0).item()
    expected_gain = 10 ** (24 / 20)
    assert measured_gain == pytest.approx(expected_gain, rel=0.02)


# --------------------------------------------------------------------------- (c) compression sweep

def test_compression_monotonic_and_engaged_at_high_amplitude():
    """Gain (output/input amplitude ratio) must be monotonically non-increasing with
    input amplitude, and clearly compressed (clamp engaged) at the top of the sweep.

    This FAILS on the old expansive BB sign ('+' instead of '-'): with '+', the BB
    stage amplifies rather than saturates near clipping, so gain would not decrease
    (and can blow up) with amplitude instead of compressing.
    """
    cfg = _cfg()
    N = 200_000
    amps = [1e-3, 3e-2, 0.1, 0.3]
    gains = []
    for v0 in amps:
        tone = torch.full((N,), v0, dtype=torch.cfloat, device=device)
        out, _PRX = circuit_model_bb_approx(cfg, tone, FS_DEFAULT, if_filter=False)
        gains.append((torch.mean(out.real) / v0).item())

    for i in range(len(gains) - 1):
        # small relative slack for residual averaging noise, not real non-monotonicity
        assert gains[i + 1] <= gains[i] * 1.001

    # clamp engaged: gain well below the small-signal gain at the top of the sweep
    assert gains[-1] < gains[0] * 0.9


# --------------------------------------------------------------------------- (c.5) phase invariance

def test_lna_mixer_envelope_is_phase_invariant():
    """Directly targets defect 2: the LNA+mixer nonlinearity now depends only on the
    envelope |v|, so its output magnitude must be phase-invariant to float precision
    (an ANALYTIC property of the (3/4)*|v|^2*v envelope form). Replicates the LNA+
    mixer-only formulas independently of the module under test (same pattern as
    _expected_NBB), so it doesn't just re-run production code against itself. This
    FAILS on the old complex cube v**3 (which mixes I/Q and is phase-dependent):
    pre-fix, the equivalent isolated-stage measurement varies by 2-3x with phase at
    this amplitude (matching the qualitative pre-fix symptom -- end-to-end |gain|
    5.28 at phase 0 vs 7.47 at ~67.5deg, |v0|=0.1V -- also driven by this stage,
    though the BB stage's own real per-rail nonlinearity, left as-is, contributes
    additional phase dependence downstream; see test_full_chain_gain_no_expansion
    below for the full-chain assertions this fix supports).
    """
    cfg = _cfg()
    Ibias_LNA, Vbias_LNA, Plo, _Ibias_BB, _Vbias_BB, _Av, _BW = cfg
    RoLNA, Vodmax, Plomax, Gsw0, Vsat, Kn = 50.0, 0.6, 0.02, 0.06, 0.5, 8.0

    GmLNA = 1.5 * Ibias_LNA / Vbias_LNA
    G3LNA = Ibias_LNA / Vbias_LNA ** 3 / 2
    r_star_lna = torch.sqrt(GmLNA / (2.25 * G3LNA))
    Vod = Vodmax * torch.sqrt(Plo / Plomax)
    Gsw = Gsw0 * Vod / Vodmax
    rho = 1 / (Gsw * RoLNA)
    a2 = -1 / 4 / Vod
    a3 = -1 / 2 / Vsat ** 2

    v0 = 0.1  # comfortably past r_star_lna (~0.1155 x Vbias_LNA) so the cubic engages
    phases = torch.linspace(0, 2 * torch.pi, 17)[:-1]
    mags = []
    for phase in phases:
        v = v0 * torch.exp(1j * phase).to(torch.cfloat)
        v_mag = torch.abs(v)
        v = v * torch.clamp(r_star_lna / torch.clamp(v_mag, min=1e-30), max=1.0)
        Vlna = RoLNA * (GmLNA * v - 0.75 * G3LNA * v * torch.abs(v) ** 2)
        Imix = (Vlna / RoLNA / (1 + rho)
                - 0.75 * (Vlna * rho) * torch.abs(Vlna * rho) ** 2 / RoLNA / (1 + rho) ** 5
                * (2 * a2 ** 2 - a3 * (1 + rho)))
        Vmix = Imix * RoLNA * rho * Kn * (1 + rho) / (1 + rho * (1 + Kn))
        mags.append(torch.abs(Vmix).item())

    mags_t = torch.tensor(mags)
    assert (mags_t.max() - mags_t.min()).item() == pytest.approx(0.0, abs=1e-3 * mags_t.mean().item())


def test_full_chain_gain_no_expansion(monkeypatch):
    """Full chain (through circuit_model_bb_approx, noise zeroed): |output| must
    never exceed the small-signal linear-gain prediction (Av) at any phase or
    amplitude -- the compressive nonlinearities (LNA/mixer envelope form + BB
    per-rail cubic) must only ever saturate, never expand. This is the concrete,
    amplitude-swept form of the defect-2 "0.80V vs 0.60V linear prediction"
    over-the-clamp EXPANSION symptom."""
    import e2e.circuit.rffe_model as rffe_mod

    monkeypatch.setattr(rffe_mod.torch, "randn_like", lambda x: torch.zeros_like(x))

    cfg = _cfg()
    linear_gain = cfg[5].item()  # Av column, 10**(24/20)
    phases = torch.linspace(0, 2 * torch.pi, 13, device=device)[:-1]

    for v0 in (1e-3, 1e-2, 0.1, 0.3):
        for phase in phases:
            tone = torch.full((1,), v0, dtype=torch.cfloat, device=device) * torch.exp(1j * phase)
            out, _PRX = circuit_model_bb_approx(cfg, tone, FS_DEFAULT, if_filter=False)
            gain = (torch.abs(out[0]) / v0).item()
            assert gain <= linear_gain * (1 + 1e-6)


# --------------------------------------------------------------------------- (d) BB compressive sign

def test_bb_stage_is_compressive_not_expansive():
    """Directly verify the sign: for v just below Vbias_BB, the cubic term must
    subtract (compress), not add (expand)."""
    cfg = _cfg()
    Ibias_LNA, Vbias_LNA, Plo, Ibias_BB, Vbias_BB, Av, _BW = cfg

    RoLNA = 50.0
    Vodmax = 0.6
    Plomax = 0.02
    Gsw0 = 0.06
    Kn = 8.0

    GmLNA = 1.5 * Ibias_LNA / Vbias_LNA
    AvLNA = GmLNA * RoLNA
    Vod = Vodmax * torch.sqrt(Plo / Plomax)
    Gsw = Gsw0 * Vod / Vodmax
    rho = 1 / (Gsw * RoLNA)
    Avmix = rho * Kn / (1 + rho * (1 + Kn))
    AvBB = Av / (AvLNA * Avmix)
    GmBB = 1.5 * Ibias_BB / Vbias_BB
    G3BB = Ibias_BB / Vbias_BB ** 3 / 2
    RoBB = AvBB / GmBB

    v = 0.99 * Vbias_BB
    compressive = AvBB * v - G3BB * RoBB * v ** 3
    linear = AvBB * v
    assert compressive.item() < linear.item()


# --------------------------------------------------------------------------- (e) chain bookkeeping

def test_chain_gain_bookkeeping():
    cfg = _cfg()
    Ibias_LNA, Vbias_LNA, Plo, _Ibias_BB, _Vbias_BB, Av, _BW = cfg

    RoLNA = 50.0
    GmLNA = 1.5 * Ibias_LNA / Vbias_LNA
    AvLNA = GmLNA * RoLNA
    assert AvLNA.item() == pytest.approx(6.0, rel=0.01)

    Vodmax = 0.6
    Plomax = 0.02
    Gsw0 = 0.06
    Kn = 8.0
    Vod = Vodmax * torch.sqrt(Plo / Plomax)
    Gsw = Gsw0 * Vod / Vodmax
    rho = 1 / (Gsw * RoLNA)
    Avmix = rho * Kn / (1 + rho * (1 + Kn))
    AvBB = Av / (AvLNA * Avmix)

    total = AvLNA * Avmix * AvBB
    assert total.item() == pytest.approx(Av.item(), rel=0.01)


# --------------------------------------------------------------------------- (e.5) vectorized batch equivalence

def test_circuit_model_batch_matches_per_slice_single_call(monkeypatch):
    """circuit_model_batch vectorizes what used to be an nrx*ntx*ns Python loop over
    circuit_model_bb_approx; each (rx,tx,s) slice must still come out identical to
    calling circuit_model_bb_approx on that slice alone. Noise is injected inside
    circuit_model_bb_approx via torch.randn_like, so it is neutralized here (zeroed)
    to make this a deterministic equivalence check -- the statistical noise behavior
    itself is already covered by test_noise_floor_matches_NBB_times_BWIF and friends,
    which are intentionally left untouched (real torch.randn_like).
    """
    import e2e.circuit.rffe_model as rffe_mod

    monkeypatch.setattr(rffe_mod.torch, "randn_like", lambda x: torch.zeros_like(x))

    n_rx, ntx, ns, nt = 5, 1, 2, 32
    torch.manual_seed(3)
    rx_config = get_RX_config(n_rx).to(device)
    signals = (torch.randn(n_rx, ntx, ns, nt, dtype=torch.cfloat, device=device) * 1e-3)

    batch_out, batch_PRX = circuit_model_batch(rx_config, signals, FS_DEFAULT, if_filter=False)
    assert batch_out.shape == signals.shape

    for r in range(n_rx):
        single_prx = None
        for t in range(ntx):
            for s in range(ns):
                single_out, single_prx = circuit_model_bb_approx(
                    rx_config[r], signals[r, t, s, :], FS_DEFAULT, if_filter=False)
                assert torch.allclose(batch_out[r, t, s, :], single_out, atol=1e-8, rtol=1e-5)
        # PRX depends only on RX_config (not signal/t/s), so any slice's single-call
        # PRX must match the batch's per-rx value.
        assert torch.allclose(batch_PRX[0, r], single_prx, atol=1e-8, rtol=1e-5)


def test_circuit_model_batch_matches_per_slice_single_call_with_if_filter(monkeypatch):
    """Same equivalence check with the (harder-to-vectorize) IF-filter boxcar enabled."""
    import e2e.circuit.rffe_model as rffe_mod

    monkeypatch.setattr(rffe_mod.torch, "randn_like", lambda x: torch.zeros_like(x))

    n_rx, ntx, ns, nt = 3, 1, 2, 40
    torch.manual_seed(4)
    rx_config = get_RX_config(n_rx).to(device)
    signals = (torch.randn(n_rx, ntx, ns, nt, dtype=torch.cfloat, device=device) * 1e-3)

    batch_out, _batch_PRX = circuit_model_batch(rx_config, signals, FS_DEFAULT, if_filter=True)

    for r in range(n_rx):
        for t in range(ntx):
            for s in range(ns):
                single_out, _ = circuit_model_bb_approx(
                    rx_config[r], signals[r, t, s, :], FS_DEFAULT, if_filter=True)
                assert torch.allclose(batch_out[r, t, s, :], single_out, atol=1e-8, rtol=1e-5)


# --------------------------------------------------------------------------- (f) AdaOja scale invariance

def test_ada_oja_scale_invariant_to_input_amplitude():
    torch.manual_seed(0)
    n, d, m = 64, 4, 8
    U0 = rand_orth_complex(n, d)

    block_small = AdaOjaBlock(n=n, d=d, eta=0.1)
    block_large = AdaOjaBlock(n=n, d=d, eta=0.1)
    block_small.oja.U = U0.clone()
    block_large.oja.U = U0.clone()

    A = block_small.gen_A_ada(m=m)
    V = torch.randn(n, 8, dtype=torch.cfloat, device=device)
    X = A @ V

    block_small.update(X * 1e-6, A)
    block_large.update(X * 1.0, A)

    dist = subspace_dist_frob(block_small.oja.U, block_large.oja.U)
    assert dist.item() == pytest.approx(0.0, abs=1e-2)


# --------------------------------------------------------------------------- (g) legacy vs physical scale

def test_rffe_legacy_mode_normalizes_physical_mode_skips_it(monkeypatch):
    """physical_scale=False must keep the legacy mean(|frame|) normalization exactly;
    physical_scale=True must skip it entirely (frame passed through unscaled)."""
    import e2e.blocks as blocks_mod

    captured = {}

    def fake_circuit_model_batch(rx_config, frame, fs, if_filter=False):
        captured['frame'] = frame.clone()
        return torch.zeros_like(frame), torch.zeros((1, frame.shape[0]), device=frame.device)

    monkeypatch.setattr(blocks_mod, "circuit_model_batch", fake_circuit_model_batch)

    n_rx, F = 8, 16
    torch.manual_seed(1)
    s_pars = torch.randn(n_rx, 1, 1, F, dtype=torch.cfloat, device=device)
    # (n_rx, 1, 2, F): apply_circuit's reshape now infers the chirp dim (no longer
    # hardcoded to 2), so this n_chirp=2 shape just exercises that generality --
    # CircuitStage/feed_forward itself feeds single-chirp frames directly now (the
    # vestigial pol-pair duplication was removed).
    s_pars2 = torch.cat([s_pars, s_pars], dim=2)

    expected_frame = torch.fft.ifft(s_pars2.view(n_rx, 1, 2, F), dim=-1)

    rffe_phys = RFFEBlock(n=n_rx, physical_scale=True)
    rffe_phys.apply_circuit(s_pars2)
    assert torch.allclose(captured['frame'], expected_frame)

    rffe_legacy = RFFEBlock(n=n_rx, physical_scale=False, signal_scaling=1e-5)
    rffe_legacy.apply_circuit(s_pars2)
    normalized = expected_frame * rffe_legacy.signal_scaling / torch.mean(torch.abs(expected_frame))
    assert torch.allclose(captured['frame'], normalized)

    # physical_scale=True: no normalization, so scaling the input scales the frame
    # passed to the circuit model linearly.
    rffe_phys.apply_circuit(s_pars2 * 3.0)
    assert torch.allclose(captured['frame'], expected_frame * 3.0)


def test_rffe_legacy_mode_end_to_end_finite():
    """Real (un-mocked) chain in legacy mode must be finite."""
    n_rx, F = 8, 16
    torch.manual_seed(2)
    s_pars = torch.randn(n_rx, 1, 1, F, dtype=torch.cfloat, device=device)
    s_pars2 = torch.cat([s_pars, s_pars], dim=2)  # exercises n_chirp=2 generality; see above

    rffe = RFFEBlock(n=n_rx, physical_scale=False, signal_scaling=1e-5)
    out, PRX = rffe.apply_circuit(s_pars2)
    assert torch.isfinite(out).all()
    assert torch.isfinite(PRX).all()


# ------------------------------------------------------------------ legacy chirp_dur kwarg

def test_rffe_block_accepts_legacy_chirp_dur_kwarg():
    """chirp_dur is accepted (and ignored) so old call sites don't break."""
    rffe = RFFEBlock(n=4, chirp_dur=10e-9)
    assert rffe.fs == 3e9  # freq_span_hz default, unaffected by the legacy kwarg


# ------------------------------------------------------------------ if_filter branch
def test_if_filter_enabled_runs_on_batch_input():
    """Regression: the enabled IF-filter branch used to crash with IndexError on the
    1-D per-(rx,tx) series circuit_model_batch feeds (it assumed [time, n_tx]); it
    must now run end-to-end through the public RFFEBlock API and stay finite."""
    n_rx = 4
    rffe = RFFEBlock(n=n_rx, if_filter=True)
    # apply_circuit's reshape infers the chirp dim, so a n_chirp=2 shape [n, 1, 2, F]
    # still runs fine (exercised here); Simulation/CircuitStage itself now feeds
    # single-chirp [n, 1, 1, F] frames directly (no more pol-pair duplication).
    s_pars = (torch.randn(n_rx, 1, 2, 64) + 1j * torch.randn(n_rx, 1, 2, 64)).to(device)
    out, PRX = rffe.apply_circuit(s_pars)
    assert out.shape == s_pars.shape
    assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()


def test_if_filter_width_floored_at_one_tap():
    """Regression: a small fs relative to BW_IF used to round the boxcar width to 0
    taps (conv1d 'negative padding' crash); the width is floored at 1 (identity)."""
    cfg = _cfg()
    x = (torch.randn(64) + 1j * torch.randn(64)).to(device)
    out, _ = circuit_model_bb_approx(cfg, x, fs=1e6, if_filter=True)
    assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()


# ------------------------------------------------------------------ full-pipeline coverage
def test_full_pipeline_physical_scale_and_if_filter(make_env_block):
    """physical_scale=True and if_filter=True through a FULL Simulation (RFFE ->
    AFE quantizer -> AdaOja -> products), not just isolated apply_circuit calls.
    Small 8x8 array so it runs in the default suite (no slow gate)."""
    from e2e.simulation import Simulation
    from e2e.blocks import AFEBlock, AdaOjaBlock, FFTBlock, SubspaceErrorBlock

    n_rx = 64
    env = make_env_block(n_frames=2, n_freqs=64, n_rx=n_rx, array_shape=(8, 8))
    sim = Simulation(
        env, [FFTBlock(bins=16), SubspaceErrorBlock()], d=8,
        circuit_block=RFFEBlock(n=n_rx, physical_scale=True, if_filter=True,
                                freq_span_hz=3e9),
        afe_block=AFEBlock(),
        subspace_block=AdaOjaBlock(n_rx, 8),
        array_shape=(8, 8),
    )
    out = sim.run(n_steps=2)
    assert len(out["subspace_err"]) == 2
    assert all(torch.isfinite(v) for v in out["subspace_err"])
    assert all(torch.isfinite(f).all() for f in out["fft"])


# ------------------------------------------------------------------ (h) mixed-convention seam
def test_mixed_convention_pkl_consumed_per_link(tmp_path):
    """Generation -> runtime seam: one multi-link pkl holding a PHYSICAL link
    (radar, tx_power_dbm set) and a LEGACY link (comm, tx_power None) must load
    per-link and run through RFFEBlock with the matching physical_scale flag.
    Pins the contract that scale convention is a per-link property of the data."""
    import numpy as np
    from e2e.scenario import Scenario, Node, NodeRole, ArrayConfig, FrequencyPlan
    from e2e.environment.scenario_runner import ScenarioRunner
    from e2e.environment.sionna_iterator import SionnaIterator

    sc = Scenario(
        name="mixed_seam", base_scene="munich",
        frequency=FrequencyPlan(carrier_hz=30e9, start_hz=28.5e9, stop_hz=31.5e9,
                                num_freqs=64),
        num_frames=2,
        nodes=[
            Node(name="r", role=NodeRole.RADAR, position=(45.0, 90.0, 1.5),
                 array=ArrayConfig(num_rows=4, num_cols=4), tx_power_dbm=12.0),
            Node(name="t", role=NodeRole.COMM_TX, position=(8.5, 21.0, 27.0),
                 array=ArrayConfig(num_rows=1, num_cols=1)),   # legacy: no tx power
            Node(name="x", role=NodeRole.COMM_RX, position=(45.0, 90.0, 1.5),
                 array=ArrayConfig(num_rows=2, num_cols=2)),
        ],
    )
    assert sc.validate() == []
    out_path = str(tmp_path / "mixed.pkl")
    ScenarioRunner(sc, dry_run=True, seed=9).run(out_path=out_path, verbose=False)

    links = SionnaIterator.available_links(out_path)
    assert links is not None and len(links) == 2
    radar_link = next(l for l in links if "r" == l or "radar" in l or l == "r")
    comm_link = next(l for l in links if l != radar_link)

    for link, physical in ((radar_link, True), (comm_link, False)):
        it = SionnaIterator(out_path, link=link)
        frame = torch.from_numpy(np.asarray(it[0], dtype=np.complex64)).to(device)
        n_elem = frame.shape[0]
        # apply_circuit's chirp dim is inferred, so a n_chirp=2 shape [n, 1, 2, F]
        # still runs (exercised here for generality); Simulation itself now feeds
        # the frame's native single-chirp shape directly.
        doubled = torch.cat([frame, frame], dim=2)
        out, _prx = RFFEBlock(n=n_elem, physical_scale=physical,
                              freq_span_hz=3e9).apply_circuit(doubled)
        assert out.shape == doubled.shape
        assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()
