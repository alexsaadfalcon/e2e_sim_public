"""The three backend hooks the conference demo presets turn (notes/DEMO_DEFENSE.md).

Each was previously reachable only by monkeypatching from a scratch script, which is how
every number in the demo review was measured. These tests pin two things per hook: that
the knob does what its help text says, and that the WEBAPP actually delivers it to the
backend -- a registry param that the runner never reads is the failure mode
`test_subspace_m_is_the_value_the_tracker_is_actually_built_with` exists for.
"""

import pytest

torch = pytest.importorskip("torch")

import e2e.blocks as blocks
from e2e.blocks import InterconnectBlock, RFFEBlock
from e2e.circuit.rffe_model import get_RX_config


# ------------------------------------------------------------------------------------
# Thrust 1: LNA bias and IF bandwidth
# ------------------------------------------------------------------------------------
def test_rffe_defaults_are_bit_identical_to_the_config_table():
    """None for both knobs must leave every column exactly as `get_RX_config` wrote it."""
    blk = RFFEBlock(n=4)
    assert torch.equal(blk.rx_config.cpu(), get_RX_config(4))


def test_rffe_knobs_override_exactly_one_column_each():
    base = get_RX_config(4)
    blk = RFFEBlock(n=4, lna_bias_ma=0.5, if_bw_mhz=50.0)
    cfg = blk.rx_config.cpu()
    assert torch.allclose(cfg[:, RFFEBlock.RX_CONFIG_IBIAS_LNA], torch.full((4,), 0.5e-3))
    assert torch.allclose(cfg[:, RFFEBlock.RX_CONFIG_IF_BW], torch.full((4,), 50e6))
    untouched = [c for c in range(cfg.shape[1])
                 if c not in (RFFEBlock.RX_CONFIG_IBIAS_LNA, RFFEBlock.RX_CONFIG_IF_BW)]
    assert torch.equal(cfg[:, untouched], base[:, untouched])


def test_rffe_column_names_match_the_table_the_model_reads():
    """The circuit model indexes the table positionally; if someone reorders
    `get_RX_config`'s stack, these two constants are the only thing that would notice."""
    base = get_RX_config(2)
    assert float(base[0, RFFEBlock.RX_CONFIG_IBIAS_LNA]) == pytest.approx(8e-3)   # 8 mA
    assert float(base[0, RFFEBlock.RX_CONFIG_IF_BW]) == pytest.approx(15e6)       # 15 MHz


@pytest.mark.parametrize("kw", [{"lna_bias_ma": 0.0}, {"lna_bias_ma": -1.0},
                                {"if_bw_mhz": 0.0}, {"if_bw_mhz": -5.0}])
def test_rffe_rejects_non_positive_knobs(kw):
    """Both are divisors inside the circuit model; a zero must fail here, not as NaNs
    three blocks downstream."""
    with pytest.raises(ValueError):
        RFFEBlock(n=2, **kw)


def test_rffe_knob_changes_the_noise_the_model_adds():
    """Behavioural: a wider IF bandwidth must add MORE noise power to a fixed input.

    Averaged over a large batch so the RNG does not decide the outcome, and compared
    as a ratio against the 17 dB the help text quotes (10*log10(50)) -- loosely,
    because the model's noise is not the only term."""
    torch.manual_seed(0)
    n, n_freqs = 64, 256
    # The SAME deterministic trace on every element (an all-zero frame is 0/0 in the
    # block's own mean-normalization), so the spread across elements is noise alone.
    one = torch.randn(n_freqs, dtype=torch.complex64)
    s = one.view(1, 1, 1, n_freqs).expand(n, 1, 1, n_freqs).contiguous()
    noise_power = {}
    for bw in (1.0, 50.0):
        blk = RFFEBlock(n=n, if_bw_mhz=bw, signal_scaling=1e-7)
        out, _ = blk.apply_circuit(s.to(blk.rx_config.device))
        noise_power[bw] = float(torch.var(out, dim=0).real.mean())
    ratio_db = 10 * torch.log10(torch.tensor(noise_power[50.0] / noise_power[1.0]))
    assert 10.0 < float(ratio_db) < 20.0, f"expected ~17 dB more noise, got {float(ratio_db):.1f}"


# ------------------------------------------------------------------------------------
# Thrust 4: a synthetic interconnect with no gain
# ------------------------------------------------------------------------------------
def test_boxcar_placeholder_has_the_gain_the_demo_review_measured():
    """+20.8 dB peak -- the number DEMO_DEFENSE.md says no passive part can have."""
    H = InterconnectBlock().frequency_response(1000, torch.device("cpu"))
    peak_db = 20 * torch.log10(torch.abs(H).max())
    assert float(peak_db) == pytest.approx(20 * torch.log10(torch.tensor(11.0)).item(), abs=1e-4)


def test_normalize_gain_pins_the_peak_at_0_db_and_keeps_the_shape():
    dev = torch.device("cpu")
    raw = InterconnectBlock().frequency_response(1000, dev)
    norm = InterconnectBlock(normalize_gain=True).frequency_response(1000, dev)
    assert float(torch.abs(norm).max()) == pytest.approx(1.0, abs=1e-6)
    # Same shape: the two responses differ by exactly one complex scalar.
    assert torch.allclose(norm * torch.abs(raw).max(), raw, atol=1e-5)


def test_normalize_gain_is_applied_by_apply_interconnect_not_only_reported():
    """`frequency_response` and `apply_interconnect` must agree, or a caption could
    quote a normalized filter while the chain runs the unnormalized one."""
    dev = torch.device("cpu")
    frame = torch.ones(2, 1, 1, 64, dtype=torch.complex64, device=dev)
    blk = InterconnectBlock(normalize_gain=True)
    out = blk.apply_interconnect(frame)
    assert torch.allclose(out[0, 0, 0], blk.frequency_response(64, dev), atol=1e-6)
    assert float(torch.abs(out).max()) == pytest.approx(1.0, abs=1e-6)


def test_normalize_gain_leaves_passthrough_alone():
    dev = torch.device("cpu")
    frame = torch.randn(2, 1, 1, 64, dtype=torch.complex64, device=dev)
    out = InterconnectBlock(case="passthrough", normalize_gain=True).apply_interconnect(frame)
    assert torch.equal(out, frame)


def test_default_interconnect_output_is_unchanged_by_the_new_kwarg():
    """Bit-compatibility: the shipped default (normalize_gain=False) must produce what
    it produced before the kwarg existed, i.e. frame * FFT(11-tap boxcar)."""
    dev = torch.device("cpu")
    frame = torch.randn(1, 1, 1, 128, dtype=torch.complex64, device=dev)
    expected = frame * torch.fft.fft(
        torch.nn.functional.pad(torch.ones(11), (0, 128 - 11))).view(1, 1, 1, -1)
    out = InterconnectBlock().apply_interconnect(frame)
    assert torch.allclose(out, expected, atol=1e-5)


# ------------------------------------------------------------------------------------
# The webapp delivers each knob to the backend
# ------------------------------------------------------------------------------------
def _run_with_spies(monkeypatch, state):
    """Run the pipeline far enough to construct every block; capture ctor kwargs."""
    from webapp import pipeline_runner
    import e2e.simulation as simulation

    seen = {}

    def spy(cls, name):
        real = cls

        def wrapped(*a, **kw):
            seen[name] = kw
            return real(*a, **kw)
        return wrapped

    monkeypatch.setattr(blocks, "RFFEBlock", spy(blocks.RFFEBlock, "rffe"))
    monkeypatch.setattr(blocks, "InterconnectBlock", spy(blocks.InterconnectBlock, "interconnect"))
    monkeypatch.setattr(simulation, "Simulation", spy(simulation.Simulation, "simulation"))
    try:
        pipeline_runner.run_pipeline(state, n_steps=1)
    except pipeline_runner.PipelineError:
        pass  # no frames on this machine is fine -- construction happens first
    return seen


def test_runner_passes_the_rffe_knobs_through(monkeypatch):
    from webapp.pipeline_registry import default_block_state
    st = default_block_state()
    st["rffe"]["params"]["lna_bias_ma"] = 0.5
    st["rffe"]["params"]["if_bw_mhz"] = 50.0
    seen = _run_with_spies(monkeypatch, st)
    assert seen["rffe"]["lna_bias_ma"] == 0.5
    assert seen["rffe"]["if_bw_mhz"] == 50.0


def test_runner_falls_back_to_defaults_for_non_positive_rffe_knobs(monkeypatch):
    """Zero from a cleared spinner must not reach the ctor (which would raise)."""
    from webapp.pipeline_registry import default_block_state
    st = default_block_state()
    st["rffe"]["params"]["lna_bias_ma"] = 0
    st["rffe"]["params"]["if_bw_mhz"] = None
    seen = _run_with_spies(monkeypatch, st)
    assert seen["rffe"]["lna_bias_ma"] == 8.0
    assert seen["rffe"]["if_bw_mhz"] == 15.0


def test_runner_passes_normalize_gain_through(monkeypatch):
    from webapp.pipeline_registry import default_block_state
    st = default_block_state()
    st["interconnect"]["enabled"] = True
    st["interconnect"]["params"]["normalize_gain"] = True
    seen = _run_with_spies(monkeypatch, st)
    assert seen["interconnect"]["normalize_gain"] is True


@pytest.mark.parametrize("choice, expected", [("warm", True), ("cold", False)])
def test_runner_maps_warm_start_choice_onto_simulation(monkeypatch, choice, expected):
    from webapp.pipeline_registry import default_block_state
    st = default_block_state()
    st["subspace"]["params"]["warm_start"] = choice
    seen = _run_with_spies(monkeypatch, st)
    assert seen["simulation"]["warm_start"] is expected


def test_registry_defaults_reproduce_the_pre_knob_backend(monkeypatch):
    """A fresh state must build the exact objects the old runner built."""
    from webapp.pipeline_registry import default_block_state
    seen = _run_with_spies(monkeypatch, default_block_state())
    assert seen["rffe"]["lna_bias_ma"] == 8.0 and seen["rffe"]["if_bw_mhz"] == 15.0
    assert seen["simulation"]["warm_start"] is True
