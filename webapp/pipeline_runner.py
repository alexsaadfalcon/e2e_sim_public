"""
Runs the runtime block pipeline from a UI block-state dict, and turns the
outputs into Plotly figures.

ALL heavy imports (torch, e2e.blocks, e2e.simulation) happen lazily inside
:func:`run_pipeline` so this module — and the whole app shell — imports cleanly
on a machine without torch installed. The functions here are the only place the
"Run" action touches the real simulator.

Failure modes are surfaced as :class:`PipelineError` with a friendly message:
  * torch / e2e not importable          -> "torch not installed ..."
  * the scenario .pkl frames are missing -> "generate frames first ..."
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from webapp.pipeline_registry import BLOCKS_BY_ID

# Speed of light (m/s), used to convert the frequency-FFT axis to physical range.
_C = 2.99792458e8


class PipelineError(Exception):
    """Raised with a user-facing message when a run cannot complete."""


def _p(state: Dict[str, Dict[str, Any]], block_id: str, key: str) -> Any:
    """Fetch a param value for a block from UI state, falling back to default."""
    spec = BLOCKS_BY_ID[block_id]
    params = state.get(block_id, {}).get("params", {})
    if key in params and params[key] is not None:
        return params[key]
    for ps in spec.params:
        if ps.key == key:
            return ps.default
    raise KeyError(key)


# Params that MUST be strictly positive: a zero/negative value here produces a
# divide-by-zero (e.g. RFFEBlock scales by signal_scaling / mean(|frame|), and a
# 0 scaling yields 0/0 NaNs downstream) or an empty FFT. For these we treat a
# non-positive (or None) value as "unset" and fall back to the registry default,
# rather than letting it poison the run. Other params keep the plain _p policy.
_POSITIVE_PARAMS = {
    ("rffe", "signal_scaling"),
    ("rffe", "freq_span_hz"),
    ("fft", "bins"),
    ("range_az", "bins"),
    ("range_el", "bins"),
    ("comms", "fft_size"),
}


def _p_positive(state: Dict[str, Dict[str, Any]], block_id: str, key: str) -> Any:
    """Like :func:`_p` but falls back to the default when the value is <= 0."""
    val = _p(state, block_id, key)
    if val is None or val <= 0:
        spec = BLOCKS_BY_ID[block_id]
        for ps in spec.params:
            if ps.key == key:
                return ps.default
    return val


def _enabled(state: Dict[str, Dict[str, Any]], block_id: str) -> bool:
    return bool(state.get(block_id, {}).get("enabled", False))


def _resolve_physical_scale(mode: str, env_block: Any) -> bool:
    """Map the rffe scale_mode param to RFFEBlock's physical_scale bool.

    'physical'/'legacy' force the value; 'auto' (or anything else) follows the
    environment block's own ``physical_scale`` metadata (v2 pkls only -- legacy
    pkls expose it as absent/None, which falls back to False, i.e. unchanged
    legacy behavior).
    """
    if mode == "physical":
        return True
    if mode == "legacy":
        return False
    return bool(getattr(env_block, "physical_scale", None))


def _resolve_freq_span_hz(state: Dict[str, Dict[str, Any]], env_block: Any) -> float:
    """The RFFE buffer's true sample-rate span (Hz).

    Prefers the environment block's own ``freq_plan`` metadata (v2 pkls) so the
    frontend automatically matches the frames' actual band; falls back to the
    rffe freq_span_hz UI param for legacy pkls (no freq_plan).
    """
    freq_plan = getattr(env_block, "freq_plan", None)
    if freq_plan:
        return float(freq_plan["stop_hz"]) - float(freq_plan["start_hz"])
    return float(_p_positive(state, "rffe", "freq_span_hz"))


def _comms_freqs(state: Dict[str, Dict[str, Any]], env_block: Any) -> np.ndarray:
    """Frequency grid (Hz) for the comms head's `ModemBlock`.

    Prefers the environment block's own ``freq_plan`` metadata (v2 pkls):
    ``linspace(start_hz, stop_hz, num_freqs)`` -- mirrors :func:`_resolve_freq_span_hz`.
    Falls back (legacy pkls, no ``freq_plan``) to a span centered at 30 GHz built
    from the rffe ``freq_span_hz`` UI param, matching the reference scenarios'
    carrier. Either way the point count follows the actual frame's frequency-
    sample count when available (best-effort; a fixed default otherwise).
    """
    try:
        n_freqs = int(env_block.get_S_pars().shape[-1])
    except Exception:
        n_freqs = 64
    freq_plan = getattr(env_block, "freq_plan", None)
    if freq_plan:
        num = int(freq_plan.get("num_freqs") or n_freqs)
        return np.linspace(float(freq_plan["start_hz"]), float(freq_plan["stop_hz"]), num)
    span = _resolve_freq_span_hz(state, env_block)
    carrier = 30e9
    return np.linspace(carrier - span / 2.0, carrier + span / 2.0, n_freqs)


def run_pipeline(state: Dict[str, Dict[str, Any]], n_steps: int = 10) -> Dict[str, Any]:
    """
    Build blocks from ``state`` and run ``n_steps`` of the simulation.

    Returns the simulation ``outputs`` dict (lists of per-frame torch tensors).
    Raises :class:`PipelineError` for the expected, recoverable failure modes.
    """
    # --- lazy heavy imports -----------------------------------------------------
    try:
        import torch  # noqa: F401
        from e2e.frames import FrameContractError
        from e2e.simulation import Simulation
        from e2e.blocks import (
            SionnaEnvironmentBlock,
            RFFEBlock,
            InterconnectBlock,
            AFEBlock,
            AdaOjaBlock,
            FFTBlock,
            RangeAzBlock,
            RangeElBlock,
            SubspaceErrorBlock,
        )
    except ImportError as e:  # torch / sionna deps not present
        raise PipelineError(
            "Could not import the simulation backend (torch / e2e). "
            "The Run action needs torch installed. Underlying error: " + str(e)
        )

    N_TX = 1

    scenario_name = _p(state, "environment", "scenario_name")
    d = int(_p(state, "subspace", "d"))

    # The simulation backend now handles RFFE-off (PRX is initialized to None),
    # AFE-off (the no-AFE subspace branch calls subspace.update(X, A) with two args),
    # and even subspace-off (the measurement stage is skipped) cleanly. The webapp
    # still always constructs a subspace block below because its results view always
    # includes SubspaceErrorBlock, which consumes the tracker's 'U'.

    # --- environment block (loads the .pkl frames; can raise FileNotFoundError) -
    try:
        environment_block = SionnaEnvironmentBlock(scenario_name)
    except FileNotFoundError as e:
        raise PipelineError(
            f"No precomputed frames found for scenario '{scenario_name}'. "
            "Generate frames first (Scenario tab -> Generate frames), or pick a "
            f"scenario whose .pkl exists. Missing file: {e}"
        )
    except ValueError as e:
        raise PipelineError(str(e))

    # Derive the receive-array size from the environment block's array_shape so the
    # Oja tracker dimension and Simulation's view() agree with the actual frames.
    # A non-(32,32) array would otherwise silently disagree with a hardcoded 32*32.
    array_shape = getattr(environment_block, "array_shape", (32, 32)) or (32, 32)
    N_RX = int(array_shape[0]) * int(array_shape[1])

    # --- optional serial blocks -------------------------------------------------
    circuit_block = None
    if _enabled(state, "rffe"):
        circuit_block = RFFEBlock(
            n=N_RX * N_TX,
            freq_span_hz=_resolve_freq_span_hz(state, environment_block),
            signal_scaling=float(_p_positive(state, "rffe", "signal_scaling")),
            physical_scale=_resolve_physical_scale(
                _p(state, "rffe", "scale_mode"), environment_block
            ),
        )

    interconnect_block = None
    if _enabled(state, "interconnect"):
        case = _p(state, "interconnect", "case")
        interconnect_block = InterconnectBlock(case=None if case == "default" else case)

    afe_block = None
    if _enabled(state, "afe"):
        afe_block = AFEBlock(
            exp=int(_p(state, "afe", "exp")),
            mantissa=int(_p(state, "afe", "mantissa")),
        )

    # Subspace is required whenever AFE is on (simulation enforces this too).
    subspace_block = None
    if _enabled(state, "subspace") or afe_block is not None:
        subspace_block = AdaOjaBlock(N_RX, d)
    if afe_block is not None and subspace_block is None:
        raise PipelineError("AFE block requires the AdaOja Subspace block to be enabled.")
    if subspace_block is None:
        # The results view always runs SubspaceErrorBlock, which needs the tracker's
        # 'U'; keep a subspace block even when the user toggles the stage off.
        subspace_block = AdaOjaBlock(N_RX, d)

    # --- downstream product blocks (always present) -----------------------------
    fft_bins = int(_p_positive(state, "fft", "bins"))
    range_az_bins = int(_p_positive(state, "range_az", "bins"))
    range_el_bins = int(_p_positive(state, "range_el", "bins"))
    downstream_blocks = [
        FFTBlock(bins=fft_bins),
        RangeAzBlock(bins=range_az_bins),
        RangeElBlock(bins=range_el_bins),
        SubspaceErrorBlock(),
    ]

    # --- optional comms head (swappable "product": OFDM demod instead of / -------
    # alongside the radar products above). Appended AFTER the radar products so it
    # composes without disturbing their output ordering.
    comms_combining = None
    if _enabled(state, "comms"):
        try:
            from e2e.comms.blocks import ModemBlock, BERBlock
        except ImportError as e:
            raise PipelineError(
                "Could not import the comms backend (e2e.comms). "
                "Underlying error: " + str(e)
            )
        comms_combining = _p(state, "comms", "combining")
        comms_snr_db = float(_p(state, "comms", "snr_db"))
        comms_fft_size = int(_p_positive(state, "comms", "fft_size"))
        comms_freqs = _comms_freqs(state, environment_block)
        try:
            modem_block = ModemBlock(
                comms_freqs, fft_size=comms_fft_size, snr_db=comms_snr_db,
                combining=comms_combining,
            )
        except ValueError as e:
            raise PipelineError(str(e))
        downstream_blocks.append(modem_block)
        downstream_blocks.append(BERBlock())

    sim = Simulation(
        environment_block,
        downstream_blocks,
        d,
        circuit_block,
        interconnect_block,
        afe_block,
        subspace_block,
        array_shape=array_shape,
    )

    try:
        outputs = sim.run(n_steps=max(1, int(n_steps)))
    except FileNotFoundError as e:
        raise PipelineError(
            "A required data file was missing during the run (generate frames "
            f"first). Underlying error: {e}"
        )
    except (AssertionError, FrameContractError) as e:
        # FrameContractError: the e2e/frames.py shape-contract guards (no MIMO,
        # single chirp, aperture factorization) that used to be bare asserts.
        raise PipelineError(f"Pipeline constraint failed: {e}")
    except Exception as e:  # surface anything else cleanly to the UI
        raise PipelineError(f"Pipeline run failed: {type(e).__name__}: {e}")

    # Stash the axis metadata figures_from_outputs needs to label heatmaps physically
    # (bins used per product + the raw frame's frequency-sample count/span) alongside
    # the product outputs. Reserved-key checks in Simulation.feed_forward only guard
    # per-frame state_dict keys, not this top-level dict, so a leading-underscore key
    # here is safe and figures_from_outputs never copies it into its returned figs.
    n_freqs = None
    try:
        n_freqs = int(environment_block.get_S_pars().shape[-1])
    except Exception:
        pass  # axis metadata is best-effort; figures_from_outputs falls back to bins
    # Prefer the environment block's own freq_plan (v2 pkls) so the range axis matches
    # the frames' true band automatically, regardless of whether rffe is enabled.
    freq_plan = getattr(environment_block, "freq_plan", None)
    if freq_plan:
        freq_span_hz = float(freq_plan["stop_hz"]) - float(freq_plan["start_hz"])
        if freq_plan.get("num_freqs") is not None:
            n_freqs = int(freq_plan["num_freqs"])
    else:
        freq_span_hz = (
            float(_p_positive(state, "rffe", "freq_span_hz")) if _enabled(state, "rffe") else 3e9
        )
    outputs["_axis_meta"] = {
        "fft_bins": fft_bins,
        "range_az_bins": range_az_bins,
        "range_el_bins": range_el_bins,
        "n_freqs": n_freqs,
        "freq_span_hz": freq_span_hz,
        # True when the values above came from the frames' own v2 metadata (env
        # block freq_plan), not from UI params/fallbacks -- lets the UI say so.
        "from_meta": bool(getattr(environment_block, "freq_plan", None)),
    }
    if comms_combining is not None:
        # Small metadata dict figures_from_outputs reads to label the BER figure
        # (mirrors "_axis_meta" above); leading underscore keeps it out of the
        # reserved per-frame state_dict keys Simulation.feed_forward guards.
        outputs["_comms_meta"] = {"combining": comms_combining}
    return outputs


# --------------------------------------------------------------------------------
# Output -> Plotly figure helpers. Kept torch-tolerant: they only import torch
# when actually given tensors, and accept the outputs dict produced above.
# --------------------------------------------------------------------------------

def _to_numpy_abs_db(tensor):
    import torch
    # Guard the normalization against an all-zero product: dividing by max(|.|)==0
    # would yield NaNs across the whole heatmap. Clamp the denominator to a small
    # epsilon so an all-zero product becomes a finite, floored (-inf-clamped) map.
    peak = torch.max(torch.abs(tensor))
    peak = torch.clamp(peak, min=1e-12)
    t = tensor / peak
    # The fft/range_az/range_el products are POWER maps (|.|^2, non-coherent
    # integration), so dB relative to peak is 10*log10(P/Pmax). (Complex amplitude
    # inputs would take 20*log10; all heatmap products routed here are power.)
    db = 10 * torch.log10(torch.abs(t) + 1e-12)
    return db.T.detach().cpu().numpy()


def _to_numpy_complex(tensor):
    """Flatten a (possibly torch) complex tensor to a 1-D numpy array.

    Torch-tolerant like `_to_numpy_abs_db`: only touches `.detach()`/`.cpu()` when
    the value actually carries them (a torch tensor), so a plain numpy array/list
    (e.g. a hand-built outputs dict in a test) also works.
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor).reshape(-1)


def _sin_angle_axis(n_bins: int):
    """fftshifted aperture-FFT bin index -> normalized sine-angle u = sin(theta).

    Half-wavelength element spacing puts the unambiguous field of view at
    u in [-1, 1); bin 0 of the fftshifted axis is the most-negative angle, bin
    n_bins//2 is broadside (u=0).
    """
    return (np.arange(n_bins) - n_bins // 2) / (n_bins / 2)


def _range_axis(n_bins: int, freq_span_hz: float):
    """fftshifted frequency-FFT bin index -> physical range (meters).

    Valid only when the FFT length equals the raw frame's frequency-sample count
    (no zero-pad/truncate): the n_bins frequency samples then span bandwidth
    freq_span_hz (B) with sample spacing df = B / n_bins, so the FFT's delay-bin
    spacing is 1/B and range-per-bin (round-trip) is c / (2*B). Callers must check
    bins == n_freqs before using this -- with padding/truncation the resolution
    changes in a way that depends on truncation semantics, so we intentionally
    don't attempt it (see figures_from_outputs).

    Sign: the range blocks take a FORWARD fft over frequency, so a physical delay
    +tau (a target at +R) lands on the NEGATIVE side of the fftshifted axis; the
    axis is negated here so physical targets read at positive range.
    """
    return -(np.arange(n_bins) - n_bins // 2) * (_C / (2.0 * freq_span_hz))


def _heatmap(data_db, title: str, *, x=None, y=None,
             xlabel: str = "Bin", ylabel: str = "Bin") -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=data_db, x=x, y=y, colorbar=dict(title="power (dB)"), zmin=-40, zmax=0
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        margin=dict(l=40, r=20, t=40, b=40),
        height=360,
    )
    return fig


def figures_from_outputs(outputs: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Build a dict of named Plotly figures from a simulation outputs dict."""
    figs: Dict[str, go.Figure] = {}

    # Axis metadata (bins per product + the raw frame's freq-sample count/span),
    # stashed by run_pipeline; absent (e.g. a hand-built outputs dict in a test)
    # means every axis falls back to raw bin indices.
    meta = outputs.get("_axis_meta") or {}
    n_freqs = meta.get("n_freqs")
    freq_span_hz = meta.get("freq_span_hz")

    if outputs.get("fft"):
        bins = meta.get("fft_bins") or outputs["fft"][-1].shape[0]
        u = _sin_angle_axis(bins)
        # Coherent 2D aperture FFT, non-coherent (power) integration over range --
        # a target shows up regardless of its range, not just one at range 0.
        figs["fft"] = _heatmap(
            _to_numpy_abs_db(outputs["fft"][-1]),
            "Azimuth-Elevation power (non-coherent over range)",
            x=u, y=u, xlabel="azimuth sin(θ)", ylabel="elevation sin(θ)",
        )

    for key, title, aperture_label in [
        ("range_az", "Range-Azimuth power (non-coherent over elevation)", "azimuth sin(θ)"),
        ("range_el", "Range-Elevation power (non-coherent over azimuth)", "elevation sin(θ)"),
    ]:
        if outputs.get(key):
            bins = meta.get(f"{key}_bins") or outputs[key][-1].shape[0]
            x = _sin_angle_axis(bins)
            if n_freqs and freq_span_hz and bins == n_freqs:
                y = _range_axis(bins, freq_span_hz)
                ylabel = "range (m)"
            else:
                # bins != n_freqs (or metadata unavailable): mapping is nontrivial
                # (zero-pad/truncate before the DFT), fall back to raw bin indices.
                y = np.arange(bins)
                ylabel = "range (bins)"
            figs[key] = _heatmap(
                _to_numpy_abs_db(outputs[key][-1]), title,
                x=x, y=y, xlabel=aperture_label, ylabel=ylabel,
            )

    if outputs.get("subspace_err"):
        errs = [float(e) for e in outputs["subspace_err"]]
        fig = go.Figure(data=go.Scatter(y=errs, mode="lines+markers"))
        fig.update_layout(
            title="Subspace error (Frobenius) per frame",
            xaxis_title="Frame",
            yaxis_title="Error",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        figs["subspace_err"] = fig

    # Comms head (opt-in "product" -- see webapp/pipeline_registry.py "comms"):
    # BER/EVM-per-frame lines + a constellation snapshot of the last frame.
    if outputs.get("ber"):
        bers = [float(b) for b in outputs["ber"]]
        comms_meta = outputs.get("_comms_meta") or {}
        combining = comms_meta.get("combining", "?")
        title = f"Comms head BER ({combining}"
        gains = [float(g) for g in (outputs.get("comm_array_gain_db") or [])
                 if g is not None and np.isfinite(float(g))]
        if gains:
            title += f", array gain {np.mean(gains):.1f} dB"
        title += ")"
        fig = go.Figure(data=go.Scatter(y=bers, mode="lines+markers"))
        fig.update_layout(
            title=title,
            xaxis_title="Frame",
            yaxis_title="BER",
            yaxis_type="log",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        figs["ber"] = fig

    if outputs.get("evm"):
        evms = [float(e) for e in outputs["evm"]]
        fig = go.Figure(data=go.Scatter(y=evms, mode="lines+markers"))
        fig.update_layout(
            title="Comms head EVM per frame",
            xaxis_title="Frame",
            yaxis_title="EVM",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        figs["evm"] = fig

    if outputs.get("comm_data_eq"):
        data_np = _to_numpy_complex(outputs["comm_data_eq"][-1])
        fig = go.Figure(data=go.Scatter(
            x=data_np.real, y=data_np.imag, mode="markers",
            marker=dict(size=4),
        ))
        fig.update_layout(
            title="Comms head constellation (last frame, equalized)",
            xaxis_title="I",
            yaxis_title="Q",
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        figs["comm_const"] = fig

    return figs


def placeholder_figure(message: str) -> go.Figure:
    """An empty figure carrying a centered message (used for errors / no-data)."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper",
        x=0.5, y=0.5, font=dict(size=14),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20), height=360,
    )
    return fig
