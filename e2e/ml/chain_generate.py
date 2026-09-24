"""
Radar-ML corpus generation AS A COMPOSED `e2e.simulation.Simulation` RUN.

This is the migration `report/chain_integration_design.html`'s "The result" section
specifies: the corpus generator no longer calls the analytic point-target synthesizer
(`e2e.chain.rd_synth.synthesize_adc`, still available as `e2e.ml.dataset`'s explicitly-
labelled CI/offline fallback) directly. Instead it builds an `e2e.simulation.Simulation`
out of the SAME blocks the runtime pipeline uses, ray-traces via
`e2e.environment.blocks.RTEnvironmentBlock`, and runs it frame by frame:

    RTEnvironmentBlock (ray-traced CFR + labels)
        -> TxPowerStage                   # sqrt(P_tx) at the SOURCE (link budget on)
        -> InterconnectStage(InterconnectBlock)   # ON by default
        -> DechirpBlock                   # crossing: CFR -> RX-time ADC
        -> FrontEndBlock                  # RF front end ON THE BEAT RECORD -- ON by default
        -> ThermalNoiseBlock(mode="once") # the ONE thermal injection
        -> ImpairmentBlock                # phase noise / leakage / clutter, per-frame
        -> IFHighPassBlock                # ON by default
        -> QuantizerBlock                 # ADC digitization
        -> SinkBlock                      # persists adc + labels + meta (serial)
        -> RangeTransformBlock            # crossing: RX-time ADC -> range cube
        -> RadarCubeBlock                 # Doppler half -> range-Doppler (downstream)

THAT ORDER IS THE ONE-CHAIN CONTRACT's (notes/ONE_CHAIN_CONTRACT_2026-09-24.md section
1.2), and it is what this module builds by DEFAULT as of 2026-09-24. The order the
stored corpora were generated under -- front end on `ifft(CFR)` before the dechirp,
`sqrt(P_tx)` applied inside the thermal block -- is reachable as
`build_chain_simulation(..., composition="legacy_impulse")` and exists only for those
corpora's bit-parity gates; see `COMPOSITION_LEGACY`.

RFFE/interconnect are config-gated (`use_rffe`/`use_interconnect`) but default ON --
that is the point of this module: the pre-migration generator realized zero of the
chain's analog stages (see the design notes' "The problem"). `RFFEBlock` is normally
sized for the 1024-element IMAGING array; `build_chain_simulation` forces its `n` to
`cfg.n_rx` (the radar's actual receive-channel count, e.g. 4 for `ti_iwr1443`) unless
the caller overrides `rffe_kwargs["n"]` explicitly -- see that function's docstring.

`generate_chain_corpus` is the CLI/script entry point (mirrors `e2e.ml.dataset.
generate_dataset`'s signature/on-disk contract as closely as the two producers'
different internals allow) and needs real Sionna RT (`RTEnvironmentBlock`); tests build
`Simulation`s directly via `build_chain_simulation(..., environment_block=<stub>)` with a
synthetic CFR-emitting stand-in, so the composition itself is exercised without Sionna.

Written samples share ONE on-disk manifest schema with `e2e.ml.dataset.generate_dataset`
(the "On-disk dataset layout" that module documents): `write_manifest` -- which THIS
module calls, not duplicates -- is the shared tail. `SinkBlock` writes the per-frame
`.npz` (`adc`/`labels`/`meta`, matching field names) directly; this module is the glue
that names files consistently with `write_manifest`'s `sequences` structure.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from e2e import frames
from e2e.blocks import CircuitStage, InterconnectBlock, InterconnectStage, RFFEBlock
from e2e.chain.dechirp import DechirpBlock
from e2e.chain.receive import (IFHighPassBlock, ImpairmentBlock, QuantizerBlock,
                               RadarCubeBlock)
from e2e.chain.transforms import range_transform_for
from e2e.environment.blocks import RTEnvironmentBlock
from e2e.frames import FrameCapabilities
from e2e.ml.blocks import CFRCaptureStage, SinkBlock
from e2e.ml.dataset import DATASETS_DIR, finalize_input_scale
from e2e.simulation import Simulation

DEFAULT_LABEL_CLASSES = ("vehicle", "pedestrian")

#: THE composition this module builds by default (owner 2026-09-24, ballot answer 1B;
#: notes/ONE_CHAIN_CONTRACT_2026-09-24.md section 1.2). The front end acts on the
#: SAMPLED BEAT RECORD after the dechirp, `sqrt(P_tx)` is applied at the source, and
#: thermal noise is injected exactly ONCE -- which is what makes
#: `tests/test_ml_link_budget.py`'s two F81 properties hold instead of being xfails.
COMPOSITION_FULL = "full"

#: The v1.0 order: `CircuitStage(RFFEBlock)` on `ifft(CFR)` BEFORE the dechirp,
#: `ThermalNoiseBlock(mode="legacy")` carrying `sqrt(P_tx)` itself, and no
#: `TxPowerStage`. It exists for exactly ONE reason -- the stored corpora
#: (`b1_demo_cfr`, `b1_demo_cfr_ka`, every `benchmark_v1*`) were generated that way and
#: `tests/test_ml_store_cfr.py` / `tests/test_webapp_live_chain.py` read max |diff| = 0
#: CODES against them, a zero tolerance that a differently-ordered RNG consumption
#: fails at any noise level. It is a recorded fact about files on disk, not an
#: alternative physics: F97c measures the two placements' signal paths as agreeing to
#: BELOW ONE LSB (8.6e-05 rel-RMSE vs a 1.76e-04 LSB) and their noise floors to
#: 8.1e-7 dB. Nothing new should select it.
COMPOSITION_LEGACY = "legacy_impulse"

COMPOSITIONS = (COMPOSITION_FULL, COMPOSITION_LEGACY)

#: Interconnect transfer function used for ML corpora. See build_chain_simulation for why
#: the block's own placeholder default is not used here.
DEFAULT_INTERCONNECT_CSV = (Path(__file__).resolve().parent.parent
                            / "data" / "interconnect" / "tessera_case3_s21_77ghz.csv")

#: Width of the frequency window the interconnect S21(f) is resampled over (see
#: `_interconnect_band_hz`), matching the historical 77 GHz literal's 6 GHz span
#: (75-81 GHz) -- wide enough to characterize insertion-loss ripple around the carrier,
#: not the (much narrower) chirp sweep itself.
_INTERCONNECT_EVAL_SPAN_HZ = 6e9

#: The literal band `build_chain_simulation` used for every 77 GHz config before
#: 2026-09-23 (Ka-band re-founding, owner decision). Preserved EXACTLY (not re-derived
#: as `f0_hz +- span/2`, which would give (74e9, 80e9), not (75e9, 81e9)) so every
#: corpus generated at f0=77e9 -- b1_bench_v3, benchmark_v1_D2/D4, b1_demo_cfr --
#: regenerates identically. See `_interconnect_band_hz`.
_LEGACY_77GHZ_INTERCONNECT_BAND_HZ = (75e9, 81e9)


def _interconnect_band_hz(cfg) -> Tuple[float, float]:
    """The frequency span the interconnect's S21(f) is resampled onto for `cfg`.

    Three cases, in order: (1) a config that carries explicit `f_start_hz`/`f_stop_hz`
    fields uses them directly -- `RadarConfig` has none today, but a future config that
    does should not be second-guessed; (2) the historical f0=77e9 configs (benchmark_v1,
    ti_iwr1443, radial_like, ddma_wide_v1) get back the EXACT literal band every corpus
    generated before this change used, bit-for-bit (see
    `_LEGACY_77GHZ_INTERCONNECT_BAND_HZ`); (3) everything else -- including
    `benchmark_v1_ka` (owner decision 2026-09-23, Ka-band re-founding) -- derives the
    band from its own carrier: `f0_hz +- _INTERCONNECT_EVAL_SPAN_HZ / 2`, so a config
    generated for a different band is not silently mapped over 75-81 GHz.
    """
    start = getattr(cfg, "f_start_hz", None)
    stop = getattr(cfg, "f_stop_hz", None)
    if start is not None and stop is not None:
        return (float(start), float(stop))
    if float(cfg.f0_hz) == 77e9:
        return _LEGACY_77GHZ_INTERCONNECT_BAND_HZ
    half = _INTERCONNECT_EVAL_SPAN_HZ / 2.0
    return (float(cfg.f0_hz) - half, float(cfg.f0_hz) + half)


# --------------------------------------------------------------------------------
# ImpairmentBlock -> SinkBlock provenance glue
# --------------------------------------------------------------------------------
def default_domain_randomizer(
    *, phase_noise_dbc_hz_range: Tuple[float, float] = (-95.0, -75.0),
    # Re-derived 2026-08-18 for the ABSOLUTE thermal reference (F35 flip). These are now
    # dB ABOVE the thermal floor, not fractions of the cube's own peak, so the old
    # negative ranges would have injected impairments ~60 dB too weak to exist.
    #   leakage: 30-40 dB TX-RX isolation at P_tx 12 dBm over an -85 dBm floor -> 57-67 dB.
    #   clutter: a clutter-to-noise spread around the +10 dB nominal (re-anchored
    #            2026-08-24, plan A15; see ClutterParams.total_relative_db for the
    #            corrected justification -- the level is a difficulty ASSUMPTION
    #            anchored on road-discrete cell CNR of ~10-30 dB with a heavy tail,
    #            +10 at the hot edge). The range is deliberately WIDE (0-18 dB,
    #            median drawn-cell CNR ~24-42 across it): the anchor itself is
    #            uncertain, so the corpus randomizes across the plausible band
    #            rather than pretending one number is known.
    leakage_relative_db_range: Tuple[float, float] = (57.0, 67.0),
    clutter_relative_db_range: Tuple[float, float] = (0.0, 18.0),
) -> Callable[[int, "torch.Generator"], Dict[str, Any]]:
    """Build the `chain_params` callable `ImpairmentBlock` expects (see its docstring):
    `(frame_index, rng) -> {"phase_noise": {...}, "leakage": {...}, "clutter": {...}}`,
    one uniform draw per stage per frame from `rng` (a `torch.Generator` on the ADC's
    device, seeded deterministically by `ImpairmentBlock` from its own `seed` -- see
    that class). This is the per-frame domain randomization requirement 1 of this
    shard's brief: every frame gets its own draw, and the resolved values are recorded
    (via `_ImpairmentStage`) into the written sample.
    """

    def _u(rng: "torch.Generator", lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=rng, device=rng.device).item())

    def _sample(frame_index: int, rng: "torch.Generator") -> Dict[str, Any]:
        return {
            "phase_noise": {"psd_dbc_hz_at_ref": _u(rng, *phase_noise_dbc_hz_range)},
            "leakage": {"leakage_relative_db": _u(rng, *leakage_relative_db_range)},
            "clutter": {"total_relative_db": _u(rng, *clutter_relative_db_range)},
        }

    return _sample


# --------------------------------------------------------------------------------
# Chain-topology provenance: what a frame does NOT otherwise record
# --------------------------------------------------------------------------------
class _ChainFlagsStage:
    """Pass-through serial stage: stamp every frame's state with the CONFIG-LEVEL
    chain flags this composition actually ran with -- which analog stages were
    wired in, and the ADC bit depth. These are constants of the whole `Simulation`
    run, not per-frame products, but `SinkBlock` only writes what `state` carries at
    the point it runs, so they are re-asserted on every `apply()` to reach every
    frame's meta (see `e2e.ml.blocks._EXTRA_META_KEYS`).

    Before this, a frame recorded its noise seeds, its impairment severities and its
    IF high-pass corner, but NOT whether the RF front end, the interconnect or the
    link budget ran, nor the ADC's bit depth -- so `webapp.pipeline_runner`'s
    live-chain gate had to guess at a mismatch's cause instead of naming it. A frame
    written before this stage existed simply carries none of these keys, and the
    gate's wording for that case is unchanged.

    `f0_hz`/`band_hz` (added 2026-09-23, F91) close a second gap the same finding named:
    a frame's meta recorded `if_hpf_corner_hz` and these chain flags but never the
    carrier itself, so which sensing band a corpus was generated at was only
    recoverable by cross-referencing the manifest's `config` block by hand.
    """

    frame_capabilities = FrameCapabilities(
        accepts_mimo=True, chirps=frames.CHIRP_NATIVE, domain=frames.DOMAIN_CFR,
    )

    def __init__(self, use_rffe: bool, use_interconnect: bool, use_link_budget: bool,
                quant_bits: int, f0_hz: float, band_hz: Tuple[float, float],
                composition: str = COMPOSITION_FULL):
        self._extra: Dict[str, Any] = {
            "use_rffe": bool(use_rffe),
            "use_interconnect": bool(use_interconnect),
            "use_link_budget": bool(use_link_budget),
            "quant_bits": int(quant_bits),
            "f0_hz": float(f0_hz),
            "band_hz": [float(band_hz[0]), float(band_hz[1])],
            # WHICH chain made this frame (added 2026-09-24, one-chain contract
            # section 3.5 / 5.4). Without it a stored corpus cannot say whether its
            # front end sat on `ifft(CFR)` before the dechirp or on the beat record
            # after it, and a live-vs-stored gate reading max |diff| != 0 has no way
            # to name the cause. See `COMPOSITIONS`.
            "composition": str(composition),
            "front_end_placement": ("impulse" if composition == COMPOSITION_LEGACY
                                    else "beat"),
        }

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self._extra)


# --------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------
def build_chain_simulation(
    scenario, cfg, out_dir, *, tag: str = "sample",
    use_rffe: bool = True, use_interconnect: bool = True,
    rffe_kwargs: Optional[Dict[str, Any]] = None,
    interconnect_kwargs: Optional[Dict[str, Any]] = None,
    impairment_chain_params=None, impairment_seed: int = 0,
    quant_bits: int = 12, environment_block=None,
    label_grid=None, label_classes: Optional[Sequence[str]] = DEFAULT_LABEL_CLASSES,
    device=None, k: int = 1, base_scene: str = "flat",
    use_transmit_chain: bool = False, tx_pa_config=None,
    coherent_targets: bool = True, antenna_pattern: Optional[str] = None,
    ground_scattering_coefficient: Optional[float] = None,
    samples_per_src: Optional[int] = None,
    use_link_budget: bool = True,
    use_if_hpf: bool = True, if_hpf_kwargs: Optional[Dict[str, Any]] = None,
    store_cfr: bool = False, store_paths: bool = False,
    composition: str = COMPOSITION_FULL,
) -> Simulation:
    """Compose ONE radar-ML `Simulation` run (see module docstring for the block list).

    `composition` (added 2026-09-24) picks the stage ORDER, and it is the one-chain
    contract's central decision rather than a style flag -- see `COMPOSITION_FULL` /
    `COMPOSITION_LEGACY`. `"full"` (the default) is the physics: `sqrt(P_tx)` at the
    source, the front end on the beat record, ONE thermal injection. `"legacy_impulse"`
    reproduces the order the stored corpora were generated under, for their bit-parity
    gates and nothing else.

    `coherent_targets` / `antenna_pattern` are the 2026-08-17 target-physics fix and are
    forwarded to `RTEnvironmentBlock` (they are ignored when the caller supplies its own
    `environment_block`). `coherent_targets=True` adds each object's coherent specular
    return -- without it a ray-traced target is pure Monte-Carlo speckle and earns almost
    none of the chain's coherent processing gain (see the "HYBRID RT" banner in
    `e2e.environment.rt_signal_chain`). `antenna_pattern=None` takes
    `rt_scene_build.DEFAULT_ANTENNA_PATTERN` (`"tr38901"`, directive), which is what keeps
    the flat scene's nadir ground bounce from owning the cube peak -- and therefore from
    owning the reference `ImpairmentBlock`'s relative-power stages calibrate against. Pass
    `coherent_targets=False, antenna_pattern="iso"` to reproduce a pre-2026-08-17 corpus.

    `environment_block=None` (default) builds `RTEnvironmentBlock(scenario, cfg,
    device=device, label_grid=label_grid, label_classes=label_classes)` -- real ray
    tracing, needs Sionna. Pass a stand-in (anything exposing `get_S_pars`/`step`/
    `reset`, optionally `get_state_updates`/`array_shape`) to run the same composition
    without Sionna -- this is how the test suite exercises it (no `RUN_SIONNA` gate on
    the composition itself, only on real ray tracing).

    `use_rffe`/`use_interconnect` (default True/True -- ON by default is the point of
    this module, see the module docstring) gate `CircuitStage(RFFEBlock(...))` /
    `InterconnectStage(InterconnectBlock(...))`. `rffe_kwargs` passes through to
    `RFFEBlock`, with `n` forced to `cfg.n_rx` UNLESS the caller already set it --
    `RFFEBlock.apply_circuit` requires `n == s_pars.shape[0]` (the RX axis), and the
    block's own default sizing is for the 1024-element imaging array, not a 4-RX
    automotive radar; passing the imaging-array default here would raise a `view()`
    shape error on the very first frame. `interconnect_kwargs` passes through to
    `InterconnectBlock` unchanged (no array-size dependency -- it broadcasts over every
    leading axis).

    `impairment_chain_params` is `ImpairmentBlock`'s `chain_params` (a fixed dict, or a
    `(frame_index, rng) -> dict` callable for per-frame domain randomization -- see
    `default_domain_randomizer`); `impairment_seed` seeds its per-frame sub-seed
    sequence. The block chain runs through `_ImpairmentStage` (see that class) so its
    JSON-unserializable dataclass provenance still reaches the written sample.

    `store_cfr=True` (owner-directed 2026-09-23) inserts a `CFRCaptureStage` as the
    FIRST serial stage and puts the `SinkBlock` into sidecar mode, so each written
    sample is accompanied by `<stem>.cfr.npy`: the ray-traced frame exactly as it
    entered the chain, before the RFFE. That is the expensive half of generation
    (~11-13 s/scene of ray tracing, vs ~2 s for everything after it), so storing it
    lets the whole analog/digital chain be re-run live from a corpus frame. Costs 64
    MiB/frame at `benchmark_v1` and nothing else: the `.npz` keys are unchanged, and
    with the flag off not one byte of the corpus differs.

    `store_paths=True` additionally writes each frame's RAY-TRACED PATH LIST to a
    `<stem>.paths.npz` sidecar (MEASURED ~1-2 MiB/frame on a `benchmark_v1`/D2 scene,
    ~30x smaller than the dense CFR's 64 MiB): the owner's preferred durable form, from
    which the channel can be re-synthesised without re-tracing. It requires the
    environment block to EMIT that list from `get_state_updates()` under
    `blocks.PATHS_CAPTURE_KEY`; `RTEnvironmentBlock` does so via `rt_cfr_frame`'s
    `capture=` out-parameter (`e2e.environment.rt_signal_chain._fill_rt_paths_capture`,
    landed 2026-09-23). Re-synthesis FROM the sidecar is a separate, not-yet-built
    step -- the sidecar only carries what the closed-form `range_migration=True`
    branch (`cfr_from_paths` + `coherent_target_cfr`) would need; nothing here or in
    `RTEnvironmentBlock` reads it back yet.

    `k` is `Simulation`'s required subspace-tracking-rank argument; this composition
    has no `subspace_block`, so `k` only sizes the (otherwise-unused) `U_true`/rank
    diagnostic `Simulation.feed_forward`'s frequency-domain branch always computes --
    kept small (default 1) to minimize that overhead.
    """
    env = environment_block if environment_block is not None else RTEnvironmentBlock(
        scenario, cfg, base_scene=base_scene, device=device,
        label_grid=label_grid, label_classes=label_classes,
        coherent_targets=coherent_targets, antenna_pattern=antenna_pattern,
        ground_scattering_coefficient=ground_scattering_coefficient,
        samples_per_src=samples_per_src,
    )

    if composition not in COMPOSITIONS:
        raise ValueError(
            f"unknown composition {composition!r}; expected one of {COMPOSITIONS}. "
            f"{COMPOSITION_FULL!r} is the contract's chain; {COMPOSITION_LEGACY!r} "
            f"exists only to reproduce a stored corpus bit-for-bit."
        )
    legacy = composition == COMPOSITION_LEGACY

    serial_stages: List[Any] = []

    # Chain-topology provenance (see _ChainFlagsStage): a pass-through, so its
    # position relative to the rest of the list has no effect on any stored payload --
    # placed first only so it reads first.
    serial_stages.append(_ChainFlagsStage(use_rffe, use_interconnect, use_link_budget,
                                          quant_bits, cfg.f0_hz,
                                          _interconnect_band_hz(cfg),
                                          composition=composition))

    # FIRST, ahead of even the transmit tributary: the frame as it ENTERS the chain is
    # what "store the ray-traced channel" means -- anything later would have the RF
    # front end (or a TX modulation) already folded in, and could not be replayed
    # through those same stages.
    if store_cfr:
        serial_stages.append(CFRCaptureStage())

    # The transmit tributary, when enabled: generate the waveform, distort it in the
    # amplifier, then merge its spectrum into the channel response. These come first
    # because everything after them is the receive side. Off by default so the composed
    # chain stays on the long-proven path unless a caller asks for the TX stage.
    if use_transmit_chain:
        from e2e.chain.waveform import ModulateBlock, TxPABlock, WaveformBlock
        from e2e.circuit.tx_pa import TxPA, TxPAConfig
        pa = TxPA(tx_pa_config if tx_pa_config is not None else TxPAConfig())
        serial_stages.append(WaveformBlock(
            kind="fmcw", n_tx=cfg.n_tx, n_chirp=cfg.n_chirps, n_t=cfg.n_samples,
            bw=float(cfg.bandwidth_hz),
        ))
        serial_stages.append(TxPABlock(pa))
        serial_stages.append(ModulateBlock(tx_pa=pa, bandwidth_hz=float(cfg.bandwidth_hz)))
    # sqrt(P_tx) AT THE SOURCE under the FULL composition (contract section 1.4): the
    # transmit-power scalar belongs to the transmitted waveform, and applying it here
    # -- ahead of everything receive-side -- is what stops receiver noise scaling with
    # transmit power. Under the legacy order it stays inside ThermalNoiseBlock, where
    # it was when the corpora were generated, and where F81 measured its coupling.
    if use_link_budget and not legacy:
        from e2e.chain.link_budget import TxPowerStage
        serial_stages.append(TxPowerStage(cfg))
    if use_rffe:
        kwargs = dict(rffe_kwargs or {})
        kwargs.setdefault("n", int(cfg.n_rx))
        # F63: the block's OWN default is physical_scale=False, which divides the frame
        # by its mean magnitude -- fine for the classic imaging pipeline, fatal here,
        # because this chain then installs an absolute kTBF floor beneath it and every
        # impairment dB is referenced to that floor. The ML corpus takes the absolute
        # scale. A caller may still override explicitly, and `ThermalNoiseBlock` will
        # refuse the resulting composition rather than silently produce it.
        kwargs.setdefault("physical_scale", True)
        # Same base seed ThermalNoiseBlock/ImpairmentBlock take below (impairment_seed):
        # each block seeds its OWN torch.Generator from it, so the three noise sources
        # stay independent while a corpus frame's recorded seed alone determines all of
        # them. Without this, RFFEBlock drew its thermal noise from the global torch RNG,
        # so replaying a stored s_pars through the composed chain twice reproduced
        # nothing downstream of it (measured rel-RMSE 0.58-0.62 on the ADC output).
        kwargs.setdefault("seed", impairment_seed)
        rffe = RFFEBlock(**kwargs)
        if legacy:
            serial_stages.append(CircuitStage(rffe))
    if use_interconnect:
        # The block's own default is an unnormalised 11-tap boxcar placeholder, which
        # convolves the range profile with an 11-sample boxcar: a point target smears
        # across 11 bins with 0 dB sidelobes, destroying range resolution. Measured, then
        # replaced here with a real simulated interconnect (0.51-0.55 dB passive loss,
        # 0.034 dB ripple in band) that leaves the range response intact -- width 1 bin,
        # sidelobes -75 dB. The classic pipeline's default is deliberately untouched.
        ic_kwargs = dict(interconnect_kwargs) if interconnect_kwargs else {}
        ic_kwargs.setdefault("transfer_csv", str(DEFAULT_INTERCONNECT_CSV))
        ic_kwargs.setdefault("band_hz", _interconnect_band_hz(cfg))
        serial_stages.append(InterconnectStage(InterconnectBlock(**ic_kwargs)))
    serial_stages.append(DechirpBlock(cfg))
    # THE FRONT END, on the beat record (contract section 1.1 fact 3, F97b/F97c). For
    # an ideal linear chirp the dechirp commutes with a memoryless ENVELOPE
    # nonlinearity, so the LNA/mixer cascade applies unchanged to the beat samples --
    # and there it sees the signal a real amplifier sees, referenced to the beat sample
    # rate, instead of `ifft(CFR)`, a signal no amplifier sees (F96). `from_rffe`
    # carries every knob across, including a hand-edited per-element config table.
    if use_rffe and not legacy:
        from e2e.chain.frontend import FrontEndBlock
        serial_stages.append(FrontEndBlock.from_rffe(rffe, cfg))
    # The link budget goes BETWEEN the front end and the impairments, and the position
    # is the whole point. Impairments are specified relative to a reference; before this
    # stage runs there is no absolute reference in the chain for them to be relative TO,
    # so they were calibrated against the only thing available -- the cube's own
    # contents, which scale with the target. That is F35's ceiling and F42's missing
    # floor, and both dissolve once the cube is on an absolute scale with a real k*T*B*F
    # floor beneath it.
    #
    # mode: under the FULL composition this is ONE injection in the whole chain --
    # `"once"` adds nothing when the front end already injected (it stamps
    # `noise_injected_by`) and is the only floor when no front end is configured, so
    # the two mechanisms F81 measured cannot both run. `"legacy"` is both of them, as
    # the corpora were generated.
    if use_link_budget:
        from e2e.chain.link_budget import ThermalNoiseBlock
        serial_stages.append(ThermalNoiseBlock(
            cfg, seed=impairment_seed, mode=("legacy" if legacy else "once")))
    serial_stages.append(
        ImpairmentBlock(cfg, impairment_chain_params, seed=impairment_seed)
    )
    # The IF high-pass sits AFTER the impairments (the close-in leakage/bumper tones it
    # exists to suppress must be present) and BEFORE the quantizer (protecting the ADC's
    # dynamic range is its purpose: QuantizerBlock AGCs full scale off the frame peak,
    # which without this filter is the leakage tone, not a target). ON by default --
    # every real FMCW receiver has one, and its absence biased every detection number
    # pessimistically (physics audit entry 9; release-plan A2).
    if use_if_hpf:
        serial_stages.append(IFHighPassBlock(cfg, **(if_hpf_kwargs or {})))
    serial_stages.append(QuantizerBlock(bits=quant_bits))

    # The SINK is a serial stage, not a downstream block, and the position is load
    # bearing: what a corpus sample stores is the DIGITISED BEAT RECORD (`adc`), and
    # the range transform that follows crosses into the cube domain, at which point
    # `Simulation` drops the previous domain's payload from state (it must -- a stale
    # `adc` outliving the crossing is how a block computes silently on pre-transform
    # data). So the sink runs where the record it persists still exists. `SinkBlock`
    # is documented as working in either position and returns `{}` either way, so no
    # output moves; the `.npz` keys and contents are unchanged.
    serial_stages.append(SinkBlock(out_dir, tag=tag, store_cfr=store_cfr,
                                   store_paths=store_paths))
    # THE range transform -- the one range FFT (contract section 1.2 row 11), built on
    # the SCORED protocol (`transforms.RD_RANGE_PROTOCOL`: hann, DC removal,
    # uncropped), which is the protocol every stored corpus was compressed under and
    # `RadarCubeBlock` now refuses a departure from by name. Present in BOTH
    # compositions: `RangeTransformBlock` + `rd_from_cube` IS `adc_to_rd`, operation
    # for operation, so the legacy arm's `radar_cube` is bit-identical to what it was
    # when `RadarCubeBlock` ran the range FFT itself.
    serial_stages.append(range_transform_for(cfg))

    downstream_blocks = [RadarCubeBlock(cfg)]

    return Simulation(
        environment_block=env,
        downstream_blocks=downstream_blocks,
        k=k,
        serial_stages=serial_stages,
        radar_cfg=cfg,
        composition=composition,
    )


def _corpus_identity_tag(dataset_dir: Path) -> str:
    """The scene-identity salt for a corpus at `dataset_dir`: its last TWO path
    components (`parent/name`), machine-stable (never an absolute path).

    Two components, not just the leaf (B1 corpus review finding, 2026-08-25): the
    leaf alone is `f"{cfg_name}_{tier}"`, so two corpora under different `--out`
    roots sharing (config, tier, seed) salted identically and drew BIT-IDENTICAL
    scenes -- a 2-frame smoke run collided with the real corpus's train split
    exactly that way, undeclared by the overlap gate's allowlist. NOTE: this
    derivation change alters the drawn scenes of any corpus regenerated at the same
    (config, tier, seed) relative to pre-2026-08-25 runs -- deliberate; the
    manifest's `corpus_tag` records which salt a corpus actually used.
    """
    return f"{dataset_dir.parent.name}/{dataset_dir.name}"


# --------------------------------------------------------------------------------
# Corpus generation entry point
# --------------------------------------------------------------------------------
def generate_chain_corpus(
    cfg_name: str, tier: str, n_scenes: int, out_dir=None, *, seed: int = 0,
    frames_per_scene: int = 1, use_rffe: bool = True, use_interconnect: bool = True,
    rffe_kwargs: Optional[Dict[str, Any]] = None,
    interconnect_kwargs: Optional[Dict[str, Any]] = None,
    quant_bits: int = 12, splits: Tuple[float, ...] = (0.8, 0.1, 0.1),
    range_stride: int = 4, n_azimuth: int = 192, device=None,
    label_classes: Optional[Sequence[str]] = DEFAULT_LABEL_CLASSES,
    randomizer: Optional[Callable[[int, "torch.Generator"], Dict[str, Any]]] = None,
    use_local_assets: bool = True, use_transmit_chain: bool = False,
    coherent_targets: bool = True, antenna_pattern: Optional[str] = None,
    ground_scattering_coefficient: Optional[float] = None,
    samples_per_src: Optional[int] = None,
    allow_unanswerable: bool = False,
    use_if_hpf: bool = True, if_hpf_kwargs: Optional[Dict[str, Any]] = None,
    store_cfr: bool = False, store_paths: bool = False,
) -> Path:
    """Generate a radar-ML corpus by RUNNING THE COMPOSED CHAIN, one `Simulation` per
    scene (real ray tracing -- needs Sionna; see `build_chain_simulation`).

    Mirrors `e2e.ml.dataset.generate_dataset`'s signature and on-disk contract: `n_scenes`
    independent scenes (`e2e.ml.scenes.sample_scene`, same tier/seed convention), each
    yielding `frames_per_scene` frames of ONE `e2e.simulation.Simulation.run()` call,
    written under `<out_dir>/<cfg_name>_<tier>/` with a `manifest.json` via `e2e.ml.
    dataset.write_manifest` -- so a directory this function writes loads via
    `e2e.ml.dataset.RadarFrameDataset` exactly like one `generate_dataset` wrote.

    Impairments are domain-randomized PER FRAME via `randomizer` (default
    `default_domain_randomizer()`) -- each scene's `ImpairmentBlock` gets a distinct
    seed (`seed + i * frames_per_scene`) so frames across the whole corpus don't repeat
    a randomization draw.

    `coherent_targets` / `antenna_pattern` (defaults `True` / `None` -> directive) are the
    2026-08-17 target-physics fix; see `build_chain_simulation`. **Every corpus generated
    before 2026-08-17 used the broken combination** (speckle-only targets, isotropic
    elements, mirror ground) and its targets sit below their own map background --
    regenerate rather than reuse. Pass `coherent_targets=False, antenna_pattern="iso"`
    only to reproduce one of those deliberately.

    `store_cfr=True` writes each sample's entering ray-traced frame to an uncompressed
    `<stem>.cfr.npy` sidecar (see `build_chain_simulation` and `e2e.ml.storage.
    write_cfr_sidecar`): +64 MiB/frame at `benchmark_v1`, and the `.npz` files and the
    manifest are bit-identical to a run without it.

    `store_paths=True` writes the ray-traced PATH LIST sidecar instead of/alongside the
    dense one -- see `build_chain_simulation`; it needs an environment-side capture
    hook that does not exist yet.

    `use_transmit_chain` defaults to **False**, and that default changed in v1.1. It used
    to be True, which silently cancelled the target-physics fix above: MEASURED, with the
    TX tributary on, median target-vs-background is **-1.6 dB**; with it off, **+7.4 dB**.
    `ModulateBlock` multiplies `s_pars` by the transmitted chirp's spectrum, but
    `rt_cfr_frame`'s `s_pars` axis is already a beat/time index, not a conventional
    frequency response -- so enabling it applies a second, incompatible modulation. That
    convention clash is UNRESOLVED; until it is, RT corpora carry no TX-side PA
    distortion, and turning this back on will produce a corpus whose targets do not stand
    out from their own background.
    """
    from e2e.ml.dataset import write_manifest
    from e2e.ml.labels import LabelGrid
    from e2e.radar_config import PRESETS
    from e2e.environment.rt_scenes import RT_DIFFICULTY_TIERS, build_rt_tier_scenario

    if cfg_name not in PRESETS:
        raise ValueError(f"unknown radar config {cfg_name!r}; choices: {sorted(PRESETS)}")
    if tier not in RT_DIFFICULTY_TIERS:
        raise ValueError(f"unknown RT difficulty tier {tier!r}; "
                         f"choices: {sorted(RT_DIFFICULTY_TIERS)}")
    if frames_per_scene < 1:
        raise ValueError(f"frames_per_scene must be >= 1, got {frames_per_scene}")
    cfg = PRESETS[cfg_name]

    # Answerability guard (release-plan A3): refuse to generate a corpus on which the
    # detection benchmark is arithmetically incapable of being answered -- F43 made this
    # class of invalid corpus a discovered fact; this guard makes it an impossible state.
    # `allow_unanswerable=True` is the deliberate escape hatch (ablations, regression
    # reproductions); the resulting corpus must never back a detection benchmark.
    from e2e.radar_config import answerability_problems
    problems = answerability_problems(
        cfg, top_speed_mps=RT_DIFFICULTY_TIERS[tier].speed_mps[1])
    if problems and not allow_unanswerable:
        raise ValueError(
            f"({cfg_name}, {tier}) cannot support a detection benchmark:\n  - "
            + "\n  - ".join(problems)
            + "\nPass allow_unanswerable=True (CLI: --allow-unanswerable) only for a "
              "corpus that will never back a benchmark (see F43 / radar_config."
              "answerability_problems)")

    grid = LabelGrid.for_config(cfg, range_stride=range_stride, n_azimuth=n_azimuth)
    randomize = randomizer if randomizer is not None else default_domain_randomizer()

    out_root = Path(out_dir) if out_dir is not None else DATASETS_DIR
    dataset_dir = out_root / f"{cfg_name}_{tier}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    corpus_tag = _corpus_identity_tag(dataset_dir)

    sequences: List[List[str]] = []
    for i in range(n_scenes):
        # The RAY-TRACED tier builder, not the analytic one: this is what places real
        # decimated vehicle meshes (car/truck/bus/trolley) and, for the city tier, sets
        # the base scene. Sampling from e2e.ml.scenes here would have produced a corpus
        # of spheres with none of the mesh work in it.
        # dt MUST be the real frame period: sampled speeds are physical m/s, and
        # frame_scatterers differences the stored per-frame displacement by this same
        # dt. Omitting it inflated every velocity by frame_rate_hz (0-8 m/s tier ->
        # 0-80 m/s at the solver, past the unambiguous-velocity limit). Fixed 2026-08-16.
        scenario = build_rt_tier_scenario(
            tier, corpus_tag=corpus_tag, frame_idx=i, seed=seed,
            num_frames=frames_per_scene,
            dt=1.0 / float(cfg.frame_rate_hz),
            use_local_assets=use_local_assets,
        )
        tag = f"sample_scene{i:05d}"

        sim = build_chain_simulation(
            scenario, cfg, dataset_dir, tag=tag,
            base_scene=RT_DIFFICULTY_TIERS[tier].base_scene,
            use_transmit_chain=use_transmit_chain,
            use_rffe=use_rffe, use_interconnect=use_interconnect,
            rffe_kwargs=rffe_kwargs, interconnect_kwargs=interconnect_kwargs,
            impairment_chain_params=randomize, impairment_seed=seed + i * frames_per_scene,
            quant_bits=quant_bits, label_grid=grid, label_classes=label_classes, device=device,
            coherent_targets=coherent_targets, antenna_pattern=antenna_pattern,
            ground_scattering_coefficient=ground_scattering_coefficient,
            samples_per_src=samples_per_src,
            use_if_hpf=use_if_hpf, if_hpf_kwargs=if_hpf_kwargs,
            store_cfr=store_cfr, store_paths=store_paths,
        )
        sim.run(n_steps=frames_per_scene)

        scene_files = [f"{tag}_frame_{t:05d}.npz" for t in range(frames_per_scene)]
        sequences.append(scene_files)

    # F63 piece 2. The manifest is written first WITHOUT input_scale, then completed:
    # the constant is MEASURED through the production load path, which needs a manifest
    # to read. See `dataset.measure_input_scale` for why an analytic ADC-domain constant
    # was wrong by ~6600x and produced val_AP 0.0000 for thirteen epochs.
    manifest_path = write_manifest(dataset_dir, cfg, tier, sequences, grid=grid, seed=seed,
                                   snr_db=None, frames_per_scene=frames_per_scene,
                                   splits=splits, label_classes=label_classes or (),
                                   corpus_tag=corpus_tag)
    finalize_input_scale(manifest_path)
    return manifest_path


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m e2e.ml.chain_generate",
        description="Generate a labeled FMCW radar corpus by running the composed "
                    "block chain (RTEnvironmentBlock -> RFFE -> interconnect -> "
                    "dechirp -> impairments -> quantizer -> radar cube -> sink). "
                    "Needs Sionna RT.",
    )
    p.add_argument("--config", required=True, help="radar config preset name (see e2e.radar_config.PRESETS)")
    p.add_argument("--tier", required=True,
                   help="RT difficulty tier (see e2e.environment.rt_scenes.RT_DIFFICULTY_TIERS)")
    p.add_argument("--n", type=int, required=True, help="number of scenes to generate")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed (scene i uses seed + i)")
    p.add_argument("--out", default=None, help="output root directory (default: e2e/ml/datasets)")
    p.add_argument("--frames-per-scene", type=int, default=1,
                   help="consecutive motion-consistent frames per scene (default: 1)")
    p.add_argument("--no-rffe", action="store_true", help="disable the RFFE front-end stage")
    p.add_argument("--no-interconnect", action="store_true", help="disable the interconnect stage")
    p.add_argument("--quant-bits", type=int, default=12, help="ADC quantizer bit depth")
    # --- target physics (2026-08-17 fix; correct behaviour is the DEFAULT) -------------
    p.add_argument("--no-coherent-targets", action="store_true",
                   help="drop each object's coherent specular return, leaving only the "
                        "ray-traced diffuse speckle. REPRODUCES THE PRE-2026-08-17 BUG "
                        "(targets below their own map background) -- regression use only")
    p.add_argument("--antenna-pattern", default=None,
                   help="antenna ELEMENT pattern (iso|tr38901|dipole|hw_dipole); default "
                        "is rt_scene_build.DEFAULT_ANTENNA_PATTERN (directive). 'iso' "
                        "reproduces the pre-2026-08-17 nadir-ground-bounce behaviour")
    # DEFAULT FLIPPED 2026-08-24 to match generate_chain_corpus's own v1.1 default:
    # the CLI used to pass use_transmit_chain=True unless --no-transmit-chain was
    # given, silently re-enabling the ModulateBlock convention clash the function
    # default was flipped to avoid (targets sink to -1.6 dB median vs +7.4 dB; see
    # generate_chain_corpus's docstring). Caught while preparing the B1 launch
    # command -- a corpus generated through the bare CLI would have been broken.
    p.add_argument("--transmit-chain", action="store_true",
                   help="OPT IN to the TX tributary (waveform -> PA -> modulate). "
                        "OFF by default: MEASURED 2026-08-17, leaving it on costs ~9 dB "
                        "of target-to-background and erases the target-physics fix "
                        "entirely (suspected convention clash in ModulateBlock -- see "
                        "generate_chain_corpus's docstring). Do not enable for corpora")
    p.add_argument("--no-transmit-chain", action="store_true",
                   help="deprecated no-op (the TX tributary is now off by default; "
                        "kept so recorded pre-2026-08-24 command lines still run)")
    p.add_argument("--no-local-assets", action="store_true",
                   help="exclude the USER-SUPPLIED local mesh pool (use_local_assets="
                        "False), whose licences are unestablished (F21/F51). It does "
                        "NOT exclude the registered downloaded pool (Kenney CC0, the "
                        "cleared freestl meshes, dl_tram_google CC-BY): those are part "
                        "of every tier's vehicle sampler regardless of this flag. Frames "
                        "record their asset provenance either way, so a figure's licence "
                        "status is auditable per scene (F56, 2026-08-27)")
    p.add_argument("--ground-scattering", type=float, default=None,
                   help="ground-plane scattering coefficient for the flat base scene "
                        "(default: rt_scene_build.DEFAULT_GROUND_SCATTERING_COEFFICIENT). "
                        "0.2-0.4 is the physically honest range for asphalt at 3.8 mm but "
                        "costs ~37x the per-frame CFR time and ~8 dB of target margin -- "
                        "read that constant before setting it")
    p.add_argument("--samples-per-src", type=int, default=None,
                   help="Sionna PathSolver Monte-Carlo ray budget (default: Sionna's own "
                        "1e6). The dominant cost knob when the ground scatters diffusely: "
                        "1e5 cut the path count 10x with the target metric unchanged")
    p.add_argument("--no-if-hpf", action="store_true",
                   help="disable the IF high-pass between mixer and ADC (on by "
                        "default -- every real FMCW receiver has one; without it the "
                        "TX-RX leakage tone sets the ADC full scale). Reproduces "
                        "pre-A2 corpora")
    p.add_argument("--if-hpf-corner-range", type=float, default=None,
                   help="IF high-pass corner expressed as a RANGE in metres "
                        "(default 1.0; see IFHighPassBlock)")
    p.add_argument("--allow-unanswerable", action="store_true",
                   help="override the answerability guard (F43): generate even though "
                        "the (config, tier) pair cannot support a detection benchmark "
                        "(targets alias in Doppler, or the azimuth tolerance outresolves "
                        "the array). For ablations/regressions only")
    p.add_argument("--store-cfr", action="store_true",
                   help="also store each frame's RAY-TRACED CFR (the frame entering "
                        "the chain, before the RFFE) as an uncompressed "
                        "'<stem>.cfr.npy' sidecar, so the whole analog/digital chain "
                        "can be re-run live from a stored frame. +64 MiB/frame at "
                        "benchmark_v1 (measured); the .npz files are unchanged")
    p.add_argument("--store-paths", action="store_true",
                   help="also store each frame's RAY-TRACED PATH LIST as a compressed "
                        "'<stem>.paths.npz' sidecar (measured ~1-2 MiB/frame, ~30x "
                        "smaller than the dense CFR): the durable form the channel's "
                        "closed-form (range_migration=True) branch can be "
                        "re-synthesised from -- re-synthesis itself is a separate, "
                        "not-yet-built step. RTEnvironmentBlock emits the capture via "
                        "PATHS_CAPTURE_KEY")
    p.add_argument("--dry-run", action="store_true",
                   help="print the generation plan without ray-tracing/writing anything")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    import sys

    args = build_arg_parser().parse_args(argv)

    from e2e.radar_config import PRESETS

    if args.config not in PRESETS:
        print(f"unknown --config {args.config!r}; choices: {sorted(PRESETS)}", file=sys.stderr)
        return 2
    cfg = PRESETS[args.config]

    # RT_DIFFICULTY_TIERS, not scenes.DIFFICULTY_TIERS: this CLI drives the RAY-TRACED
    # corpus path, whose tier set (D0-D4, incl. the Munich city tier) is what
    # `generate_chain_corpus` itself validates against. Checking the analytic dict here
    # (the pre-2026-08-23 bug) made D4 unreachable from the CLI.
    from e2e.environment.rt_scenes import RT_DIFFICULTY_TIERS

    if args.tier not in RT_DIFFICULTY_TIERS:
        print(f"unknown --tier {args.tier!r}; choices: {sorted(RT_DIFFICULTY_TIERS)}", file=sys.stderr)
        return 2

    if args.dry_run:
        total_frames = args.n * args.frames_per_scene
        print("=" * 70)
        print(f"config:       {args.config}  (mimo={cfg.mimo}, n_rx={cfg.n_rx}, n_tx={cfg.n_tx})")
        print(f"tier:         {args.tier}")
        print(f"scenes:       {args.n}  x  frames_per_scene={args.frames_per_scene}  "
              f"= {total_frames} frames")
        print(f"rffe:         {'off' if args.no_rffe else 'on'}   "
              f"interconnect: {'off' if args.no_interconnect else 'on'}   "
              f"tx chain: {'ON (broken for corpora -- see --transmit-chain help)' if args.transmit_chain else 'off'}   "
              f"if hpf: {'OFF (pre-A2)' if args.no_if_hpf else 'on'}")
        # Say what the flag actually does: it gates the user-supplied LOCAL pool only.
        # The registered downloaded pool (Kenney CC0, the cleared freestl meshes,
        # dl_tram_google CC-BY) is sampled either way, so "Sionna-bundled only" was
        # never true and "public-figure safe" overstated it.
        print("local assets: " + ("OFF (user-supplied local pool excluded; the "
                                  "registered downloaded pool is still sampled -- "
                                  "check per-frame scene_provenance before publishing "
                                  "a render)"
                                  if args.no_local_assets else
                                  "ON (includes the licence-unestablished local pool)"))
        print(f"seed:         {args.seed}   quant_bits: {args.quant_bits}")
        print(f"store cfr:    {'ON (+64 MiB/frame sidecars at benchmark_v1)' if args.store_cfr else 'off'}"
              f"   store paths: {'ON (needs the RT capture hook -- see --store-paths)' if args.store_paths else 'off'}")
        from e2e.environment.rt_scene_build import (DEFAULT_ANTENNA_PATTERN,
                                                    DEFAULT_GROUND_SCATTERING_COEFFICIENT)
        print(f"coherent targets: {'OFF (pre-2026-08-17 bug)' if args.no_coherent_targets else 'on'}"
              f"   antenna pattern: "
              f"{args.antenna_pattern or DEFAULT_ANTENNA_PATTERN + ' (default)'}"
              f"   ground S: "
              f"{DEFAULT_GROUND_SCATTERING_COEFFICIENT if args.ground_scattering is None else args.ground_scattering}"
              f"   samples_per_src: {args.samples_per_src or 'sionna default'}")
        from e2e.radar_config import answerability_problems
        spec = RT_DIFFICULTY_TIERS.get(args.tier)
        if spec is not None:
            problems = answerability_problems(cfg, top_speed_mps=spec.speed_mps[1])
            if problems:
                print("answerable:   NO -- the real run will REFUSE without "
                      "--allow-unanswerable:")
                for prob in problems:
                    print(f"                - {prob}")
            else:
                print("answerable:   yes (Doppler unaliased, azimuth tolerance within "
                      "the Rayleigh limit)")
        out_root = Path(args.out) if args.out is not None else DATASETS_DIR
        print(f"out:          {out_root / f'{args.config}_{args.tier}'}  (NOT written -- dry-run)")
        print("this path ray-traces with Sionna RT -- see report/chain_integration_design.html")
        print("=" * 70)
        return 0

    manifest_path = generate_chain_corpus(
        args.config, args.tier, args.n, out_dir=args.out, seed=args.seed,
        frames_per_scene=args.frames_per_scene, use_rffe=not args.no_rffe,
        use_interconnect=not args.no_interconnect, quant_bits=args.quant_bits,
        coherent_targets=not args.no_coherent_targets,
        antenna_pattern=args.antenna_pattern,
        ground_scattering_coefficient=args.ground_scattering,
        samples_per_src=args.samples_per_src,
        use_transmit_chain=args.transmit_chain,
        use_local_assets=not args.no_local_assets,
        allow_unanswerable=args.allow_unanswerable,
        use_if_hpf=not args.no_if_hpf,
        store_cfr=args.store_cfr, store_paths=args.store_paths,
        if_hpf_kwargs=(None if args.if_hpf_corner_range is None
                       else {"corner_range_m": args.if_hpf_corner_range}),
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
