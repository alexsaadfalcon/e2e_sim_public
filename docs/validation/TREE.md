# Validation tree — three levels, one node per thing that can be wrong

**Tree state (2026-09-23, wave 0):** HEAD `88f6240` + DIRTY working tree (a Ka re-founding
wave was editing `sionna_simple_channel.py`, `sionna_iterator.py`, `radar_config.py`,
`chain_generate.py`, `link_budget.py`, `ml/blocks.py`, `webapp/*`). Every `path:line` was
opened at read time; in files marked **[IF]** (in flight) the `(symbol)` is authoritative.
Nothing ran on the GPU.

**Legend.** Status `UNV`/`PART`/`VER`/`RET` per `notes/RIGOR_STANDARD.md`; `VER` only with a
named test or F-number. Tier: L2 Sonnet, L1 Opus, L0 Fable. `★Tn` = demo path, Thrust n
(`webapp/demo_presets.py`); `★` = every preset. `x::t` = `tests/test_x.py::t`; `x::N` = N
tests there. `reg:` = `notes/PHYSICAL_ASSUMPTIONS.md` row; `Fnn` = ledger entry; `§n` =
`notes/PHYSICS_JUSTIFICATION_AUDIT.md` entry. `Ev` = status · evidence.

## Level 0 — purpose and UX (tier F)

| ID | Node (where) | User does / is told | "Correct" means | Ev |
|---|---|---|---|---|
| L0-1 ★ | Purpose: ONE live chain from stored RT (`README.md:1`, `notes/VISION.md`, STATE §0.1) | every product is the same block chain run live on a traced channel | no screen replays what the chain did not compute; every number reproduces from HEAD | PART · F76, F75 |
| L0-2 ★ | Block Diagram tab (`webapp/app.py:136` [IF]; `pipeline_registry.py`) | enable blocks, edit params | each knob = physical parameter with a §-entry, in units; no impossible state | PART · §23; webapp::84, e2e_ui journeys::10 |
| L0-3 | Scenario tab (`app.py:138`, `_validate_scenario:1042`, `_generate_frames:1074` dry-run) | author/validate JSON, generate frames | frames carry declared band, array, provenance | PART · scenario::43, scenario_runner::39 (real RT gated) |
| L0-4 ★ | Results + A/B + notes (`app.py:140`, `_run_pipeline:486`, `_render_results:912`) | run, compare arms, read a statistic | arms differ only by the named knob; axes physical | PART · webapp_ab::45, figures_wave4::6, F59 |
| L0-5 ★T1 | Thrust 1 circuit knobs (`demo_presets.py:198` [IF]) | LNA bias 8→0.5 mA A/B; IF BW | dB deltas = Friis; trade-off shown is real | PART · STATE §5 (0.17 dB; no trade-off; gm∝I) |
| L0-6 ★T2 | Thrust 2 feature reduction (`:269`) | mantissa 6→1 | degradation monotone; AFE models the cited chip | PART · F47; demo_presets::test_thrust2_* |
| L0-7 ★T3 | Thrust 3 cold start (`:342`) | cold vs warm, k=8 | converges; error vs a stated oracle | PART · simulation::test_warm_start_*; F25, F65 |
| L0-8 ★T4 | Thrust 4 interconnect (`:403`) | Tessera height A/B; range profile | change is Ka physics via scale ×2, not a label | PART · F89, F91; F90 |
| L0-9 ★T5 | Thrust 5 detectors (`:472/:539/:614`) | stored CFR → live chain → CFAR/ML; 12→3-bit A/B; offline PR | live cube = stored at 12 bit (gate); offline numbers reproduce; A/B direction honest | PART · webapp_live_chain::18; F85–F88; 3-bit pinned because 4-bit moves the wrong way (file comment) |
| L0-10 | Scripted entries (`e2e/main/*`, `scenario_runner`, `chain_generate`, `beat_cfar`) | run a module | each states band/array; `main_sionna_blocks.py:31` `N_FREQS` dead (reg: example band); `:86` `'case3'`=passthrough | PART · main_sionna_blocks::4 |
| L0-11 ★ | Operator tooling (`webapp/preflight.py`, `rehearse.py:83`) | preflight; PNG rehearsal | fails on missing asset/off-grid override; a human reads PNGs | PART · webapp_preflight::25 |
| L0-12 | Public claims (`README.md`, `docs/PHYSICS.md` 21 entries, `GLOSSARY.md`) | reads numbers, verdicts | each number has date, conditions, command | PART · F75–F78 |

## Level 1 — contracts (tier O)

| ID | Node | I/O · claim | Invariants | reg | Oracle | Ev |
|---|---|---|---|---|---|---|
| L1-C1 ★ | Frame contract (`e2e/frames.py:41 dims`, `:168 FrameCapabilities`, `:296 check_capabilities`, `:357 to_aperture_grid`) | `[n_rx,n_tx,n_chirp,n_freqs]` c64 → same/aperture view | domain+dimension bridges named; subspace path single-chirp/no-MIMO; views not copies | pipeline shape | misordered chain raises naming the bridge | VER · frames::33, simulation::test_mimo_assertion |
| L1-C2 ★ | Frequency plan (`scenario.py:163 FrequencyPlan` 30 GHz/28.5–31.5/5000; pkl `meta.freq_plan` `scenario_runner.py:421`; `blocks.py:426 band_hz`; `chain_generate.py:82` [IF]) | declared plan → the axis every block assumes | ONE axis travels with the frame; band labels derive from it | declared band, munich carrier, munich grid, shipped frames, example, ML, Case bands | `band_audit` (register design, unbuilt) | RET · F91–F93 (munich traced at 3.5 GHz, no axis in pkl); Ka re-trace [IF] `sionna_simple_channel.py:59` |
| L1-C3 ★ | Array geometry/ordering (`blocks.py:47-55`; `frames.py:379`; `dechirp.py:44 ANTENNA_INDEX_REVERSED`; `rt_scene_build.py:982-987`; `pipeline_runner.py:1544` [IF]) | element index → (az,el)/u | column-first flat axis; dim0=az; +u=+az (`scatterers.py:86`); λ/2 assumed once (runner), computed elsewhere | munich RX, pipeline shape, corpus spacing, angle-axis | known angle → known bin on every path | PART · frames::test_to_aperture_grid_recovers_sionna_column_first_ordering, rd_synth::test_beamforming_peak_at_target_angle, rt_gen::test_known_scatterer_lands_in_expected_range_doppler_angle_bins; F93 |
| L1-C4 ★T1,T5 | Units/absolute scale (`scenario_runner.py:304` A=√(N·P·Z0/n_tx); `blocks.py:145`; `link_budget.py:53/:57/:62`; `rffe_model.py:12` T0 dup) | CFR → volts → kTBF floor | volts iff `physical_scale`; one T0; noise BW=fs | TX power, NF, T0, noise BW vs BW_IF | hand RF calc | PART · F42, F63, F72, F80, F81 |
| L1-C5 ★ | Determinism (`blocks.py:154`; `impairments.py:803`; `rt_scenes.py:278`, `dataset.py:155` sha256; `algorithms.py:246` global RNG) | seed → bit-identical run | stage seeds never collide | — | rerun = stored | PART · rffe_determinism::5, impairments::test_stage_seeds_never_collide_across_frames_or_stages, webapp_live_chain::test_each_frame_is_rerun_with_its_own_stored_seed; sensing matrix unseeded |
| L1-C6 ★T5 | RadarConfig (`radar_config.py:29` [IF]; `:249 answerability_problems`; presets 77 GHz + `benchmark_v1_ka:461`) | fields → slope, Δr, R_max, v_max | closed forms; unanswerable pairs refused | preset carrier, BW, ADC rate, T_c | textbook | VER · radar_config::31, F33, F43 |
| L1-B1 ★ | RT propagation (`rt_scene_build.py:973-979`; `scenario_runner.py:360`; legacy generator [IF]) | scene+arrays+f → paths | geometry f-independent (F92); no backscatter off curved meshes (F16a) | munich carrier/grid, legacy env carrier | our CFR vs Sionna `cfr()` 2.3e-4; retrace | PART · rt_doppler_validity::13, rt_gen::test_retrace_reference_*, F92, F93 |
| L1-B2 ★ | Materials/band policy (`city_scenes.py:131/:112`; `rt_scene_build.py:581` concrete, `:165` S_ground=0, `:68` S=0.3, `:253` skin) | table+f → ε,σ,S | ITU in validity; OOB refit `a·f^b`; ground specular by decision | ITU validity, OOB, ground, roughness, tissue, pattern | ITU-R P.2040 at 30 GHz | PART · city_scenes::13; §3 |
| L1-B3 ★T5 | CFR synthesis (`rt_signal_chain.py:286/:352/:435`; `coherent_target_cfr:792`, `:862` 1−S², `:887`; gain `:732`) | paths → `[rx,tx,chirp,f]` c64 | RCS conserved across split; Doppler receding-positive; radar equation | RCS table | analytic point target (`rd_synth`) | PART · rt_coherent::25, F32; f64 reference not on generation call site |
| L1-B4 ★T5 | Motion/scene draws (`motion.py:147`; `scatterers.py:128`, dt=1 `:55`; `rt_scenes.py:537`; `geometry.py:251/:291`) | scenario → kinematics, yaw, surface | 1 frame = 1 s; speeds guarded downstream | scene speeds | finite difference = track velocity | PART · motion::19, scatterers::20, geometry::20, object_yaw::6, ml_scenes::20; F1, F20, F23, F61 |
| L1-B5 ★ | Serialisation/replay (`scenario_runner.py:421/:449`; `sionna_iterator.py:6/:46/:54` [IF]; `ml/storage.py:170/:208/:246`; `ml/blocks.py SourceBlock` [IF]) | frames ↔ disk | replay == stored, with band/array/power/provenance | shipped frames | round trip bit-for-bit | PART · sionna_iterator::17, ml_storage::16, ml_store_cfr::14; legacy `munich.pkl` has no meta; F30, F31, F51, F55, F84 |
| L1-B6 | TX waveform/PA (`waveform.py:111/:147/:242`; `tx_pa.py:105/:126/:153`) | t → envelope → CFR×spectrum | linear ramp (nonlinearity stated absent); Rapp; ideal path bit-identical | — | constant envelope through PA | PART · chain_waveform::13, tx_pa::15; §1–2 |
| L1-B7 ★T1 | RFFE (`rffe_model.py:18/:63/:237`; `blocks.py:134`) | CFR → distorted CFR, PRX | clamp r*=√(Gm/2.25G3) `:123`; BB compressive `:200`; noise NBB·BW_IF/nt `:230` | IF BW, Av, carrier-free, T0 dup | Friis; 24 dB | PART · rffe_physics::18, rffe_determinism::5, F42; STATE §5 (no bias–linearity; IF filter never enabled) |
| L1-B8 ★T4 | Interconnect (`blocks.py:371/:426/:501`; CSV `:174/:525`; boxcar `:515`; Tessera `:250/:329/:537`; `tessera.py:121/:461`) | CFR × H(f) | H on `linspace(band_hz)` over the INDEX axis; CSV phase = min-phase reconstruction; scale ×2 at Ka | ML band, Case band, boxcar, Tessera geometry/scale | Case3 re-derivation 5e-7 dB; wrapper = upstream (F89) | PART · interconnect::18, interconnect_tessera::18, interconnect_surrogate::44; F5, F6, F10, F89–F91 |
| L1-B9 ★T2 | AFE/compression (`compress.py:139/:150/:191/:230/:352`; `afe_utils.py:117`, `:153` NotImplemented; `blocks.py:561/:600/:874`) | `[N,f]` → `[M,f]` (→ pinv) | weights quantised (float model); MAC output NEVER quantised | — | M=N exact | PART · afe::11, chain_compress::36; F47, §12 |
| L1-B10 ★T2,T3 | Subspace tracker+oracle (`blocks.py:617/:705/:757`; `algorithms.py:10/:82`; `simulation.py:106/:122/:164`; `subspace_utils.py:16`; `spectrum_estimator.py:74` unwired) | X,A → U | "reestimate" = warm power iteration, not AdaOja; gate reads the ORACLE SVD | — | static subspace; k-SVD floor | PART · subspace::12, subspace_tracking::9, simulation::34; F25, F38, F65 |
| L1-B11 ★T5 | Dechirp bridge (`dechirp.py:47/:60/:88`) | CFR → `adc[rx,chirp,n]` | conj+flip; TDM select/DDMA code; bit-exact, invertible | — | reference math | VER · chain_dechirp::16 |
| L1-B12 ★T5 | Link budget/thermal noise (`link_budget.py:87/:106/:131/:170/:192` [IF]) | cfg → kTBF, SNR; adc+noise | independent of cube; B=fs | TX power, NF, T0, noise BW | hand calc | PART · link_budget::13, F42; F81 open; Ka carry-over unmeasured |
| L1-B13 ★T5 | Impairments (`impairments.py:277/:508/:632/:824/:803/:224/:124`) | adc → adc | τ=0 exactly cancelled; per-(TX,RX) taps; clutter K-dist, R⁻³, persistent, steered | — | azimuth recovered through TDM/DDMA | VER · impairments::29; F35, F52, F54, F73 |
| L1-B14 ★T5 | IF HPF & ADC (`receive.py:147/:274/:319`; `:325/:401/:424`) | adc → adc | causal Butterworth, linear conv, 0.92–0.98 taper; mid-tread; AGC +6 dB | IF corner, taper, bits, full scale | Butterworth; 6.02b+1.76 | VER · chain_receive::28 |
| L1-B15 ★T1,T2,T4 | Radar cube & FFT products (`transforms.py:35/:82/:129`; `blocks.py:1042/:1096/:1156/:1196/:983/:1003`; runner axes `:1544/:1594` [IF]) | adc → RD cube; aperture → maps | Hann range/Doppler (cube); NO range window (maps); angle FFT zero-padded 32→256; range = forward FFT + fftshift, axis negated | angle-axis | point target → known bin | PART · transforms::16, blocks::20, blocks_range_profile::7; F22, F59; zero-padded angle vs native-resolution rule untested |
| L1-B16 ★T5 | Classical detector (`baseline.py:183/:286/:349/:378/:397/:472`) | cube → `[3,R,A]` | native angle FFT+Hann; sum collapse; notch ≥5 m/s; CA-CFAR 0–20 dB→[0,1], no Pfa | — | synthesised target at every fine offset | PART · ml_baseline::22; F26, F44, F48, F53, F57, F60, F71, F74 |
| L1-B17 ★T5 | Labels & metrics (`labels.py:197/:304/:347`; `metrics.py:194/:214/:242/:444/:457`) | scene → GT; dets → AP, FA | surface-aware match (2 m, 0.06 sin, widened); VOC AP; pessimistic ties | — | GT as detections → AP=1 | VER · ml_metric_oracle::7, ml_metrics::50, ml_labels::19; F2, F17, F58, F61 |
| L1-B18 ★T5 | Learned detectors/training (`fftradnet.py:364/:248`; `raddetnet.py:83/:135`; `cfar_head.py:412/:533`; `controls.py:89`; `bootstrap_ci.py:155`) | cube → `[3,R,A]`; runs → AP, CI | reads the frame (controls); fingerprint/recertify | — | overfit-5-frames; controls | PART · ml_{train,fftradnet,ssmradnet,cfar_head,controls,bootstrap_ci,pipeline_fingerprint,recertify}::122; F69, F83–F88; corpora 77 GHz |
| L1-B19 ★T5 | Corpus generation/identity (`chain_generate.py:193/:150/:82/:394/:474` [IF]; `dataset.py:155`; `export_corpus.py`) | scenario+cfg → npz+manifest | fixed stage order; meta names flags, f0, band [IF]; unanswerable refused | corpus metadata | manifest names its code | PART · ml_chain_generate::25, ml_dataset::30, ml_export_corpus::15; F30, F31, F49, F51, F55 |
| L1-B20 | Comms head (`ofdm.py:169/:198`; `channel.py:141/:167/:228/:235/:267`; `beamforming.py:87/:104/:132`; `comms/blocks.py:239`; `isac.py:82/:114`) | CFR → bits, BER/EVM | AWGN per element before combine; MRC≡EGC here; no phase noise/converter | — | zero noise → zero BER | PART · comms_*::81; F41, F46 |
| L1-B21 ★ | Webapp runner & gate (`pipeline_runner.py:673/:812/:205/:1521/:1544/:1594/:1667/:1648/:447` [IF]) | state → figures, notes | live cube == stored (LSB diff); axes physical; fallbacks (32,32)/64 explicit | angle-axis | stored cube | PART · webapp_live_chain::18, webapp_ab::45, webapp::84; display numerics have no oracle |
| L1-B22 ★T5 | Scoreboard/stored numbers (`detector_scoreboard.py:130/:59/:84/:91`) | run + `beat_cfar.json` → panel | live counts recomputed; AP/PR/CI read from JSON; F83/F86 constants hardcoded | — | `python -m e2e.ml.beat_cfar --skip-train` | PART · webapp_detector::14, figures_wave4::6; F85–F88 |

## Level 2 — numerics (tier S)

Loud: `raise` = named error, `silent` = none, `—` = n/a.

| ID | Function | Algorithm · dtype | Loud | Tests | Falsifier | St |
|---|---|---|---|---|---|---|
| L2-C1.1 | `frames.to_aperture_grid:357` | `view(rx_x,rx_y,chirp,f)` c64 view | raise | frames::test_to_aperture_grid_* | (rows,cols) pkl imaged transposed | VER |
| L2-C2.1 | `sionna_simple_channel.build_frequencies:59` [IF] | `linspace(start−fc,stop−fc,N)` f64 rel. `scene.frequency` | silent | sionna_simple_channel::11 (untracked) | pkl axis ≠ plan; sideband not symmetric | UNV |
| L2-C2.2 | `scenario.FrequencyPlan.linspace:170` | absolute `linspace` f64 | silent | scenario_runner::test_dry_run_* | absolute Hz handed where Sionna wants baseband | UNV |
| L2-C2.3 | `chain_generate._interconnect_band_hz:82` [IF] | 77 GHz literal else f0±3 GHz | silent | ml_chain_generate::test_interconnect_band_hz_*; webapp_live_chain_interconnect_band::4 | band ≠ beat-frequency span | PART |
| L2-C3.1 | `rt_signal_chain.beat_frequencies:209` | `f0+S·n/fs` rel. `f0+B/2` f64 | silent | rt_gen::test_beat_frequency_grid_spans_the_ramp | centre ≠ `cfg.wavelength_m` | PART |
| L2-C3.2 | `dechirp.beat_from_cfr:47` | `conj`, `flip(0,1)` | — | chain_dechirp::test_beat_from_cfr_is_exactly_invertible | angle sign vs `rd_synth` | VER |
| L2-C3.3 | runner sin axis `:1544` [IF] | `(k−N/2)/(N/2)`, λ/2 hardcoded | silent | none | non-λ/2 pkl (munich, 3.5 GHz traced) mislabels angle | UNV |
| L2-C3.4 | `rt_signal_chain._array_element_positions:590` | `p+(j−(n−1)/2)·s·λ·ŷ` f64 | silent | rt_coherent::test_coherent_term_is_a_clean_aperture_phase_ramp | handedness ≠ `rd_synth.array_axis:103` | PART |
| L2-C4.1 | `scenario_runner.tx_power_amplitude_scale:304` | `√(N·P·Z0/n_tx)` | silent | scenario_runner::test_tx_power_amplitude_scale_matches_derivation | mean power ≠ P_rx·Z0 | VER |
| L2-C4.2 | `rffe_model` noise `:230` | `√(NBB·BW/nt)` pre-divided for unnormalised fft | silent | rffe_physics::test_frequency_bin_noise_matches_NBB_times_BWIF_after_fft | bin noise ≠ NBB·BW_IF | VER |
| L2-C4.3 | `link_budget.noise_bandwidth_hz:87` | `=fs` | — | link_budget::test_noise_bandwidth_is_the_sample_rate_not_the_bin_width | two floors (fs vs 15 MHz) | PART |
| L2-C5.1 | `impairments.stage_seed:803` | seed⊕stage hash | — | impairments::test_stage_seed_is_stable_across_processes | Python `hash()` salt | VER |
| L2-C5.2 | `algorithms.gen_A_ada:246` | `randn(d,m−k)` GLOBAL RNG cfloat | — | subspace::test_gen_A_ada_shape_and_rows | runs differ with every seed set | UNV |
| L2-C6.1 | `radar_config.max_velocity_mps:152` [IF] | `λ/(4·n_tx_eff·T_c)` | `validate()` list | radar_config::test_tdm_max_velocity_penalty_vs_single_tx | aliasing below v_max | VER |
| L2-B1.1 | `rt_scene_build.build_rt_scene:973-987` | `scene.frequency=f0+B/2`; arrays λ/2 built AFTER | silent | rt_gen::test_build_rt_scene_wires_arrays_objects_and_velocity | array built before frequency (the F93 mechanism) | PART |
| L2-B1.2 | `rt_signal_chain._solve:220` | Sionna `PathSolver`, seed 41, diffuse on | — | rt_gen::test_repeated_generation_is_reproducible | driver nondeterminism | PART |
| L2-B2.1 | `city_scenes.prepare_scene_for_frequency:131` | `ε=a·f^b, σ=c·f^d` refit / stand-in | raise (policy) | city_scenes::test_extrapolated_policy_is_frequency_independent_for_marble | refit ≠ ITU at in-table f | PART |
| L2-B2.2 | `rt_signal_chain._element_field_amplitude:732` | Sionna pattern eval f64 | raise | rt_coherent::test_element_gain_enters_amplitude_squared | tr38901 boresight ≠ 8.00 dB | PART |
| L2-B3.1 | `cfr_sum_over_paths:286` / `_budgeted:352` | Σaᵢe^{−j2πf_cτ}e^{j2πf_Dt}e^{−j2πfτ}; f64 phase → c64 | silent | rt_doppler_validity::13 | the two implementations disagree; c64 wrap at 77 GHz | PART |
| L2-B3.2 | `coherent_target_cfr:792` | `g²√σλ/((4π)^{1.5}R_rR_t)`; 1−S²; 1\|5 centres | raise | rt_coherent::test_coherent_term_range_and_rcs_follow_the_radar_equation | energy ≠ RCS | PART |
| L2-B4.1 | `scatterers.frame_scatterers:128` | finite difference, dt=1.0 | raise (OOB) | scatterers::test_linearly_moving_object_finite_difference_matches_motion_velocity | dt ≠ 1/frame_rate (F23) | PART |
| L2-B4.2 | `geometry.nearest_surface_point:291` | ellipsoid hit, yaw, clamp 0.99 | silent | geometry::test_surface_point_lies_on_the_radar_to_centre_line | sphere offset ≠ radius | VER |
| L2-B5.2 | `sionna_iterator.SionnaIterator:6` [IF] | legacy ndarray / v2 dict; ka preference `:117` | raise (link) | sionna_iterator::17, sionna_simple_channel::test_munich_iterator_* | legacy pkl served under a Ka label | PART |
| L2-B6.1 | `tx_pa._am_am:105` / `_am_pm:126` | Rapp p=2; φ=5°u²/(1+u²) | raise | tx_pa::test_large_signal_saturates_at_a_sat | non-monotone | VER |
| L2-B7.1 | `rffe_model.circuit_model_bb_approx:63` | LNA/mixer/BB cubic, clamp r*, f32 | silent | rffe_physics::test_bb_stage_is_compressive_not_expansive | IIP3 constant vs bias | PART |
| L2-B7.2 | `RFFEBlock.apply_circuit:134` | ifft→circuit→fft; legacy mean-abs normalise `:145` | raise (bias/BW) | rffe_physics::test_rffe_legacy_mode_normalizes_physical_mode_skips_it | 1/N seam double-counted | PART |
| L2-B8.1 | `InterconnectBlock._resampled_response:525` | `np.interp` re/im on `linspace(band_hz)` | raise (file) | interconnect::test_transfer_mode_applies_resampled_s21_over_band | phase interp across a wrap | PART |
| L2-B8.2 | `_resolve_tessera_scale:250` + `:284` | round(max_f/20 GHz); four lengths ×s | raise (box) | interconnect_tessera::test_scale_two_evaluates_model_at_scaled_geometry_and_half_frequency | S21 not scale-invariant | PART |
| L2-B8.4 | `tessera.s_matrix:519` + `_passivity_scale:461` | GNN per f, c128; raise if \|S21\|>1 | raise | interconnect_surrogate::test_passivity_guard_raises_on_gain | wrapper ≠ upstream | VER |
| L2-B9.1 | `afe_utils.quantizer_fp:117` | sign/exp/mantissa, round-nearest | NaN propagates | afe::11 | magnitude bias | VER |
| L2-B9.2 | `compress.reconstruct_aperture:150` | `pinv(A)@x` | raise (no A) | chain_compress::test_decompress_is_exact_when_not_actually_compressing | exact at M<N | VER |
| L2-B10.1 | `AdaOjaBlock.update:757` | `Z=Y(YᴴU)`, `U=qr(Z)`; zero-norm skip | raise (method, m≤k) | subspace::test_adaoja_reestimate_zero_frame_does_not_swap_the_basis | wrong subspace when Row(A)⊅U | PART |
| L2-B10.2 | `Oja.add_data:110` | −(p+r)wᴴ, η₀/t, QR | silent skip | subspace::test_oja_tracks_a_static_subspace | diverges on static input | PART |
| L2-B10.3 | `simulation.rank_diagnostic:122` `:164` | full SVD; `(S[k−1]−S[k])/S[0]` | raise (k<1) | simulation::test_rank_diagnostic_values | gate fed by ground truth (F25) | PART |
| L2-B10.4 | `subspace_dist_frob:16` | `√max(d−‖AᴴB‖²_F,0)` | bare `assert` | subspace::test_subspace_dist_zero_for_identical_basis | ≠ chordal distance | PART |
| L2-B11.1 | `dechirp.mimo_combine:60` | TDM select; DDMA `e^{j2πtc/n_tx}` | raise | chain_dechirp::test_ddma_combine_matches_reference | code sign ≠ `rd_synth` | VER |
| L2-B12.1 | `link_budget.add_thermal_noise:170` | complex AWGN kTBF, seeded | raise (domain) | link_budget::test_injected_noise_power_matches_the_budget | power ≠ budget | VER |
| L2-B13.1 | `apply_phase_noise:277` | φ(t)−φ(t−τ), log bands, gate 0 alone | — | impairments::test_phase_noise_zero_delay_gate_is_exactly_cancelled | energy not conserved | VER |
| L2-B13.2 | `apply_clutter:632`, `_k_distributed_gain:201`, `_sample_gamma:164` | gamma×speckle f64; R⁻³; steering `:224` | raise (reference) | impairments::test_clutter_follows_the_surface_range_law | kurtosis ≠ K-law | VER |
| L2-B14.1 | `IFHighPassBlock.analog_response:274`, `apply:319` | Butterworth f64→f32; zero-pad linear conv | raise | chain_receive::test_if_hpf_nulls_dc_and_matches_butterworth_oracle | circular wrap leaks | VER |
| L2-B14.2 | `QuantizerBlock.apply:409` `:424`, AGC `:401` | round(x/LSB), clamp top code | raise (bits<2) | chain_receive::test_quantization_error_is_bounded_by_half_an_lsb | bias or wrap | VER |
| L2-B15.1 | `transforms.adc_to_rd:35` | DC `:61`; Hann×fft range `:64`; Hann×fft+shift Doppler `:69` | raise (shape) | transforms::test_range_bin_location, ::test_doppler_bin_location | bin ≠ 2R/Δr | VER |
| L2-B15.2 | `RangeAzBlock._map:1138`, `_power_bin:1003` | fft(az,n=bins)+shift; fft(f)+shift; Σ\|·\|²; gate-sum | warns (short band) | blocks::test_range_az_nondivisible_nfreqs_zero_gate_and_energy | zero-pad sidelobes read as targets | PART |
| L2-B15.3 | `FFTBlock._map:1073` | fft(f); 2-D angle fft padded to bins; power over range | — | blocks::test_fft_block_noncoherent_range_shows_target_at_10m | angle axis not u=sinθ | PART |
| L2-B15.4 | runner range axis `:1594-1596` [IF] | `per·c/(2·span)`; zero gate `(N//2)//per`; negated | silent | webapp_rehearsal::test_range_figures_show_only_nonnegative_range | known delay lands at wrong metre on munich grid | UNV |
| L2-B15.5 | `_aperture_window:983` | Hann/Hamming f32, aperture only | raise | blocks::test_aperture_window_reduces_sidelobes | applied to range | VER |
| L2-B16.1 | `baseline.cfar_objectness:349` | annulus mean (avg_pool2d); `(ratio_dB−0)/20` clamp | — | ml_baseline::test_cfar_objectness_edge_cells_are_not_padding_biased | not scale-invariant | PART |
| L2-B16.2 | `baseline._to_grid:286` `:345` | angle NN upsample; range amax pool | — | ml_baseline::test_classical_map_localizes_a_point_target_at_every_fine_bin_offset | point-sampling returns (F53) | VER |
| L2-B16.3 | notch `baseline.py:265-279` | ±bins zeroed when v_max≥5 | raise (tdm comp) | ml_baseline::test_default_notch_suppresses_a_stationary_target | stationary invisible (F74) | PART |
| L2-B17.1 | `metrics._interpolated_ap:444` | VOC running max | — | ml_metrics::test_interpolated_ap_hand_computed_five_nine | ≠ hand value | VER |
| L2-B17.2 | `metrics._normalized_distance:214` | max(\|Δr\|/2 m, \|Δsin\|/(0.06+ext/R)) | — | ml_metric_oracle::test_surface_tolerance_does_not_make_the_criterion_unbounded | unbounded widening | VER |
| L2-B18.1 | `RADDetNet.forward:135` | Doppler as channels; bilinear to grid | raise | ml_train::29 | reads a stripe prior (F83) | PART |
| L2-B18.2 | `CFARHead.forward:533` | autocast off f32; stage1 = baseline algebra | raise | ml_cfar_head::test_cfar_from_rad_matches_the_classical_arm | fp16 moves objectness | VER |
| L2-B19.1 | `chain_generate.build_chain_simulation:193` [IF] | flags→RFFE→interconnect→dechirp→noise→impair→HPF→ADC→cube | raise `:474` | ml_chain_generate::test_if_hpf_in_default_composition_between_impairments_and_quantizer | webapp order ≠ this (gate non-zero) | PART |
| L2-B20.1 | `beamforming.mrc_weights:87`; AWGN `comms/blocks.py:239` | `H/‖H‖`; noise before combine | silent | comms_beamforming::test_mrc_array_gain_matches_10log10N_monte_carlo | gain ≠ 10log10N | VER |
| L2-B21.1 | `pipeline_runner` dB `:1521` [IF] | `10log10(\|t\|+1e-12)` on POWER products | silent | webapp_ab::test_peak_minus_median_matches_direct_computation_to_1e_minus_6 | amplitude product routed here | PART |
| L2-B21.2 | `_StoredADCGateBlock:447` [IF] | both cubes in run LSB; max\|diff\| | raise (sidecar) | webapp_live_chain::test_the_gate_is_not_vacuous_a_moved_knob_shows_up_as_a_difference | reads 0 with a moved knob | VER |
| L2-B22.1 | `detector_scoreboard.score_frames:130` | pooled `match_detections` | — | webapp_detector::test_cfar_block_emits_map_detections_and_ground_truth | counts ≠ metrics' | PART |

**Counts.** L0 12 · L1 28 · L2 58. Demo path: L0 9, L1 24, L2 ≈45. Evidence (F or test): all
but the five UNV rows. VER by the test-name criterion: L1 6, L2 26 — provisional until a card
exists (README.md); the status column is regenerated from cards, not edited here. Length is
measured as `tr -d '|' < TREE.md | wc -w` (table delimiters excluded).
