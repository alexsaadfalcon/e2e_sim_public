# Radar corpus data format

This document lets a recipient decode every frame in an `e2e.ml.dataset`-style radar
corpus **with numpy alone** — no access to this repository is required, though every
claim below cites the source file:line it was derived from, in case you do have the
repo and want to check.

A corpus is a directory (in a shipped bundle: `data/`, alongside this file) containing:

```
manifest.json
frame_00000.npz
frame_00001.npz
...
```

or, when `frames_per_scene > 1` (a corpus of short motion-consistent sequences rather
than independent single-instant frames):

```
manifest.json
frame_00000_t00.npz
frame_00000_t01.npz
...
```

`manifest.json`'s `"files"`/`"sequences"` entries are **filenames only**, relative to
`manifest.json`'s own directory (`e2e/ml/dataset.py:85`) — the two always sit side by
side, wherever the directory is copied.

## `manifest.json`

Written by `write_manifest` (`e2e/ml/dataset.py:357-403`). Top-level keys:

| Key | Meaning |
| --- | --- |
| `manifest_version` | `2` for everything described here. A directory with this key absent (or `1`) is an older format: its npz files hold a precomputed `"input"` array, not raw `"adc"` (`e2e/ml/dataset.py:96-100`); this document does not cover it. |
| `config` | `RadarConfig.to_dict()` (`e2e/ml/radar_config.py:201-202`) — the exact radar timing/geometry this corpus was synthesized with. See "Radar config fields" below. |
| `tier` | Difficulty-tier name the scenes were drawn from (`e2e.ml.scenes.DIFFICULTY_TIERS`). Informational only; every field needed to decode a frame is elsewhere in the manifest. |
| `grid` | `dataclasses.asdict(LabelGrid)` (`e2e/ml/dataset.py:383-384`): `{"n_range": int, "n_azimuth": int, "max_range_m": float}` — the label map's geometry. See "Label array" below. |
| `snr_db` | Synthesis SNR passed to the frame generator, or `null`. |
| `seed` | Base RNG seed; scene `i`'s RNG is `seed + i` (`e2e/ml/dataset.py:324`). |
| `frames_per_scene` | Consecutive motion-consistent frames drawn per scene (`e2e/ml/dataset.py:56`, `:276-300`). `1` unless stated otherwise. |
| `label_classes` | Object classes that became ground truth, e.g. `["vehicle", "pedestrian"]` (`e2e/ml/dataset.py:153-158`, `:396`), or `null` if every class was labeled. Classes not in this list (typically `"scatterer"` = background clutter) still contribute *signal* to `"adc"` but never appear in `"targets"` or the label map. |
| `files` | `{"train": [...], "val": [...], "test": [...]}`, flat filename lists, a deterministic **scene-level** split (`e2e/ml/dataset.py:251-266`, `:375-380`) — never splits one motion sequence across splits. |
| `sequences` | `[[filenames...], ...]`, one inner list per scene in frame order — the same files as `files`, grouped by scene instead of split. |

## `.npz` per-frame keys

Written by `storage.write_sample_npz` (`e2e/ml/storage.py:161-191`), called from
`e2e/ml/dataset.py:345-349`. Every frame `.npz` has:

| Key | Contents |
| --- | --- |
| `meta` | A **0-d numpy unicode array** holding a JSON string (`np.array(json.dumps(...))`, `e2e/ml/storage.py:190`). Decode with `json.loads(str(npz["meta"].item()))`. See "meta fields" below. |
| `labels` | `float32 [3, n_range, n_azimuth]`, stored uncompressed/unmodified (`e2e/ml/storage.py:181` — not the codec-compressed key). See "Label array" below. |
| `adc` **or** `adc_code_re` (+ `adc_code_im`) | The raw ADC cube, `complex64 [n_rx, n_chirps, n_samples]` (`e2e/ml/dataset.py:64`), encoded per `meta["codec"]`. See "ADC codec" below — **which key(s) are present tells you the codec**, independent of `meta["codec"]` (`e2e/ml/dataset.py:490-499`). |

Older (`manifest_version` 1) files instead have an `"input"` key (the precomputed
network-input tensor, no raw ADC) — out of scope here (`e2e/ml/dataset.py:96-100`).

## ADC codec (`e2e/ml/storage.py`)

`meta["codec"]` is either `"raw"` or `"int16"` (`e2e/ml/storage.py:67-68`; absent ==
`"raw"`, `:205`). Both are **exactly lossless** — this is a storage-size optimization,
never a quality tradeoff (`e2e/ml/storage.py:24-34`, `:45-53`).

**`"raw"`**: the `adc` array *is* the complex64 cube, unmodified. Nothing to decode.

**`"int16"`**: two arrays, `adc_code_re` and `adc_code_im`, both `int16`, same shape as
the cube. `meta["codec_meta"]` holds `{"scale": float, "dtype": "complex64"}`
(`e2e/ml/storage.py:142`). Dequantize with **exactly**:

```python
scale = meta["codec_meta"]["scale"]
re = adc_code_re.astype(np.float32) * np.float32(scale)
im = adc_code_im.astype(np.float32) * np.float32(scale)
adc = (re + 1j * im).astype(np.complex64)   # meta["codec_meta"]["dtype"]
```

(`e2e/ml/storage.py:100-105`, the exact body of `_int16_decode`.) `scale` is the
original quantizer's LSB step (`full_scale / 2**15`); each `int16` code is that
step's exact integer multiple, so this recovers the pre-quantization float bit-for-bit
(`e2e/ml/storage.py:24-34`). There is no case where `adc_code_im` is absent for an ADC
cube — the payload is always complex (`e2e/ml/storage.py:95-97`) — but a reader that
wants to be robust to other `payload_key`s using this same codec on a *real*-valued
array should treat a missing `adc_code_im`/`*_code_im` key as "real output only"
(`e2e/ml/storage.py:100-103`).

## `meta` fields

Written per-frame in `e2e/ml/dataset.py:235-244` (`generate_sample`) and extended in
`e2e/ml/dataset.py:334-337` (`generate_dataset`), plus the codec fields added by
`write_sample_npz` (`e2e/ml/storage.py:177-179`):

| Key | Type / units | Meaning |
| --- | --- | --- |
| `frame_idx` | int | Frame-within-scene index (`0` unless `frames_per_scene > 1`). |
| `snr_db` | float or `null` | Synthesis SNR for this frame. |
| `seed` | int or `null` | This frame's own synthesis RNG seed. |
| `config` | str | `RadarConfig.name` used (matches `manifest.json["config"]["name"]`). |
| `mimo` | str | `"tdm"` / `"ddma"` / `"single"` — same as `manifest.json["config"]["mimo"]`. |
| `pose_position` | `[x, y, z]`, metres | Radar array position, world frame. |
| `pose_boresight` | `[x, y, z]`, unit vector | Radar boresight direction, world frame. |
| `target_extras` | list, parallel to `targets` (below), of `{"rcs_dbsm": float, "velocity_mps": [vx, vy, vz]}` | Per-target RCS and world-frame velocity, same order/length as `targets` (`e2e/ml/dataset.py:161-188`). |
| `scene_index` | int | Index of the scene (0-based) this frame's scene belongs to. |
| `targets` | list of tuples | Ground-truth target list — see "Target tuple schema" below. |
| `scene` | dict | `e2e.ml.scenes.scene_summary()` (`e2e/ml/scenes.py:286-304`): `{"n_vehicles": int, "n_pedestrians": int, "n_clutter": int, "clutter": [{"position": [x,y,z], "rcs_dbsm": float}, ...], "placement_attempts": int or null}`. `clutter` is background signal, not a label — it contributes to `adc` but never to `targets`/the label map. |
| `codec` | `"raw"` or `"int16"` | See "ADC codec" above. |
| `codec_meta` | dict | `{}` for `"raw"`; `{"scale": float, "dtype": str}` for `"int16"`. |

## Array shapes / axis order

* **`adc`**: `complex64 [n_rx, n_chirps, n_samples]` (`e2e/ml/dataset.py:64`).
  * `n_rx` = `config.n_rx` — the number of **physical** RX antennas, *not* the virtual
    MIMO array size (`n_virtual = n_tx * n_rx`, `e2e/ml/radar_config.py:67-70`). For
    TDM configs the virtual array is only formed by de-interleaving chirps by which TX
    fired them — not done here; the raw axis order stays TX-interleaved.
  * `n_chirps` = `config.n_chirps` — total chirps in the frame (all TX rounds, for
    TDM).
  * `n_samples` = `config.n_samples` — ADC samples per chirp (fast time / range axis).
  * Fast time (last axis) maps to **range**; chirp index (middle axis) maps to
    **Doppler/velocity** after the relevant FFTs — see "Radar config fields" for the
    bin-to-metre/bin-to-m/s formulas. This repo's own derivation (`adc_to_rd` /
    `tdm_deinterleave`, `e2e/ml/transforms.py`, not reproduced here) is one specific,
    documented choice of how to turn this cube into a range-Doppler map; a recipient
    is free to process the raw cube differently.
* **`labels`**: `float32 [3, n_range, n_azimuth]` — see "Label array" below.

## Label array (`e2e/ml/labels.py`)

Grid geometry comes from `manifest.json["grid"]`: `n_range`, `n_azimuth`,
`max_range_m`. Bin sizes (`e2e/ml/labels.py:147-155`):

```python
range_bin_m = max_range_m / n_range
az_bin      = 2.0 / n_azimuth          # sin(azimuth) spans a uniform [-1, 1) axis
```

Bin `(i, j)`'s centre is at `range_m = (i + 0.5) * range_bin_m`,
`sin_azimuth = -1.0 + (j + 0.5) * az_bin` (`e2e/ml/labels.py:282-284`). This is a
**`(range, sin(azimuth))` grid, not `(range, angle_degrees)`** — `sin_azimuth` is the
ULA direction cosine (see "Coordinate / angle conventions" below), uniformly spaced;
recover an angle with `theta = arcsin(sin_azimuth)` if needed, noting the resulting
angle grid is *not* uniform.

Channels (`e2e/ml/labels.py:48-118`, `:252-289`):

* **channel 0 — objectness**: `1.0` on a dense 3x3-cell footprint centred on each
  target's *surface* `(range, sin_azimuth)` cell (clipped, not wrapped, at the grid
  edge), `0.0` elsewhere.
* **channels 1-2 — range / azimuth regression residuals**, defined *per footprint
  cell* `(i, j)`, toward the target's *centre* (not its surface):
  ```python
  residual_range = (centre_range_m - (i + 0.5) * range_bin_m) / range_bin_m
  residual_az    = (sin_azimuth    - (-1.0 + (j + 0.5) * az_bin)) / az_bin
  ```
  i.e. `centre_range_m = (i + 0.5) * range_bin_m + residual_range * range_bin_m`, and
  similarly for azimuth. Because footprint cells span roughly 1.5 bins from the
  target's own cell and the surface-vs-centre offset is one-sided toward larger range
  (a target's reflecting surface is always at or in front of its centre), the range
  residual is **not** bounded to `[-0.5, 0.5]` — it can be several bins for a large,
  close vehicle (`e2e/ml/labels.py:62-76` gives measured worst cases per asset). This
  is expected, not a bug in a reader that assumes a wider range.
  Only footprint cells actually written (channel 0 == 1.0) carry a meaningful
  residual; elsewhere channels 1-2 are `0.0` by construction (never written).

## Target tuple schema (3-tuple vs 4-tuple)

`meta["targets"]` is a list of ground-truth tuples, one per in-grid labeled target
(clutter excluded — see `label_classes` above), produced by `targets_in_grid`
(`e2e/ml/labels.py:219-246`). **Two schema versions exist; both are still emitted by
current code, and any corpus written by this repo since 2026-08-17 uses the 4-tuple.**

* **4-tuple** (current, `e2e/ml/labels.py:224-245`):
  `(centre_range_m, sin_azimuth, object_class, surface_range_m)`
  * `centre_range_m` — range to the object's geometric centre, metres.
  * `sin_azimuth` — direction cosine, shared by centre and surface (the surface point
    lies on the radar-to-centre line, `e2e/ml/labels.py:32-33`).
  * `object_class` — string, e.g. `"vehicle"` / `"pedestrian"`.
  * `surface_range_m` — range to the object's nearest reflecting surface along the
    line of sight; equals `centre_range_m` for a point target with no known extent
    (`e2e/ml/labels.py:190-212`). This is where the label footprint (channel 0) is
    actually written, and what a matcher should compare a detection's range against
    (`e2e/ml/labels.py:99-109`).
* **3-tuple** (older corpora, pre-2026-08-17): `(centre_range_m, sin_azimuth,
  object_class)` — no surface range. **A reader can tell which schema a given corpus
  uses by checking `len(t)` on any entry of `meta["targets"]`** (3 vs 4); the first
  three fields have identical meaning and values in both, so `t[0]`, `t[1]`, `t[2]`
  are always safe to read positionally. When only a 3-tuple is available, treat the
  target as a point (`surface_range_m := centre_range_m`) — this is exactly what
  `e2e.ml.metrics` falls back to (`e2e/ml/labels.py:92-93`).

`target_extras[i]` (`meta["target_extras"]`, see the `meta` table above) is parallel
to `targets[i]` regardless of which schema: `{"rcs_dbsm": float, "velocity_mps": [vx,
vy, vz]}` (`e2e/ml/dataset.py:161-188`).

If you are decoding predictions from a model trained with `e2e.ml.labels.
decode_detections` rather than reading `meta["targets"]`: its output tuples are
`(range_m, sin_azimuth, score, surface_range_m)` (`e2e/ml/labels.py:295-322`) —
`range_m` there is the *regressed centre* (sub-bin precision), and `surface_range_m`
is the *kept cell's own bin centre* (cell-quantized) — not the same pairing as the
ground-truth 4-tuple's `(centre_range_m, ..., surface_range_m)`, but analogous.

## Radar config fields — mapping bins to metres and m/s

From `manifest.json["config"]` (a `RadarConfig.to_dict()`,
`e2e/ml/radar_config.py:27-56`, all base/stored fields):

| Field | Units | Meaning |
| --- | --- | --- |
| `f0_hz` | Hz | Chirp start frequency. |
| `bandwidth_hz` | Hz | Swept bandwidth over the sampled window. |
| `n_tx`, `n_rx` | count | Physical TX / RX antenna counts. |
| `n_chirps` | count | Chirps per frame (all TX rounds combined, for TDM). |
| `n_samples` | count | ADC samples per chirp. |
| `fs_hz` | Hz | ADC sample rate. |
| `chirp_period_s` | s | Chirp-to-chirp period (ramp + idle time). |
| `mimo` | str | `"tdm"` / `"ddma"` / `"single"` (`e2e/ml/radar_config.py:31-44`). |
| `frame_rate_hz` | Hz | Frame repetition rate. |

None of the derived quantities below are stored in the manifest (only base fields
are) — recompute them with these exact formulas (`e2e/ml/radar_config.py:66-159`;
`C = 299_792_458.0` m/s, `e2e/ml/radar_config.py:22`):

```python
n_chirps_per_tx     = n_chirps // n_tx if mimo == "tdm" else n_chirps
sweep_time_s        = n_samples / fs_hz
ramp_slope_hzps     = bandwidth_hz / sweep_time_s
f_center_hz         = f0_hz + bandwidth_hz / 2.0
wavelength_m        = C / f_center_hz

range_resolution_m  = C / (2.0 * bandwidth_hz)
max_range_m         = fs_hz * C / (2.0 * ramp_slope_hzps)   # == n_samples * range_resolution_m

n_tx_eff            = n_tx if mimo in ("tdm", "ddma") else 1   # both MIMO schemes pay
                                                                 # this Doppler-span penalty
t_slow_eff_s        = n_tx_eff * chirp_period_s
max_velocity_mps    = wavelength_m / (4.0 * t_slow_eff_s)      # unambiguous +-v_max
velocity_resolution_mps = wavelength_m / (2.0 * n_chirps * chirp_period_s)
```

`range_resolution_m` maps a fast-time FFT bin index `k` (of `n_samples` bins, or
`n_samples` output range bins before any downsampling) to `range_m = k *
range_resolution_m`. `velocity_resolution_mps` maps a (zero-centred, `fftshift`ed)
Doppler-FFT bin index `d` (of `n_chirps_per_tx` bins) to `velocity_mps = d *
velocity_resolution_mps`, wrapping at `+-max_velocity_mps`. The **output label grid**
(`manifest.json["grid"]`) is a *separate*, coarser range axis (`n_range` bins over the
same `[0, max_range_m)` span, typically `n_samples / range_stride` — see "Label array"
above); it has no azimuth or Doppler analogue to `n_chirps_per_tx`/`velocity_resolution_mps`
— azimuth in the label grid comes only from the array geometry (`n_azimuth` is a free
parameter, not derived from the radar config, `e2e/ml/labels.py:157-170`).

## Coordinate / angle conventions

* World frame: right-handed, **+z up** (`e2e/ml/rd_synth.py:73-74`, `e2e/ml/labels.py:176-187`).
* The ULA (array) axis for a given radar pose is `u = normalize(z_up x boresight)`
  (`e2e/ml/rd_synth.py:78-89`, formula at `:89`) — i.e. the horizontal axis
  perpendicular to where the radar is pointing.
* For a world point `p` and radar position `origin`, `sin_azimuth = ((p - origin) /
  |p - origin|) . u` — the direction cosine of the line of sight onto the array axis
  (`e2e/ml/labels.py:176-187`). This is the standard quantity a ULA of `n_virtual`
  elements actually resolves (uniformly, unlike an angle-degrees axis) — see
  `n_virtual = n_tx * n_rx` (`e2e/ml/radar_config.py:67-70`) for the array's angular
  resolution, roughly `2 / n_virtual` in `sin_azimuth` units.
* `range_m` is always straight-line Euclidean distance from `pose_position` to the
  point in question (target centre, target surface, or a clutter point) — not a
  ground-projected range.
* Elevation is not modeled: a ULA measures only the azimuthal direction cosine, so an
  elevated point is indistinguishable from a coplanar one at the same `sin_azimuth`
  (`e2e/ml/rd_synth.py:41`).
