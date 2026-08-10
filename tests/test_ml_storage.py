"""
Tests for `e2e.ml.storage` (the lossless int16-code corpus codec) and its wiring
into `e2e.ml.blocks.SinkBlock`/`SourceBlock` and `e2e.ml.dataset`.

Synthetic only -- no Sionna, no GPU-only assumptions (uses `torch_device`/CPU
fallback like the rest of the suite).
"""
import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e2e.ml import storage


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------
def _quantized_complex_array(shape=(4, 12, 32), bits=12, fs=2.5, seed=0):
    """A genuinely uniformly-quantized complex64 array: every component is an exact
    multiple of `lsb = fs / 2**(bits-1)`, mimicking `e2e.chain.receive.QuantizerBlock`'s
    own output (mid-tread uniform quantization) without importing that module."""
    rng = np.random.default_rng(seed)
    top = 2 ** (bits - 1) - 1
    lsb = fs / (2 ** (bits - 1))
    re_code = rng.integers(-top - 1, top + 1, size=shape)
    im_code = rng.integers(-top - 1, top + 1, size=shape)
    array = (re_code.astype(np.float32) + 1j * im_code.astype(np.float32)) * np.float32(lsb)
    return array.astype(np.complex64), fs


def _continuous_complex_array(shape=(4, 12, 32), seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


# --------------------------------------------------------------------------------
# encode_payload / decode_payload
# --------------------------------------------------------------------------------
def test_int16_codec_is_exact_for_a_genuinely_quantized_array():
    array, fs = _quantized_complex_array()
    codec, arrays, codec_meta = storage.encode_payload(array, full_scale=fs)
    assert codec == storage.CODEC_INT16
    assert arrays["code_re"].dtype == np.int16
    assert arrays["code_im"].dtype == np.int16

    back = storage.decode_payload(codec, arrays, codec_meta)
    assert back.dtype == array.dtype
    assert np.array_equal(back, array)   # bit-for-bit, not approximate


def test_int16_codec_exact_regardless_of_original_bit_depth():
    """16-bit storage codes must exactly represent an array quantized at ANY bit
    depth <= 16 (see storage.py's docstring: bits_storage - bits_original just
    changes how many low bits are structurally zero, not correctness)."""
    for bits in (4, 8, 10, 12, 16):
        array, fs = _quantized_complex_array(bits=bits, seed=bits)
        codec, arrays, codec_meta = storage.encode_payload(array, full_scale=fs)
        assert codec == storage.CODEC_INT16, f"bits={bits} failed to verify as lossless"
        back = storage.decode_payload(codec, arrays, codec_meta)
        assert np.array_equal(back, array)


def test_raw_fallback_for_continuous_unquantized_data_stays_exact():
    array = _continuous_complex_array()
    codec, arrays, codec_meta = storage.encode_payload(array)   # no full_scale hint
    assert codec == storage.CODEC_RAW   # continuous data never verifies as int16-exact
    back = storage.decode_payload(codec, arrays, codec_meta)
    assert np.array_equal(back, array)


def test_encode_payload_never_lossy_even_with_a_wrong_full_scale_hint():
    """A bogus `full_scale` (e.g. stale/mismatched provenance) must never produce a
    silently-wrong int16 encoding -- either it still verifies exact, or it falls back
    to CODEC_RAW; decode must always reproduce the original array bit-for-bit."""
    array, _true_fs = _quantized_complex_array()
    codec, arrays, codec_meta = storage.encode_payload(array, full_scale=123.456)
    back = storage.decode_payload(codec, arrays, codec_meta)
    assert np.array_equal(back, array)


def test_encode_payload_real_valued_array():
    """Non-complex (real float) payloads are supported too (e.g. a labels grid, or
    any real-valued sensor array): int16 codes with no imaginary component."""
    top = 2 ** 11 - 1
    lsb = 1.0 / (2 ** 11)
    rng = np.random.default_rng(0)
    codes = rng.integers(-top - 1, top + 1, size=(3, 5))
    array = (codes.astype(np.float32) * lsb).astype(np.float32)
    codec, arrays, codec_meta = storage.encode_payload(array, full_scale=1.0)
    assert codec == storage.CODEC_INT16
    assert "code_im" not in arrays
    back = storage.decode_payload(codec, arrays, codec_meta)
    assert np.array_equal(back, array)


def test_encode_payload_empty_array_falls_back_to_raw():
    array = np.zeros((0,), dtype=np.complex64)
    codec, arrays, codec_meta = storage.encode_payload(array)
    assert codec == storage.CODEC_RAW
    assert storage.decode_payload(codec, arrays, codec_meta).shape == (0,)


def test_decode_payload_unknown_codec_raises():
    with pytest.raises(ValueError):
        storage.decode_payload("bogus_codec", {}, {})


# --------------------------------------------------------------------------------
# write_sample_npz / read_payload
# --------------------------------------------------------------------------------
def test_write_read_sample_npz_roundtrip_records_codec(tmp_path):
    array, fs = _quantized_complex_array(shape=(4, 192, 512), bits=12)
    labels = np.zeros((3, 4, 5), dtype=np.float32)
    path = tmp_path / "frame_00000.npz"

    storage.write_sample_npz(
        path, {"adc": array, "labels": labels}, {"frame_idx": 0},
        payload_key="adc", full_scale=fs,
    )

    with np.load(path) as data:
        meta = json.loads(str(data["meta"].item()))
        assert meta["codec"] == storage.CODEC_INT16
        assert "scale" in meta["codec_meta"]
        assert "adc" not in data                 # int16 codec: not stored under the raw key
        assert "adc_code_re" in data and "adc_code_im" in data
        back = storage.read_payload(data, meta, "adc")
        assert np.array_equal(back, array)
        assert np.array_equal(data["labels"], labels)   # untouched, own key


def test_write_read_sample_npz_raw_fallback_keeps_payload_key(tmp_path):
    array = _continuous_complex_array()
    path = tmp_path / "frame_00000.npz"
    storage.write_sample_npz(path, {"adc": array}, {}, payload_key="adc")

    with np.load(path) as data:
        meta = json.loads(str(data["meta"].item()))
        assert meta["codec"] == storage.CODEC_RAW
        assert "adc" in data   # raw codec: payload stays under its own key, back-compat
        back = storage.read_payload(data, meta, "adc")
    assert np.array_equal(back, array)


def test_compression_reduces_bytes_on_disk(tmp_path):
    """A real acceptance criterion: the int16 codec must actually shrink the file on
    disk relative to storing the same (quantized) array raw, measured in real bytes."""
    array, fs = _quantized_complex_array(shape=(4, 192, 512), bits=12, seed=1)

    compressed_path = tmp_path / "compressed.npz"
    storage.write_sample_npz(compressed_path, {"adc": array}, {}, payload_key="adc", full_scale=fs)

    raw_path = tmp_path / "raw.npz"
    np.savez_compressed(raw_path, adc=array)

    compressed_size = compressed_path.stat().st_size
    raw_size = raw_path.stat().st_size
    assert compressed_size < raw_size
    # Structural floor: int16 (2B) vs complex64 (4B) per real/imag component is a 2x
    # win before any zlib bonus -- assert we are meaningfully better than that alone
    # isn't required, but the codec must at least clear the raw complex64 zlib size
    # by a comfortable margin (regression guard against an accidental no-op codec).
    assert compressed_size < raw_size * 0.8


def test_read_payload_back_compat_no_codec_key_in_meta(tmp_path):
    """A .npz written before e2e.ml.storage existed has no "codec" key in meta at
    all -- read_payload must treat that as CODEC_RAW (the only thing such a file
    could be) and return the payload unchanged."""
    array = _continuous_complex_array()
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, adc=array, meta=np.array(json.dumps({"frame_idx": 0})))

    with np.load(path) as data:
        meta = json.loads(str(data["meta"].item()))
        assert "codec" not in meta
        back = storage.read_payload(data, meta, "adc")
    assert np.array_equal(back, array)


# --------------------------------------------------------------------------------
# SinkBlock / SourceBlock integration
# --------------------------------------------------------------------------------
def _quantized_rx_time_state(torch_device, n_rx=4, n_chirp=8, n_samples=32, bits=12, seed=0):
    """Runs a real `e2e.chain.receive.QuantizerBlock` over random ADC data -- the
    genuine on-disk shape (`state["adc_full_scale"]` set exactly as the real chain
    leaves it) without depending on the rest of the chain."""
    from e2e import frames
    from e2e.chain.receive import QuantizerBlock

    g = torch.Generator(device="cpu").manual_seed(seed)
    adc = (torch.randn(n_rx, n_chirp, n_samples, generator=g)
          + 1j * torch.randn(n_rx, n_chirp, n_samples, generator=g)).to(torch.complex64).to(torch_device)
    quantizer = QuantizerBlock(bits=bits)
    out = quantizer.apply({"adc": adc})
    state = {"signal_domain": frames.DOMAIN_RX_TIME, "adc": out["adc"],
             "adc_full_scale": out["adc_full_scale"]}
    return state


def test_sink_source_roundtrip_quantized_adc_uses_int16_codec_and_is_exact(tmp_path, torch_device):
    from e2e.ml.blocks import SinkBlock, SourceBlock

    state = _quantized_rx_time_state(torch_device)
    SinkBlock(tmp_path, tag="rt").apply(state)

    with np.load(tmp_path / "rt_frame_00000.npz") as data:
        meta = json.loads(str(data["meta"].item()))
    assert meta["codec"] == storage.CODEC_INT16

    src = SourceBlock(tmp_path, tag="rt")
    loaded = src.get_S_pars()
    assert torch.equal(loaded, state["adc"])   # exact, not approximate
    assert loaded.device.type == torch_device.type


def test_sink_source_roundtrip_continuous_adc_falls_back_to_raw(tmp_path, torch_device):
    """Mirrors the pre-existing SinkBlock/SourceBlock contract test but explicitly
    checks the recorded codec: unquantized data (no QuantizerBlock upstream, no
    adc_full_scale in state) must land as CODEC_RAW and still round-trip exactly."""
    from e2e import frames
    from e2e.ml.blocks import SinkBlock, SourceBlock

    g = torch.Generator(device="cpu").manual_seed(0)
    adc = (torch.randn(4, 6, 32, generator=g) + 1j * torch.randn(4, 6, 32, generator=g)
          ).to(torch.complex64).to(torch_device)
    state = {"signal_domain": frames.DOMAIN_RX_TIME, "adc": adc}
    SinkBlock(tmp_path, tag="rt").apply(state)

    with np.load(tmp_path / "rt_frame_00000.npz") as data:
        meta = json.loads(str(data["meta"].item()))
    assert meta["codec"] == storage.CODEC_RAW

    loaded = SourceBlock(tmp_path, tag="rt").get_S_pars()
    assert torch.equal(loaded, adc)


def test_sink_compression_reduces_on_disk_size_for_quantized_corpus(tmp_path, torch_device):
    state = _quantized_rx_time_state(torch_device, n_rx=4, n_chirp=192, n_samples=512, bits=12)
    from e2e.ml.blocks import SinkBlock

    SinkBlock(tmp_path, tag="rt").apply(state)
    compressed_size = (tmp_path / "rt_frame_00000.npz").stat().st_size

    raw_path = tmp_path / "raw_reference.npz"
    np.savez_compressed(raw_path, adc=state["adc"].detach().cpu().numpy())
    raw_size = raw_path.stat().st_size

    assert compressed_size < raw_size


# --------------------------------------------------------------------------------
# e2e.ml.dataset integration (writer + RadarFrameDataset reader)
# --------------------------------------------------------------------------------
def test_dataset_generate_dataset_records_codec_in_meta(tmp_path, torch_device):
    scenes = pytest.importorskip("e2e.ml.scenes", reason="sibling shard e2e.ml.scenes not present")
    from e2e.ml import dataset as ml_dataset
    from e2e.ml.radar_config import PRESETS, TI_IWR1443

    tiny_cfg = dataclasses.replace(TI_IWR1443, name="test_storage_tiny_tdm", n_chirps=12, n_samples=64)
    PRESETS[tiny_cfg.name] = tiny_cfg
    try:
        tier = sorted(scenes.DIFFICULTY_TIERS)[0]
        manifest_path = ml_dataset.generate_dataset(
            tiny_cfg.name, tier, 2, out_dir=tmp_path, seed=0, device=torch_device,
        )
        manifest = json.loads(manifest_path.read_text())
        fname = manifest["files"]["train"][0] if manifest["files"]["train"] else \
            manifest["sequences"][0][0]
        with np.load(manifest_path.parent / fname) as data:
            meta = json.loads(str(data["meta"].item()))
            assert "codec" in meta   # analytic-fallback floats -> CODEC_RAW, but recorded
            assert meta["codec"] == storage.CODEC_RAW
            assert "adc" in data     # CODEC_RAW keeps the original on-disk key: back-compat

        # RadarFrameDataset (this module's reader) round-trips through the recorded codec.
        ds = ml_dataset.RadarFrameDataset(manifest_path, split="train" if manifest["files"]["train"]
                                           else "val", input_format="adc")
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
    finally:
        PRESETS.pop(tiny_cfg.name, None)


def test_dataset_reader_loads_a_legacy_manifest_v2_corpus_without_codec_key(tmp_path):
    """A manifest_version-2 corpus written before e2e.ml.storage existed: "adc" is
    the raw array directly, and meta has no "codec" key at all. RadarFrameDataset
    must still load it (existing on-disk contract, unbroken)."""
    from e2e.ml import dataset as ml_dataset
    from e2e.ml.radar_config import TI_IWR1443

    cfg = dataclasses.replace(TI_IWR1443, name="test_storage_legacy", n_chirps=12, n_samples=64)
    dataset_dir = tmp_path / f"{cfg.name}_D0"
    dataset_dir.mkdir()

    adc = _continuous_complex_array(shape=(cfg.n_rx, cfg.n_chirps, cfg.n_samples), seed=3)
    labels = np.zeros((3, 4, 5), dtype=np.float32)
    meta = {"frame_idx": 0, "targets": [], "target_extras": [], "scene": {}}
    np.savez_compressed(
        dataset_dir / "frame_00000.npz",
        adc=adc, labels=labels, meta=np.array(json.dumps(meta)),
    )
    manifest_path = ml_dataset.write_manifest(
        dataset_dir, cfg, "D0", [["frame_00000.npz"]], splits=(1.0, 0.0, 0.0),
    )

    ds = ml_dataset.RadarFrameDataset(manifest_path, split="train", input_format="adc")
    x, y = ds[0]
    assert x.shape[0] == 2 * cfg.n_rx
