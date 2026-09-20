"""Unit tests for M5 depth model adapters — GPU- and weight-free by default.

Coverage:
* DEPTH_SPECS table + resolution selection (TAS calculateAspectRatio /
  limboResolution parity) + output mapping vs TAS's exact math.
* DepthWriter quantization + a real FFmpeg round-trip (10-bit mono MP4 and
  lossless 16-bit PNG sequence) — no GPU involved.
* Loader strictness against the REAL checkpoints, on CPU only — skipped when
  the weights are not in the local cache (first use downloads them).

Real-model + TRT validation lives in tests/test_depth_gpu.py (GPU-marked).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aive.config import model_cache_dir
from aive.stages.depth.stage import (
    DEPTH_SPECS,
    PerFrameDepthRunner,  # noqa: F401 - import check only
    StreamingDepthRunner,  # noqa: F401 - import check only
    dav2_process_size,
    disparity_stretch,
    minmax_stretch,
    normalize_nchw,
    spec_for_depth,
    streaming_shape,
)

# -- specs table ------------------------------------------------------------------


def test_specs_table():
    assert set(DEPTH_SPECS) == {
        "depth_anything_v2_small", "limbo_v1", "limbo_v2",
        "depth_anything_3_small", "depth_anything_3_base",
    }
    assert DEPTH_SPECS["depth_anything_v2_small"].mode == "per_frame"
    assert DEPTH_SPECS["depth_anything_v2_small"].arch == "dav2-small"
    for name in ("limbo_v1", "limbo_v2"):
        spec = DEPTH_SPECS[name]
        assert spec.mode == "streaming" and spec.arch == "da3-small"
        assert not spec.experimental
    assert DEPTH_SPECS["depth_anything_3_small"].arch == "da3-small"
    assert DEPTH_SPECS["depth_anything_3_base"].arch == "da3-base"
    for name in ("depth_anything_3_small", "depth_anything_3_base"):
        assert DEPTH_SPECS[name].experimental


def test_spec_for_depth_unknown_raises():
    with pytest.raises(ValueError):
        spec_for_depth("nope")


# -- resolution selection (TAS parity) ---------------------------------------------


@pytest.mark.parametrize(
    ("w", "h", "quality", "expected"),
    [
        (1920, 1080, "high", (1932, 1092)),   # each axis up to a multiple of 14
        (640, 360, "high", (644, 364)),
        (1932, 1092, "high", (1932, 1092)),   # already multiples: identity
        (640, 360, "medium", (700, 700)),     # fixed squares
        (640, 360, "low", (518, 518)),
    ],
)
def test_dav2_process_size(w, h, quality, expected):
    assert dav2_process_size(w, h, quality) == expected


def test_dav2_process_size_bad_quality():
    with pytest.raises(ValueError):
        dav2_process_size(640, 360, "ultra")


@pytest.mark.parametrize(
    ("w", "h", "expected_hw"),
    [
        (1920, 1080, (280, 504)),   # 16:9 -> widescreen export
        (1280, 720, (280, 504)),
        (1920, 1200, (280, 504)),   # 1.6:1 still closer to 1.8 than 1.32 in log space
        (720, 540, (378, 504)),     # 4:3 -> squarer export
        (1080, 1080, (378, 504)),   # square
        (1080, 1920, (378, 504)),   # portrait
    ],
)
def test_streaming_shape_aspect_pick(w, h, expected_hw):
    assert streaming_shape(w, h) == expected_hw


# -- output mapping vs TAS's exact math ---------------------------------------------


def _tas_write_depth_reference(depth: np.ndarray) -> np.ndarray:
    """TAS da3_streaming._writeDepth math, inlined as the cross-check."""
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() <= 10:
        return np.zeros_like(depth)
    disparity = np.full_like(depth, np.nan)
    disparity[valid] = 1.0 / depth[valid]
    low, high = np.percentile(disparity[valid], (2, 98))
    gray = ((disparity - low) / max(high - low, 1e-6)).clip(0, 1)
    gray[~valid] = 0
    return gray


def test_disparity_stretch_matches_tas_reference():
    rng = np.random.default_rng(3)
    # Depth-like map: positive values with a few invalid holes.
    depth = (rng.random((64, 96)) * 5 + 0.5).astype(np.float32)
    depth[10:12, 20:24] = np.nan
    depth[40, :] = -1.0
    assert np.allclose(disparity_stretch(depth), _tas_write_depth_reference(depth))


def test_disparity_stretch_inverts_order():
    """Closer (smaller positive depth) -> higher disparity -> brighter gray."""
    # >= 11 valid pixels so the TAS <=10-pixels black guard does not trigger.
    depth = np.arange(1.0, 13.0, dtype=np.float32).reshape(1, -1)
    gray = disparity_stretch(depth)
    assert (np.diff(gray[0]) < 0).all()


def test_disparity_stretch_all_invalid_is_black():
    d = np.zeros((8, 8), dtype=np.float32)
    d[0, 0] = 1.0  # <= 10 valid pixels -> whole frame black (TAS guard)
    assert (disparity_stretch(d) == 0).all()


def test_disparity_stretch_flat_frame_safe():
    d = np.full((16, 16), 3.0, dtype=np.float32)
    gray = disparity_stretch(d)
    assert np.isfinite(gray).all() and gray.min() >= 0 and gray.max() <= 1


def test_minmax_stretch():
    d = np.array([[0.0, 5.0], [10.0, 7.0]], dtype=np.float32)
    out = minmax_stretch(d)
    assert abs(out[0, 0]) < 1e-6 and abs(out[0, 1] - 0.5) < 1e-6 and abs(out[1, 0] - 1.0) < 1e-6


def test_minmax_stretch_flat_frame_safe():
    out = minmax_stretch(np.full((4, 4), 2.0, dtype=np.float32))
    assert np.isfinite(out).all() and (out == 0).all()


def test_normalize_nchw_math():
    frame = np.zeros((2, 8, 16, 3), dtype=np.uint8)
    frame[..., 0] = 123  # R channel
    x = normalize_nchw(frame)
    assert x.shape == (2, 3, 8, 16) and x.dtype == np.float32
    expected_r = (123 / 255.0 - 0.485) / 0.229
    assert np.allclose(x[:, 0], expected_r)


# -- writer: quantization + real FFmpeg round-trips ---------------------------------


def _probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)["streams"][0]


def test_writer_mp4_10bit_roundtrip(tmp_path):
    from aive.stages.depth.writer import DepthWriter

    out = tmp_path / "depth.mp4"
    rng = np.random.default_rng(1)
    frames = [rng.random((36, 64), dtype=np.float32) for _ in range(5)]
    with DepthWriter(out, width=64, height=36, fps="30", fmt="mp4_10bit") as w:
        for f in frames:
            w.write(f)

    st = _probe(out)
    assert (int(st["width"]), int(st["height"])) == (64, 36)
    # HEVC monochrome (I400): ffprobe reports a yuv400p/gray-ish mono format.
    assert "400" in st.get("pix_fmt", "") or "gray" in st.get("pix_fmt", ""), st.get("pix_fmt")


def test_writer_png16_lossless_roundtrip(tmp_path):
    import cv2

    from aive.stages.depth.writer import DepthWriter

    out_dir = tmp_path / "depth_seq"
    rng = np.random.default_rng(2)
    frames = [rng.random((32, 48), dtype=np.float32) for _ in range(4)]
    with DepthWriter(out_dir, width=48, height=32, fps="30", fmt="png16") as w:
        for f in frames:
            w.write(f)

    files = sorted(out_dir.glob("*.png"))
    assert len(files) == 4
    for i, (f, path) in enumerate(zip(frames, files)):
        expected = (np.clip(f, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        got = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert got.dtype == np.uint16 and got.shape == (32, 48)
        assert np.array_equal(got, expected), f"frame {i} not bit-exact"


def test_writer_rejects_wrong_shape(tmp_path):
    from aive.stages.depth.writer import DepthWriter

    with pytest.raises(ValueError):
        with DepthWriter(tmp_path / "d.mp4", width=64, height=36, fps="30") as w:
            w.write(np.zeros((10, 20), dtype=np.float32))


def test_writer_unknown_format():
    from aive.stages.depth.writer import DepthWriter

    with pytest.raises(ValueError):
        DepthWriter(Path("x.mp4"), width=8, height=8, fps="30", fmt="avi")


# -- loader strictness against the REAL checkpoints (CPU; skipped if not cached) -----

_CACHED = {
    "depth_anything_v2_small": ("vits", "depth_anything_v2_vits.pth"),
    "limbo_v1": ("V1", "Limbo.safetensors"),
    "limbo_v2": ("V2", "Limbo.safetensors"),
    "depth_anything_3_small": ("vits", "depth_anything_v3_vits.safetensors"),
    "depth_anything_3_base": ("vitb", "depth_anything_v3_vitb.safetensors"),
}


def _cached_weights(name: str) -> Path | None:
    version, fname = _CACHED[name]
    p = model_cache_dir() / name / version / fname
    return p if p.is_file() else None


@pytest.mark.parametrize("name", ["limbo_v1", "limbo_v2"])
def test_limbo_checkpoint_strict_verify_cpu(name):
    path = _cached_weights(name)
    if path is None:
        pytest.skip(f"{name} weights not in local cache")
    from aive.stages.depth._da3_loader import verify_da3_weights

    result = verify_da3_weights("da3-small", path)  # raises on disallowed mismatch
    assert result["missing"] == [] and result["unexpected"] == []


@pytest.mark.parametrize("name", ["depth_anything_3_small", "depth_anything_3_base"])
def test_official_da3_checkpoint_strict_verify_cpu(name):
    path = _cached_weights(name)
    if path is None:
        pytest.skip(f"{name} weights not in local cache")
    from aive.stages.depth._da3_loader import verify_da3_weights

    preset = "da3-small" if name.endswith("small") else "da3-base"
    result = verify_da3_weights(preset, path)
    assert result["missing"] == [] and result["unexpected"] == []


def test_dav2_checkpoint_strict_load_cpu():
    path = _cached_weights("depth_anything_v2_small")
    if path is None:
        pytest.skip("DAv2 weights not in local cache")
    from aive.stages.depth.stage import load_dav2

    model = load_dav2(path, device="cpu")  # strict=True inside; raises on mismatch
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    assert abs(n_params - 24.8) < 0.3, f"DAv2-small params {n_params:.2f}M unexpected"
