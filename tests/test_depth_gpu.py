"""GPU validation for M5 depth models — skipped without CUDA or cached weights.

Covers what unit tests cannot: real model inference on CUDA, ONNX export + TRT
engine builds for both runner families, and a full E2E pass (mono 10-bit MP4 +
lossless PNG sequence) on synthetic clips.

TRT-vs-eager acceptance mirrors TAS's LIMBO_TENSORRT_VALIDATION.md: global
relative MAE < 2% (mean abs error / mean abs reference depth), per-frame
relative MAE < 5%, display-MAE < 0.01 after the disparity + percentile stretch.

When VRAM headroom is insufficient for an engine build (another process holds
the GPU — common on this machine), the runners fall back to PyTorch-CUDA and
the TRT comparison skips with a clear message; engines build on first real use.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")  # noqa: F401 - presence check only

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _cached(name: str, version: str, fname: str):
    from aive.config import model_cache_dir

    p = model_cache_dir() / name / version / fname
    return p if p.is_file() else None


LIMBO_V2 = ("limbo_v2", "V2", "Limbo.safetensors")
DAV2_SMALL = ("depth_anything_v2_small", "vits", "depth_anything_v2_vits.pth")


@pytest.fixture(autouse=True)
def _clear_runners():
    """Free model VRAM between tests (this machine shares the GPU with an LLM)."""
    from aive.stages.depth.stage import clear_depth_runner_cache

    yield
    clear_depth_runner_cache()


def _synthetic_frames(n: int, h: int = 240, w: int = 426, seed: int = 7):
    """Structured anime-ish frames (gradient + moving bar) — deterministic."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0.15, 0.85, w)[None, :, None] * np.ones((h, 1, 3))
    frames = []
    for i in range(n):
        f = (base + rng.normal(0, 0.02, size=(h, w, 3))).clip(0, 1)
        x0 = int(w * (i / max(n - 1, 1)))
        f[:, x0 : x0 + 24, :] *= 0.35  # moving dark bar
        frames.append((f * 255).astype(np.uint8))
    return frames


# -- per-frame DAv2 -----------------------------------------------------------------


def test_dav2_runner_matches_fp32_reference():
    path = _cached(*DAV2_SMALL)
    if path is None:
        pytest.skip("DAv2 weights not in local cache")

    from aive.stages.depth.stage import PerFrameDepthRunner, dav2_process_size, load_dav2, minmax_stretch

    w, h = 336, 192  # -> process 336x196 (high quality)
    frame = _synthetic_frames(1, h, w)[0]

    runner = PerFrameDepthRunner(path, w, h, model_name="depth_anything_v2_small", quality="high")
    got = runner(frame)

    # FP32 eager reference (no autocast), same resize + stretch pipeline.
    import cv2

    pw, ph = dav2_process_size(w, h, "high")
    proc = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)
    from aive.stages.depth.stage import normalize_nchw

    model = load_dav2(path, device="cuda", dtype=None).eval()
    with torch.no_grad():
        ref_t = model(torch.from_numpy(normalize_nchw(proc[None])).to("cuda"))
    ref = minmax_stretch(cv2.resize(ref_t[0, 0].float().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR))

    assert got.shape == (h, w) and np.isfinite(got).all()
    mad = float(np.abs(got - ref).mean())
    maxd = float(np.abs(got - ref).max())
    # bf16-autocast (or fp16 TRT) vs fp32 eager on a stretched [0,1] map.
    assert mad < 0.01 and maxd < 0.05, f"DAv2 backend diverges: mean {mad:.4f}, max {maxd:.4f}"


# -- streaming Limbo ------------------------------------------------------------------


def test_streaming_runner_infer_chunk_sane():
    path = _cached(*LIMBO_V2)
    if path is None:
        pytest.skip("Limbo v2 weights not in local cache")

    from aive.stages.depth.stage import StreamingDepthRunner, streaming_shape

    runner = StreamingDepthRunner(path, 640, 360, model_name="limbo_v2", window=8)
    mh, mw = streaming_shape(640, 360)
    assert (mh, mw) == (280, 504)

    frames = _synthetic_frames(8)
    out = runner.infer_chunk(frames)
    assert out.shape == (8, mh, mw) and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert (out > 0).mean() > 0.95, "DA3-family depth should be positive almost everywhere"

    # Tail lengths down to 1 must work (real tail chunks are never padded).
    for n in (4, 2, 1):
        out_n = runner.infer_chunk(frames[:n])
        assert out_n.shape == (n, mh, mw) and np.isfinite(out_n).all()


def test_streaming_trt_vs_eager_acceptance():
    """TAS acceptance: global rel MAE <2%, per-frame <5%, display MAE <0.01."""
    path = _cached(*LIMBO_V2)
    if path is None:
        pytest.skip("Limbo v2 weights not in local cache")

    from aive.stages.depth.stage import StreamingDepthRunner, disparity_stretch, normalize_nchw, streaming_shape

    runner = StreamingDepthRunner(path, 640, 360, model_name="limbo_v2", window=8)
    if runner._trt_runner is None:
        pytest.skip("TRT engine not built (VRAM headroom guard); PyTorch-CUDA fallback active")

    mh, mw = streaming_shape(640, 360)
    frames = _synthetic_frames(8)

    got = runner.infer_chunk(frames)  # TRT fp16 path

    # Unautocast FP32 eager reference on the identical normalized input.
    from aive.stages.depth._da3_loader import load_da3
    from aive.stages.depth._limbo_export import MultiViewDepth
    from aive.stages.depth.stage import _resize_frame

    wrapper = load_da3("da3-small", path, device="cuda", dtype=None)
    mv = MultiViewDepth(wrapper.model, mh, mw)
    imgs = np.stack([_resize_frame(f, mw, mh) for f in frames])
    x = torch.from_numpy(normalize_nchw(imgs)).to("cuda")
    try:
        with torch.no_grad():
            ref_t = mv(x)  # pure fp32 (no autocast context)
    except torch.cuda.OutOfMemoryError:
        pytest.skip("fp32 eager reference OOM (GPU shared); TRT-vs-eager check deferred")
    ref = ref_t.float().cpu().numpy()

    diff = np.abs(got - ref)
    global_rel_mae = float(diff.mean()) / max(float(np.abs(ref).mean()), 1e-9)
    per_frame_rel = [float(d.mean()) / max(float(r.mean()), 1e-9) for d, r in zip(diff, np.abs(ref))]
    display_mae = float(
        np.abs(disparity_stretch(got[0]) - disparity_stretch(ref[0])).mean()
    )

    assert global_rel_mae < 0.02, f"global relative MAE {global_rel_mae:.4f} >= 2%"
    assert max(per_frame_rel) < 0.05, f"worst per-frame relative MAE {max(per_frame_rel):.4f} >= 5%"
    assert display_mae < 0.01, f"display MAE {display_mae:.4f} >= 0.01"


# -- E2E pass ---------------------------------------------------------------------------


def test_e2e_depth_pass_mp4_and_png(test_clip: str, tmp_path):
    from aive.stages.depth.depth_pass import DepthPassConfig, run_depth_pass
    from aive.stages.ingest import probe

    info = probe(test_clip)  # 640x360 @30fps, 90 frames
    assert info.num_frames == 90

    out_mp4 = tmp_path / "e2e_depth.mp4"
    result = run_depth_pass(
        test_clip, out_mp4,
        DepthPassConfig(model="limbo_v2", output="mp4_10bit", window=4),
        info, scene_split=False,
    )
    assert result["frames"] == 90 and result["width"] == 640 and result["height"] == 360
    assert out_mp4.exists() and out_mp4.stat().st_size > 10_000

    # PNG sequence: lossless writer path on a short segment (writer math is
    # unit-tested separately; here we prove the pass drives it end-to-end).
    import subprocess

    short = tmp_path / "short.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", test_clip, "-t", "1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(short)],
        check=True, capture_output=True,
    )
    out_png = tmp_path / "e2e_depth"  # folder for png16
    result2 = run_depth_pass(
        str(short), out_png,
        DepthPassConfig(model="limbo_v2", output="png16", window=4),
        probe(str(short)), scene_split=False,
    )
    assert result2["frames"] == 30
    pngs = sorted(out_png.glob("*.png"))
    assert len(pngs) == 30
