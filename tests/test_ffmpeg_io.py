"""M0 acceptance: FFmpeg probe / decode / encode round-trips."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from aive.stages.encode import CodecPreset, encode_video, video_args
from aive.stages.ingest import decode_frames, probe


def test_probe(test_clip):
    info = probe(test_clip)
    assert info.width == 640 and info.height == 360
    assert abs(info.fps_float - 30.0) < 0.01
    assert info.num_frames >= 85  # ~90 frames for 3s@30fps (encoder rounding)
    assert info.video_codec == "h264"
    assert len(info.audio_streams) == 1 and info.audio_streams[0]["codec"] == "aac"


def test_decode_all_frames(test_clip):
    info = probe(test_clip)
    frames = list(decode_frames(test_clip))
    assert len(frames) == info.num_frames
    f0 = frames[0]
    assert f0.shape == (360, 640, 3) and f0.dtype == np.uint8
    # testsrc2 is not a constant image
    assert float(f0.std()) > 10.0


def test_decode_start_count(test_clip):
    frames = list(decode_frames(test_clip, start_frame=10, count=5))
    assert len(frames) == 5
    all_frames = list(decode_frames(test_clip))
    # Frame content must match the corresponding frame of a full decode.
    np.testing.assert_array_equal(frames[0], all_frames[10])


def test_encode_roundtrip_with_audio(test_clip, tmp_path):
    info = probe(test_clip)
    frames = list(decode_frames(test_clip))
    out = tmp_path / "out.mp4"
    encode_video(
        iter(frames), out, info.width, info.height, info.fps,
        preset=CodecPreset.H264, crf=18, container="mp4", audio_src=test_clip,
    )
    out_info = probe(out)
    assert out_info.width == 640 and out_info.height == 360
    assert out_info.num_frames == info.num_frames
    assert len(out_info.audio_streams) == 1


def test_video_args_pix_fmts():
    # Both HEVC presets must force 10-bit output; H.264 stays 8-bit.
    assert "yuv420p10le" in video_args(CodecPreset.X265_10BIT_ANIME, 19)
    assert "yuv420p10le" in video_args(CodecPreset.NVENC_HEVC_10BIT, 23)
    args = video_args(CodecPreset.H264, 18)
    assert "yuv420p" in args and "yuv420p10le" not in args


def _has_encoder(name: str) -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return name in out


@pytest.mark.skipif(not _has_encoder("hevc_nvenc"), reason="hevc_nvenc encoder not available")
def test_encode_nvenc_hevc_10bit(test_clip, tmp_path):
    # Regression: NVENC must emit 10-bit HEVC even when fed an 8-bit source.
    info = probe(test_clip)
    frames = list(decode_frames(test_clip))[:30]
    out = tmp_path / "out_nv.mp4"
    encode_video(
        iter(frames), out, info.width, info.height, info.fps,
        preset=CodecPreset.NVENC_HEVC_10BIT, crf=23, container="mp4",
    )
    out_info = probe(out)
    assert out_info.video_codec == "hevc"
    assert out_info.pix_fmt == "yuv420p10le"
