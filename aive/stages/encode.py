"""Frame encoding via FFmpeg stdin pipe, with audio muxing and codec presets."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from .ingest import find_ffmpeg, probe

log = logging.getLogger(__name__)


class CodecPreset:
    X265_10BIT_ANIME = "x265-10bit-anime"  # default (design.md D4)
    NVENC_HEVC_10BIT = "nvenc-hevc-10bit"  # fast option
    H264 = "h264"


def video_args(preset: str, crf: int) -> list[str]:
    if preset == CodecPreset.X265_10BIT_ANIME:
        return [
            "-c:v", "libx265",
            "-profile:v", "main10",
            "-pix_fmt", "yuv420p10le",
            "-crf", str(crf),
            "-preset", "slow",
            # x265 v4 removed the `anime`/`stillimage` tunes; emulate their intent
            # (flat colors, sharp line art, static scenes) with explicit params.
            # aq-mode=3 (variance-bias AQ per the x265 manual): measured +1.3 dB SSIM /
            # +1.24 dB PSNR at CRF 19 on flat-color anime content at identical encode
            # speed (+~11% size) — see docs/benchmarks.md, M7 A4.
            "-x265-params", "bframes=8:rc-lookahead=60:aq-mode=3",
        ]
    if preset == CodecPreset.NVENC_HEVC_10BIT:
        return [
            "-c:v", "hevc_nvenc",
            "-preset", "p7",
            "-rc", "vbr",
            "-cq", str(crf),
            "-b:v", "0",
            # Without this, hevc_nvenc defaults to 8-bit when fed an 8-bit
            # source (e.g. rgb24 pipe input) and the output is yuv420p.
            "-pix_fmt", "yuv420p10le",
        ]
    if preset == CodecPreset.H264:
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow"]
    raise ValueError(f"Unknown codec preset: {preset}")


def encode_video(
    frames,
    out_path: str | Path,
    width: int,
    height: int,
    fps: str,
    preset: str = CodecPreset.X265_10BIT_ANIME,
    crf: int = 19,
    container: str = "mp4",
    audio_src: str | Path | None = None,
    pix_fmt_in: str = "rgb24",
) -> Path:
    """Encode an iterable of HxWx3 uint8 frames to `out_path`.

    If `audio_src` is given and contains audio streams, they are muxed with
    passthrough (stream copy). Returns the output path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "." + container if not out_path.suffix else ""

    cmd = [find_ffmpeg(), "-v", "error", "-y"]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt_in, "-s", f"{width}x{height}", "-r", fps, "-i", "pipe:0"]

    audio_map: list[str] = []
    if audio_src is not None:
        src_info = probe(audio_src)
        if src_info.audio_streams:
            cmd += ["-i", str(audio_src)]
            for a in src_info.audio_streams:
                audio_map += ["-map", f"1:{a['index']}"]
            audio_map += ["-c:a", "copy"]

    cmd += ["-map", "0:v:0"] + audio_map + video_args(preset, crf)
    if container == "mp4" and preset != CodecPreset.H264:
        # 10-bit HEVC in MP4 is fine on modern players; keep mp4 as requested.
        pass
    cmd += [str(out_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame, dtype=np.uint8)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed (rc={rc}): {err[:3000]}")
    return out_path


def concat_encode(
    list_file: str | Path,
    out_path: str | Path,
    fps: str,
    preset: str = CodecPreset.X265_10BIT_ANIME,
    crf: int = 19,
    container: str = "mp4",
    audio_src: str | Path | None = None,
) -> Path:
    """Assemble lossless segment chunks (concat demuxer list) into the final encode."""
    cmd = [find_ffmpeg(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]

    audio_map: list[str] = []
    if audio_src is not None:
        src_info = probe(audio_src)
        if src_info.audio_streams:
            cmd += ["-i", str(audio_src)]
            for a in src_info.audio_streams:
                audio_map += ["-map", f"1:{a['index']}"]
            audio_map += ["-c:a", "copy"]

    cmd += ["-map", "0:v:0"] + audio_map + video_args(preset, crf) + [str(out_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat encode failed (rc={proc.returncode}): {proc.stderr[:3000]}")
    return Path(out_path)


def encode_lossless(
    frames,
    out_path: str | Path,
    width: int,
    height: int,
    fps: str,
    pix_fmt_in: str = "rgb24",
) -> Path:
    """Encode frames to a lossless FFV1-in-MKV chunk (checkpoint store format)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_ffmpeg(), "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", pix_fmt_in, "-s", f"{width}x{height}", "-r", fps, "-i", "pipe:0",
        "-c:v", "ffv1", "-level", "3", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame, dtype=np.uint8)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg lossless encode failed (rc={rc}): {err[:3000]}")
    return out_path


def stream_copy_passthrough(src: str | Path, out_path: str | Path) -> Path:
    """Fast sanity passthrough: remux without re-encoding (-c copy)."""
    cmd = [find_ffmpeg(), "-v", "error", "-y", "-i", str(src), "-c", "copy", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg passthrough failed (rc={proc.returncode}): {proc.stderr[:3000]}")
    return Path(out_path)
