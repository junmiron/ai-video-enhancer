"""Media probing and frame decoding via FFmpeg subprocess pipes."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install FFmpeg (full build) and ensure it is on PATH."
        )
    return exe


def find_ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not found on PATH.")
    return exe


@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    fps: str  # exact rational string, e.g. "24000/1001"
    num_frames: int
    duration_s: float
    video_codec: str
    pix_fmt: str
    audio_streams: list[dict] = field(default_factory=list)

    @property
    def fps_float(self) -> float:
        num, _, den = self.fps.partition("/")
        return float(num) / (float(den) if den else 1.0)


def probe(path: str | Path) -> MediaInfo:
    path = str(path)
    cmd = [find_ffprobe(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)

    vstream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if vstream is None:
        raise ValueError(f"No video stream in {path}")

    fps_raw = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "30/1"
    num, _, den = fps_raw.partition("/")
    try:
        fps_val = float(num) / (float(den) if den else 1.0)
    except ZeroDivisionError:
        fps_val = 30.0
    # Keep the exact rational for -r to avoid drift; fall back to "30".
    fps = fps_raw if "/" in fps_raw and float(den or 1) > 0 else f"{int(round(fps_val))}/1"

    num_frames = int(vstream.get("nb_frames") or 0)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if num_frames == 0 and duration > 0:
        num_frames = int(round(duration * fps_val))

    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]

    return MediaInfo(
        path=path,
        width=int(vstream["width"]),
        height=int(vstream["height"]),
        fps=fps,
        num_frames=num_frames,
        duration_s=duration,
        video_codec=str(vstream.get("codec_name", "")),
        pix_fmt=str(vstream.get("pix_fmt", "")),
        audio_streams=[{"index": s["index"], "codec": s.get("codec_name", "")} for s in audio_streams],
    )


def _read_exact(stream, n: int) -> bytes | None:
    """Read exactly n bytes from a pipe; returns None at EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_frames(
    path: str | Path,
    start_frame: int = 0,
    count: int | None = None,
    pix_fmt: str = "rgb24",
    hwaccel: bool = False,
):
    """Yield frames as numpy uint8 arrays (H x W x C) from an FFmpeg decode pipe.

    `count=None` decodes to end of stream. Uses rawvideo over stdout so no
    intermediate files are written.
    """
    path = str(path)
    info = probe(path)
    w, h = info.width, info.height
    channels = 3 if pix_fmt == "rgb24" else (1 if pix_fmt in ("gray", "gray10l") else 3)

    cmd = [find_ffmpeg(), "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", path]
    if start_frame > 0:
        # Frame-accurate selection by frame index (robust for CFR content).
        cmd += ["-vf", f"select='gte(n\\,{start_frame})'"]
    if count is not None:
        cmd += ["-frames:v", str(count)]
    if start_frame > 0 or count is not None:
        # Keep exactly the selected frames (FFmpeg >= 7 name for -vsync 0).
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = w * h * channels
    try:
        while True:
            buf = _read_exact(proc.stdout, frame_size)
            if buf is None:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, channels)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0 and err.strip():
            log.error("ffmpeg decode failed (rc=%s): %s", rc, err[:2000])
