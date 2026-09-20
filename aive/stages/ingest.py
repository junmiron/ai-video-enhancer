"""Media probing and frame decoding via FFmpeg subprocess pipes."""
from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _bundled_tool_dir() -> Path | None:
    """Folder holding bundled ffmpeg/ffprobe for packaged builds (M7 B).

    PyInstaller onedir: ``<app>/ffmpeg/`` next to the executable. Dev checkout:
    ``<repo>/tools/ffmpeg/``. Lets the installer ship FFmpeg without touching PATH.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "ffmpeg"
    return Path(__file__).resolve().parents[2] / "tools" / "ffmpeg"


def _find_tool(name: str) -> str | None:
    exe = shutil.which(name)
    if exe:
        return exe
    d = _bundled_tool_dir()
    for suffix in (".exe", ""):  # Windows first, then POSIX
        p = d / f"{name}{suffix}"
        if p.exists():
            return str(p)
    return None


def find_ffmpeg() -> str:
    exe = _find_tool("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found on PATH or in the bundled ffmpeg folder. Install FFmpeg "
            "(full build) and add it to PATH, or place ffmpeg.exe under tools\\ffmpeg "
            "(dev) / <app>\\ffmpeg (installed)."
        )
    return exe


def find_ffprobe() -> str:
    exe = _find_tool("ffprobe")
    if not exe:
        raise RuntimeError(
            "ffprobe not found on PATH or in the bundled ffmpeg folder. Install FFmpeg "
            "(full build) and add it to PATH, or place ffprobe.exe under tools\\ffmpeg "
            "(dev) / <app>\\ffmpeg (installed)."
        )
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


def grab_frame(path: str | Path, index: int):
    """Fast single-frame grab for the M6 preview compare view (design.md §7.3).

    Input-seeks ~2 s before the target frame and decodes forward until the
    requested absolute frame number is reached — matched via ``showinfo``
    ``pts_time`` values, which stay accurate after an input seek. Only a couple
    of seconds of video are decoded instead of everything up to `index`, so
    this stays interactive even on long 1080p/4K files.

    Returns the frame as HxWx3 uint8 RGB, or ``None`` when out of range / on error.
    """
    import time as _time

    path = str(path)
    info = probe(path)
    if index < 0 or (info.num_frames and index >= info.num_frames):
        return None
    fps = info.fps_float or 30.0
    t_target = index / fps
    seek = max(0.0, t_target - 2.0)
    limit = int((t_target - seek) * fps) + 8  # decode margin past the target

    cmd = [
        find_ffmpeg(), "-v", "info",
        "-ss", f"{seek:.6f}",
        "-i", path,
        # Keep ORIGINAL (absolute) timestamps after the input seek — without it
        # showinfo's pts_time is relative to the seek point and late frames in a
        # file never match their absolute index.
        "-copyts",
        "-vf", "showinfo", "-an", "-sn",
        "-frames:v", str(limit),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = info.width * info.height * 3
    frames_q: queue.Queue[bytes | None] = queue.Queue(maxsize=16)

    def _reader():
        try:
            while True:
                buf = _read_exact(proc.stdout, frame_size)
                if buf is None:
                    break
                frames_q.put(buf)
        except Exception:  # pragma: no cover - pipe closed early
            pass
        finally:
            frames_q.put(None)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    deadline = _time.time() + 60.0
    try:
        for line in proc.stderr:
            if _time.time() > deadline:
                return None
            m = re.search(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)", line.decode("utf-8", "replace"))
            if not m or "Parsed_showinfo" not in line.decode("utf-8", "replace"):
                continue
            idx = int(round(float(m.group(1)) * fps))
            buf = frames_q.get(timeout=30.0)
            if buf is None:  # stream ended before the target frame
                return None
            if idx >= index:
                return np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3).copy()
    except (queue.Empty, OSError):
        return None
    finally:
        try:
            proc.kill()
        except OSError:  # pragma: no cover
            pass
        proc.wait()
