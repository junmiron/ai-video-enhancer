"""Depth-map output writers: 10-bit grayscale MP4 and 16-bit PNG sequence.

Both are FFmpeg stdin pipes fed with raw little-endian gray frames (verified on
the bundled ffmpeg 9.x build):

* ``mp4_10bit`` — ``rawvideo gray10le -> libx265 -pix_fmt gray10le``: true
  monochrome HEVC (I400), the default depth video. No explicit profile flag:
  x265 rejects ``main10`` for mono input and auto-selects its mono profile.
* ``png16`` — ``rawvideo gray16le -> png`` into a folder: lossless 16-bit
  sequence for editing/3D workflows (round-trips bit-exact).

Frames arrive as [H,W] float32 in [0,1]; quantization happens once here.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from ..ingest import find_ffmpeg, popen_hidden

log = logging.getLogger(__name__)


class DepthWriter:
    """Writes one depth map per call; close() finalizes the container."""

    def __init__(self, out_path: str | Path, width: int, height: int, fps: str, fmt: str = "mp4_10bit"):
        self.out_path = Path(out_path)
        self.width, self.height = width, height
        self.fps = fps
        if fmt not in ("mp4_10bit", "png16"):
            raise ValueError(f"Unknown depth output format: {fmt}")
        self.fmt = fmt

        cmd = [find_ffmpeg(), "-v", "error", "-y"]
        if fmt == "mp4_10bit":
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["-f", "rawvideo", "-pix_fmt", "gray10le", "-s", f"{width}x{height}", "-r", fps, "-i", "pipe:0"]
            cmd += [
                "-c:v", "libx265",
                "-pix_fmt", "gray10le",
                "-crf", "18",
                "-preset", "slow",
                str(self.out_path),
            ]
        else:  # png16
            self.out_path.mkdir(parents=True, exist_ok=True)
            cmd += ["-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{width}x{height}", "-r", fps, "-i", "pipe:0"]
            cmd += ["-c:v", "png", str(self.out_path / "%05d.png")]

        self._proc = popen_hidden(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        log.info("Depth writer started (%s): %s", fmt, self.out_path)

    def write(self, depth_f32: np.ndarray) -> None:
        """Write one [H,W] float32 depth map in [0,1]."""
        if depth_f32.shape != (self.height, self.width):
            raise ValueError(f"Depth frame {depth_f32.shape} does not match writer {(self.height, self.width)}")
        d = np.clip(depth_f32, 0.0, 1.0)
        if self.fmt == "mp4_10bit":
            raw = (d * 1023.0 + 0.5).astype(np.uint16)
        else:
            raw = (d * 65535.0 + 0.5).astype(np.uint16)
        try:
            self._proc.stdin.write(raw.tobytes())
        except BrokenPipeError:
            err = self._read_err()
            raise RuntimeError(f"Depth writer pipe broken (ffmpeg died): {err[:2000]}") from None

    def _read_err(self) -> str:
        try:
            return self._proc.stderr.read().decode("utf-8", "replace") if self._proc.stderr else ""
        except Exception:  # pragma: no cover - pipe already closed
            return ""

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):  # pragma: no cover - ffmpeg died early
            pass
        err = self._read_err()
        rc = self._proc.wait()
        self._proc = None
        if rc != 0:
            raise RuntimeError(f"ffmpeg depth encode failed (rc={rc}): {err[:3000]}")
        log.info("Depth writer closed: %s", self.out_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:  # don't mask the original exception with a possibly-broken pipe close
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError, AttributeError):
                pass
            self._proc.wait()
            self._proc = None
