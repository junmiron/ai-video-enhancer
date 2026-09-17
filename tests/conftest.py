"""Shared fixtures: synthetic test clips generated with FFmpeg lavfi."""
from __future__ import annotations

import subprocess

import pytest


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[:2000]}")


@pytest.fixture(scope="session")
def test_clip(tmp_path_factory) -> str:
    """3s 640x360@30fps clip with moving pattern + sine audio (H.264)."""
    out = tmp_path_factory.mktemp("clips") / "test.mp4"
    _run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-shortest", str(out),
    ])
    return str(out)


@pytest.fixture(scope="session")
def cut_clip(tmp_path_factory) -> str:
    """2s 640x360@30fps clip with one hard cut at t=1s (red -> blue)."""
    out = tmp_path_factory.mktemp("clips") / "cut.mp4"
    _run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:size=640x360:rate=30:duration=1",
        "-f", "lavfi", "-i", "color=c=blue:size=640x360:rate=30:duration=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(out),
    ])
    return str(out)


@pytest.fixture(scope="session")
def static_clip(tmp_path_factory) -> str:
    """1s 640x360@30fps fully static gray clip (no cuts, no motion)."""
    out = tmp_path_factory.mktemp("clips") / "static.mp4"
    _run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=gray:size=640x360:rate=30:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(out),
    ])
    return str(out)
