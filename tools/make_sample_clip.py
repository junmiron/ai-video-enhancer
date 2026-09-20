"""Generate the app's sample test clip (design.md §7.5 first-run offer).

640x360 @ 30fps, 90 frames (3 s): camera pans right across a parallax scene
(sky/sun/clouds -> mountains -> hills+trees -> grass) while a chibi character
walks in place with swinging limbs and cel line-art. Encoded H.264 + sine audio
so the pipeline's audio passthrough is exercised too.

Usage::

    python tools/make_sample_clip.py [output.mp4]   (default: samples/sample_anime.mp4)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 640, 360
FPS, N = 30, 90

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def circle(cx, cy, r):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def ellipse(cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def rect(x0, y0, w, h):
    m = np.zeros((H, W), bool)
    m[int(round(y0)):int(round(y0 + h)), int(round(x0)):int(round(x0 + w))] = True
    return m


def capsule(p1, p2, r):
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    tseg = np.clip(((xx - x1) * dx + (yy - y1) * dy) / (dx * dx + dy * dy + 1e-9), 0.0, 1.0)
    return (xx - (x1 + tseg * dx)) ** 2 + (yy - (y1 + tseg * dy)) ** 2 <= r * r


def erode(m, t=2):
    e = m.copy()
    for dx in range(-t, t + 1):
        for dy in range(-t, t + 1):
            if dx or dy:
                e &= np.roll(np.roll(m, dx, axis=1), dy, axis=0)
    return e


def paint(img, mask, color):
    img[mask] = color


def stroke(img, mask, t=2, color=(42, 42, 58)):
    o = mask & ~erode(mask, t)
    img[o] = color


def sky() -> np.ndarray:
    top, bot = np.array([126, 200, 240], float), np.array([223, 243, 251], float)
    f = np.clip(yy / 280.0, 0, 1)[:, :, None]
    return (top[None, None, :] + (bot - top)[None, None, :] * f).astype(np.uint8)


def mountain_top(x):
    a = np.abs(np.sin(2 * np.pi * x / 320.0)) ** 1.4
    b = np.abs(np.sin(2 * np.pi * x / 97.0 + 1.3)) ** 1.8
    return 280 - (55 * a + 26 * b)


def hill_top(x):
    u = (x % 480.0) - 240.0
    return 280 - 62.0 * np.exp(-(u ** 2) / (2 * 135.0 ** 2))


def cloud_mask(cx, cy, s):
    return (ellipse(cx, cy, 46 * s, 15 * s) | ellipse(cx - 30 * s, cy + 7 * s, 28 * s, 11 * s)
            | ellipse(cx + 32 * s, cy + 6 * s, 30 * s, 12 * s))


def render(t: int) -> np.ndarray:
    img = sky().copy()

    # sun (far layer, slow parallax)
    sx = 540 - t * 0.35
    paint(img, circle(sx, 66, 26), (255, 241, 189))
    paint(img, circle(sx, 66, 34) & ~circle(sx, 66, 27), (255, 246, 210))

    # clouds: own wind drift + wrap
    for cx0, cy0, s in [(90, 78, 1.0), (310, 48, 0.75), (520, 100, 1.2)]:
        cx = ((cx0 - t * 1.6 + 100) % (W + 200)) - 100
        paint(img, cloud_mask(cx, cy0, s), (255, 255, 255))

    # mountains (parallax 0.5x camera speed)
    paint(img, yy >= mountain_top(xx + t * 0.5), (104, 116, 178))

    # hills (parallax 1.0x)
    paint(img, yy >= hill_top(xx + t * 1.0), (95, 170, 105))

    # trees riding the hill layer at fixed world positions
    for wx in (240.0, 640.0, 1120.0):
        tx = ((wx - t * 1.0) % (W + 80)) - 40
        by = float(hill_top(np.array([wx]))[0])
        trunk = capsule((tx, by), (tx, by - 20), 3.5)
        canopy = circle(tx, by - 31, 15) | circle(tx - 9, by - 24, 11) | circle(tx + 9, by - 24, 11)
        paint(img, trunk, (122, 84, 52))
        stroke(img, trunk)
        paint(img, canopy, (56, 128, 74))
        stroke(img, canopy)

    # ground
    img[yy >= 280] = (139, 204, 128)
    img[(yy >= 280) & (yy <= 283)] = (96, 158, 92)

    # chibi character walking in place (foreground)
    phase = 2 * np.pi * t / 15.0
    xc = 320 + 6 * np.sin(2 * np.pi * t / 45.0)
    yf = 308 - 2.5 * (1 + np.cos(phase)) / 2
    hip_y = yf - 24
    torso_top = hip_y - 26
    head_cy = torso_top - 15

    leg_l = capsule((xc - 4, hip_y), (xc - 4 + 7 * np.sin(phase), yf), 3.5)
    leg_r = capsule((xc + 4, hip_y), (xc + 4 - 7 * np.sin(phase), yf), 3.5)
    paint(img, leg_l | leg_r, (66, 88, 148))
    stroke(img, leg_l)
    stroke(img, leg_r)

    arm_l = capsule((xc - 12, torso_top + 5), (xc - 12 - 7 * np.sin(phase), torso_top + 20), 3.0)
    arm_r = capsule((xc + 12, torso_top + 5), (xc + 12 - 7 * np.sin(phase), torso_top + 20), 3.0)
    paint(img, arm_l | arm_r, (236, 104, 132))
    stroke(img, arm_l)
    stroke(img, arm_r)

    body = rect(xc - 13, torso_top, 26, hip_y - torso_top + 2)
    paint(img, body, (236, 104, 132))
    stroke(img, body)

    head = circle(xc, head_cy, 17)
    paint(img, head, (255, 221, 191))
    stroke(img, head)

    hair = ((circle(xc, head_cy - 3, 18.5) & (yy <= head_cy - 4))
            | circle(xc - 15, head_cy + 7, 6.5) | circle(xc + 15, head_cy + 7, 6.5))
    paint(img, hair, (72, 48, 38))
    stroke(img, hair)

    eyes = ellipse(xc - 6, head_cy + 3, 2.0, 3.0) | ellipse(xc + 6, head_cy + 3, 2.0, 3.0)
    paint(img, eyes, (45, 45, 60))

    return img


def make_clip(dest: str | Path) -> Path:
    """Render the scene and encode it to `dest` (MP4). Returns the path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(tempfile.mkdtemp(prefix="aive-sample-"))
    try:
        for t in range(N):
            Image.fromarray(render(t)).save(frames_dir / f"f_{t:03d}.png")
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-framerate", str(FPS), "-i", str(frames_dir / "f_%03d.png"),
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={N / FPS:.2f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "17",
                "-c:a", "aac", "-b:a", "96k", "-shortest", str(dest),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the AI Video Enhancer sample clip.")
    default_out = Path(__file__).resolve().parent.parent / "samples" / "sample_anime.mp4"
    parser.add_argument("dest", nargs="?", default=str(default_out), help="output MP4 path")
    args = parser.parse_args()
    out = make_clip(args.dest)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
