"""Artifact-injection harness (M4 acceptance gate).

Takes clean frames and injects *synthetic* artifacts into known regions so we can
verify each detector catches what it is supposed to catch (recall) without lighting
up elsewhere (false positives). This is the key quality gate for the differentiator:
the plan requires **≥90% of injected regions caught with <10% false positives**.

All functions are pure numpy/cv2 over HWC uint8 frames at full resolution; the test
converts to proxy and runs the *real* detectors on the result, then uses :func:`measure`
to score recall / false-positive rate against the known injected bbox.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from .findings import ArtifactType


@dataclass
class Injection:
    """A clean window with one synthetic artifact stamped into a known region."""

    artifact: ArtifactType
    bbox: tuple[int, int, int, int]  # x, y, w, h (full res) where the artifact was injected
    frames: list[np.ndarray]  # HWC uint8 window with the injection applied
    is_interpolated: bool = False


def make_clean_window(n_frames: int = 7, h: int = 360, w: int = 640, seed: int = 0) -> list[np.ndarray]:
    """A mostly-static scene with mild structure (low flow) for injection tests.

    Static so the flicker detector's low-flow gate is satisfied; a few soft blobs and
    a gentle gradient give it non-degenerate content without introducing motion.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (xx / w * 120 + yy / h * 60) % 255
    img = np.stack([base] * 3, axis=-1).astype(np.uint8)
    for _ in range(4):
        cy = int(rng.integers(h // 4, 3 * h // 4))
        cx = int(rng.integers(w // 4, 3 * w // 4))
        r = int(rng.integers(20, 60))
        col = tuple(int(c) for c in rng.integers(80, 200, size=3))
        cv2.circle(img, (cx, cy), r, col, -1)
    return [img.copy() for _ in range(n_frames)]


def inject_flicker(frames: list[np.ndarray], bbox: tuple[int, int, int, int], amp: float = 14.0, seed: int = 0) -> Injection:
    """Add independent per-frame noise inside ``bbox`` → high temporal σ there."""
    rng = np.random.default_rng(seed)
    x, y, w, hh = bbox
    out = []
    for f in frames:
        g = f.copy()
        noise = (rng.uniform(-amp, amp, size=(hh, w, 3))).astype(np.int16)
        region = g[y : y + hh, x : x + w].astype(np.int16) + noise
        g[y : y + hh, x : x + w] = np.clip(region, 0, 255).astype(np.uint8)
        out.append(g)
    return Injection(ArtifactType.FLICKER, bbox, out)


def inject_banding(frame: np.ndarray, bbox: tuple[int, int, int, int], levels: int = 6) -> Injection:
    """Replace ``bbox`` with a posterized horizontal gradient → long plateaus."""
    x, y, w, hh = bbox
    g = frame.copy()
    grad = np.linspace(0.0, 255.0, w, dtype=np.float32)
    step = 255.0 / max(1, levels - 1)
    quant = (np.round(grad / step) * step).astype(np.uint8)  # (w,)
    band = np.repeat(quant[:, None], 3, axis=1)  # (w, 3)
    g[y : y + hh, x : x + w] = np.repeat(band[None, :, :], hh, axis=0)
    return Injection(ArtifactType.BANDING, bbox, [g])


def inject_ghosting(center: np.ndarray, prev: np.ndarray, nxt: np.ndarray, bbox: tuple[int, int, int, int], seed: int = 0) -> Injection:
    """Fill ``bbox`` in ``center`` with content that matches neither neighbor → dual-diff.

    A saturated color far from the (grayish) base guarantees a large per-pixel distance to
    both neighbors across the *whole* region, which is exactly what the detector measures.
    """
    rng = np.random.default_rng(seed)
    x, y, w, hh = bbox
    g = center.copy()
    fill = np.array([250, 30, 210], dtype=np.int16) + rng.integers(-8, 9, size=(hh, w, 3))
    g[y : y + hh, x : x + w] = np.clip(fill, 0, 255).astype(np.uint8)
    return Injection(ArtifactType.GHOSTING, bbox, [prev, g, nxt], is_interpolated=True)


def inject_halo(frame: np.ndarray, src: np.ndarray, bbox: tuple[int, int, int, int]) -> Injection:
    """Add ringing overshoot around a strong edge inside ``bbox`` (vs the source)."""
    x, y, w, hh = bbox
    g = frame.copy()
    # A sharp vertical edge in the source; the processed copy gets overshoot lobes.
    ex = x + w // 2
    src[y : y + hh, max(0, ex - 1) : min(w, ex + 8)] = 30
    g[y : y + hh, max(0, ex - 1) : min(w, ex + 8)] = 225
    # Overshoot: bright ring just outside the dark edge on both sides.
    for off in (9, 10, 11):
        if ex + off < x + w:
            g[y : y + hh, ex + off] = np.clip(g[y : y + hh, ex + off].astype(int) + 70, 0, 255).astype(np.uint8)
    return Injection(ArtifactType.HALO, bbox, [g])


def inject_color_drift(frames: list[np.ndarray], bbox: tuple[int, int, int, int], shift: float = 0.18) -> Injection:
    """Shift the whole frame's palette (per-channel gain/offset) vs its first frame."""
    ref = frames[0].astype(np.float32)
    out = []
    for f in frames:
        g = f.astype(np.float32)
        g[..., 0] *= 1.0 + shift
        g[..., 2] -= shift * 90.0
        out.append(np.clip(g, 0, 255).astype(np.uint8))
    return Injection(ArtifactType.COLOR_DRIFT, bbox, out)


def inject_regression(frames: list[np.ndarray], bad_index: int = 3, blur_ksize: int = 9) -> Injection:
    """Make one frame an outlier (heavy blur) vs its neighbors → IQA z-score spike."""
    h, w = frames[bad_index].shape[:2]
    out = [f.copy() for f in frames]
    out[bad_index] = cv2.GaussianBlur(frames[bad_index], (blur_ksize, blur_ksize), 0)
    bbox = (w // 4, h // 4, w // 2, h // 2)
    return Injection(ArtifactType.REGRESSION, bbox, out)


def inject_lineart(frame: np.ndarray, src: np.ndarray, bbox: tuple[int, int, int, int]) -> Injection:
    """Blur/thicken the lines inside ``bbox`` relative to a crisp source."""
    x, y, w, hh = bbox
    g = frame.copy()
    g[y : y + hh, x : x + w] = cv2.GaussianBlur(frame[y : y + hh, x : x + w], (5, 5), 1.0)
    return Injection(ArtifactType.LINEART, bbox, [g])


def measure(mask: np.ndarray, bbox: tuple[int, int, int, int], fw: int, fh: int, pw: int, ph: int, thresh: float = 0.3) -> tuple[float, float]:
    """Score a proxy-res detector mask against the known injected region.

    Returns ``(recall, fp_rate)``: the fraction of injected pixels flagged (catch) and
    the fraction of non-injected pixels falsely flagged. ``bbox`` is full-res; it is
    scaled into proxy coordinates using the given full/proxy dims.
    """
    x, y, w, hh = bbox
    sx, sy = pw / max(1, fw), ph / max(1, fh)
    px0, py0 = int(x * sx), int(y * sy)
    px1, py1 = int((x + w) * sx), int((y + hh) * sy)
    inside = mask[py0:py1, px0:px1]
    region = np.zeros_like(mask)
    region[py0:py1, px0:px1] = 1.0
    outside = region == 0
    recall = float((inside > thresh).mean()) if inside.size else 0.0
    fp_rate = float((mask[outside] > thresh).mean()) if outside.any() else 0.0
    return recall, fp_rate
