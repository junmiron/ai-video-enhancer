"""FastLineDarken — TAS line-art fix as an exact OpenCV/numpy port (M4.x).

TAS ships this as a torch module (src/extraArches/fastlinedarken.py) whose
"TRT" variant merely exports the fixed conv pipeline to an engine; it is NOT a
learned model. We therefore port the math directly instead of building an
engine for fixed kernels — exact match, milliseconds per frame, no download.

Exact op sequence (matches TAS ``FastLineDarkenWithStreams``, default
``darkenStrength=0.7``, thinEdges=True, gaussian=True):

1. luma = 0.2989*R + 0.5870*G + 0.1140*B   (BT.601)
2. Sobel X/Y: 3x3 kernels [[-1,0,1],[-2,0,2],[-1,0,1]] / [[-1,-2,-1],[0,0,0],[1,2,1]],
   ZERO-padded cross-correlation (torch F.conv2d padding=1 semantics).
3. edges = sqrt(gx^2 + gy^2), then GLOBAL min-max normalize to [0,1].
4. Thinning: 3x3 max-pool of the NEGATED map with zero padding, negated back —
   i.e. ``-dilate(-edges)``; the outer 1px border ring becomes exactly 0
   (the zero pad dominates at the frame edge). cv2.dilate's default
   BORDER_CONSTANT(0) reproduces this exactly.
5. Gaussian blur k=3 sigma=1.0, ZERO-padded grouped conv — cv2.filter2D with
   BORDER_CONSTANT (NOT cv2.GaussianBlur, which reflect-pads by default and
   would change the border).
6. image -= strength * softened_edges (same edge map on all 3 channels),
   clamp to [0,1].

One deliberate deviation: TAS divides by ``edges.max() - edges.min()`` without
a guard, so a perfectly flat frame yields NaN everywhere. This port returns
the input unchanged when the edge range is ~0 — the sane result for "no edges".

The lineart repair applies this *masked* (edge map computed globally for the
correct normalization, subtraction only inside the detector's region) to keep
repairs targeted per design.md §4.3.
"""
from __future__ import annotations

import cv2
import numpy as np

_LUMA_W = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
_MAXPOOL_K = np.ones((3, 3), dtype=np.uint8)


def _gaussian_kernel_3x3(sigma: float = 1.0) -> np.ndarray:
    """TAS's gaussianBlur kernel: exp(-x^2/2sigma^2) over x in {-1,0,1}, normalized."""
    x = np.arange(-1.0, 2.0)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    return (k[:, None] * k[None, :]).astype(np.float32)


def fast_line_darken_f32(img_hwc: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """FastLineDarken on an HWC float [0,1] image; returns HWC float clamped to [0,1]."""
    gray = (img_hwc * _LUMA_W[None, None, :]).sum(axis=2)  # [H,W]

    # Zero-padded cross-correlation == torch F.conv2d(padding=1).
    gx = cv2.filter2D(gray, -1, _SOBEL_X, borderType=cv2.BORDER_CONSTANT)
    gy = cv2.filter2D(gray, -1, _SOBEL_Y, borderType=cv2.BORDER_CONSTANT)
    edges = np.sqrt(gx * gx + gy * gy)

    lo = float(edges.min())
    hi = float(edges.max())
    if hi - lo <= 1e-9:
        return img_hwc.copy()  # flat frame: no edges to darken (TAS would NaN here)
    edges = (edges - lo) / (hi - lo)

    # Zero-padded max-pool thinning via dilate on the negated map.
    thinned = -cv2.dilate(-edges, _MAXPOOL_K)  # default border: constant 0

    softened = cv2.filter2D(thinned, -1, _gaussian_kernel_3x3(), borderType=cv2.BORDER_CONSTANT)

    out = img_hwc - strength * softened[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fast_line_darken(frame_hwc_u8: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """FastLineDarken on an HWC uint8 frame; returns HWC uint8 (full-frame result)."""
    img = frame_hwc_u8.astype(np.float32) / 255.0
    out = fast_line_darken_f32(img, strength=strength)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
