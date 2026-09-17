"""Tiled inference helpers for SR (M3): grid planning and feathered blending.

Full-frame inference is the default on a 32 GB card; tiling kicks in only when
the estimated VRAM for one full frame exceeds ``threshold`` of the free VRAM
(see :func:`choose_tile_size`).

Geometry: tiles are placed on a grid with nominal cover size ``C`` and spacing
``S = C - overlap``. Each tile is *responsible* for its (clamped) cover region;
its model input is that cover expanded by ``overlap`` px on every side (clamped
to the frame, edge-padded to the fixed engine size). The model output's central
part — exactly the cover — is blended into a shared accumulator with a linearly
ramped feather mask anchored to the nominal boundaries.

With linear ramps of length ``overlap`` at each nominal cover edge, adjacent
tiles' weights sum to exactly 1 in their overlap strip (partition of unity), so
the normalized blend has no seam: every output pixel is a weighted average whose
weights add up to 1. Frame-edge pixels are covered by a single tile and resolve
exactly to that tile's value; :func:`apply_edge_floor` floors only the mask
rows/cols on frame edges (where a tile origin would otherwise leave a corner
pixel with zero total weight) so normalization never divides by zero without
disturbing any overlap region.

Note on global ops: ShuffleCUGAN's SE blocks pool over the whole input, so a
tile's output depends on its full content — overlap + blending is what keeps
the result continuous across tile borders (same approach TAS uses for SR).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

OVERLAP = 32  # px at input resolution; feather ramp length on each cover edge
MASK_EPS = 1e-4  # floor for frame-edge mask rows/cols so wsum is never zero


def upscale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-upscale a feather mask by the model's output scale.

    Exact under nearest sampling: every output pixel inherits the weight of its
    input-resolution parent pixel, so partition of unity (and positivity) carry
    over unchanged to output resolution.
    """
    if scale == 1:
        return mask
    return np.kron(mask, np.ones((scale, scale), dtype=mask.dtype))


@dataclass(frozen=True)
class Tile:
    """One tile's responsibility region within the frame."""

    top: int
    left: int
    cover_h: int
    cover_w: int


def plan_tiles(h: int, w: int, tile_cover: int, overlap: int = OVERLAP) -> list[Tile]:
    """Grid of overlapping tiles covering [0,h) x [0,w).

    ``tile_cover`` is the nominal cover size; spacing is ``tile_cover - overlap``.
    Last row/column covers are clamped to the frame edge.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame size {h}x{w}")
    if tile_cover < 1:
        raise ValueError("tile_cover must be >= 1")
    overlap = min(overlap, max(0, tile_cover // 2))
    step = tile_cover - overlap

    tiles: list[Tile] = []
    top = 0
    while top < h:
        left = 0
        while left < w:
            ch = min(tile_cover, h - top)
            cw = min(tile_cover, w - left)
            tiles.append(Tile(top, left, ch, cw))
            left += step
        top += step
    return tiles


def feather_mask(
    cover_h: int,
    cover_w: int,
    nominal_h: int | None = None,
    nominal_w: int | None = None,
    overlap: int = OVERLAP,
) -> np.ndarray:
    """Linearly-ramped weight map for one tile's (possibly clamped) cover region.

    1.0 in the interior; ramps linearly to ~0 over ``overlap`` px at each edge;
    floored at MASK_EPS (see module doc). Shape [cover_h, cover_w], float32.

    Ramps are anchored to the NOMINAL cover boundaries — not the clamped size —
    so adjacent tiles' weights sum to exactly 1 even when the last row/column
    covers are clamped smaller than ``overlap`` (partition of unity).
    """
    nominal_h = nominal_h or cover_h
    nominal_w = nominal_w or cover_w
    overlap = min(overlap, max(0, min(nominal_h, nominal_w) // 2))

    def axis(n: int, nominal: int) -> np.ndarray:
        if n <= 0 or overlap == 0:
            return np.ones(max(n, 1), dtype=np.float32)
        x = np.arange(n, dtype=np.float32)  # position within the cover
        left = np.clip(x / overlap, 0.0, 1.0)
        right = np.clip((nominal - x) / overlap, 0.0, 1.0)
        return (left * right).astype(np.float32)

    rows = axis(cover_h, nominal_h)[:, None]
    cols = axis(cover_w, nominal_w)[None, :]
    return (rows * cols).astype(np.float32)


def apply_edge_floor(
    mask: np.ndarray,
    tile_top: int,
    tile_left: int,
    cover_h: int,
    cover_w: int,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Floor the mask rows/cols that lie on a frame edge (in place, returns mask).

    A pixel at a tile origin has zero ramp weight; when that origin coincides
    with a frame boundary (the four corners), the pixel would have no positive
    coverage at all and normalization would divide by zero. Flooring only those
    edge rows/cols keeps partition of unity exact in every overlap region while
    guaranteeing wsum > 0 everywhere — single-coverage edge pixels normalize to
    exactly their covering tile's value regardless of the floor magnitude.
    """
    if tile_top == 0:
        mask[0] = np.maximum(mask[0], MASK_EPS)
    if tile_top + cover_h >= frame_h:
        mask[-1] = np.maximum(mask[-1], MASK_EPS)
    if tile_left == 0:
        mask[:, 0] = np.maximum(mask[:, 0], MASK_EPS)
    if tile_left + cover_w >= frame_w:
        mask[:, -1] = np.maximum(mask[:, -1], MASK_EPS)
    return mask


def choose_tile_size(
    h: int,
    w: int,
    free_vram_bytes: int,
    bytes_per_pixel: float,
    pad_mod: int = 4,
    threshold: float = 0.75,
) -> int | None:
    """Pick a nominal tile cover size, or ``None`` when the full frame fits.

    The model input for one tile is ``(C + 2*overlap)^2`` pixels; we require its
    estimated activation memory to stay under ``threshold * free VRAM``.
    """
    budget = int(free_vram_bytes * threshold)
    if h * w * bytes_per_pixel <= budget:
        return None

    max_input_side = int(math.sqrt(max(budget, 1) / max(bytes_per_pixel, 1e-9)))
    c = max_input_side - 2 * OVERLAP
    c = (c // pad_mod) * pad_mod
    # Sane bounds: never smaller than a workable tile, never larger than the frame.
    c = max(c, min(64, h, w))
    c = min(c, h, w)
    return c


def blend_into(
    acc: np.ndarray,
    wsum: np.ndarray,
    out_top: int,
    out_left: int,
    tile_out_nchw: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Accumulate one tile's (NCHW float32 [0,1]) output into the frame accumulators.

    ``out_top``/``out_left`` locate the tile's cover in OUTPUT resolution;
    ``mask`` is the feather mask for that cover at OUTPUT resolution
    ([ch*scale, cw*scale], see :func:`upscale_mask`).
    """
    ch, cw = mask.shape
    region = acc[:, :, out_top : out_top + ch, out_left : out_left + cw]
    region += tile_out_nchw[0] * mask[None]
    wsum[out_top : out_top + ch, out_left : out_left + cw] += mask
