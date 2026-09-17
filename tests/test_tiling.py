"""Tests for SR tiling: grid planning, feather masks and blending geometry.

The key invariant is *partition of unity*: summing every tile's feather mask
over its cover region must give exactly 1 at every frame pixel (within float
tolerance) — that is what guarantees the normalized blend has no seams, even
when last row/column covers are clamped smaller than the overlap.
"""
from __future__ import annotations

import numpy as np
import pytest

from aive.stages.upscale.tiling import (
    MASK_EPS,
    OVERLAP,
    Tile,
    apply_edge_floor,
    blend_into,
    choose_tile_size,
    feather_mask,
    plan_tiles,
    upscale_mask,
)


def _wsum(h: int, w: int, tile_cover: int, overlap: int = OVERLAP) -> np.ndarray:
    """Accumulate all tiles' masks into a per-pixel weight sum (no model needed).

    Mirrors the production blend path in SRRunner._run_tiled: feather mask +
    geometry-aware edge floor before accumulation.
    """
    acc = np.zeros((h, w), dtype=np.float64)
    for t in plan_tiles(h, w, tile_cover, overlap):
        m = feather_mask(t.cover_h, t.cover_w, nominal_h=tile_cover, nominal_w=tile_cover, overlap=overlap)
        apply_edge_floor(m, t.top, t.left, t.cover_h, t.cover_w, h, w)
        acc[t.top : t.top + t.cover_h, t.left : t.left + t.cover_w] += m.astype(np.float64)
    return acc


# -- plan_tiles ---------------------------------------------------------------


def test_plan_tiles_covers_frame_exactly():
    h, w = 100, 160
    tiles = plan_tiles(h, w, tile_cover=64, overlap=32)
    # Every pixel is covered by at least one tile.
    cov = np.zeros((h, w), dtype=bool)
    for t in tiles:
        assert 0 <= t.top and t.top + t.cover_h <= h
        assert 0 <= t.left and t.left + t.cover_w <= w
        cov[t.top : t.top + t.cover_h, t.left : t.left + t.cover_w] = True
    assert cov.all()


def test_plan_tiles_last_row_clamped():
    tiles = plan_tiles(100, 160, tile_cover=64, overlap=32)
    # Spacing is 32; last origins at 96 (y) and 128 (x).
    assert max(t.top for t in tiles) == 96
    assert max(t.left for t in tiles) == 128
    corner = [t for t in tiles if t.top == 96 and t.left == 128][0]
    assert (corner.cover_h, corner.cover_w) == (4, 32)


def test_plan_tiles_single_tile_small_frame():
    # Frame smaller than the grid spacing (tile_cover - overlap) in both dims.
    tiles = plan_tiles(30, 30, tile_cover=64, overlap=32)
    assert len(tiles) == 1
    assert tiles[0] == Tile(0, 0, 30, 30)


def test_plan_tiles_rejects_bad_input():
    with pytest.raises(ValueError):
        plan_tiles(0, 64, tile_cover=64)
    with pytest.raises(ValueError):
        plan_tiles(64, 64, tile_cover=0)


# -- feather masks / partition of unity ---------------------------------------


@pytest.mark.parametrize("h,w", [(128, 192), (100, 160), (95, 97), (33, 200), (64, 64)])
def test_partition_of_unity(h: int, w: int):
    """Interior weight sums are exactly 1 for several frame/tile combos.

    Includes frames whose last covers clamp to less than the overlap (95x97) —
    the case that breaks cover-size-anchored ramps. Frame-boundary pixels are
    covered by a single tile with sub-unit weight; they still normalize exactly
    in the blend, so we only require positivity there.
    """
    ov = 32
    s = _wsum(h, w, tile_cover=64, overlap=ov)
    assert (s > 0).all(), "some pixel has zero total weight -> division by zero in blend"
    # Pixels at least `overlap` away from every frame edge are covered by the
    # overlapping tile grid only -> weights sum to exactly 1.
    if h > 2 * ov and w > 2 * ov:
        inner = s[ov : h - ov, ov : w - ov]
        assert np.allclose(inner, 1.0, atol=1e-5), f"max deviation {np.abs(inner - 1).max()}"


def test_feather_mask_interior_and_edges():
    """Tent-shape ramp: 0 at the tile origin edge, peak 1.0 at x = overlap,
    tapering to 1/overlap (not zero) at the far edge.

    The exact-zero origin row/col is intentional — an adjacent tile's
    complementary ramp supplies the missing weight there (partition of unity).
    Frame-edge tiles have no neighbor beyond the frame; that case is handled by
    apply_edge_floor, not here (a blanket floor would break interior partition
    of unity in overlap strips).
    """
    m = feather_mask(64, 64, nominal_h=64, nominal_w=64, overlap=32)
    assert m.shape == (64, 64) and m.dtype == np.float32
    # Origin edge is exactly 0; peak at x = overlap is exactly 1.
    assert m[0, 32] == 0.0 and m[32, 0] == 0.0
    assert m[32, 32] == 1.0
    # Far edge tapers to 1/overlap but stays positive (no zero-division risk).
    np.testing.assert_allclose(m[63, 32], 1.0 / 32, rtol=1e-6)
    np.testing.assert_allclose(m[32, 63], 1.0 / 32, rtol=1e-6)
    # Monotone up to the peak, monotone down after it.
    assert (m[:33, 32] == np.sort(m[:33, 32])).all()
    assert (m[32:, 32][::-1] == np.sort(m[32:, 32][::-1])).all()


def test_apply_edge_floor_fixes_frame_edges_only():
    """Floor raises zero-weight frame-edge rows/cols to MASK_EPS and nothing else."""
    h, w = 128, 192
    for t in plan_tiles(h, w, tile_cover=64, overlap=OVERLAP):
        m = feather_mask(t.cover_h, t.cover_w, nominal_h=64, nominal_w=64, overlap=OVERLAP)
        before = m.copy()
        apply_edge_floor(m, t.top, t.left, t.cover_h, t.cover_w, h, w)
        # Flooring only raises values.
        assert (m >= before).all()
        changed = m > before
        if not changed.any():
            continue
        # Changed entries: were exactly 0 and became exactly the floor.
        assert (before[changed] == 0).all()
        np.testing.assert_allclose(m[changed], MASK_EPS, rtol=1e-6)
        # Changes are confined to this tile's frame-edge rows/cols.
        edge_rows = {i for i in range(t.cover_h)
                     if (t.top == 0 and i == 0) or (t.top + t.cover_h >= h and i == t.cover_h - 1)}
        edge_cols = {j for j in range(t.cover_w)
                     if (t.left == 0 and j == 0) or (t.left + t.cover_w >= w and j == t.cover_w - 1)}
        # A floored row spans every column (and vice versa), so each changed
        # pixel must sit on an edge row OR an edge col.
        rows_hit, cols_hit = np.where(changed)
        bad = [(i, j) for i, j in zip(rows_hit, cols_hit) if i not in edge_rows and j not in edge_cols]
        assert not bad, f"changed pixels off the frame-edge lines: {bad[:8]}"
    # A tile strictly inside the frame is never modified.
    inner = [t for t in plan_tiles(h, w, tile_cover=64, overlap=OVERLAP)
             if t.top > 0 and t.left > 0 and t.top + t.cover_h < h and t.left + t.cover_w < w]
    assert inner, "expected at least one interior tile for this geometry"
    t = inner[0]
    m = feather_mask(t.cover_h, t.cover_w, nominal_h=64, nominal_w=64, overlap=OVERLAP)
    before = m.copy()
    apply_edge_floor(m, t.top, t.left, t.cover_h, t.cover_w, h, w)
    np.testing.assert_array_equal(before, m)


def test_feather_mask_clamped_cover_uses_nominal_ramp():
    """A cover clamped to 34 px (overlap is 32) still ramps over the nominal 64."""
    full = feather_mask(64, 64, nominal_h=64, nominal_w=64, overlap=32)
    clamped = feather_mask(34, 64, nominal_h=64, nominal_w=64, overlap=32)
    # The clamped mask is the top slice of the full mask (same anchor).
    np.testing.assert_allclose(clamped, full[:34], atol=1e-6)


def test_feather_mask_no_overlap_is_ones():
    m = feather_mask(48, 96, nominal_h=48, nominal_w=96, overlap=0)
    assert (m == 1.0).all()


# -- choose_tile_size ----------------------------------------------------------


def test_choose_tile_size_full_frame_fits():
    # 720p with a tiny per-pixel cost fits in 32 GB easily -> no tiling.
    assert choose_tile_size(720, 1280, free_vram_bytes=32 * 2**30, bytes_per_pixel=600) is None


def test_choose_tile_size_tiles_when_needed():
    # Force a small budget: full frame must tile.
    c = choose_tile_size(1080, 1920, free_vram_bytes=4 * 2**30, bytes_per_pixel=2500, pad_mod=4)
    assert c is not None
    # The resulting engine input (c + 2*overlap)^2 must fit the budget.
    side = c + 2 * OVERLAP
    assert side * side * 2500 <= int(4 * 2**30 * 0.75)
    assert c % 4 == 0 and c >= 64


def test_choose_tile_size_never_exceeds_frame():
    c = choose_tile_size(100, 80, free_vram_bytes=1024, bytes_per_pixel=1000, pad_mod=4)
    assert c is not None and c <= 80


# -- blend_into geometry --------------------------------------------------------


def test_blend_into_writes_cover_region_only():
    acc = np.zeros((1, 3, 64, 96), dtype=np.float32)  # NCHW accumulator
    wsum = np.zeros((64, 96), dtype=np.float32)
    tile_out = np.full((1, 3, 8, 12), 0.5, dtype=np.float32)
    mask = feather_mask(8, 12, nominal_h=8, nominal_w=12, overlap=4)

    blend_into(acc, wsum, out_top=16, out_left=24, tile_out_nchw=tile_out, mask=mask)

    # Exact accumulation: region == tile_out * mask (origin row/col are exactly
    # 0 by tent-shape design; interior is positive).
    np.testing.assert_allclose(acc[0, :, 16:24, 24:36], tile_out[0] * mask[None], atol=1e-7)
    assert (acc[0, :, 18:22, 26:34] > 0).all()
    # Everything outside the cover region is untouched.
    probe = acc.copy()
    probe[0, :, 16:24, 24:36] = 0
    assert (probe == 0).all()
    np.testing.assert_allclose(wsum[16:24, 24:36], mask, atol=1e-7)
    outside = wsum.copy()
    outside[16:24, 24:36] = 0
    assert (outside == 0).all()


def test_blend_normalization_recovers_constant_image():
    """A constant input must come back exactly constant after blend + normalize.

    Mirrors the production tiled path: mask upscaled to output resolution and
    edge-floored before blending, so frame-edge pixels (single coverage,
    sub-unit weight) still normalize exactly.
    """
    h, w, s = 96, 128, 2
    acc = np.zeros((1, 3, h * s, w * s), dtype=np.float32)  # NCHW accumulator
    wsum = np.zeros((h * s, w * s), dtype=np.float32)
    for t in plan_tiles(h, w, tile_cover=64, overlap=OVERLAP):
        # "Model output" = constant 0.75 over the cover (what a real model would
        # return for a constant input, modulo global-op effects).
        out_crop = np.full((1, 3, t.cover_h * s, t.cover_w * s), 0.75, dtype=np.float32)
        mask = feather_mask(t.cover_h, t.cover_w, nominal_h=64, nominal_w=64, overlap=OVERLAP)
        apply_edge_floor(mask, t.top, t.left, t.cover_h, t.cover_w, h, w)
        blend_into(acc, wsum, t.top * s, t.left * s, out_crop, upscale_mask(mask, s))
    assert (wsum > 0).all(), "edge floor must keep every output pixel covered"
    result = acc[0] / wsum[None]
    np.testing.assert_allclose(result, 0.75, atol=1e-5)
