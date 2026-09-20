"""Bounded depth streaming over overlapping chunks (TAS-validated pattern).

Ported from TheAnimeScripter's ``src/depth/chunk_stream.py`` (AGPL project —
this file is a clean reimplementation of the same algorithm, not copied code):
a sliding window of ``chunk_size`` frames with 50% overlap. Each chunk is one
model forward over N ordered views of ONE scene; repeated views between chunks
are aligned by a robust depth-scale fit (median ratio on sampled pixels) and
linearly blended so the output has no visible seam.

Only predictions of the SAME input frame are ever blended, and windows never
cross scene cuts because the caller feeds one scene segment at a time.
"""
from __future__ import annotations

from itertools import islice
from typing import Callable, Iterable, Iterator

import numpy as np


def align_depth_overlap(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Align a new chunk to its predecessor and blend their repeated views.

    ``previous`` holds the last ``overlap`` depth maps of the prior chunk;
    ``current[:overlap]`` are the model's fresh predictions for those same
    frames. Monocular depth is only defined up to a per-chunk scale, so we fit
    a single median ratio on deterministically-sampled pixels (tolerant of
    moving objects and outliers) and rescale the whole chunk before blending.
    """
    overlap = len(previous)
    repeated = current[:overlap]
    if repeated.shape != previous.shape:
        raise ValueError("Depth overlap dimensions changed between chunks")

    # Deterministic spatial sampling bounds the temporary ratio array. Median
    # ratios tolerate moving-object errors and outliers in either prediction.
    stride = max(1, (previous.size + 99999) // 100000)
    old = previous.reshape(-1)[::stride].astype(np.float64)
    new = repeated.reshape(-1)[::stride].astype(np.float64)
    valid = np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)
    if valid.sum() < 32:
        # No reliable shared scale: use the new prediction without blending
        # two unrelated coordinate systems.
        return current
    scale = float(np.median(old[valid] / new[valid]))
    if not np.isfinite(scale) or scale <= 0:
        return current
    current *= scale
    if not np.isfinite(current).all():
        raise ValueError("Non-finite depth after chunk scale alignment")

    weight = np.arange(1, overlap + 1, dtype=np.float32)[:, None, None]
    weight /= overlap + 1
    old_valid = previous > 0
    new_valid = repeated > 0
    current[:overlap] = np.where(
        old_valid & new_valid,
        previous * (1 - weight) + repeated * weight,
        np.where(new_valid, repeated, previous),
    )
    return current


def stream_depth_chunks(
    frames: Iterable[np.ndarray],
    infer_chunk: Callable[[list[np.ndarray]], np.ndarray],
    chunk_size: int = 32,
    overlap: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield exactly one depth map per input frame, holding at most one RGB chunk.

    ``infer_chunk`` receives ordered views of one scene and returns [N,H,W]
    positive depth (one map per view). A half-window overlap is retained until
    the next prediction, including at EOF — no extra inference or duplicate
    outputs are needed for an exact frame-count fit: a short tail chunk uses
    its real length (duplicate padding would change attention and bias depth).
    """
    if overlap is None:
        overlap = chunk_size // 2
    if chunk_size < 2 or not 0 < overlap <= chunk_size // 2:
        raise ValueError("Require chunkSize >= 2 and 0 < overlap <= chunkSize // 2")

    frames = iter(frames)
    window = list(islice(frames, chunk_size))
    previous: np.ndarray | None = None
    while window:
        depths = np.array(infer_chunk(window), dtype=np.float32, copy=True)
        if depths.ndim != 3 or len(depths) != len(window):
            raise ValueError("Chunk inference must return one [H,W] depth map per frame")
        depths[~np.isfinite(depths) | (depths <= 0)] = 0
        if previous is not None:
            depths = align_depth_overlap(previous, depths)

        if len(window) < chunk_size:
            yield from depths
            return

        yield from depths[:-overlap]
        previous = depths[-overlap:].copy()
        del depths
        window = window[-overlap:]
        new_frames = list(islice(frames, chunk_size - overlap))
        if not new_frames:
            yield from previous
            return
        window.extend(new_frames)
        del new_frames
