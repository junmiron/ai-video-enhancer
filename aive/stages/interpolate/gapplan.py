"""Frame-slot planning for frame interpolation (pure Python, no torch).

Ported from TheAnimeScripter `src/interpolate/_timesteps.py` (MIT license,
https://github.com/NevermindNilas/TheAnimeScripter) with light renaming.

Positions count SOURCE frames. For a source interval ``(prevPos, curPos]`` the
planner returns how many output slots fall inside it and the timestep ``t in
(0, 1)`` for each slot relative to that interval (t=0 is prevPos's frame,
t=1 is curPos's frame). The interpolation driver emits intermediates for the
interval *ending* at the frame it was just handed.
"""
from __future__ import annotations


def gap_plan(prev_pos: int, cur_pos: int, factor_num: int, factor_den: int) -> tuple[int, list[float]]:
    """Frames to synthesize for the SOURCE interval ``(prevPos, curPos]``.

    A de-duplicated gap that swallowed duplicate frames is simply a wider
    interval and gets proportionally more output slots. For an integer factor
    ``f = factor_num/factor_den`` over a unit interval this yields exactly
    ``[1/f, 2/f, ..., (f-1)/f]``.

    Returns ``(frames_to_insert, timesteps)`` with timesteps in ``(0, 1)``,
    relative to the interval ending at ``cur_pos``.
    """
    out_prev = (prev_pos * factor_num) // factor_den
    out_cur = (cur_pos * factor_num) // factor_den
    # Keep the whole numerator in integers: dividing an absolute source position
    # first and subtracting after loses the exactness of ratios like 1/3 once
    # prevPos is large, and those values must match the driver's own ladder.
    base = prev_pos * factor_num
    scale = factor_num * (cur_pos - prev_pos)
    timesteps = [(o * factor_den - base) / scale for o in range(out_prev + 1, out_cur)]
    return len(timesteps), timesteps


def cut_hold_split(prev_pos: int, cur_pos: int, factor_num: int, factor_den: int) -> int:
    """How many of a gap's synthesized slots precede its final source step.

    A gap widened by de-duplication covers several source frames, and every one
    of them except the last was a duplicate of the frame at ``prev_pos``. When
    a scene cut lands on ``cur_pos``, only that final source step is the cut;
    the slots before it still belong to the outgoing shot and must be filled
    from the previous frame, not the incoming one. Returns 0 for a one-frame-
    wide gap (a plain cut fills entirely from the incoming frame).
    """
    out_prev = (prev_pos * factor_num) // factor_den
    last_step = ((cur_pos - 1) * factor_num) // factor_den
    return max(0, last_step - out_prev)


def interpolate_timestep(index: int, frames_to_insert: int, timesteps: list[float] | None = None) -> float:
    """Timestep for the ``index``-th inserted frame of an interval."""
    if timesteps is not None and index < len(timesteps):
        return timesteps[index]
    return (index + 1) / (frames_to_insert + 1)


def fps_ratio(source_fps: float, target_fps: float) -> tuple[int, int]:
    """Reduce ``target/source`` to a small integer ratio (num, den).

    Used as the (factor_num, factor_den) pair for gap_plan. Exact for common
    cases like 24->60 (5/2), 30->120 (4/1); falls back to a continued-fraction
    approximation with denominator <= 64 for odd ratios.
    """
    from fractions import Fraction

    frac = Fraction(round(target_fps, 6) / round(source_fps, 6)).limit_denominator(64)
    return frac.numerator, frac.denominator


def estimate_output_frames(num_source_frames: int, factor_num: int, factor_den: int) -> int:
    """Total output frames the interpolation stage yields for a segment.

    Mirrors the driver exactly: source frames sit at positions 0..N-1 and the
    first is emitted as-is; each interval ``(k, k+1]`` adds its planned slots
    plus the endpoint frame. Source frame ``k`` lands on output slot
    ``floor(k * factor_num / factor_den)`` (exact timing by construction).
    """
    if num_source_frames <= 0:
        return 0
    total = 1
    prev = 0
    for cur in range(1, num_source_frames):
        n, _ = gap_plan(prev, cur, factor_num, factor_den)
        total += n + 1
        prev = cur
    return total
