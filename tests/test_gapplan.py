"""Unit tests for gap-plan math (pure-Python port of TAS _timesteps.py)."""
from aive.stages.interpolate.gapplan import (
    cut_hold_split,
    estimate_output_frames,
    fps_ratio,
    gap_plan,
    interpolate_timestep,
)


def test_fps_ratio_common_cases():
    assert fps_ratio(24.0, 60.0) == (5, 2)   # 24 -> 60
    assert fps_ratio(30.0, 120.0) == (4, 1)  # 30 -> 120
    assert fps_ratio(24.0, 48.0) == (2, 1)   # integer x2
    assert fps_ratio(29.97, 59.94) == (2, 1)  # NTSC doubling
    assert fps_ratio(24.0, 80.0) == (10, 3)  # non-integer ratio


def test_fps_ratio_odd_rate_small_denominator():
    target = 59.94 / 23.976
    num, den = fps_ratio(23.976, 59.94)
    assert den <= 64
    assert abs(num / den - target) < 0.001


def test_gap_plan_unit_interval_integer_factor():
    n, ts = gap_plan(0, 1, 2, 1)
    assert (n, ts) == (1, [0.5])
    n, ts = gap_plan(3, 4, 4, 1)
    assert (n, ts) == (3, [0.25, 0.5, 0.75])


def test_gap_plan_24_to_60_pattern():
    # factor 5/2: unit intervals alternate 1 and 2 inserts with exact timesteps
    n0, ts0 = gap_plan(0, 1, 5, 2)
    assert (n0, ts0) == (1, [0.4])
    n1, ts1 = gap_plan(1, 2, 5, 2)
    assert (n1, ts1) == (2, [0.2, 0.6])


def test_gap_plan_timesteps_in_open_interval_and_sorted():
    for k in range(48):
        n, ts = gap_plan(k, k + 1, 5, 2)
        assert all(0.0 < t < 1.0 for t in ts)
        assert ts == sorted(ts)


def test_gap_plan_wide_interval_covers_all_slots():
    # A de-duplication-widened gap covers every output slot inside (prev, cur],
    # including the slots that would have been endpoint frames for the removed
    # duplicates. For f=2 over (0,4]: slots 1..7 -> 7 inserts = 4 unit inserts
    # + 3 internal endpoints (source frames 1..3 were duplicates).
    n_wide, ts = gap_plan(0, 4, 2, 1)
    assert n_wide == 7
    total_units = sum(gap_plan(k, k + 1, 2, 1)[0] for k in range(4))
    internal_endpoints = 4 - 1
    assert n_wide == total_units + internal_endpoints


def test_interpolate_timestep_fallback():
    assert interpolate_timestep(0, 3) == 1 / 4
    assert interpolate_timestep(2, 3) == 3 / 4
    assert interpolate_timestep(1, 2, [0.25, 0.75]) == 0.75


def test_estimate_output_frames_identity():
    for n in (1, 8, 96):
        assert estimate_output_frames(n, 1, 1) == n


def test_estimate_output_frames_matches_driver_simulation():
    # Simulate the stage driver: first frame as-is, then per interval inserts + endpoint.
    def simulate(n_src, num, den):
        out = 1
        prev = 0
        for cur in range(1, n_src):
            k, _ = gap_plan(prev, cur, num, den)
            out += k + 1
            prev = cur
        return out

    assert estimate_output_frames(96, 5, 2) == simulate(96, 5, 2)
    assert estimate_output_frames(8, 2, 1) == simulate(8, 2, 1)


def test_estimate_output_frames_96_at_24_to_60():
    # Last source frame (k=95) lands on slot floor(95 * 5/2) = 237 -> 238 total.
    assert estimate_output_frames(96, 5, 2) == 238


def test_cut_hold_split_unit_gap_is_zero():
    assert cut_hold_split(0, 1, 2, 1) == 0
    assert cut_hold_split(7, 8, 5, 2) == 0


def test_cut_hold_split_wide_gap_bounded():
    n_insert, _ = gap_plan(0, 4, 2, 1)
    hold = cut_hold_split(0, 4, 2, 1)
    assert 0 <= hold < n_insert + 1
