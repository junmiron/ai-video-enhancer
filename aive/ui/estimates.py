"""Rough output-size estimates for the settings panel (design.md §7.2 "Est. size").

Deliberately simple heuristics — good enough to warn about a 40 GB surprise, not
a bitrate calculator. Pure Python: no Qt, no torch.
"""
from __future__ import annotations

# Approximate bits-per-pixel at CRF ~19 for our encode presets (anime content is
# very compressible; these are mid-range values from typical x265/NVENC runs).
# M7 A4: the x265 anime preset gained aq-mode=3 (+~11% size, +1.3 dB SSIM) -> 0.10 -> 0.11.
_BPP_BASE = {
    "x265-10bit-anime": 0.11,
    "nvenc-hevc-10bit": 0.13,
    "h264": 0.16,
}


def fi_ratio(fi_enabled: bool, target_fps: float | None, multiplier: int | None, src_fps: float) -> float:
    """Output/source frame-count ratio for the interpolation settings."""
    if not fi_enabled or src_fps <= 0:
        return 1.0
    if target_fps and target_fps > 0:
        return max(1.0, target_fps / src_fps)
    return float(max(1, int(multiplier or 2)))


def estimate_output_bytes(
    src_width: int,
    src_height: int,
    src_fps: float,
    num_frames: int,
    audio_bitrate_bps: int = 0,
    sr_enabled: bool = False,
    sr_scale: int | None = None,
    fi_enabled: bool = False,
    fi_target_fps: float | None = None,
    fi_multiplier: int | None = None,
    codec: str = "x265-10bit-anime",
    crf: int = 19,
) -> float:
    """Estimated output file size in bytes (video + audio when passthrough)."""
    scale = float(sr_scale or 2) if sr_enabled else 1.0
    out_w, out_h = src_width * scale, src_height * scale
    ratio = fi_ratio(fi_enabled, fi_target_fps, fi_multiplier, src_fps)
    out_frames = max(1, int(round(num_frames * ratio)))

    bpp = _BPP_BASE.get(codec, 0.12)
    # Every +6 CRF roughly halves the size (and vice versa), relative to CRF 19.
    crf_factor = 2 ** ((19 - crf) / 6.0)
    video_bytes = out_w * out_h * out_frames * bpp * crf_factor / 8.0

    audio_bytes = 0.0
    if audio_bitrate_bps > 0 and src_fps > 0:
        duration_s = num_frames / src_fps
        audio_bytes = duration_s * audio_bitrate_bps / 8.0
    return video_bytes + audio_bytes


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"
