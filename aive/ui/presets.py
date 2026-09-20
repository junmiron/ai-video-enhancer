"""Preset definitions (design.md §7.2).

Presets drive all defaults; every field stays overridable in the settings panel.
A preset is a plain dict of :class:`aive.config.Settings` fields — applying one
overwrites exactly those fields and leaves everything else untouched, so a user's
tweaks survive switching presets for unrelated groups.

Pure data module: no Qt imports (usable from tests and the worker).
"""
from __future__ import annotations

PRESET_ORDER = ["anime", "live_action", "mixed", "fast_preview"]

PRESETS: dict[str, dict] = {
    "anime": {
        "label": "Anime",
        "description": (
            "CUGAN-class upscale + RIFE interpolation tuned for 2D animation. "
            "The default choice for anime content."
        ),
        "fields": {
            # TAS-proven stage order: interpolate at source res (fast), then upscale.
            "stage_order": "fi_first",
            # Frame interpolation: 60 fps target, light de-duplication of held frames.
            "fi_enabled": True,
            "fi_model": "rife_425",
            "fi_target_fps": 60.0,
            "fi_multiplier": None,
            "fi_dedup_threshold": 0.01,
            # Upscale: the TAS workhorse (tiny, fast, anime-tuned).
            "sr_enabled": True,
            "sr_model": "shufflecugan_x2",
            "sr_scale": 2,
            # Artifact QA: full auto with AI repairs.
            "qa_enabled": True,
            "qa_mode": "auto",
            "qa_ai_repairs": True,
            # Depth: off by default; Limbo v2 is the anime-tuned model when enabled.
            "depth_enabled": False,
            "depth_model": "limbo_v2",
            # Output (design.md D4).
            "output_codec": "x265-10bit-anime",
            "crf": 19,
            "container": "mp4",
            "audio_passthrough": True,
        },
    },
    "live_action": {
        "label": "Live Action",
        "description": (
            "Real-ESRGAN upscale + RIFE interpolation for photographic content. "
            "Upscale runs before interpolation (design §3.1) for maximum motion "
            "detail; no anime de-duplication."
        ),
        "fields": {
            # Upscale first: RIFE at target resolution — slower, best motion fidelity.
            "stage_order": "sr_first",
            "fi_enabled": True,
            "fi_model": "rife_425",
            "fi_target_fps": 60.0,
            "fi_multiplier": None,
            "fi_dedup_threshold": 0.0,
            "sr_enabled": True,
            "sr_model": "realesrgan_x4plus",
            "sr_scale": 2,
            "qa_enabled": True,
            "qa_mode": "auto",
            "qa_ai_repairs": True,
            "depth_enabled": False,
            "depth_model": "depth_anything_v2_small",
            "output_codec": "x265-10bit-anime",
            "crf": 19,
            "container": "mp4",
            "audio_passthrough": True,
        },
    },
    "mixed": {
        "label": "Mixed / Unknown",
        "description": (
            "General-purpose: Real-ESRGAN upscale + 2× frame interpolation. "
            "Safe defaults when the content type is unclear."
        ),
        "fields": {
            "stage_order": "fi_first",
            "fi_enabled": True,
            "fi_model": "rife_425",
            "fi_target_fps": None,
            "fi_multiplier": 2,
            "fi_dedup_threshold": 0.0,
            "sr_enabled": True,
            "sr_model": "realesrgan_x4plus",
            "sr_scale": 2,
            "qa_enabled": True,
            "qa_mode": "auto",
            "qa_ai_repairs": True,
            "depth_enabled": False,
            "depth_model": "depth_anything_v2_small",
            "output_codec": "x265-10bit-anime",
            "crf": 19,
            "container": "mp4",
            "audio_passthrough": True,
        },
    },
    "fast_preview": {
        "label": "Fast Preview",
        "description": (
            "Quick test run: no interpolation, fast CUGAN upscale, detect-only QA "
            "(report without repairs) and NVENC encoding. Use it to check the "
            "pipeline on a clip before committing to a long job."
        ),
        "fields": {
            # FI is off here, so stage order has no effect; keep the fast default.
            "stage_order": "fi_first",
            "fi_enabled": False,
            "fi_model": "rife_425",
            "fi_target_fps": None,
            "fi_multiplier": 2,
            "fi_dedup_threshold": 0.0,
            "sr_enabled": True,
            "sr_model": "shufflecugan_x2",
            "sr_scale": 2,
            "qa_enabled": True,
            "qa_mode": "detect_only",
            "qa_ai_repairs": False,
            "depth_enabled": False,
            "depth_model": "limbo_v2",
            "output_codec": "nvenc-hevc-10bit",
            "crf": 23,
            "container": "mp4",
            "audio_passthrough": True,
        },
    },
}


def preset_keys() -> list[str]:
    return list(PRESET_ORDER)


def apply_preset(settings_dict: dict, preset_key: str) -> dict:
    """Return a copy of `settings_dict` with the preset's fields applied.

    Accepts/returns plain dicts (asdict(Settings)) so this stays Qt-free and
    trivially testable; unknown preset keys raise KeyError.
    """
    if preset_key not in PRESETS:
        raise KeyError(f"Unknown preset '{preset_key}'. Known: {PRESET_ORDER}")
    out = dict(settings_dict)
    out.update(PRESETS[preset_key]["fields"])
    return out


def default_models_for(preset_key: str, settings_dict: dict | None = None) -> list[str]:
    """Registry names the first-run 'Get started' download should fetch.

    The preset's SR + FI models; depth model only when depth generation is on in
    `settings_dict` (defaults to off). Restore-model weights (SCUNet / deepDeband-f)
    auto-download on first use inside the QA stage, so they are not pre-fetched.
    """
    s = dict(settings_dict or {})
    s.update(PRESETS[preset_key]["fields"])
    names: list[str] = []
    if s.get("sr_enabled"):
        names.append(s["sr_model"])
    if s.get("fi_enabled"):
        names.append(s["fi_model"])
    if s.get("depth_enabled"):
        names.append(s["depth_model"])
    # De-duplicate, keep order.
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]
