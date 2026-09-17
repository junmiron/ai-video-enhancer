"""Scene (cut) detection via PySceneDetect content-aware analysis."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def detect_scenes(path: str, threshold: float = 27.0, min_scene_len_s: float = 1.0):
    """Return a list of (start_frame, end_frame_exclusive) covering the whole video.

    Falls back to a single segment when no cuts are found or detection fails.
    """
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(path)
    fps = video.frame_rate
    dur = video.duration
    if hasattr(dur, "frame_num"):  # FrameTimecode (scenedetect >= 0.7)
        total_frames = int(dur.frame_num)
    elif hasattr(dur, "get_frames"):  # older scenedetect
        total_frames = int(dur.get_frames())
    else:
        total_frames = int(float(dur) * fps)

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=max(1, int(min_scene_len_s * fps))))
    try:
        scene_manager.detect_scenes(video, show_progress=False)
        raw = scene_manager.get_scene_list()
    except Exception as exc:  # pragma: no cover - detector edge cases
        log.warning("Scene detection failed (%s); using single segment.", exc)
        return [(0, total_frames)]

    def _tc_frames(tc) -> int:
        # scenedetect >= 0.7: frame_num property; older: get_frames()
        return int(tc.frame_num) if hasattr(tc, "frame_num") else int(round(tc.get_frames()))

    if not raw:
        return [(0, total_frames)]

    segments = [(_tc_frames(s), _tc_frames(e)) for s, e in raw]
    # Defensive: ensure full coverage of the video.
    if segments[0][0] != 0:
        segments.insert(0, (0, segments[0][0]))
    if segments[-1][1] < total_frames:
        segments.append((segments[-1][1], total_frames))

    # Drop tiny leading/trailing slivers (encoder padding artifacts).
    min_len = max(1, int(min_scene_len_s * fps))
    cleaned = [(s, e) for s, e in segments if (e - s) >= min_len] or [(0, total_frames)]
    log.info("Scene detection: %d segment(s) over %d frames @ %sfps", len(cleaned), total_frames, fps)
    return cleaned
