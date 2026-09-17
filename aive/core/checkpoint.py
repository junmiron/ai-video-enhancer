"""Segment checkpoint store.

Completed scene segments are written as lossless FFV1-in-MKV chunks so a job
can resume from the last completed segment boundary after crash/cancel.
The same chunks feed the final encode (concat demuxer), so there is one
mechanism for both durability and assembly.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Segment:
    index: int
    start_frame: int
    end_frame: int  # exclusive

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class _SegmentRecord:
    index: int
    start_frame: int
    end_frame: int
    file: str
    status: str = "pending"  # pending | done


class CheckpointStore:
    def __init__(self, workspace_dir: str | Path):
        self.dir = Path(workspace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.dir / "manifest.json"
        self._records: list[_SegmentRecord] = []
        self._fingerprint = ""  # identity of the job that produced the stored chunks
        if self._manifest_path.exists():
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            self.job_id = data.get("job_id", "")
            self.input_path = data.get("input_path", "")
            self.width = int(data.get("width", 0))
            self.height = int(data.get("height", 0))
            self.fps = str(data.get("fps", "30"))
            self._fingerprint = data.get("fingerprint", "")
            self._records = [_SegmentRecord(**r) for r in data.get("segments", [])]

    # -- lifecycle ---------------------------------------------------------
    def init_job(
        self,
        job_id: str,
        input_path: str,
        width: int,
        height: int,
        fps: str,
        segments: list[Segment],
        fingerprint: str = "",
    ) -> None:
        """(Re)initialize the workspace for a job.

        Already-done segment records are preserved ONLY when `fingerprint`
        matches the one stored in the manifest — i.e. same input file and same
        content-affecting pipeline settings. Otherwise the stored chunks belong
        to a different job and are discarded, so a re-run can never silently
        reuse stale segments.
        """
        self.job_id = job_id
        self.input_path = input_path
        self.width, self.height, self.fps = width, height, str(fps)
        same_job = bool(fingerprint) and fingerprint == self._fingerprint
        if same_job:
            done = {r.index: r for r in self._records if r.status == "done" and (self.dir / r.file).exists()}
        else:
            for r in self._records:
                stale = self.dir / r.file
                if stale.exists():
                    log.info("Discarding stale checkpoint %s (job fingerprint changed).", stale.name)
                    stale.unlink()
            done = {}
        self._fingerprint = fingerprint
        self._records = []
        for seg in segments:
            existing = done.get(seg.index)
            self._records.append(
                existing or _SegmentRecord(seg.index, seg.start_frame, seg.end_frame, f"seg_{seg.index:04d}.mkv")
            )
        self._save()

    def _save(self) -> None:
        payload = {
            "job_id": self.job_id,
            "input_path": self.input_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fingerprint": self._fingerprint,
            "segments": [asdict(r) for r in self._records],
        }
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self._manifest_path)

    # -- queries -----------------------------------------------------------
    def is_done(self, index: int) -> bool:
        for r in self._records:
            if r.index == index:
                return r.status == "done" and (self.dir / r.file).exists()
        return False

    @property
    def done_count(self) -> int:
        return sum(1 for r in self._records if self.is_done(r.index))

    @property
    def total_segments(self) -> int:
        return len(self._records)

    def pending_indices(self) -> list[int]:
        return [r.index for r in self._records if not self.is_done(r.index)]

    def segment_file(self, index: int) -> Path:
        for r in self._records:
            if r.index == index:
                return self.dir / r.file
        raise KeyError(index)

    # -- writing -----------------------------------------------------------
    def mark_done(self, index: int) -> None:
        for r in self._records:
            if r.index == index:
                r.status = "done"
        self._save()

    def concat_list_file(self) -> Path:
        """Write an ffmpeg concat-demuxer list of all segment chunks (in order)."""
        lines = []
        for r in sorted(self._records, key=lambda x: x.index):
            p = (self.dir / r.file).as_posix().replace("\\", "/")
            # Escape single quotes per ffmpeg concat demuxer rules.
            lines.append(f"file '{p.replace(chr(39), chr(39) + chr(92) + chr(39))}'")
        out = self.dir / "concat.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def clear(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
