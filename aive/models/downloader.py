"""Model weight downloading with progress, checksum verification and caching.

Supports plain HTTP(S) URLs and Google Drive share links (RIFE weights live on
Drive). First download records the computed SHA-256 in a local verified.json;
subsequent runs verify against it (and against the manifest hash when present).

Some releases ship as archive bundles (e.g. a zipped folder); set
``extract_member`` in the manifest and the configured member is extracted once
as the canonical weights file.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.request
from pathlib import Path

from .registry import ModelInfo, default_manifest_path  # noqa: F401 - re-export convenience

log = logging.getLogger(__name__)


def resolve_gdrive(url: str) -> str | None:
    """Convert a Google Drive share URL to a direct download URL (files < ~100 MB)."""
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url) or re.search(r"id=([A-Za-z0-9_-]+)", url)
    if not m:
        return None
    file_id = m.group(1)
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _filename_for(url: str, model_name: str) -> str:
    direct = resolve_gdrive(url) or url
    name = direct.split("/")[-1].split("?")[0]
    return name if "." in name else f"{model_name}.bin"


class DownloadError(RuntimeError):
    pass


def download_model(
    info: ModelInfo,
    dest_dir: str | Path | None = None,
    progress_cb=None,
) -> Path:
    """Download (or return cached) weights for `info`. Returns the local path."""
    from ..config import model_cache_dir

    if not info.urls:
        raise DownloadError(
            f"{info.name}: no download URLs configured "
            "(experimental model; adapter work pending)"
        )

    base = Path(dest_dir) if dest_dir else model_cache_dir() / info.name / info.version
    base.mkdir(parents=True, exist_ok=True)

    url = info.urls[0]
    direct = resolve_gdrive(url) or url
    fname = _filename_for(direct, info.name)
    target = base / fname
    verified_file = base / "verified.json"
    verified: dict = {}
    if verified_file.exists():
        try:
            verified = json.loads(verified_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            verified = {}

    # Cache hit?
    if target.exists():
        expected = info.sha256 or verified.get(fname)
        if expected is None or _sha256_file(target) == expected.lower():
            log.info("Model cached: %s", target)
            return _maybe_extract(info, base, target, verified, verified_file)
        raise DownloadError(f"Cache hash mismatch for {target}; delete it and re-download.")

    req = urllib.request.Request(direct, headers={"User-Agent": "Mozilla/5.0 (aive model downloader)"})
    log.info("Downloading %s from %s", info.name, direct)
    part = target.with_suffix(target.suffix + ".part")
    h = hashlib.sha256()
    total: int | None = None
    done = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as f:
            total_hdr = resp.headers.get("Content-Length")
            total = int(total_hdr) if total_hdr else None
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress_cb is not None and (total or done > 0):
                    progress_cb(done, total or -1)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"Download failed for {info.name}: {exc}") from exc

    digest = h.hexdigest()
    expected = (info.sha256 or verified.get(fname) or "").lower() or None
    if expected is not None and digest != expected:
        part.unlink(missing_ok=True)
        raise DownloadError(f"SHA-256 mismatch for {info.name}: got {digest[:16]}…, expected {expected[:16]}…")

    part.replace(target)
    verified[fname] = digest
    verified_file.write_text(json.dumps(verified, indent=1), encoding="utf-8")
    log.info("Downloaded & verified %s (sha256=%s…)", target.name, digest[:16])
    return _maybe_extract(info, base, target, verified, verified_file)


def _maybe_extract(
    info: ModelInfo,
    base: Path,
    artifact: Path,
    verified: dict,
    verified_file: Path,
) -> Path:
    """Extract the configured archive member as the canonical weights file.

    Returns `artifact` unchanged when ``extract_member`` is not set. The
    extracted file gets its own entry in verified.json so re-runs skip the
    (cheap, but unnecessary) extraction.
    """
    if not info.extract_member:
        return artifact
    import zipfile

    member_name = Path(info.extract_member).name
    out_path = base / member_name
    expected = verified.get(member_name)
    if out_path.exists() and (expected is None or _sha256_file(out_path) == expected.lower()):
        log.info("Extracted weights cached: %s", out_path)
        return out_path

    try:
        with zipfile.ZipFile(artifact) as zf:
            data = zf.read(info.extract_member)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DownloadError(
            f"{info.name}: expected archive containing '{info.extract_member}', got {exc}"
        ) from exc

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_bytes(data)
    digest = _sha256_file(tmp)
    tmp.replace(out_path)
    verified[member_name] = digest
    verified_file.write_text(json.dumps(verified, indent=1), encoding="utf-8")
    log.info("Extracted %s -> %s (sha256=%s…)", info.extract_member, out_path.name, digest[:16])
    return out_path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
