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
import socket
import time
import urllib.error
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


# M7: transient network failures are retried with backoff; every URL in the
# manifest is a mirror and is tried when an earlier one fails permanently.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0


def _is_transient(exc: Exception) -> bool:
    """Network-level failures worth retrying; checksum errors are not."""
    if "SHA-256 mismatch" in str(exc):
        return False
    inner = exc.__cause__ or exc
    if isinstance(inner, (socket.timeout, TimeoutError, ConnectionError)):
        return True
    if isinstance(inner, urllib.error.HTTPError):  # subclass of URLError — check first
        code = getattr(inner, "code", 0) or 0
        return code in (408, 429) or code >= 500
    if isinstance(inner, urllib.error.URLError):
        return True
    return False


def _fetch_one(name: str, direct_url: str, part: Path, progress_cb, expected_sha: str | None) -> str:
    """Download one URL to `part`; returns the SHA-256 hex digest.

    Resumes an existing partial file when a manifest checksum is known and the
    server honors HTTP Range (HuggingFace does) — important for multi-GB weights;
    without a known checksum a stale partial could mask corruption, so it restarts.
    Raises DownloadError on network failure (partial file removed) or when the
    digest does not match `expected_sha` (when given).
    """
    headers = {"User-Agent": "Mozilla/5.0 (aive model downloader)"}
    h = hashlib.sha256()
    resume_pos = 0
    if expected_sha is not None and part.exists():
        # Hash the existing partial so the final digest covers the whole file.
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        resume_pos = part.stat().st_size
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"
    req = urllib.request.Request(direct_url, headers=headers)
    log.info(
        "Downloading %s from %s%s",
        name, direct_url,
        f" (resuming at {resume_pos >> 20} MiB)" if resume_pos else "",
    )
    total: int | None = None
    done = resume_pos
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = getattr(resp, "status", 200) or 200
            resumed = resume_pos > 0 and status == 206
            if not resumed:
                h = hashlib.sha256()
                resume_pos = 0
                done = 0
            cr_hdr = resp.headers.get("Content-Range")
            cl_hdr = resp.headers.get("Content-Length")
            if cr_hdr and "/" in cr_hdr:
                tail = cr_hdr.split("/")[-1]
                total = int(tail) if tail.isdigit() else None
            elif resumed and cl_hdr:
                total = resume_pos + int(cl_hdr)
            elif cl_hdr:
                total = int(cl_hdr)
            with open(part, "ab" if resumed else "wb") as f:
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
        raise DownloadError(f"Download failed for {name}: {exc}") from exc

    digest = h.hexdigest()
    if expected_sha is not None and digest != expected_sha.lower():
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"SHA-256 mismatch for {name}: got {digest[:16]}…, expected {expected_sha[:16]}…"
        )
    return digest


def _resolve_bundled(info: ModelInfo, base: Path) -> Path:
    """Copy a locally-bundled model file into the cache (sha256-verified)."""
    from .registry import bundled_source_path

    src = bundled_source_path(info)
    if src is None:
        raise DownloadError(
            f"{info.name}: bundled file '{info.bundled}' not found next to the manifest "
            "(personal-use model; place it under assets/models/)"
        )
    target = base / Path(info.bundled).name
    expected = (info.sha256 or "").lower() or None
    if target.exists():
        if expected is None or _sha256_file(target) == expected:
            log.info("Bundled model cached: %s", target)
            return target
        raise DownloadError(f"Cache hash mismatch for bundled {target}; delete it and retry.")
    import shutil

    tmp = target.with_suffix(target.suffix + ".part")
    shutil.copyfile(src, tmp)
    digest = _sha256_file(tmp)
    if expected is not None and digest != expected:
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"SHA-256 mismatch for bundled {info.name}: got {digest[:16]}…, expected {expected[:16]}…"
        )
    tmp.replace(target)
    log.info("Copied bundled model %s -> %s (sha256=%s…)", src.name, target.name, digest[:16])
    return target


def download_model(
    info: ModelInfo,
    dest_dir: str | Path | None = None,
    progress_cb=None,
) -> Path:
    """Download (or return cached) weights for `info`. Returns the local path."""
    from ..config import model_cache_dir

    if not info.urls and not info.bundled and not info.files:
        raise DownloadError(
            f"{info.name}: no download URLs configured "
            "(experimental model; adapter work pending)"
        )

    base = Path(dest_dir) if dest_dir else model_cache_dir() / info.name / info.version
    base.mkdir(parents=True, exist_ok=True)

    if info.files:
        return _download_files(info, base, progress_cb)
    if info.bundled:
        return _resolve_bundled(info, base)

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

    part = target.with_suffix(target.suffix + ".part")
    expected = (info.sha256 or verified.get(fname) or "").lower() or None

    # Mirrors: every manifest URL is a fallback source; the primary URL drives
    # the cache layout so mirror content lands in the same place.
    candidates: list[str] = []
    for u in info.urls:
        d = resolve_gdrive(u) or u
        if d not in candidates:
            candidates.append(d)

    digest: str | None = None
    last_err: Exception | None = None
    for direct_url in candidates:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                digest = _fetch_one(info.name, direct_url, part, progress_cb, expected)
                break
            except DownloadError as exc:
                last_err = exc
                log.warning(
                    "Download attempt %d/%d for %s via %s failed: %s",
                    attempt, _MAX_ATTEMPTS, info.name, direct_url[:120], str(exc)[:300],
                )
                if not _is_transient(exc) or attempt == _MAX_ATTEMPTS:
                    break
                time.sleep(_RETRY_BACKOFF_S * attempt)
        if digest is not None:
            break
    if digest is None:
        raise DownloadError(f"Download failed for {info.name} (all mirrors tried): {last_err}")

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
    """Extract the configured archive member(s) as the canonical weights.

    Returns `artifact` unchanged when no extraction is configured. Extracted
    files get their own entries in verified.json so re-runs skip the (cheap,
    but unnecessary) extraction. Multi-file bundles (``extract_members``)
    return the directory containing all extracted checkpoints.
    """
    if info.extract_members:
        return _extract_members(info, base, artifact, verified, verified_file)
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


def _download_files(info: ModelInfo, base: Path, progress_cb=None) -> Path:
    """Download every configured file into `base` (sha256-verified); returns `base`.

    Multi-file models (e.g. SeedVR2's DiT + VAE + text embeddings): each manifest
    entry is {url, sha256, name}. Existing files are verified against their pin;
    missing ones are downloaded with resume support and retried like the
    single-file path.
    """
    verified_file = base / "verified.json"
    verified: dict = {}
    if verified_file.exists():
        try:
            verified = json.loads(verified_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            verified = {}

    changed = False
    for entry in info.files:
        fname = entry["name"]
        url = entry["url"]
        expected = (entry.get("sha256") or "").lower() or None
        target = base / fname
        if target.exists():
            if expected is None or _sha256_file(target) == expected:
                log.info("Model file cached: %s", target)
                continue
            raise DownloadError(f"Cache hash mismatch for {target}; delete it and re-download.")
        part = target.with_suffix(fname + ".part")
        digest: str | None = None
        last_err: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                digest = _fetch_one(info.name, url, part, progress_cb, expected)
                break
            except DownloadError as exc:
                last_err = exc
                log.warning(
                    "Download attempt %d/%d for %s (%s) failed: %s",
                    attempt, _MAX_ATTEMPTS, info.name, fname, str(exc)[:300],
                )
                if not _is_transient(exc) or attempt == _MAX_ATTEMPTS:
                    break
                time.sleep(_RETRY_BACKOFF_S * attempt)
        if digest is None:
            raise DownloadError(f"Download failed for {info.name}/{fname}: {last_err}")
        part.replace(target)
        verified[fname] = digest
        changed = True
    if changed:
        verified_file.write_text(json.dumps(verified, indent=1), encoding="utf-8")
    log.info("Model files ready in %s", base)
    return base


def cached_weights_path(info: ModelInfo, dest_dir: str | Path | None = None) -> Path | None:
    """Local path of already-downloaded weights for `info`, or ``None`` if not cached.

    Pure filesystem check (no network) — the M6 UI uses it to mark models as
    ready in model lists and to decide what the first-run download offers.
    For multi-file bundles this is the directory holding all extracted files.
    """
    from ..config import model_cache_dir

    base = Path(dest_dir) if dest_dir else model_cache_dir() / info.name / info.version
    if info.bundled:
        target = base / Path(info.bundled).name
        expected = (info.sha256 or "").lower() or None
        return target if (target.exists() and (expected is None or _sha256_file(target) == expected)) else None
    if info.files:
        # Fast path for the UI: existence + download-time verification record.
        # (No re-hashing here — multi-GB weights make per-refresh digests painful;
        # download_model still verifies fully at job start.)
        verified_file = base / "verified.json"
        verified: dict = {}
        if verified_file.exists():
            try:
                verified = json.loads(verified_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                verified = {}
        for entry in info.files:
            target = base / entry["name"]
            if not target.exists() or entry["name"] not in verified:
                return None
        return base
    if not info.urls:
        return None  # experimental model without a configured URL
    url = info.urls[0]
    direct = resolve_gdrive(url) or url
    target = base / _filename_for(direct, info.name)
    if not target.exists():
        return None
    if info.extract_members:
        outs = [base / out_name for out_name in info.extract_members.values()]
        return base if all(p.exists() for p in outs) else None
    if info.extract_member:
        out_path = base / Path(info.extract_member).name
        return out_path if out_path.exists() else None
    return target


def _extract_members(
    info: ModelInfo,
    base: Path,
    artifact: Path,
    verified: dict,
    verified_file: Path,
) -> Path:
    """Extract every configured archive member (renamed per the mapping).

    Returns `base` — the directory holding all extracted checkpoints. Each
    output file is tracked in verified.json under its output name; a re-run
    skips extraction when every output exists and matches.
    """
    import zipfile

    pending: list[str] = []
    for member, out_name in info.extract_members.items():
        out_path = base / out_name
        expected = verified.get(out_name)
        if not (out_path.exists() and (expected is None or _sha256_file(out_path) == expected.lower())):
            pending.append(member)
    if not pending:
        log.info("Extracted weights cached: %s", base)
        return base

    try:
        with zipfile.ZipFile(artifact) as zf:
            for member in pending:
                out_name = info.extract_members[member]
                data = zf.read(member)
                tmp = (base / out_name).with_suffix(Path(out_name).suffix + ".part")
                tmp.write_bytes(data)
                digest = _sha256_file(tmp)
                tmp.replace(base / out_name)
                verified[out_name] = digest
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DownloadError(
            f"{info.name}: expected archive containing {pending}, got {exc}"
        ) from exc

    verified_file.write_text(json.dumps(verified, indent=1), encoding="utf-8")
    log.info("Extracted %d member(s) -> %s", len(pending), base)
    return base


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
