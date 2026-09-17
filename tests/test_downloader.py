"""Model downloader: real (small) download, caching, hash verification."""
from __future__ import annotations

import urllib.request

import pytest

from aive.models.downloader import DownloadError, download_model, resolve_gdrive
from aive.models.registry import ModelInfo

SMALL_URL = "https://raw.githubusercontent.com/hzwer/Practical-RIFE/main/README.md"


def _info(name="testfile", url=SMALL_URL, sha256=None) -> ModelInfo:
    return ModelInfo(name=name, task="sr", version="1", urls=[url], sha256=sha256)


def test_gdrive_url_resolution():
    assert resolve_gdrive("https://drive.google.com/file/d/ABC123_-xyz/view?usp=share_link") == (
        "https://drive.usercontent.google.com/download?id=ABC123_-xyz&export=download&confirm=t"
    )
    assert resolve_gdrive("https://example.com/x.pth") is None


def test_download_and_cache(tmp_path, monkeypatch):
    dest = tmp_path / "cache"
    path = download_model(_info(), dest_dir=dest)
    assert path.exists() and path.stat().st_size > 100

    # Second call must be served from cache WITHOUT any network access.
    def _no_network(*a, **k):
        raise AssertionError("network accessed during cached run")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    path2 = download_model(_info(), dest_dir=dest)
    assert path2 == path


def test_no_urls_rejected_without_network_or_dir(tmp_path):
    dest = tmp_path / "cache"
    info = ModelInfo(name="nope", task="fi", version="1", urls=[])
    with pytest.raises(DownloadError, match="no download URLs"):
        download_model(info, dest_dir=dest)
    assert not dest.exists()  # guard fires before any filesystem work


def test_hash_mismatch_rejected(tmp_path):
    dest = tmp_path / "cache"
    bad = _info(sha256="0" * 64)
    with pytest.raises(DownloadError, match="SHA-256 mismatch"):
        download_model(bad, dest_dir=dest)
    # No partial file left behind.
    leftovers = [p for p in dest.rglob("*") if p.name.endswith(".part")]
    assert not leftovers
