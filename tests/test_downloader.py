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


# -- M7 retry + mirror behavior (offline; urlopen is monkeypatched) ------------


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def headers(self):
        return {"Content-Length": str(len(self._data))}

    def read(self, n: int) -> bytes:
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk


def _fail_then_succeed(monkeypatch, failures: list[Exception], data: bytes):
    """urlopen raises the given exceptions in order, then returns `data`."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        if i < len(failures):
            raise failures[i]
        return _FakeResp(data)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_transient_failures_retried_until_success(tmp_path, monkeypatch):
    import urllib.error

    data = b"weights-bytes"
    calls = _fail_then_succeed(
        monkeypatch,
        [urllib.error.URLError("connection reset"), TimeoutError("timed out")],
        data,
    )
    path = download_model(_info(), dest_dir=tmp_path / "cache")
    assert path.read_bytes() == data
    assert calls["n"] == 3  # two transient failures + one success


def test_mirror_fallback_on_permanent_failure(tmp_path, monkeypatch):
    import urllib.error

    data = b"mirror-bytes"
    info = ModelInfo(
        name="mirrored", task="sr", version="1",
        urls=["https://primary.example/x.bin", "https://mirror.example/x.bin"],
    )
    mirror_calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", str(req))
        if "primary" in url:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        mirror_calls["n"] += 1
        return _FakeResp(data)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("aive.models.downloader.time.sleep", lambda s: sleeps.append(s))

    path = download_model(info, dest_dir=tmp_path / "cache")
    assert path.read_bytes() == data
    assert mirror_calls["n"] == 1  # the mirror was tried exactly once
    assert sleeps == []            # non-transient failure -> no backoff sleep


def test_all_mirrors_fail_raises(tmp_path, monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(str(getattr(req, "full_url", req)), 404, "not found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    info = ModelInfo(
        name="dead", task="sr", version="1",
        urls=["https://a.example/x.bin", "https://b.example/x.bin"],
    )
    with pytest.raises(DownloadError, match="all mirrors tried"):
        download_model(info, dest_dir=tmp_path / "cache")


# -- multi-file bundles (extract_members) -------------------------------------


def test_extract_members_bundle(tmp_path, monkeypatch):
    """Multi-file bundles extract every member (renamed per the mapping) into a dir.

    Mirrors the GMFSS union bundle: some archive members keep their name, others
    are renamed to canonical checkpoint names. download_model returns the holding
    directory; cached_weights_path reports ready once every output exists.
    """
    import zipfile

    payload = {
        "flownet.pkl": b"flow-bytes",
        "rife.pkl": b"ifnet-bytes",
        "metric_union.pkl": b"metric-bytes",
        "feat_union.pkl": b"feat-bytes",
        "fusionnet_union.pkl": b"fusion-bytes",
    }
    src = tmp_path / "bundle.zip"
    with zipfile.ZipFile(src, "w") as zf:
        for name, data in payload.items():
            zf.writestr(name, data)

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(src.read_bytes())
    )

    info = ModelInfo(
        name="gmfss", task="fi", version="fortuna-union",
        urls=["https://example.com/gmfss-fortuna-union.zip"],
        extract_members={
            "flownet.pkl": "flownet.pkl",
            "rife.pkl": "rife.pkl",
            "metric_union.pkl": "metric.pkl",
            "feat_union.pkl": "feat.pkl",
            "fusionnet_union.pkl": "fusionnet.pkl",
        },
    )
    dest = tmp_path / "cache"
    result = download_model(info, dest_dir=dest)

    # Multi-file bundle -> the directory holding all checkpoints is returned.
    assert result == dest
    for out_name in ("flownet.pkl", "rife.pkl", "metric.pkl", "feat.pkl", "fusionnet.pkl"):
        p = dest / out_name
        assert p.exists(), f"missing extracted member {out_name}"
    # Renamed members carry the right bytes.
    assert (dest / "metric.pkl").read_bytes() == b"metric-bytes"
    assert (dest / "feat.pkl").read_bytes() == b"feat-bytes"
    assert (dest / "fusionnet.pkl").read_bytes() == b"fusion-bytes"

    # cached_weights_path reports ready once every output exists.
    from aive.models.downloader import cached_weights_path

    assert cached_weights_path(info, dest_dir=dest) == dest

    # Re-run is served from cache with no network and skips re-extraction.
    def _no_network(*a, **k):
        raise AssertionError("network accessed during cached run")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    assert download_model(info, dest_dir=dest) == dest
