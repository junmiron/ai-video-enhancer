"""Auto-update check (M7 B) — personal-use stub with friendly failure.

No update server exists yet, so the Settings page's "Check for updates" button
reports that plainly instead of failing. When one does exist (e.g. a raw GitHub
JSON file), point ``UPDATE_ENDPOINT`` at it and nothing else changes:

    {"version": "0.2.0", "url": "https://example.com/ai-video-enhancer-0.2.0.exe"}

The check is pure stdlib (urllib) so it works frozen or not, and every failure
mode returns a human-readable :class:`UpdateResult` rather than raising.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

# Empty = "not configured" (the default for personal builds). Set to a URL that
# serves the JSON shape above to enable real update checks.
UPDATE_ENDPOINT: str = ""
_TIMEOUT_S = 8.0


@dataclass
class UpdateResult:
    status: str  # up_to_date | update_available | not_configured | error
    message: str
    latest_version: str | None = None
    download_url: str | None = None


def _parse_version(v: str) -> tuple[int, ...]:
    """'0.1.0' / 'v1.2.3-beta' -> (0, 1, 0); non-numeric chunks count as 0."""
    parts = []
    for chunk in str(v).strip().lstrip("v").split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def check_for_updates(
    current_version: str, endpoint: str | None = None, timeout_s: float = _TIMEOUT_S
) -> UpdateResult:
    """Query the version endpoint and compare against ``current_version``.

    Never raises — network/parse problems become status='error' with a message.
    """
    url = (endpoint or UPDATE_ENDPOINT).strip()
    if not url:
        return UpdateResult(
            "not_configured",
            "No update server is configured for this build (personal-use stub).",
        )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"ai-video-enhancer/{current_version}"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # URLError, timeout, JSON decode, HTTP error...
        return UpdateResult(
            "error",
            f"Could not reach the update server ({type(exc).__name__}). "
            "Check your connection and try again.",
        )
    latest = str(data.get("version", "")).strip() if isinstance(data, dict) else ""
    if not latest:
        return UpdateResult("error", "Update server response is missing a 'version' field.")
    download_url = data.get("url") or None
    if _parse_version(latest) > _parse_version(current_version):
        return UpdateResult(
            "update_available",
            f"Version {latest} is available (you have {current_version}).",
            latest,
            download_url,
        )
    return UpdateResult(
        "up_to_date",
        f"You're on the latest version ({current_version}).",
        latest,
        download_url,
    )
