"""Parse Redacted image-host responses without calling the site."""

from __future__ import annotations

import re
from typing import Any

_HTTPS_URL = re.compile(r"https://[^\s\[\]\"'<>]+", re.IGNORECASE)
_URL_KEYS = ("url", "link", "src", "image", "full", "bbcode")


def image_url_from_response(payload: Any) -> str:
    """Return the hosted HTTPS URL from a Redacted ``upload_image`` body.

    The image host has answered with a bare URL, a ``url`` field, and BBCode.
    Any of those is accepted. The API key is never read from the body.
    """
    if not isinstance(payload, dict):
        raise ValueError("Redacted image host returned a malformed response.")
    if payload.get("status") != "success":
        error = str(payload.get("error") or "").strip()
        raise ValueError(f"Redacted image host rejected the upload: {error}" if error else "Redacted image host rejected the upload.")
    url = _find_https_url(payload.get("response"))
    if not url:
        raise ValueError("Redacted image host accepted the upload but returned no HTTPS URL.")
    return url


def _find_https_url(value: Any) -> str:
    if isinstance(value, str):
        match = _HTTPS_URL.search(value)
        return match.group(0).rstrip(".,);") if match else ""
    if isinstance(value, dict):
        for key in _URL_KEYS:
            if key in value:
                found = _find_https_url(value[key])
                if found:
                    return found
        for item in value.values():
            found = _find_https_url(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_https_url(item)
            if found:
                return found
    return ""
