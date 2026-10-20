"""Store and database links for a Redacted group description.

The line matches smoked-salmon's More info row. ptpimg.me no longer serves the
old icons, so the pictures are the ones salmon uses now.
"""

from __future__ import annotations

import re

from src.music.models import MusicRelease

SOURCE_ICONS = {
    "Apple Music": "https://i.ibb.co/KpqNkPn2/applemusic.png",
    "Bandcamp": "https://i.ibb.co/jd8Ysrm/bandcamp.png",
    "Beatport": "https://i.ibb.co/fdbp8V7S/beatport.png",
    "Deezer": "https://i.ibb.co/zHbd0GMm/deezer.png",
    "Discogs": "https://i.ibb.co/TMPGqrKj/discogs.png",
    "MusicBrainz": "https://i.ibb.co/Y42MftV1/musicbrainz.png",
    "Qobuz": "https://i.ibb.co/39M2cBx9/qobuz.png",
    "Tidal": "https://i.ibb.co/bjnnJcFm/tidal.png",
}

# Bandcamp stays last. Its path shape is loose, so a store URL must not land there first.
_SOURCES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("MusicBrainz", re.compile(r"^https?://(?:www\.)?musicbrainz\.org/release/([a-f0-9-]{36})/?$")),
    ("Apple Music", re.compile(r"^https?://(?:itunes|music)\.apple\.com/(?:\w{2,4}/)?album/(?:[^/?#]*/)?([^?#]+)")),
    ("Deezer", re.compile(r"^https?://(?:[a-z0-9-]+\.)*deezer\.com/(?:[a-z]+/)?(?:album|playlist|track)/([0-9]+)(?=$|[/?#])")),
    ("Discogs", re.compile(r"^https?://(?:www\.)?discogs\.com/(?:[^/]+/)?(?:release|master)/(\d+)")),
    ("Beatport", re.compile(r"^https?://(?:(?:www|classic)\.)?beatport\.com/release/.+?/(\d+)/?$")),
    ("Qobuz", re.compile(r"^https?://(?:www\.|play\.|open\.)?qobuz\.com/(?:(?:[^/]+/)?album/(?:[^/]+/)?|album/(?:-/)?)([a-zA-Z0-9]+)/?$")),
    ("Tidal", re.compile(r"^https?://(?:www\.)?(?:tidal|wimpmusic)\.com/.+?/(?:album|track|playlist)/([0-9a-z-]+)")),
    ("Bandcamp", re.compile(r"^https?://(?:[^/]+\.)?bandcamp\.com/(?:album|track)/([^/?#]+)/?$")),
)
_URL_TAGS = {"url", "www", "website", "comment", "woas"}
_HTTP_URL = re.compile(r"https?://[^\s\[\]<>\"]+")
_MB_ID = re.compile(r"^[a-f0-9-]{36}$", re.I)


def source_links(release: MusicRelease) -> list[tuple[str, str]]:
    """Known store and database pages, store first and MusicBrainz last."""
    found: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        for match in _HTTP_URL.findall(str(value or "")):
            url = match.rstrip(".,)")
            if url not in seen:
                seen.add(url)
                found.append(url)

    add(release.get("store_url"))
    add(release.external_ids.get("store_url", ""))
    for track in release.tracks:
        for key, values in track.tags.items():
            folded = key.casefold().replace(" ", "")
            if folded in {"musicbrainz_albumid", "musicbrainz_releaseid"}:
                for item in values:
                    if _MB_ID.fullmatch(item.strip()):
                        add(f"https://musicbrainz.org/release/{item.strip()}")
            elif folded in _URL_TAGS or folded.endswith("url"):
                add(" ".join(values))
    for key in ("discogs_release_url", "discogs_master_url"):
        add(release.external_ids.get(key, ""))
    musicbrainz = str(release.external_ids.get("musicbrainz_release") or release.get("musicbrainz_release") or "").strip()
    if _MB_ID.fullmatch(musicbrainz):
        add(f"https://musicbrainz.org/release/{musicbrainz}")

    chosen: dict[str, tuple[str, str]] = {}
    order: list[str] = []
    for url in found:
        name, identity = _match_source(url)
        if not name or identity in chosen:
            continue
        chosen[identity] = (name, url)
        order.append(identity)
    links = [chosen[identity] for identity in order]
    links.sort(key=lambda item: item[0] == "MusicBrainz")
    return links


def _match_source(url: str) -> tuple[str, str]:
    for name, pattern in _SOURCES:
        match = pattern.match(url)
        if match:
            return name, f"{name}:{match.group(1).casefold()}"
    return "", ""


def more_info_bbcode(links: list[tuple[str, str]]) -> str:
    """One More info row. An empty list stays empty."""
    if not links:
        return ""
    parts = [f"[pad=0|3][url={url}][img]{SOURCE_ICONS[name]}[/img] {name}[/url][/pad]" for name, url in links]
    return "[b]More info:[/b] " + " ".join(parts)
