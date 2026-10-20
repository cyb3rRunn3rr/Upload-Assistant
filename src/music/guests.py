"""Guest performers that the file tags do not name.

Streamrip writes only the main artist. Qobuz and Deezer still list a featured
performer on the track. Those names become Redacted guest artists. Composers,
producers and the album's main artist stay out.
"""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections.abc import Awaitable, Callable
from html import unescape
from typing import Any
from urllib.parse import quote

import httpx

from src.console import logger
from src.music.models import MetadataSource, MusicRelease
from src.music.sources import _music_cache_path, _read_music_cache, _write_music_cache

_FEATURED_ROLE = re.compile(r"featured|guest", re.I)
_FEAT_SPLIT = re.compile(r"(?:^|[\s(\[])(?:feat\.?|ft\.?|featuring)\s+", re.I)
_QOBUZ_ID = re.compile(r"qobuz\.com/.*/album/(?:[^/?#]+/)?([a-z0-9]+)(?:[/?#]|$)", re.I)
_ALLOWED_HOSTS = {"www.qobuz.com", "qobuz.com", "api.deezer.com"}
_TRACK_FETCHES = 4


def fold_name(value: object) -> str:
    """Compare artist and title text without accents, case or punctuation."""
    text = unicodedata.normalize("NFD", str(value or ""))
    plain = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", "", plain.casefold())


def format_guest_artists(value: object) -> str:
    """Render stored guests as ``Projota (15)``."""
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            numbers = [str(number) for number in item.get("tracks") or [] if str(number).strip()]
        else:
            name = str(item).strip()
            numbers = []
        if not name:
            continue
        parts.append(f"{name} ({', '.join(numbers)})" if numbers else name)
    return ", ".join(parts)


def featured_names(credit_line: str) -> list[str]:
    """Names whose role on one credit line is featured or guest."""
    names: list[str] = []
    for piece in credit_line.split(" - "):
        if "," not in piece:
            continue
        name, roles = piece.split(",", 1)
        name = name.strip()
        if name and _FEATURED_ROLE.search(roles) and name not in names:
            names.append(name)
    return names


def parse_qobuz_album(html: str) -> list[dict[str, Any]]:
    """Track title and featured performers from a public Qobuz album page."""
    tracks: list[dict[str, Any]] = []
    for match in re.finditer(r'data-track-v2="([^"]+)"', html):
        try:
            meta = json.loads(unescape(match.group(1)))
        except json.JSONDecodeError:
            continue
        if not isinstance(meta, dict):
            continue
        title = str(meta.get("item_name") or "").strip()
        block = html[match.end() : match.end() + 6000]
        number_match = re.search(r'track__item--number">\s*<span>(\d+)</span>', block)
        info_match = re.search(r'<p class="track__info">([^<]*)</p>', block)
        featured = featured_names(unescape(info_match.group(1))) if info_match else []
        if title:
            tracks.append({"title": title, "number": int(number_match.group(1)) if number_match else None, "featured": featured})
    return tracks


def parse_qobuz_search(html: str) -> list[tuple[str, str]]:
    """Album name and URL pairs from a Qobuz search page."""
    found: list[tuple[str, str]] = []
    for match in re.finditer(r'<script type="application/ld\+json">\s*(.*?)\s*</script>', html, re.S):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        entity = data.get("mainEntity") if isinstance(data, dict) else None
        items = entity.get("itemListElement") if isinstance(entity, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            album = item.get("item") if isinstance(item.get("item"), dict) else {}
            name = str(album.get("name") or "").strip()
            url = str(album.get("url") or item.get("url") or "").strip()
            if name and "/album/" in url:
                found.append((name, url))
    return found


def choose_qobuz_url(results: list[tuple[str, str]], artist: str, album: str) -> str:
    """One search hit whose title and artist slug match. Several hits stay unused."""
    artist_key = fold_name(artist)
    chosen: list[str] = []
    for name, url in results:
        if fold_name(name) != fold_name(album):
            continue
        path = fold_name(url)
        if artist_key and artist_key not in path:
            continue
        if url not in chosen:
            chosen.append(url)
    return chosen[0] if len(chosen) == 1 else ""


def guests_from_tags(release: MusicRelease) -> list[dict[str, Any]]:
    """Featured names already written in a track artist or title tag."""
    mains = {fold_name(name) for name in _main_names(release)}
    found: list[dict[str, Any]] = []
    for track in release.tracks:
        values = _tag_values(track.tags, "artist", "performer")
        if not values and track.artist:
            values = [track.artist]
        names: list[str] = []
        for value in [*values, track.title or ""]:
            names.extend(_featured_from_text(value))
        if len(values) > 1:
            for value in values:
                cleaned = value.strip()
                if cleaned and "," not in cleaned and fold_name(cleaned) not in mains:
                    names.append(cleaned)
        elif len(values) == 1 and "," not in values[0] and fold_name(values[0]) not in mains and not _FEAT_SPLIT.search(values[0]):
            names.append(values[0].strip())
        _add_guests(found, names, track.track_number, mains)
    return found


def credits_for_release(release: MusicRelease, remote_tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach featured names only when most local titles match the remote album."""
    local = [(fold_name(track.title), track.track_number) for track in release.tracks if fold_name(track.title)]
    remote: dict[str, list[str]] = {}
    for row in remote_tracks:
        if not isinstance(row, dict):
            continue
        key = fold_name(row.get("title"))
        featured = [str(name).strip() for name in row.get("featured") or [] if str(name).strip()]
        if key and key not in remote:
            remote[key] = featured
    if not local or len([key for key, _number in local if key in remote]) * 2 < len(local):
        return []
    mains = {fold_name(name) for name in _main_names(release)}
    found: list[dict[str, Any]] = []
    for key, number in local:
        _add_guests(found, remote.get(key, []), number, mains)
    return found


def merge_guests(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union guest names and keep every track number, in first-seen order."""
    found: list[dict[str, Any]] = []
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            numbers = [int(number) for number in item.get("tracks") or [] if isinstance(number, int) or (isinstance(number, str) and number.isdigit())]
            if not numbers:
                _add_guests(found, [name], None, set())
                continue
            for number in numbers:
                _add_guests(found, [name], number, set())
    return found


def qobuz_album_id(url: str) -> str:
    match = _QOBUZ_ID.search(str(url or ""))
    return match.group(1) if match else ""


def store_tag_guests(release: MusicRelease) -> None:
    """Keep featured names that are already in the files."""
    found = guests_from_tags(release)
    if found:
        release.set_field("guest_artists", found, MetadataSource.FILE_TAG, 0.93)


class GuestCreditEnricher:
    """Read featured performers from Qobuz and Deezer when enrichment is on."""

    def __init__(self, base_dir: str = "") -> None:
        self.base_dir = base_dir
        self._client: httpx.AsyncClient | None = None

    async def enrich(self, release: MusicRelease) -> None:
        artist = str(release.get("artist") or "").strip()
        album = str(release.get("album") or "").strip()
        if not artist or not album or not release.tracks:
            return
        local = release.get("guest_artists")
        local_guests = local if isinstance(local, list) else []
        try:
            remote, origins = await self._remote_guests(release, artist, album)
        except httpx.HTTPError:
            logger.warning("MUSIC: guest lookup failed. The upload continues without store guests.")
            return
        merged = merge_guests(local_guests, remote)
        if not merged or merged == local_guests:
            return
        release.set_field("guest_artists", merged, MetadataSource.EXTERNAL, 0.86, force=True)
        origin = " and ".join(origins) if origins else "the store"
        logger.info(f"[cyan]MUSIC: guest artists from {origin}: {format_guest_artists(merged)}.[/cyan]")

    async def _remote_guests(self, release: MusicRelease, artist: str, album: str) -> tuple[list[dict[str, Any]], list[str]]:
        key = "\x1f".join([fold_name(artist), fold_name(album), str(len(release.tracks)), *sorted(fold_name(track.title) for track in release.tracks)])
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0), follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
            self._client = client
            try:
                qobuz = await self._cached_tracks("qobuz", key, lambda: self._qobuz_tracks(release, artist, album))
                deezer = await self._cached_tracks("deezer", key, lambda: self._deezer_tracks(release, artist, album))
            finally:
                self._client = None
        guests: list[dict[str, Any]] = []
        origins: list[str] = []
        qobuz_guests = credits_for_release(release, qobuz)
        if qobuz_guests:
            guests = merge_guests(guests, qobuz_guests)
            origins.append("Qobuz")
        deezer_guests = credits_for_release(release, deezer)
        if deezer_guests:
            guests = merge_guests(guests, deezer_guests)
            origins.append("Deezer")
        return guests, origins

    async def _cached_tracks(self, provider: str, key: str, loader: Callable[[], Awaitable[list[dict[str, Any]] | None]]) -> list[dict[str, Any]]:
        path = _music_cache_path(self.base_dir, "guests", provider, key)
        cached = await _read_music_cache(path)
        if isinstance(cached, dict) and isinstance(cached.get("tracks"), list):
            return [item for item in cached["tracks"] if isinstance(item, dict)]
        try:
            tracks = await loader()
        except httpx.HTTPError:
            logger.warning(f"MUSIC: {provider} guest lookup failed.")
            return []
        if tracks is None:
            return []
        await _write_music_cache(path, {"tracks": tracks})
        return tracks

    async def _qobuz_tracks(self, release: MusicRelease, artist: str, album: str) -> list[dict[str, Any]] | None:
        album_id = _known_qobuz_id(release)
        if not album_id:
            query = quote(f"{artist} {album}")
            search = await self._text(f"https://www.qobuz.com/us-en/search/albums/{query}")
            url = choose_qobuz_url(parse_qobuz_search(search), artist, album)
            album_id = qobuz_album_id(url)
        if not album_id:
            return []
        html = await self._text(f"https://www.qobuz.com/us-en/album/x/{album_id}")
        tracks = [{"title": row["title"], "featured": row["featured"]} for row in parse_qobuz_album(html)]
        return tracks or None

    async def _deezer_tracks(self, release: MusicRelease, artist: str, album: str) -> list[dict[str, Any]] | None:
        payload = await self._json("https://api.deezer.com/search/album", {"q": f'artist:"{artist}" album:"{album}"', "limit": 25})
        albums = payload.get("data") if isinstance(payload, dict) else None
        chosen = _choose_deezer_album(albums, artist, album, len(release.tracks))
        if not chosen:
            return []
        listing = await self._json(f"https://api.deezer.com/album/{chosen}/tracks", {"limit": 100})
        rows = listing.get("data") if isinstance(listing, dict) else None
        if not isinstance(rows, list):
            return []
        semaphore = asyncio.Semaphore(_TRACK_FETCHES)
        failed = False

        async def one(row: dict[str, Any]) -> dict[str, Any] | None:
            track_id = str(row.get("id") or "")
            title = str(row.get("title") or row.get("title_short") or "").strip()
            if not track_id or not title:
                return None
            async with semaphore:
                detail = await self._json(f"https://api.deezer.com/track/{track_id}")
            return {"title": title, "featured": _deezer_featured(detail)}

        results = await asyncio.gather(*(one(row) for row in rows if isinstance(row, dict)), return_exceptions=True)
        tracks: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                failed = True
                continue
            if result:
                tracks.append(result)
        if failed:
            return None
        return tracks

    async def _text(self, url: str) -> str:
        response = await self._request(url)
        return response.text

    async def _json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = await self._request(url, params)
        try:
            return response.json()
        except ValueError as exc:
            raise httpx.HTTPError("guest lookup returned invalid JSON") from exc

    async def _request(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        if self._client is None:
            raise httpx.HTTPError("guest lookup has no client")
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        if response.url.host.casefold() not in _ALLOWED_HOSTS:
            raise httpx.HTTPError("guest lookup left the expected host")
        return response


def _known_qobuz_id(release: MusicRelease) -> str:
    candidates = [release.get("store_url"), release.external_ids.get("store_url", "")]
    for track in release.tracks:
        for key, values in track.tags.items():
            folded = key.casefold().replace(" ", "")
            if folded in {"url", "www", "website", "comment", "woas"} or folded.endswith("url"):
                candidates.extend(values)
    for value in candidates:
        album_id = qobuz_album_id(str(value or ""))
        if album_id:
            return album_id
    return ""


def _choose_deezer_album(albums: Any, artist: str, album: str, track_count: int) -> str:
    if not isinstance(albums, list):
        return ""
    chosen: list[str] = []
    for item in albums:
        if not isinstance(item, dict):
            continue
        person = item.get("artist") if isinstance(item.get("artist"), dict) else {}
        if fold_name(item.get("title")) != fold_name(album) or fold_name(person.get("name")) != fold_name(artist):
            continue
        try:
            count = int(item.get("nb_tracks") or 0)
        except TypeError, ValueError:
            count = 0
        if track_count and count != track_count:
            continue
        identifier = str(item.get("id") or "")
        if identifier and identifier not in chosen:
            chosen.append(identifier)
    return chosen[0] if len(chosen) == 1 else ""


def _deezer_featured(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    names: list[str] = []
    contributors = payload.get("contributors")
    if not isinstance(contributors, list):
        return names
    for person in contributors:
        if not isinstance(person, dict) or not _FEATURED_ROLE.search(str(person.get("role") or "")):
            continue
        name = str(person.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _main_names(release: MusicRelease) -> list[str]:
    values = release.get("artists")
    names = [str(value).strip() for value in values if str(value).strip()] if isinstance(values, list) else []
    if names:
        return names
    artist = str(release.get("artist") or "").strip()
    return [artist] if artist else []


def _tag_values(tags: dict[str, list[str]], *keys: str) -> list[str]:
    wanted = {key.casefold().replace(" ", "") for key in keys}
    values: list[str] = []
    for key, items in tags.items():
        if key.casefold().replace(" ", "") not in wanted:
            continue
        for item in items:
            cleaned = str(item).strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return values


def _featured_from_text(value: str) -> list[str]:
    parts = _FEAT_SPLIT.split(value)
    if len(parts) < 2:
        return []
    names: list[str] = []
    for part in parts[1:]:
        for name in re.split(r"\s*(?:,|&|/)\s*", part):
            cleaned = name.strip(" ()[]")
            if cleaned and cleaned not in names:
                names.append(cleaned)
    return names


def _add_guests(found: list[dict[str, Any]], names: list[str], number: int | None, mains: set[str]) -> None:
    for name in names:
        cleaned = name.strip()
        key = fold_name(cleaned)
        if not cleaned or not key or key in mains:
            continue
        current = next((item for item in found if fold_name(item["name"]) == key), None)
        if current is None:
            current = {"name": cleaned, "tracks": []}
            found.append(current)
        if number and number not in current["tracks"]:
            current["tracks"].append(number)
            current["tracks"].sort()
