"""Redacted music adapter.

The upload form matches Gazelle, with Redacted's own edition fields, formats
and image host. Spectrograms are optional SoX plots uploaded to that host and
linked from the release description.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any, ClassVar

import httpx

from src.console import logger
from src.meta import Meta
from src.music.models import AudioTrack, MusicRelease
from src.music.validation import OrpheusMusicValidator, ValidationIssue, ValidationLevel
from src.trackers.GAZELLE.orpheus import Orpheus
from src.trackers.GAZELLE.red_images import image_url_from_response
from src.trackers.GAZELLE.red_spectrograms import output_name, render_spectrogram, spectrogram_bbcode, zoom_window

IMAGE_UPLOAD_INTERVAL_SECONDS = 2.0
REDACTED_FORMATS = {"FLAC", "MP3", "AAC", "AC3", "DTS", "DSD"}
DSD_BITRATES = (
    (2_822_400, "DSD64"),
    (5_644_800, "DSD128"),
    (11_289_600, "DSD256"),
    (22_579_200, "DSD512"),
)
MEDIA_NAMES = {
    "cd": "CD",
    "dvd": "DVD",
    "vinyl": "Vinyl",
    "soundboard": "Soundboard",
    "sacd": "SACD",
    "dat": "DAT",
    "cassette": "Cassette",
    "web": "WEB",
    "blu-ray": "Blu-Ray",
    "bluray": "Blu-Ray",
    "bd": "Blu-Ray",
    "laserdisc": "LaserDisc",
}


class RedactedMusicValidator(OrpheusMusicValidator):
    """Orpheus checks adjusted to the Redacted upload form."""

    def validate(self, release: MusicRelease) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for issue in super().validate(release):
            if issue.code == "unsupported_format" and "DSD" in issue.message:
                continue
            message = issue.message.replace("Orpheus", "Redacted")
            issues.append(issue if message == issue.message else ValidationIssue(issue.level, issue.code, message))
        for track in release.tracks:
            if track.format not in REDACTED_FORMATS:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "unsupported_format",
                        f"{track.relative_path}: {track.format} is not an allowed Redacted music format.",
                    )
                )
            if track.format == "DSD" and Path(track.path).suffix.lower() not in {".dsf", ".dff"}:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "invalid_container",
                        f"{track.relative_path}: DSD must use the .dsf or .dff container.",
                    )
                )
        media = str(release.get("media", ""))
        if media in {"Blu-Ray", "LaserDisc"} and release.is_lossless and not release.auxiliary.logs:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "missing_log",
                    "A lossless physical-media rip has no rip log; it may be trumpable or require review.",
                )
            )
        return issues


class Redacted(Orpheus):
    """Redacted is a private music tracker."""

    tracker = "REDACTED"
    display_name = "Redacted"
    source_flag = "RED"
    base_url = "https://redacted.sh"
    validator_cls = RedactedMusicValidator
    request_category = "0"
    release_types: ClassVar[dict[str, int]] = {
        "Album": 1,
        "Soundtrack": 3,
        "EP": 5,
        "Anthology": 6,
        "Compilation": 7,
        "Single": 9,
        "Live album": 11,
        "Remix": 13,
        "Bootleg": 14,
        "Interview": 15,
        "Mixtape": 16,
        "Demo": 17,
        "Concert Recording": 18,
        "Concert recording": 18,
        "DJ Mix": 19,
        "Unknown": 21,
    }
    blocked_music_artists: ClassVar[dict[str, str]] = {
        "pauldvr": "Paul_DVR",
        "vap0rwave": "Vap0rwave",
        "firmensprecher": "Firmensprecher",
        "phyllomedusa": "Phyllomedusa",
        "stretches": "stretches",
        "drizzywap": "Drizzy Wap",
        "cracksnow": "CracksNow",
        "tsx": "TSX",
        "hiphoplosslesscom": "Hiphoplossless.com",
        "funkerman": "Funkerman",
        "beatstreetsny": "BeatstreetsNY",
        "midfloridawholesale2015": "MidFloridaWholesale2015",
        "mixunit": "MixUnit",
    }
    blocked_music_releases: ClassVar[tuple[tuple[str, str], ...]] = ()
    blocked_music_labels: ClassVar[tuple[str, ...]] = (
        "CracksNow",
        "TSX",
        "Hiphoplossless.com",
        "Funkerman",
        "BeatstreetsNY",
        "MidFloridaWholesale2015",
        "MixUnit",
    )
    cd_only_blocked_labels: ClassVar[tuple[str, ...]] = ("BELLUM",)
    comment_hosts = ("redacted.sh", "redacted.ch")
    tracker_urls = ("redacted.sh", "redacted.ch")

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._spectrogram_bbcode = ""
        self._next_image_at = 0.0

    def _headers(self, meta: Meta) -> dict[str, str]:
        headers = super()._headers(meta)
        headers["Authorization"] = self.api_key
        return headers

    @classmethod
    def _blocked_labels(cls, release: MusicRelease) -> list[str]:
        labels = super()._blocked_labels(release)
        if str(release.get("media", "")).upper() != "CD":
            return labels
        present = {cls._normalise_artist_name(value) for value in (release.get("release_label", ""), release.get("label", "")) if str(value or "").strip()}
        for label in cls.cd_only_blocked_labels:
            if cls._normalise_artist_name(label) in present and label not in labels:
                labels.append(label)
        return labels

    async def upload(self, meta: Meta) -> bool:
        self._spectrogram_bbcode = ""
        release = self._release(meta)
        if not await self.get_additional_checks(meta):
            return False
        issues = self.validator_cls().validate(release)
        errors = [issue.message for issue in issues if issue.level == ValidationLevel.ERROR]
        if errors:
            meta.tracker_status.setdefault(self.tracker, {})["status_message"] = "Validation failed: " + " | ".join(errors)
            return False
        if not meta.debug and self.api_key:
            await self._rehost_cover(meta)
            if self._wants_spectrograms():
                self._spectrogram_bbcode = await self._upload_spectrograms(meta, release)
        return await super().upload(meta)

    def _wants_spectrograms(self) -> bool:
        settings = self.config.get("TRACKERS", {}).get(self.tracker, {})
        return bool(settings.get("add_spectrograms", True))

    async def _rehost_cover(self, meta: Meta) -> None:
        """Put the cover on Redacted's image host when it is not already there."""
        current = str(meta.artwork_url or "").strip()
        if self._is_site_image(current):
            return
        artwork_path = Path(str(meta.artwork_path or ""))
        try:
            if artwork_path.is_file():
                hosted = await self._upload_image_file(meta, artwork_path)
            elif current.startswith("https://"):
                hosted = await self._rehost_image_url(meta, current)
            else:
                return
        except (OSError, ValueError, httpx.HTTPError) as error:
            logger.warning(f"{self.tracker}: [yellow]cover rehost failed, keeping the current artwork URL: {error}[/yellow]")
            return
        meta.artwork_url = hosted

    async def _upload_spectrograms(self, meta: Meta, release: MusicRelease) -> str:
        sox = shutil.which("sox") or ""
        if not sox:
            logger.warning(f"{self.tracker}: [yellow]sox was not found, so spectrograms were skipped.[/yellow]")
            return ""
        if not meta.base_dir or not meta.uuid:
            logger.warning(f"{self.tracker}: [yellow]spectrogram output directory is unavailable.[/yellow]")
            return ""
        output_dir = Path(meta.base_dir) / "tmp" / str(meta.uuid) / "red-spectrograms"
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[tuple[str, str, str]] = []
        for index, track in enumerate(release.tracks, start=1):
            source = self._track_source(release, track)
            if source is None:
                logger.warning(f"{self.tracker}: [yellow]skipping spectrogram, audio file is missing: {track.relative_path}[/yellow]")
                continue
            title = track.title or source.stem
            full_url, zoom_url = await self._one_spectrogram_pair(meta, sox, source, output_dir, index, title, track.duration)
            if full_url or zoom_url:
                rows.append((title, full_url, zoom_url))
        return spectrogram_bbcode(rows)

    async def _one_spectrogram_pair(
        self,
        meta: Meta,
        sox: str,
        source: Path,
        output_dir: Path,
        index: int,
        title: str,
        duration: float | None,
    ) -> tuple[str, str]:
        full_path = output_dir / output_name(index, title, "full")
        zoom_path = output_dir / output_name(index, title, "zoom")
        window = zoom_window(duration)
        try:
            await asyncio.to_thread(render_spectrogram, sox, source, full_path)
        except RuntimeError as error:
            logger.warning(f"{self.tracker}: [yellow]{error}[/yellow]")
            return "", ""
        zoom_ready = False
        if window is not None:
            try:
                await asyncio.to_thread(render_spectrogram, sox, source, zoom_path, zoom=window)
                zoom_ready = True
            except RuntimeError as error:
                logger.warning(f"{self.tracker}: [yellow]zoom spectrogram skipped for {source.name}: {error}[/yellow]")
        full_url = await self._host_existing_image(meta, full_path)
        zoom_url = await self._host_existing_image(meta, zoom_path) if zoom_ready else ""
        return full_url, zoom_url

    async def _host_existing_image(self, meta: Meta, path: Path) -> str:
        if not path.is_file():
            return ""
        try:
            return await self._upload_image_file(meta, path)
        except (OSError, ValueError, httpx.HTTPError) as error:
            logger.warning(f"{self.tracker}: [yellow]could not upload {path.name} to the Redacted image host: {error}[/yellow]")
            return ""

    @staticmethod
    def _track_source(release: MusicRelease, track: AudioTrack) -> Path | None:
        direct = Path(track.path)
        if direct.is_file():
            return direct
        rooted = Path(release.root) / track.relative_path
        return rooted if rooted.is_file() else None

    async def _upload_image_file(self, meta: Meta, path: Path) -> str:
        await self._pace_images()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers=self._headers(meta)) as client:
            with path.open("rb") as handle:
                response = await client.post(f"{self.base_url}/ajax.php?action=upload_image", files={"file": (path.name, handle, mime)})
            response.raise_for_status()
            return image_url_from_response(response.json())

    async def _rehost_image_url(self, meta: Meta, url: str) -> str:
        await self._pace_images()
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers=self._headers(meta)) as client:
            response = await client.post(f"{self.base_url}/ajax.php?action=upload_image", data={"url": url})
            response.raise_for_status()
            return image_url_from_response(response.json())

    async def _pace_images(self) -> None:
        delay = self._next_image_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_image_at = time.monotonic() + IMAGE_UPLOAD_INTERVAL_SECONDS

    @staticmethod
    def _is_site_image(url: str) -> bool:
        host = url.removeprefix("https://").split("/", 1)[0].casefold()
        return host in {"redacted.sh", "redacted.ch"} or host.endswith(".redacted.sh") or host.endswith(".redacted.ch")

    def build_upload_payload(self, meta: Meta, release: MusicRelease) -> dict[str, str | list[str] | list[int] | int]:
        media = self._media_name(str(release.get("media", "")))
        if not media:
            raise ValueError("Redacted media must be one of the upload form values.")
        if len(release.formats) != 1:
            raise ValueError("Redacted uploads must contain exactly one audio format.")
        format_name = next(iter(release.formats))
        bitrate, other_bitrate, vbr = self._encoding(release, format_name)
        edition_year = self._edition_year_for_upload(release)
        if not edition_year:
            raise ValueError("Redacted requires an edition year.")
        artists = self._artists(release)
        release_type = str(release.get("release_type", "Unknown"))
        releasetype = self.release_types.get(release_type, 21)
        label = self._edition_text(release.get("edition_label") or release.get("release_label") or release.get("label"))
        catalogue = self._edition_text(release.get("edition_catalogue_number") or release.get("release_catalogue_number") or release.get("catalogue_number"))
        edition_title = self._edition_text(release.get("edition"))
        payload: dict[str, str | list[str] | list[int] | int] = {
            "album_desc": self._album_description(release),
            "artists[]": artists,
            "bitrate": bitrate,
            "format": format_name,
            "image": self._cover_url(meta),
            "importance[]": [1] * len(artists),
            "media": media,
            "release_desc": self._release_description(release),
            "releasetype": releasetype,
            "remaster_year": edition_year,
            "submit": "true",
            "tags": ",".join(str(value).replace(" ", ".").lower() for value in release.get("genres", [])),
            "title": str(release.get("album")),
            "type": 0,
            "year": str(release.get("year", "")),
        }
        if edition_title:
            payload["remaster_title"] = edition_title
        if label:
            payload["remaster_record_label"] = label
        if catalogue:
            payload["remaster_catalogue_number"] = catalogue
        if other_bitrate:
            payload["other_bitrate"] = other_bitrate
        if vbr:
            payload["vbr"] = "on"
        if meta.scene:
            payload["scene"] = "on"
        if releasetype == 21:
            payload["unknown"] = "on"
        return {key: value for key, value in payload.items() if value not in ("", None, [])}

    @staticmethod
    def _edition_text(value: Any) -> str:
        text = str(value or "").strip()
        return text if len(text) >= 2 else ""

    @staticmethod
    def _media_name(value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if cleaned in {"CD", "DVD", "Vinyl", "Soundboard", "SACD", "DAT", "Cassette", "WEB", "Blu-Ray", "LaserDisc"}:
            return cleaned
        return MEDIA_NAMES.get(cleaned.casefold().replace(" ", ""), "")

    def _encoding(self, release: MusicRelease, format_name: str) -> tuple[str, str, bool]:
        if format_name == "DSD":
            rate = next((track.sample_rate or 0 for track in release.tracks if track.sample_rate), 0)
            chosen = min(DSD_BITRATES, key=lambda item: abs(item[0] - rate)) if rate else None
            bitrate = chosen[1] if chosen is not None and abs(chosen[0] - rate) / chosen[0] < 0.05 else ""
            return (bitrate or "Other", "" if bitrate else "DSD", False)
        return super()._encoding(release, format_name)

    def _release_description(self, release: MusicRelease) -> str:
        text = super()._release_description(release)
        if not self._spectrogram_bbcode:
            return text
        return f"{text}\n\n{self._spectrogram_bbcode}".strip() if text else self._spectrogram_bbcode

    @staticmethod
    def _cover_url(meta: Meta) -> str:
        value = Orpheus._cover_url(meta)
        return value if value.startswith("https://") else ""
