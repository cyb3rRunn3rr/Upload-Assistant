"""Redacted music adapter.

The upload form matches Gazelle, with Redacted's own edition fields, formats
and image host. Spectrograms are optional SoX plots uploaded to that host and
linked from the release description. Image uploads use the logged-in user
session and leave the host in batches of three. The three go out together,
then the host gets a short pause. The Web UI progress card counts each image
once, including while that pause is running.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar

import httpx

from src.console import logger
from src.media_extensions import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from src.meta import Meta
from src.music.guests import fold_name
from src.music.models import AudioTrack, MetadataSource, MusicRelease
from src.music.validation import OrpheusMusicValidator, ValidationIssue, ValidationLevel
from src.trackers.GAZELLE.orpheus import Orpheus
from src.trackers.GAZELLE.red_images import authkey_from_index, image_url_from_response
from src.trackers.GAZELLE.red_links import more_info_bbcode, source_links
from src.trackers.GAZELLE.red_spectrograms import output_name, render_spectrogram, spectrogram_bbcode, zoom_window
from src.webui_progress import complete_progress, publish_progress

IMAGE_BATCH_SIZE = 3
IMAGE_BATCH_PAUSE_SECONDS = 4.0
IMAGE_PROGRESS_ID = "redacted-images"
_RATE_LIMIT_MARKERS = ("rate limit", "ratelimit", "too many", "slow down")
_LOGIN_MARKERS = ("logged in", "log in", "invalid auth", "bad auth")
REDACTED_FORMATS = {"FLAC", "MP3", "AAC", "AC3", "DTS", "DSD"}
DSD_RATES = {
    2_822_400: "DSD64",
    3_072_000: "DSD64",
    5_644_800: "DSD128",
    6_144_000: "DSD128",
    11_289_600: "DSD256",
    12_288_000: "DSD256",
    22_579_200: "DSD512",
    24_576_000: "DSD512",
}
DXD_RATES = {352_800, 384_000}
MEDIUM_FIRST_YEAR = {"CD": 1982, "DVD": 1996, "DAT": 1987, "SACD": 1999, "Blu-Ray": 2006, "WEB": 1999, "LaserDisc": 1978}
BANNED_SIDECAR_SUFFIXES = VIDEO_EXTENSIONS | {
    ".exe", ".dll", ".bat", ".cmd", ".msi", ".apk", ".ipa", ".dmg", ".jar", ".com", ".scr",
    ".gba", ".gbc", ".gb", ".nes", ".sfc", ".smc", ".n64", ".z64", ".nds", ".3ds", ".cia", ".wad", ".wbfs", ".rvz", ".xci", ".nsp",
}
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
ARCHIVE_SUFFIXES = {".rar", ".zip", ".7z", ".iso", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".lzh", ".cab", ".ace", ".arj", ".sit", ".sitx"}
DISC_IMAGE_SUFFIXES = {".bin", ".img", ".nrg", ".mdf", ".mds", ".cdi", ".ccd"}
EXTRA_VIDEO_SUFFIXES = {".vob", ".ifo", ".bup", ".flv", ".f4v", ".3gp", ".ogv", ".rm", ".rmvb", ".asf", ".divx", ".mts", ".mxf", ".swf"}
FORBIDDEN_AUDIO_SUFFIXES = {
    ".wma", ".m4b", ".m4p", ".mka", ".mp2", ".mpc", ".tta", ".tak", ".shn", ".aif", ".caf", ".aifc", ".w64", ".amr", ".ra", ".au", ".ofr", ".spx",
}
SHORTCUT_SUFFIXES = {".url", ".lnk", ".webloc", ".htm", ".html"}
SPLIT_ARCHIVE_RE = re.compile(r"\.(?:r\d{2}|\d{3})$", re.I)
ANALOG_HYBRID_MEDIA = {"Vinyl", "Cassette", "Soundboard"}
LINEAGE_MEDIA = {"Vinyl", "Cassette", "SACD", "Blu-Ray", "LaserDisc"}
DSD_WEB_ONLY = {"DSD128", "DSD256", "DSD512"}
_MP3_PRESETS = (
    (re.compile(r"(?:^|\s)(?:--alt-preset|--preset)\s+(?:fast\s+)?extreme\b", re.I), "APX (VBR)"),
    (re.compile(r"(?:^|\s)(?:--alt-preset|--preset)\s+(?:fast\s+)?standard\b", re.I), "APS (VBR)"),
    (re.compile(r"(?:^|\s)-V\s+0(?:\.0)?\b"), "V0 (VBR)"),
    (re.compile(r"(?:^|\s)-V\s+1(?:\.0)?\b"), "V1 (VBR)"),
    (re.compile(r"(?:^|\s)-V\s+2(?:\.0)?\b"), "V2 (VBR)"),
)
PATH_LIMIT = 180
MIN_LOSSY_BPS = 192_000
_NON_MUSIC_GENRES = {"audiobook", "audiobooks", "comedy", "podcast", "podcasts", "spoken word", "spoken.word"}


class RedactedMusicValidator(OrpheusMusicValidator):
    """File checks that can be decided from the local album.

    Sound quality, lineage, pirate status and staff exceptions stay warnings.
    Guessing those would be as dangerous as skipping them.
    """

    def validate(self, release: MusicRelease) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        media = Redacted._media_name(str(release.get("media", ""))) or str(release.get("media", ""))
        for issue in super().validate(release):
            if issue.code == "unsupported_format" and "DSD" in issue.message:
                continue
            if issue.code == "missing_lineage":
                continue
            if issue.code == "16bit_high_rate":
                if media != "CD":
                    issues.append(
                        ValidationIssue(
                            ValidationLevel.WARNING,
                            issue.code,
                            f"{issue.message} Redacted allows this only when the files were sold at that rate.",
                        )
                    )
                continue
            if issue.code == "sample_rate" and self._is_dxd_flac(release, issue):
                if media == "WEB":
                    issues.append(
                        ValidationIssue(
                            ValidationLevel.WARNING,
                            "dxd_rate",
                            f"{self._issue_path(issue)}: 352.8 and 384 kHz FLAC are allowed only for official DXD from WEB.",
                        )
                    )
                else:
                    issues.append(
                        ValidationIssue(
                            ValidationLevel.ERROR,
                            "dxd_media",
                            f"{self._issue_path(issue)}: DXD is only allowed from an official WEB source.",
                        )
                    )
                continue
            if issue.code == "hybrid_technical":
                level, message = self._hybrid_issue(release, media)
                issues.append(ValidationIssue(level, issue.code, message))
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
            suffix = Path(track.path).suffix.lower()
            if track.format == "AAC" and suffix == ".aac":
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "invalid_container",
                        f"{track.relative_path}: AAC must use .m4a; remux audio-only .mp4 to .m4a.",
                    )
                )
            if track.format == "DSD" and suffix not in {".dsf", ".dff"}:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "invalid_container",
                        f"{track.relative_path}: DSD must use the .dsf or .dff container.",
                    )
                )
            if track.format == "DSD" and track.sample_rate not in DSD_RATES:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "dsd_rate",
                        f"{track.relative_path}: DSD rate is not DSD64, DSD128, DSD256 or DSD512.",
                    )
                )
            tier = DSD_RATES.get(track.sample_rate or 0, "")
            if track.format == "DSD" and tier and media not in {"SACD", "WEB"}:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "dsd_media",
                        f"{track.relative_path}: DSD64 must come from SACD or WEB, and higher DSD only from WEB.",
                    )
                )
            elif track.format == "DSD" and tier in DSD_WEB_ONLY and media == "SACD":
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "dsd_sacd",
                        f"{track.relative_path}: an SACD DSD rip must be DSD64.",
                    )
                )
            if track.format == "DSD" and suffix == ".dff" and track.tags:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "dff_tags",
                        f"{track.relative_path}: a DFF file must not contain tags.",
                    )
                )
            if track.format == "FLAC" and track.bit_depth and track.bit_depth not in {16, 24} and track.bit_depth <= 24:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "bit_depth",
                        f"{track.relative_path}: FLAC must be 16-bit or 24-bit.",
                    )
                )
            if media == "CD" and track.format == "FLAC" and (track.bit_depth != 16 or track.sample_rate != 44_100):
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "cd_redbook",
                        f"{track.relative_path}: a CD rip must stay 16-bit 44.1 kHz.",
                    )
                )
            self._check_lossy_floor(issues, track)
            self._check_lossy_signal(issues, track, media)
        self._check_archives(issues, release)
        self._check_banned_sidecars(issues, release)
        self._check_extra_audio(issues, release)
        self._check_shortcuts(issues, release)
        self._check_names(issues, release)
        self._check_required_tags(issues, release)
        self._check_non_music_genres(issues, release)
        self._check_edition_year(issues, release, media)
        self._check_release_type(issues, release)
        self._check_mqa_tags(issues, release)
        self._check_single_inference(issues, release)
        self._check_unsplit(issues, release)
        self._check_mp3_presets(issues, release)
        self._check_embedded_size(issues, release)
        if media == "Soundboard" and release.is_lossless and not release.auxiliary.lineage:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "missing_lineage",
                    "A lossless soundboard needs lineage in the torrent. Redacted deletes it if the lineage is missing.",
                )
            )
        if media in LINEAGE_MEDIA and release.is_lossless and not self._has_lineage_file(release):
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "missing_lineage",
                    f"A lossless {media} rip has no lineage, log or notes file. Redacted can trump it.",
                )
            )
        if media == "Blu-Ray" and release.is_lossless and not str(release.get("edition", "")).strip():
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "bluray_edition",
                    "Name the lossless Blu-ray track in the edition title, for example 24-bit PCM.",
                )
            )
        if media in {"Blu-Ray", "LaserDisc"} and release.is_lossless and not release.auxiliary.logs:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "missing_log",
                    "A lossless physical-media rip has no rip log; it may be trumpable or require review.",
                )
            )
        if str(release.get("release_type", "")).casefold() == "unknown":
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "unknown_release_type",
                    "Release type is Unknown. Confirm the album category. Redacted still needs a real edition year.",
                )
            )
        return issues

    @staticmethod
    def _check_lossy_floor(issues: list[ValidationIssue], track: AudioTrack) -> None:
        if track.format not in {"MP3", "AAC", "AC3", "DTS"} or not track.bitrate or track.bitrate >= MIN_LOSSY_BPS:
            return
        if track.format == "MP3" and RedactedMusicValidator._mp3_preset(track.encoder_settings):
            return
        if track.format == "AAC" or track.bitrate_mode == "VBR":
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "low_vbr" if track.bitrate_mode == "VBR" else "low_bitrate",
                    f"{track.relative_path}: average bitrate is under 192 kbps. Only specific VBR presets are allowed.",
                )
            )
            return
        issues.append(
            ValidationIssue(
                ValidationLevel.ERROR,
                "low_bitrate",
                f"{track.relative_path}: bitrate is under the 192 kbps minimum.",
            )
        )

    @staticmethod
    def _check_lossy_signal(issues: list[ValidationIssue], track: AudioTrack, media: str) -> None:
        if track.format not in {"MP3", "AAC", "AC3", "DTS"} or not track.sample_rate:
            pass
        elif track.sample_rate not in OrpheusMusicValidator.ALLOWED_SAMPLE_RATES:
            if media == "Cassette" and track.sample_rate < 44_100:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.WARNING,
                        "cassette_rate",
                        f"{track.relative_path}: a cassette under 44.1 kHz needs staff approval before upload.",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.ERROR,
                        "sample_rate",
                        f"{track.relative_path}: sample rate is not allowed.",
                    )
                )
        if media == "CD" and track.format == "MP3" and track.sample_rate and track.sample_rate != 44_100:
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "cd_mp3_rate",
                    f"{track.relative_path}: an MP3 from CD must stay 44.1 kHz.",
                )
            )
        if track.format in {"MP3", "AAC"} and track.channels and track.channels > 2:
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "lossy_multichannel",
                    f"{track.relative_path}: MP3 and AAC cannot carry more than two channels.",
                )
            )

    @staticmethod
    def _issue_path(issue: ValidationIssue) -> str:
        return issue.message.split(":", 1)[0].strip()

    @classmethod
    def _is_dxd_flac(cls, release: MusicRelease, issue: ValidationIssue) -> bool:
        path = cls._issue_path(issue)
        return any(
            track.relative_path == path and track.format == "FLAC" and track.bit_depth == 24 and track.sample_rate in DXD_RATES
            for track in release.tracks
        )

    @staticmethod
    def _hybrid_issue(release: MusicRelease, media: str) -> tuple[ValidationLevel, str]:
        modes = {track.bitrate_mode for track in release.tracks if track.bitrate_mode}
        if len(modes) > 1:
            return ValidationLevel.ERROR, "Tracks mix bitrate modes such as CBR and VBR. Redacted does not allow that."
        depths = {track.bit_depth for track in release.tracks if track.bit_depth}
        rates = {track.sample_rate for track in release.tracks if track.sample_rate}
        if media in ANALOG_HYBRID_MEDIA and (len(depths) > 1 or len(rates) > 1):
            return (
                ValidationLevel.ERROR,
                "Vinyl, cassette and soundboard rips cannot mix bit depth or sample rate. The ripper chose those values.",
            )
        return (
            ValidationLevel.WARNING,
            "Tracks differ in bit depth, sample rate, channels or bitrate mode. Redacted allows that only when the source was delivered that way.",
        )

    @staticmethod
    def _check_banned_sidecars(issues: list[ValidationIssue], release: MusicRelease) -> None:
        blocked = BANNED_SIDECAR_SUFFIXES | EXTRA_VIDEO_SUFFIXES
        banned = [name for name in release.auxiliary.other if Path(name).suffix.lower() in blocked]
        if banned:
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "banned_sidecar",
                    f"Music uploads cannot contain video, games or programs: {', '.join(banned[:5])}.",
                )
            )

    @staticmethod
    def _has_edition_evidence(release: MusicRelease) -> bool:
        return any(str(release.get(name) or "").strip() for name in ("edition_year", "release_year", "retail_date"))

    @staticmethod
    def _check_edition_year(issues: list[ValidationIssue], release: MusicRelease, media: str) -> None:
        first_year = MEDIUM_FIRST_YEAR.get(media)
        year = Orpheus._edition_year_for_upload(release)
        if first_year and year.isdigit() and int(year) < first_year:
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "edition_year",
                    f"Edition year {year} is before {media} existed. Enter the real pressing or release year.",
                )
            )
        if RedactedMusicValidator._has_edition_evidence(release):
            return
        if media == "WEB":
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "edition_year_evidence",
                    "No release date was found. The original year is being sent as the edition year. Confirm it.",
                )
            )
            return
        if media:
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "edition_year_evidence",
                    "No edition year was found. Pass the real pressing year. Redacted does not accept a guess.",
                )
            )

    @staticmethod
    def _check_release_type(issues: list[ValidationIssue], release: MusicRelease) -> None:
        name = str(release.get("release_type", "")).strip()
        if not name:
            return
        folded = name.casefold()
        if any(key.casefold() == folded for key in Redacted.release_types):
            return
        issues.append(
            ValidationIssue(
                ValidationLevel.ERROR,
                "release_type",
                f"Release type {name} is not a Redacted music category.",
            )
        )

    @staticmethod
    def _track_has_mqa(track: AudioTrack) -> bool:
        for key, values in track.tags.items():
            folded = str(key).casefold()
            if folded.startswith("mqa") or folded == "originalsamplerate":
                return True
            blob = " ".join(str(item) for item in values)
            if re.search(r"\bmqa", blob, re.I):
                return True
        return False

    @staticmethod
    def _check_mqa_tags(issues: list[ValidationIssue], release: MusicRelease) -> None:
        if not any(RedactedMusicValidator._track_has_mqa(track) for track in release.tracks):
            return
        issues.append(
            ValidationIssue(
                ValidationLevel.WARNING,
                "mqa_tags",
                "MQA tags found. Name the store and state the unfolded resolution. Do not transcode MQA.",
            )
        )
        if any(track.format != "FLAC" for track in release.tracks):
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "mqa_format",
                    "MQA is only allowed in FLAC.",
                )
            )
        edition = str(release.get("edition", ""))
        if not re.search(r"\bmqa\b", edition, re.I):
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "mqa_edition",
                    "The edition title must contain MQA. The title is not filled in for you.",
                )
            )

    @staticmethod
    def _filename_number_dupes(release: MusicRelease) -> list[str]:
        """Warn when one directory holds two discs that share a leading file number."""
        discs_by_folder: dict[str, set[int]] = {}
        for track in release.tracks:
            folder = str(Path(track.relative_path).parent)
            discs_by_folder.setdefault(folder, set()).add(track.disc_number or 1)
        shared = {folder for folder, discs in discs_by_folder.items() if len(discs) > 1}
        seen: dict[tuple[str, str], str] = {}
        dupes: list[str] = []
        for track in release.tracks:
            folder = str(Path(track.relative_path).parent)
            if folder not in shared:
                continue
            match = re.match(r"\s*(\d+)", Path(track.relative_path).name)
            if not match:
                continue
            key = (folder, match.group(1).lstrip("0") or "0")
            if key in seen:
                dupes.append(track.relative_path)
            else:
                seen[key] = track.relative_path
        return dupes

    @staticmethod
    def _cue_track_count(release: MusicRelease) -> int:
        total = 0
        root = Path(release.root)
        for relative in release.auxiliary.cues:
            try:
                text = (root / relative).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            total += len(re.findall(r"(?im)^\s*TRACK\s+\d+", text))
        return total

    @staticmethod
    def _check_unsplit(issues: list[ValidationIssue], release: MusicRelease) -> None:
        if len(release.tracks) != 1 or RedactedMusicValidator._cue_track_count(release) < 2:
            return
        issues[:] = [issue for issue in issues if issue.code not in {"single_track", "possible_unsplit"}]
        track = release.tracks[0]
        release_type = str(release.get("release_type", "")).casefold()
        if track.format == "FLAC":
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "unsplit_image",
                    "This FLAC is an unsplit image with a cue. It is trumpable by a split rip.",
                )
            )
            return
        if track.format == "MP3" and release_type in {"dj mix", "mixtape"}:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "unsplit_mix",
                    "Unsplit MP3 is allowed for a DJ mix or mixtape when the cue belongs to that mix.",
                )
            )
            return
        if track.format == "MP3":
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "unsplit_mp3",
                    "Unsplit MP3 is only allowed for a continuous album or mix, and it needs the cue.",
                )
            )
            return
        issues.append(
            ValidationIssue(
                ValidationLevel.ERROR,
                "unsplit_image",
                f"An unsplit {track.format} image is not an allowed music upload.",
            )
        )

    @staticmethod
    def _mp3_preset(settings: str) -> str:
        for pattern, label in _MP3_PRESETS:
            if pattern.search(settings or ""):
                return label
        return ""

    @staticmethod
    def _check_mp3_presets(issues: list[ValidationIssue], release: MusicRelease) -> None:
        if not release.tracks or any(track.format != "MP3" for track in release.tracks):
            return
        labels = {RedactedMusicValidator._mp3_preset(track.encoder_settings) for track in release.tracks}
        labels.discard("")
        if len(labels) > 1:
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "mixed_preset",
                    f"MP3 tracks use different presets ({', '.join(sorted(labels))}). Redacted requires one preset.",
                )
            )

    @staticmethod
    def _check_embedded_size(issues: list[ValidationIssue], release: MusicRelease) -> None:
        heavy = [track.relative_path for track in release.tracks if track.embedded_bytes > 1024 * 1024]
        if heavy:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "embedded_art",
                    f"{len(heavy)} file(s) have more than 1024 KiB of embedded art plus padding.",
                )
            )

    @staticmethod
    def _has_lineage_file(release: MusicRelease) -> bool:
        if release.auxiliary.lineage or release.auxiliary.logs:
            return True
        return any(Path(name).suffix.lower() == ".txt" for name in release.auxiliary.other)

    @staticmethod
    def _check_single_inference(issues: list[ValidationIssue], release: MusicRelease) -> None:
        type_field = release.fields.get("release_type")
        if len(release.tracks) == 1 and type_field is not None and type_field.source == MetadataSource.INFERRED:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "inferred_single",
                    "A one-track upload is allowed only when it is an officially released single. Confirm the release type.",
                )
            )

    @staticmethod
    def _check_extra_audio(issues: list[ValidationIssue], release: MusicRelease) -> None:
        blocked = AUDIO_EXTENSIONS | FORBIDDEN_AUDIO_SUFFIXES
        found = [name for name in release.auxiliary.other if Path(name).suffix.lower() in blocked]
        if found:
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "extra_audio",
                    f"Music uploads cannot contain other or unreadable audio: {', '.join(found[:5])}.",
                )
            )

    @staticmethod
    def _check_shortcuts(issues: list[ValidationIssue], release: MusicRelease) -> None:
        found = [name for name in release.auxiliary.other if Path(name).suffix.lower() in SHORTCUT_SUFFIXES]
        if found:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "shortcut_file",
                    f"Shortcut or page files are not part of the release: {', '.join(found[:5])}.",
                )
            )

    @staticmethod
    def _check_archives(issues: list[ValidationIssue], release: MusicRelease) -> None:
        archived = [
            name
            for name in release.auxiliary.other
            if Path(name).suffix.lower() in ARCHIVE_SUFFIXES | DISC_IMAGE_SUFFIXES or SPLIT_ARCHIVE_RE.search(Path(name).name)
        ]
        if archived:
            issues.append(
                ValidationIssue(
                    ValidationLevel.ERROR,
                    "archived_file",
                    f"Music uploads cannot contain archives: {', '.join(archived[:5])}.",
                )
            )

    @staticmethod
    def _torrent_paths(release: MusicRelease) -> list[str]:
        auxiliary = release.auxiliary
        names = [track.relative_path for track in release.tracks]
        for group in (
            auxiliary.logs,
            auxiliary.cues,
            auxiliary.nfos,
            auxiliary.sfvs,
            auxiliary.playlists,
            auxiliary.artwork,
            auxiliary.scans,
            auxiliary.lineage,
            auxiliary.other,
        ):
            names.extend(group)
        return names

    @staticmethod
    def _check_names(issues: list[ValidationIssue], release: MusicRelease) -> None:
        folder = Path(release.root).name
        paths = RedactedMusicValidator._torrent_paths(release)
        long_paths = [name for name in paths if len(f"{folder}/{name}") > PATH_LIMIT]
        if long_paths:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "path_length",
                    f"{len(long_paths)} path(s) exceed Redacted's 180 character limit, including the torrent folder.",
                )
            )
        spaced = [name for name in paths if any(part.startswith(" ") for part in Path(name).parts) or folder.startswith(" ")]
        if spaced:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "leading_space",
                    f"{len(spaced)} path(s) have a leading space. Redacted torrents with that are trumpable.",
                )
            )
        if len(release.tracks) > 1:
            untitled = [track.relative_path for track in release.tracks if not re.match(r"\d{1,3}\b", Path(track.relative_path).stem.lstrip())]
            if untitled:
                issues.append(
                    ValidationIssue(
                        ValidationLevel.WARNING,
                        "filename_track_number",
                        f"{len(untitled)} filename(s) do not start with a track number. Redacted can trump them.",
                    )
                )
        seen: set[tuple[str, int, int]] = set()
        duplicates: list[str] = []
        for track in release.tracks:
            if not track.track_number:
                continue
            key = (str(Path(track.relative_path).parent), track.disc_number or 1, track.track_number)
            if key in seen:
                duplicates.append(track.relative_path)
            seen.add(key)
        if duplicates:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "duplicate_track_number",
                    f"The same disc and track number is reused in one folder: {', '.join(duplicates[:5])}.",
                )
            )
        filename_dupes = RedactedMusicValidator._filename_number_dupes(release)
        if filename_dupes:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "duplicate_filename_number",
                    f"One folder repeats the same leading track number: {', '.join(filename_dupes[:5])}.",
                )
            )
        discs = sorted({track.disc_number or 1 for track in release.tracks})
        if discs and (discs[0] != 1 or discs != list(range(discs[0], discs[-1] + 1))):
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "disc_gap",
                    f"Disc numbers are {', '.join(str(number) for number in discs)}. Confirm the set is complete.",
                )
            )

    @staticmethod
    def _check_required_tags(issues: list[ValidationIssue], release: MusicRelease) -> None:
        missing: list[str] = []
        for track in release.tracks:
            absent: list[str] = []
            if not track.title:
                absent.append("title")
            has_track_number = any(key.casefold() == "tracknumber" for key in track.tags) if track.tags else bool(track.track_number)
            if track.format == "FLAC" and track.tags and not any(key.casefold() == "tracknumber" for key in track.tags):
                absent.append("TRACKNUMBER")
            elif not has_track_number and len(release.tracks) > 1:
                absent.append("track number")
            if not track.album:
                absent.append("album")
            has_artist = any(key.casefold() in {"artist", "performer"} for key in track.tags) if track.tags else bool(track.artist or track.album_artist)
            if not has_artist:
                absent.append("artist")
            if absent:
                missing.append(f"{track.relative_path} ({', '.join(absent)})")
        if missing:
            extra = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "missing_required_tags",
                    f"Required tags are missing: {', '.join(missing[:5])}{extra}.",
                )
            )

    @staticmethod
    def _check_non_music_genres(issues: list[ValidationIssue], release: MusicRelease) -> None:
        genres = release.get("genres", [])
        if not isinstance(genres, list):
            return
        found = [str(genre) for genre in genres if str(genre).casefold() in _NON_MUSIC_GENRES]
        if found:
            issues.append(
                ValidationIssue(
                    ValidationLevel.WARNING,
                    "non_music_genre",
                    f"Genre tags look like non-music ({', '.join(found)}). Comedy, audiobooks and podcasts are not music uploads.",
                )
            )


class Redacted(Orpheus):
    """Redacted is a private music tracker."""

    tracker = "REDACTED"
    display_name = "Redacted"
    source_flag = "RED"
    base_url = "https://redacted.sh"
    validator_cls = RedactedMusicValidator
    request_category = "0"
    hosts_own_music_cover = True
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
        self._image_auth = ""
        self._images_since_pause = 0
        self._image_batching = False
        self._image_login_warned = False
        self._image_progress_logged = False
        self._image_progress_detail = ""
        self._reset_image_progress()

    def _record_upload_response(self, meta: Meta, payload: Any) -> bool:
        accepted = super()._record_upload_response(meta, payload)
        status = meta.tracker_status.get(self.tracker, {})
        message = status.get("status_message")
        if isinstance(message, str):
            status["status_message"] = message.replace("Orpheus", "Redacted")
        return accepted

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
        self._image_progress_logged = False
        self._reset_image_progress()
        release = self._release(meta)
        if not await self.get_additional_checks(meta):
            return False
        issues = self.validator_cls().validate(release)
        for issue in issues:
            if issue.level == ValidationLevel.WARNING:
                logger.warning(f"{self.tracker}: [yellow]{issue.message}[/yellow]")
        errors = [issue.message for issue in issues if issue.level == ValidationLevel.ERROR]
        if not meta.isdir:
            errors.append("Music must be inside a folder, even when there is only one file.")
        if errors:
            message = "Validation failed: " + " | ".join(errors)
            meta.tracker_status.setdefault(self.tracker, {})["status_message"] = message
            logger.info(f"{self.tracker}: [red]{message}[/red]")
            return False
        self._hosted_cover = ""
        if not meta.debug and (not self.api_key or not self.announce_url):
            message = f"Missing {self.display_name} API key or announce URL."
            meta.tracker_status.setdefault(self.tracker, {})["status_message"] = message
            logger.info(f"{self.tracker}: [red]{message}[/red]")
            return False
        if not meta.debug:
            try:
                self.build_upload_payload(meta, release)
            except ValueError as error:
                message = str(error)
                meta.tracker_status.setdefault(self.tracker, {})["status_message"] = message
                logger.info(f"{self.tracker}: [red]{message}[/red]")
                return False
        if not meta.debug and self.api_key:
            image_started = time.monotonic()
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers=self._headers(meta)) as client:
                if await self._prepare_image_session(client):
                    self._begin_image_progress(self._planned_image_count(meta, release))
                    try:
                        jobs: list[Callable[[], Awaitable[None]]] = []
                        self._queue_cover_upload(meta, client, jobs)
                        plots: list[dict[str, str]] = []
                        if self._wants_spectrograms():
                            plots = await self._render_spectrograms(meta, release)
                            self._queue_spectrogram_uploads(meta, client, plots, jobs)
                        self._image_progress_detail = ""
                        await self._run_image_batches(jobs)
                        self._spectrogram_bbcode = spectrogram_bbcode([(row["title"], row["full_url"], row["zoom_url"]) for row in plots])
                    finally:
                        self._finish_image_progress()
            await self._pause_before_next_api_call()
            meta[f"{self.tracker}_image_upload_duration"] = time.monotonic() - image_started
        uploaded = await super().upload(meta)
        if not uploaded:
            message = self._safe_status(str(meta.tracker_status.get(self.tracker, {}).get("status_message", "")))
            if message:
                logger.info(f"{self.tracker}: [red]{message.replace('Orpheus', 'Redacted')}[/red]")
        return uploaded

    def _wants_spectrograms(self) -> bool:
        settings = self.config.get("TRACKERS", {}).get(self.tracker, {})
        return bool(settings.get("add_spectrograms", True))

    def _reset_image_progress(self) -> None:
        self._image_progress_active = False
        self._image_progress_published = False
        self._image_progress_current = 0
        self._image_progress_total = 0
        self._image_progress_detail = ""
        self._image_batching = False

    def _planned_image_count(self, meta: Meta, release: MusicRelease) -> int:
        """Images that will be posted: the cover, plus each spectrogram SoX can draw."""
        total = self._cover_image_count(meta)
        if self._wants_spectrograms() and shutil.which("sox") and meta.base_dir and meta.uuid:
            total += self._spectrogram_image_count(release)
        return total

    def _cover_image_count(self, meta: Meta) -> int:
        current = str(meta.artwork_url or "").strip()
        if self._is_site_image(current):
            return 0
        if current.startswith("https://"):
            return 1
        artwork_path = Path(str(meta.artwork_path or ""))
        try:
            return 1 if artwork_path.is_file() else 0
        except OSError:
            return 0

    def _spectrogram_image_count(self, release: MusicRelease) -> int:
        total = 0
        for track in release.tracks:
            if self._track_source(release, track) is None:
                continue
            total += 2 if zoom_window(track.duration) is not None else 1
        return total

    def _begin_image_progress(self, total: int) -> None:
        self._image_progress_active = True
        self._image_progress_published = False
        self._image_progress_current = 0
        self._image_progress_total = max(0, total)
        self._publish_image_progress()

    def _advance_image_progress(self) -> None:
        """Count one image. The rate-limit retry calls the sender again and must not land here."""
        if not self._image_progress_active:
            return
        self._image_progress_current += 1
        if self._image_progress_current > self._image_progress_total:
            self._image_progress_total = self._image_progress_current
        self._publish_image_progress()

    def _skip_image_progress(self, count: int) -> None:
        """Drop images that will not be posted, such as a spectrogram SoX could not draw."""
        if not self._image_progress_active or count <= 0:
            return
        self._image_progress_total = max(0, self._image_progress_total - count)
        if self._image_progress_total < self._image_progress_current:
            self._image_progress_total = self._image_progress_current
        if self._image_progress_total > 0:
            self._publish_image_progress()

    def _publish_image_progress(self) -> None:
        if not self._image_progress_active or self._image_progress_total <= 0:
            return
        self._image_progress_published = True
        publish_progress(
            IMAGE_PROGRESS_ID,
            f"{self.display_name} images",
            current=self._image_progress_current,
            total=self._image_progress_total,
            detail=self._image_progress_detail or "This can take a while.",
            group="images",
            unit="images",
        )

    def _finish_image_progress(self) -> None:
        published = self._image_progress_published
        current = self._image_progress_current
        total = self._image_progress_total
        self._image_progress_active = False
        if not published:
            return
        complete_progress(
            IMAGE_PROGRESS_ID,
            f"{self.display_name} images",
            current=current,
            total=total if total > 0 else current,
            detail="Images uploaded." if current else "No images uploaded.",
            group="images",
            unit="images",
        )

    async def _prepare_image_session(self, client: httpx.AsyncClient) -> bool:
        """Load the auth key of the logged-in user. Image uploads require it."""
        try:
            response = await client.get(f"{self.base_url}/ajax.php?action=index")
            if response.status_code == 429:
                logger.warning(f"{self.tracker}: [yellow]image host rate limit reached while opening the user session.[/yellow]")
                await self._pause_for_image_host()
                response = await client.get(f"{self.base_url}/ajax.php?action=index")
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            logger.warning(f"{self.tracker}: [yellow]image uploads need a logged-in user and the session check failed ({error.__class__.__name__}). Images were skipped.[/yellow]")
            return False
        auth = authkey_from_index(payload)
        if not auth:
            detail = self._safe_status(self._json_error(response) or f"HTTP {response.status_code}")
            logger.warning(f"{self.tracker}: [yellow]image uploads are only available to a logged-in user ({detail}). Images were skipped.[/yellow]")
            return False
        self._image_auth = auth
        return True

    async def _rehost_cover(self, meta: Meta, client: httpx.AsyncClient) -> None:
        """Put the cover on Redacted's image host when it is not already there."""
        current = str(meta.artwork_url or "").strip()
        if self._is_site_image(current):
            return
        artwork_path = Path(str(meta.artwork_path or ""))
        try:
            if artwork_path.is_file():
                hosted = await self._upload_image_file(meta, artwork_path, client)
            elif current.startswith("https://"):
                hosted = await self._rehost_image_url(meta, current, client)
            else:
                return
        except (OSError, ValueError, httpx.HTTPError) as error:
            self._warn_image_problem(f"cover rehost failed, keeping the current artwork URL: {error}")
            return
        # Keep the hosted URL on this tracker. A shared artwork URL would leak into Orpheus.
        self._hosted_cover = hosted

    def _cover_url(self, meta: Meta) -> str:
        hosted = str(getattr(self, "_hosted_cover", "") or "")
        value = hosted or Orpheus._cover_url(meta)
        return value if value.startswith("https://") else ""

    def _queue_cover_upload(self, meta: Meta, client: httpx.AsyncClient, jobs: list[Callable[[], Awaitable[None]]]) -> None:
        """Add the cover to the next image batch instead of posting it on its own."""
        current = str(meta.artwork_url or "").strip()
        if self._is_site_image(current):
            return
        artwork_path = Path(str(meta.artwork_path or ""))
        try:
            local = artwork_path.is_file()
        except OSError:
            local = False
        if not local and not current.startswith("https://"):
            return

        async def send() -> None:
            try:
                if local:
                    hosted = await self._upload_image_file(meta, artwork_path, client)
                else:
                    hosted = await self._rehost_image_url(meta, current, client)
            except (OSError, ValueError, httpx.HTTPError) as error:
                self._warn_image_problem(f"cover rehost failed, keeping the current artwork URL: {error}")
                return
            self._hosted_cover = hosted

        jobs.append(send)

    async def _render_spectrograms(self, meta: Meta, release: MusicRelease) -> list[dict[str, str]]:
        """Draw every plot before any of them is posted, so a batch is not split by SoX."""
        sox = shutil.which("sox") or ""
        if not sox:
            logger.warning(f"{self.tracker}: [yellow]sox was not found, so spectrograms were skipped.[/yellow]")
            return []
        if not meta.base_dir or not meta.uuid:
            logger.warning(f"{self.tracker}: [yellow]spectrogram output directory is unavailable.[/yellow]")
            return []
        output_dir = Path(meta.base_dir) / "tmp" / str(meta.uuid) / "red-spectrograms"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.warning(f"{self.tracker}: [yellow]spectrograms were skipped because the output directory could not be created ({error.__class__.__name__}).[/yellow]")
            self._skip_image_progress(self._spectrogram_image_count(release))
            return []
        self._image_progress_detail = "Rendering spectrograms."
        self._publish_image_progress()
        rows: list[dict[str, str]] = []
        for index, track in enumerate(release.tracks, start=1):
            source = self._track_source(release, track)
            if source is None:
                logger.warning(f"{self.tracker}: [yellow]skipping spectrogram, audio file is missing: {track.relative_path}[/yellow]")
                continue
            title = track.title or source.stem
            full_path, zoom_path = await self._render_one_spectrogram(sox, source, output_dir, index, title, track.duration)
            if full_path or zoom_path:
                rows.append({"title": title, "full": full_path, "zoom": zoom_path, "full_url": "", "zoom_url": ""})
        return rows

    def _queue_spectrogram_uploads(
        self,
        meta: Meta,
        client: httpx.AsyncClient,
        rows: list[dict[str, str]],
        jobs: list[Callable[[], Awaitable[None]]],
    ) -> None:
        for row in rows:
            full = row["full"]
            zoom = row["zoom"]
            if full:
                jobs.append(self._spectrogram_job(meta, client, row, "full_url", full))
            if zoom:
                jobs.append(self._spectrogram_job(meta, client, row, "zoom_url", zoom))

    def _spectrogram_job(self, meta: Meta, client: httpx.AsyncClient, row: dict[str, str], field: str, path: str) -> Callable[[], Awaitable[None]]:
        async def send() -> None:
            row[field] = await self._host_existing_image(meta, Path(path), client)

        return send

    async def _run_image_batches(self, jobs: list[Callable[[], Awaitable[None]]]) -> None:
        """Post three images at once. The pause sits between batches, not between the three."""
        if not jobs:
            return
        self._image_batching = True
        try:
            for offset in range(0, len(jobs), IMAGE_BATCH_SIZE):
                if offset:
                    await self._pause_for_image_host()
                batch = jobs[offset : offset + IMAGE_BATCH_SIZE]
                results = await asyncio.gather(*(job() for job in batch), return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        self._warn_image_problem(f"image upload failed: {result}")
                self._images_since_pause = len(batch)
        finally:
            self._image_batching = False

    async def _render_one_spectrogram(
        self,
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
            self._skip_image_progress(2 if window is not None else 1)
            return "", ""
        if window is None:
            return str(full_path), ""
        try:
            await asyncio.to_thread(render_spectrogram, sox, source, zoom_path, zoom=window)
        except RuntimeError as error:
            logger.warning(f"{self.tracker}: [yellow]zoom spectrogram skipped for {source.name}: {error}[/yellow]")
            self._skip_image_progress(1)
            return str(full_path), ""
        return str(full_path), str(zoom_path)

    async def _one_spectrogram_pair(
        self,
        meta: Meta,
        client: httpx.AsyncClient,
        sox: str,
        source: Path,
        output_dir: Path,
        index: int,
        title: str,
        duration: float | None,
    ) -> tuple[str, str]:
        full_path, zoom_path = await self._render_one_spectrogram(sox, source, output_dir, index, title, duration)
        full_url = await self._host_existing_image(meta, Path(full_path), client) if full_path else ""
        zoom_url = await self._host_existing_image(meta, Path(zoom_path), client) if zoom_path else ""
        return full_url, zoom_url

    async def _host_existing_image(self, meta: Meta, path: Path, client: httpx.AsyncClient) -> str:
        if not path.is_file() or not self._image_auth:
            self._skip_image_progress(1)
            return ""
        try:
            return await self._upload_image_file(meta, path, client)
        except (OSError, ValueError, httpx.HTTPError) as error:
            self._warn_image_problem(f"could not upload {path.name} to the Redacted image host: {error}")
            return ""

    @staticmethod
    def _track_source(release: MusicRelease, track: AudioTrack) -> Path | None:
        direct = Path(track.path)
        if direct.is_file():
            return direct
        rooted = Path(release.root) / track.relative_path
        return rooted if rooted.is_file() else None

    async def _upload_image_file(self, _meta: Meta, path: Path, client: httpx.AsyncClient) -> str:
        if not self._image_auth:
            raise ValueError("Redacted image uploads require a logged-in user.")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        try:
            handle = path.open("rb")
        except OSError:
            self._skip_image_progress(1)
            raise
        with handle:
            return await self._post_image(client, data={"auth": self._image_auth}, files={"file": (path.name, handle, mime)})

    async def _rehost_image_url(self, _meta: Meta, url: str, client: httpx.AsyncClient) -> str:
        if not self._image_auth:
            raise ValueError("Redacted image uploads require a logged-in user.")
        return await self._post_image(client, data={"auth": self._image_auth, "url": url})

    async def _post_image(
        self,
        client: httpx.AsyncClient,
        *,
        data: dict[str, str],
        files: dict[str, tuple[str, Any, str]] | None = None,
    ) -> str:
        # One step per image. _send_image runs again after a rate limit and must not add another.
        self._advance_image_progress()
        response = await self._send_image(client, data, files)
        if self._is_rate_limited(response):
            logger.warning(f"{self.tracker}: [yellow]image host rate limit reached. Waiting {int(IMAGE_BATCH_PAUSE_SECONDS)}s before one retry.[/yellow]")
            await self._pause_for_image_host()
            self._rewind_uploads(files)
            response = await self._send_image(client, data, files)
        if self._needs_login(response):
            self._image_auth = ""
            raise ValueError("Redacted image uploads require a logged-in user.")
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as error:
            raise ValueError("Redacted image host returned a malformed response.") from error
        return image_url_from_response(payload)

    async def _send_image(
        self,
        client: httpx.AsyncClient,
        data: dict[str, str],
        files: dict[str, tuple[str, Any, str]] | None,
    ) -> httpx.Response:
        await self._reserve_image_slot()
        return await client.post(f"{self.base_url}/ajax.php?action=upload_image", data=data, files=files)

    async def _reserve_image_slot(self) -> None:
        if not self._image_progress_logged:
            logger.warning(f"{self.tracker}: [yellow]uploading images. This can take a while.[/yellow]")
            self._image_progress_logged = True
        # A batch posts its three images together. The pause is applied once, before the next batch.
        if self._image_batching:
            return
        if self._images_since_pause >= IMAGE_BATCH_SIZE:
            await self._pause_for_image_host()
        self._images_since_pause += 1

    async def _pause_for_image_host(self) -> None:
        await asyncio.sleep(IMAGE_BATCH_PAUSE_SECONDS)
        self._images_since_pause = 0

    async def _pause_before_next_api_call(self) -> None:
        if self._images_since_pause:
            await self._pause_for_image_host()

    def _warn_image_problem(self, message: str) -> None:
        if "logged-in user" in message:
            if self._image_login_warned:
                return
            self._image_login_warned = True
        logger.warning(f"{self.tracker}: [yellow]{self._safe_status(message)}[/yellow]")

    def _safe_status(self, message: str) -> str:
        text = " ".join(str(message).split())
        for secret in (self.api_key, self.announce_url, self._image_auth):
            if secret:
                text = text.replace(secret, "[hidden]")
        return text[:400]

    @classmethod
    def _is_rate_limited(cls, response: httpx.Response) -> bool:
        if response.status_code == 429:
            return True
        folded = cls._json_error(response).casefold()
        return any(marker in folded for marker in _RATE_LIMIT_MARKERS)

    @classmethod
    def _needs_login(cls, response: httpx.Response) -> bool:
        if response.status_code == 401:
            return True
        folded = cls._json_error(response).casefold()
        return any(marker in folded for marker in _LOGIN_MARKERS)

    @staticmethod
    def _json_error(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return ""
        if isinstance(payload, dict):
            return str(payload.get("error") or "").strip()
        return ""

    @staticmethod
    def _rewind_uploads(files: dict[str, tuple[str, Any, str]] | None) -> None:
        if not files:
            return
        for _name, handle, _mime in files.values():
            rewind = getattr(handle, "seek", None)
            if callable(rewind):
                rewind(0)

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
        artists, importance = self._upload_artists(release)
        label = self._confirmed_text(release, "edition_label", "release_label", "label")
        catalogue = self._confirmed_text(release, "edition_catalogue_number", "release_catalogue_number", "catalogue_number")
        edition_title = self._edition_text(release.get("edition"))
        # Gazelle stores every edition in the remaster fields. Without the flag
        # the site marks the torrent Unknown Release and rejects the upload.
        payload: dict[str, str | list[str] | list[int] | int] = {
            "album_desc": self._album_description(release),
            "artists[]": artists,
            "bitrate": bitrate,
            "format": format_name,
            "image": self._cover_url(meta),
            "importance[]": importance,
            "media": media,
            "release_desc": self._release_description(release),
            "releasetype": self._release_type_id(release),
            "remaster": "on",
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
        return {key: value for key, value in payload.items() if value not in ("", None, [])}

    def _upload_artists(self, release: MusicRelease) -> tuple[list[str], list[int]]:
        """Main artists stay importance 1. A featured performer is a guest, importance 2."""
        artists = self._artists(release)
        known = {fold_name(name) for name in artists}
        guests: list[str] = []
        raw = release.get("guest_artists")
        if isinstance(raw, list):
            for item in raw:
                name = str(item.get("name") if isinstance(item, dict) else item).strip()
                folded = fold_name(name)
                if name and folded not in known and folded not in {fold_name(guest) for guest in guests}:
                    guests.append(name)
                    known.add(folded)
        return artists + guests, [1] * len(artists) + [2] * len(guests)

    def _release_type_id(self, release: MusicRelease) -> int:
        name = str(release.get("release_type", "Unknown")).strip()
        if name in self.release_types:
            return self.release_types[name]
        folded = name.casefold()
        for key, value in self.release_types.items():
            if key.casefold() == folded:
                return value
        raise ValueError(f"Release type {name} is not a Redacted music category.")

    @staticmethod
    def _edition_text(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _media_name(value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if cleaned in {"CD", "DVD", "Vinyl", "Soundboard", "SACD", "DAT", "Cassette", "WEB", "Blu-Ray", "LaserDisc"}:
            return cleaned
        return MEDIA_NAMES.get(cleaned.casefold().replace(" ", ""), "")

    _EDITION_SOURCES: ClassVar[set[MetadataSource]] = {MetadataSource.USER, MetadataSource.FILE_TAG, MetadataSource.EXTERNAL, MetadataSource.TRACKER}

    @classmethod
    def _confirmed_text(cls, release: MusicRelease, *names: str) -> str:
        """Use a label or catalogue only when a tag, the user, or a tracker supplied it."""
        for name in names:
            item = release.fields.get(name)
            if item is None or item.source not in cls._EDITION_SOURCES:
                continue
            text = str(item.value or "").strip()
            if text:
                return text
        return ""

    def _encoding(self, release: MusicRelease, format_name: str) -> tuple[str, str, bool]:
        if format_name == "DSD":
            rate = next((track.sample_rate or 0 for track in release.tracks if track.sample_rate), 0)
            bitrate = DSD_RATES.get(rate, "")
            return (bitrate or "Other", "" if bitrate else "DSD", False)
        if format_name == "MP3":
            return self._mp3_encoding(release)
        return super()._encoding(release, format_name)

    @classmethod
    def _mp3_encoding(cls, release: MusicRelease) -> tuple[str, str, bool]:
        labels = [RedactedMusicValidator._mp3_preset(track.encoder_settings) for track in release.tracks]
        if labels and all(labels) and len(set(labels)) == 1:
            return labels[0], "", True
        average = cls._weighted_kbps(release.tracks)
        mode = next((track.bitrate_mode for track in release.tracks if track.bitrate_mode), None)
        if mode == "CBR" and average in {192, 256, 320}:
            return str(average), "", False
        return "Other", str(average or "VBR"), mode == "VBR"

    @staticmethod
    def _weighted_kbps(tracks: list[AudioTrack]) -> int:
        weight = 0.0
        total = 0.0
        for track in tracks:
            seconds = track.duration if track.duration and track.duration > 0 else 1.0
            total += (track.bitrate or 0) * seconds
            weight += seconds
        if weight <= 0:
            return 0
        return round(total / weight / 1000)

    def _album_description(self, release: MusicRelease) -> str:
        text = super()._album_description(release)
        more = more_info_bbcode(source_links(release))
        if not more:
            return text
        return f"{text}\n\n{more}" if text else more

    def _release_description(self, release: MusicRelease) -> str:
        text = super()._release_description(release)
        links = source_links(release)
        more = more_info_bbcode(links)
        if more:
            text = self._drop_known_store_listing(text, release, links)
            text = f"{text}\n\n{more}".strip() if text else more
        lossy = self._lossy_technical(release)
        if lossy:
            lines = text.split("\n")
            if lines and lines[0].startswith("Technical audio:"):
                lines[0] = lossy
            else:
                lines.insert(0, lossy)
            text = "\n".join(line for line in lines if line)
        note = self._hybrid_note(release)
        spelling = self._spelling_note(release)
        if note:
            text = f"{text}\n\n{note}".strip() if text else note
        if spelling:
            text = f"{text}\n\n{spelling}".strip() if text else spelling
        if not self._spectrogram_bbcode:
            return text
        return f"{text}\n\n{self._spectrogram_bbcode}".strip() if text else self._spectrogram_bbcode

    @staticmethod
    def _drop_known_store_listing(text: str, release: MusicRelease, links: list[tuple[str, str]]) -> str:
        """A recognized store already named in More info does not also stay as Store listing."""
        store = str(release.get("store_url") or "")
        if store not in {url for _name, url in links}:
            return text
        listing = f"[url={store}]Store listing[/url]"
        return "\n".join(line for line in text.split("\n") if line != listing)

    @staticmethod
    def _lossy_technical(release: MusicRelease) -> str:
        if not release.tracks or any(track.format not in {"MP3", "AAC", "AC3", "DTS"} for track in release.tracks):
            return ""
        parts: list[str] = []
        for track in release.tracks:
            rate = f"{(track.sample_rate or 0) / 1000:g} kHz" if track.sample_rate else "? kHz"
            mode = track.bitrate_mode or "bitrate"
            kbps = str(round(track.bitrate / 1000)) if track.bitrate else "?"
            parts.append(f"{kbps} kbps {mode} / {rate} / {track.channels or '?'}ch")
        return f"Technical audio: {', '.join(sorted(set(parts)))}."

    @staticmethod
    def _hybrid_note(release: MusicRelease) -> str:
        """Describe mixed bit depth, sample rate or channel count. Rule 2.1.6.2."""
        groups: dict[tuple[int | None, int | None, int | None], list[str]] = {}
        for track in release.tracks:
            key = (track.bit_depth, track.sample_rate, track.channels)
            number = str(track.track_number) if track.track_number else Path(track.relative_path).name
            groups.setdefault(key, []).append(number)
        if len(groups) < 2:
            return ""
        mixed_quality = len({track.bit_depth for track in release.tracks}) > 1 or len({track.sample_rate for track in release.tracks}) > 1
        parts: list[str] = []
        for (depth, rate, channels), numbers in groups.items():
            depth_text = f"{depth}-bit" if depth else "unknown bit depth"
            rate_text = f"{rate / 1000:g} kHz" if rate else "unknown sample rate"
            channel_text = f"{channels} channels" if channels else "unknown channels"
            joined = ", ".join(numbers)
            verb = "is" if len(numbers) == 1 else "are"
            label = "Track" if len(numbers) == 1 else "Tracks"
            parts.append(f"{label} {joined} {verb} {depth_text} {rate_text}, {channel_text}")
        prefix = "Hybrid audio, labeled by the highest bit depth and sample rate: " if mixed_quality else "Hybrid audio: "
        return prefix + "; ".join(parts) + "."

    @staticmethod
    def _spelling_note(release: MusicRelease) -> str:
        """Note a title that differs from the others only by accents. Do not rewrite it."""
        grouped: dict[str, dict[str, list[str]]] = {}
        for track in release.tracks:
            match = re.search(r"\(([^)]+)\)\s*$", track.title or "")
            if not match:
                continue
            raw = match.group(1).strip()
            folded = "".join(char for char in unicodedata.normalize("NFD", raw) if unicodedata.category(char) != "Mn").casefold()
            number = str(track.track_number) if track.track_number else Path(track.relative_path).name
            grouped.setdefault(folded, {}).setdefault(raw, []).append(number)
        notes: list[str] = []
        for variants in grouped.values():
            if len(variants) < 2:
                continue
            parts: list[str] = []
            for spelling, numbers in variants.items():
                label = "Track" if len(numbers) == 1 else "Tracks"
                verb = "uses" if len(numbers) == 1 else "use"
                parts.append(f"{label} {', '.join(numbers)} {verb} ({spelling})")
            notes.append("; ".join(parts) + ". Left as supplied.")
        return " ".join(notes)
