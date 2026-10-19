"""Redacted form mapping, image-host parsing and SoX spectrogram commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.clients import Clients
from src.meta import Meta
from src.music.models import AudioTrack, MetadataSource, MusicRelease
from src.trackers.GAZELLE.red_images import image_url_from_response
from src.trackers.GAZELLE.red_spectrograms import output_name, sox_spectrogram_command, spectrogram_bbcode, zoom_window
from src.trackers.GAZELLE.redacted import Redacted, RedactedMusicValidator


def _release(**fields: object) -> MusicRelease:
    release = MusicRelease(root=".")
    release.tracks.append(
        AudioTrack(
            path="track.flac",
            relative_path="track.flac",
            format="FLAC",
            codec="FLAC",
            bit_depth=16,
            sample_rate=44100,
            channels=2,
            duration=180,
            title="Faixa",
            track_number=1,
        )
    )
    defaults = {
        "artist": "Artista",
        "artists": ["Artista"],
        "album": "Disco",
        "year": "2011",
        "media": "WEB",
        "release_type": "Album",
        "genres": ["art rock"],
    }
    defaults.update(fields)
    for name, value in defaults.items():
        release.set_field(name, value, MetadataSource.FILE_TAG, 1.0)
    return release


def test_authorization_uses_the_raw_api_key() -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})

    assert tracker._headers(Meta(category="MUSIC"))["Authorization"] == "secret-key"  # noqa: S101


def test_payload_uses_redacted_edition_fields_and_blu_ray_spelling() -> None:
    release = _release(media="Blu-ray", release_label="Selo", release_catalogue_number="ABC-1", edition="Deluxe", edition_year="2024")
    payload = Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC", artwork_url="https://images.example/cover.jpg"), release)

    assert payload["media"] == "Blu-Ray"  # noqa: S101
    assert payload["releasetype"] == 1  # noqa: S101
    assert payload["remaster_year"] == "2024"  # noqa: S101
    assert payload["remaster_title"] == "Deluxe"  # noqa: S101
    assert payload["remaster_record_label"] == "Selo"  # noqa: S101
    assert payload["remaster_catalogue_number"] == "ABC-1"  # noqa: S101
    assert "record_label" not in payload  # noqa: S101
    assert "catalogue_number" not in payload  # noqa: S101
    assert "remaster" not in payload  # noqa: S101
    assert "scene" not in payload  # noqa: S101
    assert "unknown" not in payload  # noqa: S101
    assert payload["tags"] == "art.rock"  # noqa: S101
    assert "processed_by" not in payload  # noqa: S101


def test_descriptions_do_not_credit_upload_assistant() -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {}}})
    tracker._spectrogram_bbcode = "[b]Spectrograms[/b]\n\n[img]https://redacted.sh/image/full.png[/img]"
    payload = tracker.build_upload_payload(Meta(category="MUSIC", ua_signature="Uploaded with Upload Assistant"), _release())
    text = f"{payload['album_desc']}\n{payload['release_desc']}"

    assert "Upload Assistant" not in text  # noqa: S101
    assert "processed_by" not in payload  # noqa: S101


def test_unknown_release_sets_the_checkbox_and_http_covers_are_omitted() -> None:
    release = _release(release_type="Unknown")
    payload = Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC", artwork_url="http://images.example/cover.jpg", scene=True), release)

    assert payload["releasetype"] == 21  # noqa: S101
    assert payload["unknown"] == "on"  # noqa: S101
    assert payload["scene"] == "on"  # noqa: S101
    assert "image" not in payload  # noqa: S101


def test_dsd_bitrate_uses_the_nearest_named_rate() -> None:
    release = _release()
    release.tracks[0].format = "DSD"
    release.tracks[0].path = "track.dff"
    release.tracks[0].sample_rate = 5_644_800

    bitrate, other, vbr = Redacted({"TRACKERS": {"REDACTED": {}}})._encoding(release, "DSD")

    assert (bitrate, other, vbr) == ("DSD128", "", False)  # noqa: S101


def test_do_not_upload_names_block_and_bellum_is_cd_only() -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {}}})
    blocked = _release(artist="Paul_DVR", artists=["Paul_DVR"])
    meta = Meta(category="MUSIC", music_release=blocked.to_dict())
    assert asyncio.run(tracker.get_additional_checks(meta)) is False  # noqa: S101

    cd = _release(release_label="BELLUM", media="CD")
    web = _release(release_label="BELLUM", media="WEB")
    assert tracker._blocked_labels(cd) == ["BELLUM"]  # noqa: S101
    assert tracker._blocked_labels(web) == []  # noqa: S101


def test_ogg_is_rejected_and_dsd_container_is_required() -> None:
    ogg = _release()
    ogg.tracks[0].format = "Ogg Vorbis"
    messages = [issue.message for issue in RedactedMusicValidator().validate(ogg) if issue.level.value == "error"]
    assert any("not an allowed Redacted music format" in message for message in messages)  # noqa: S101

    dsd = _release()
    dsd.tracks[0].format = "DSD"
    dsd.tracks[0].path = "track.dsf"
    assert not any(issue.code == "unsupported_format" for issue in RedactedMusicValidator().validate(dsd))  # noqa: S101


def test_image_host_response_accepts_url_and_bbcode() -> None:
    assert image_url_from_response({"status": "success", "response": "https://redacted.sh/image/a.png"}) == "https://redacted.sh/image/a.png"  # noqa: S101
    assert image_url_from_response({"status": "success", "response": {"bbcode": "[img]https://redacted.sh/image/b.png[/img]"}}) == "https://redacted.sh/image/b.png"  # noqa: S101
    with pytest.raises(ValueError, match="rejected"):
        image_url_from_response({"status": "failure", "error": "rejected"})


def test_sox_commands_match_the_full_and_zoom_plots(tmp_path: Path) -> None:
    source = tmp_path / "faixa.flac"
    full = tmp_path / "full.png"
    zoom = tmp_path / "zoom.png"

    assert sox_spectrogram_command("sox", source, full) == [  # noqa: S101
        "sox",
        str(source),
        "-n",
        "remix",
        "1",
        "spectrogram",
        "-x",
        "3000",
        "-y",
        "513",
        "-z",
        "120",
        "-w",
        "Kaiser",
        "-o",
        str(full),
    ]
    zoom_command = sox_spectrogram_command("sox", source, zoom, zoom=("1:00", "0:02"))
    assert zoom_command[-14:] == [  # noqa: S101
        "-X",
        "500",
        "-y",
        "1025",
        "-z",
        "120",
        "-w",
        "Kaiser",
        "-S",
        "1:00",
        "-d",
        "0:02",
        "-o",
        str(zoom),
    ]
    assert zoom_window(30) == ("0:00", "2.00")  # noqa: S101
    assert zoom_window(0.2) is None  # noqa: S101
    assert output_name(1, "Faixa 01", "full") == "01-Faixa-01-full.png"  # noqa: S101
    assert "[img]https://redacted.sh/image/full.png[/img]" in spectrogram_bbcode([("Faixa", "https://redacted.sh/image/full.png", "https://redacted.sh/image/zoom.png")])  # noqa: S101


def test_comment_urls_keep_the_torrent_id() -> None:
    client = Clients({"TRACKERS": {}})

    assert client._extract_tracker_ids_from_comment("https://redacted.sh/torrents.php?id=9&torrentid=42") == {"redacted": "42"}  # noqa: S101
    assert client._extract_tracker_ids_from_comment("https://redacted.ch/torrents.php?torrentid=7") == {"redacted": "7"}  # noqa: S101


def test_base_url_stays_on_redacted() -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"base_url": "https://not-redacted.example"}}})

    assert tracker.base_url == "https://redacted.sh"  # noqa: S101
