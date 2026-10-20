"""Redacted form mapping, image-host parsing and SoX spectrogram commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from src.clients import Clients
from src.meta import Meta
from src.music.analyzer import MusicReleaseAnalyzer
from src.music.models import AudioTrack, MetadataSource, MusicRelease
from src.trackerhandle import record_upload_duration
from src.trackers.GAZELLE import redacted as redacted_module
from src.trackers.GAZELLE.red_images import authkey_from_index, image_url_from_response
from src.trackers.GAZELLE.red_spectrograms import output_name, sox_spectrogram_command, spectrogram_bbcode, zoom_window
from src.trackers.GAZELLE.redacted import IMAGE_BATCH_PAUSE_SECONDS, IMAGE_BATCH_SIZE, Redacted, RedactedMusicValidator
from src.webui_progress import clear_progress_callback, set_progress_callback


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


def test_http_upload_error_keeps_the_tracker_reason() -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    request = httpx.Request("POST", "https://redacted.sh/ajax.php?action=upload")
    response = httpx.Response(403, json={"status": "failure", "error": "bad api scope"}, request=request)
    error = httpx.HTTPStatusError("forbidden", request=request, response=response)

    assert tracker._status_error(error) == "HTTP 403: bad api scope"  # noqa: S101


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
    assert payload["remaster"] == "on"  # noqa: S101
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


def test_unknown_release_type_still_sends_a_real_edition() -> None:
    release = _release(release_type="Ep")
    payload = Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC", artwork_url="http://images.example/cover.jpg", scene=True), release)

    assert payload["releasetype"] == 5  # noqa: S101
    assert payload["remaster"] == "on"  # noqa: S101
    assert "remaster_title" not in payload  # noqa: S101
    assert "unknown" not in payload  # noqa: S101
    assert "image" not in payload  # noqa: S101
    assert payload["scene"] == "on"  # noqa: S101


def test_hybrid_note_names_the_16_and_24_bit_tracks() -> None:
    release = _release()
    release.tracks.append(
        AudioTrack(
            path="02.flac",
            relative_path="02.flac",
            format="FLAC",
            codec="FLAC",
            bit_depth=24,
            sample_rate=48000,
            channels=2,
            title="Outra",
            track_number=2,
        )
    )
    payload = Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC"), release)
    description = str(payload["release_desc"])

    assert "Track 1 is 16-bit" in description  # noqa: S101
    assert "Track 2 is 24-bit 48 kHz" in description  # noqa: S101
    assert payload["bitrate"] == "24bit Lossless"  # noqa: S101


def test_accent_difference_is_noted_and_the_title_is_not_rewritten() -> None:
    release = _release()
    release.tracks[0].title = "Jogo de Sedução (Acústico)"
    release.tracks[0].track_number = 1
    release.tracks.append(
        AudioTrack(
            path="03.flac",
            relative_path="03.flac",
            format="FLAC",
            codec="FLAC",
            bit_depth=24,
            sample_rate=48000,
            channels=2,
            title="Se Eu Largar o Freio (Acustico)",
            track_number=3,
        )
    )
    payload = Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC"), release)
    description = str(payload["release_desc"])

    assert "Track 3 uses (Acustico)" in description  # noqa: S101
    assert "Left as supplied." in description  # noqa: S101
    assert "remaster_title" not in payload  # noqa: S101


def test_archives_and_low_bitrate_block_and_official_16bit_hires_warns() -> None:
    archived = _release()
    archived.auxiliary.other.append("booklet.zip")
    assert any(issue.code == "archived_file" for issue in RedactedMusicValidator().validate(archived))  # noqa: S101

    quiet = _release()
    quiet.tracks[0].format = "MP3"
    quiet.tracks[0].path = "track.mp3"
    quiet.tracks[0].bitrate = 128_000
    quiet.tracks[0].bitrate_mode = "CBR"
    assert any(issue.code == "low_bitrate" and issue.level.value == "error" for issue in RedactedMusicValidator().validate(quiet))  # noqa: S101

    hires = _release()
    hires.tracks[0].sample_rate = 96000
    assert any(issue.code == "16bit_high_rate" and issue.level.value == "warning" for issue in RedactedMusicValidator().validate(hires))  # noqa: S101


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


def test_index_authkey_ignores_the_passkey() -> None:
    assert authkey_from_index({"status": "success", "response": {"authkey": "session-auth", "passkey": "secret-pass"}}) == "session-auth"  # noqa: S101
    assert authkey_from_index({"status": "failure", "error": "bad credentials"}) == ""  # noqa: S101


class _ImageClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.posts: list[dict[str, str]] = []
        self.gets = 0

    async def get(self, url: str) -> httpx.Response:
        del url
        self.gets += 1
        return self._responses.pop(0)

    async def post(self, url: str, data: dict[str, str] | None = None, files: object = None) -> httpx.Response:
        del url, files
        self.posts.append(dict(data or {}))
        return self._responses.pop(0)


def _response(status_code: int, payload: dict[str, object]) -> httpx.Response:
    request = httpx.Request("POST", "https://redacted.sh/ajax.php")
    return httpx.Response(status_code, json=payload, request=request)


def _hosted() -> httpx.Response:
    return _response(200, {"status": "success", "response": {"url": "https://redacted.sh/image/x.png"}})


def test_image_uploads_send_the_user_auth_and_pause_after_three(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    tracker._image_auth = "session-auth"
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(redacted_module.asyncio, "sleep", fake_sleep)
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"\xff\xd8\xff")
    client = _ImageClient([_hosted() for _ in range(IMAGE_BATCH_SIZE + 1)])

    async def send_batch() -> None:
        for _ in range(IMAGE_BATCH_SIZE + 1):
            hosted = await tracker._upload_image_file(Meta(category="MUSIC"), cover, client)  # pyright: ignore[reportArgumentType]
            assert hosted == "https://redacted.sh/image/x.png"  # noqa: S101

    asyncio.run(send_batch())

    assert sleeps == [IMAGE_BATCH_PAUSE_SECONDS]  # noqa: S101
    assert len(client.posts) == IMAGE_BATCH_SIZE + 1  # noqa: S101
    assert all(post["auth"] == "session-auth" for post in client.posts)  # noqa: S101


def test_missing_user_session_skips_image_upload(tmp_path: Path) -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"\xff\xd8\xff")
    client = _ImageClient([])
    meta = Meta(category="MUSIC", artwork_path=str(cover), artwork_url="https://images.example/cover.jpg")

    asyncio.run(tracker._rehost_cover(meta, client))  # pyright: ignore[reportArgumentType]

    assert client.posts == []  # noqa: S101
    assert meta.artwork_url == "https://images.example/cover.jpg"  # noqa: S101


def test_image_session_keeps_only_the_auth_key() -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    client = _ImageClient([_response(200, {"status": "success", "response": {"authkey": "session-auth", "passkey": "secret-pass"}})])

    ready = asyncio.run(tracker._prepare_image_session(client))  # pyright: ignore[reportArgumentType]

    assert ready is True  # noqa: S101
    assert tracker._image_auth == "session-auth"  # noqa: S101
    assert "secret-pass" not in tracker.__dict__.values()  # noqa: S101


def test_rate_limit_waits_and_retries_one_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    tracker._image_auth = "session-auth"
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(redacted_module.asyncio, "sleep", fake_sleep)
    cover = tmp_path / "cover.png"
    cover.write_bytes(b"\x89PNG")
    client = _ImageClient([_response(429, {"status": "failure", "error": "rate limit"}), _hosted()])

    hosted = asyncio.run(tracker._upload_image_file(Meta(category="MUSIC"), cover, client))  # pyright: ignore[reportArgumentType]

    assert hosted == "https://redacted.sh/image/x.png"  # noqa: S101
    assert sleeps == [IMAGE_BATCH_PAUSE_SECONDS]  # noqa: S101
    assert len(client.posts) == 2  # noqa: S101


def test_image_progress_counts_each_upload_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    tracker._image_auth = "session-auth"
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(redacted_module.asyncio, "sleep", fake_sleep)
    events: list[dict[str, object]] = []
    set_progress_callback(lambda event: events.append(dict(event)))
    try:
        cover = tmp_path / "cover.jpg"
        cover.write_bytes(b"\xff\xd8\xff")
        client = _ImageClient([_hosted() for _ in range(4)])
        tracker._begin_image_progress(4)

        async def send_batch() -> None:
            for _ in range(4):
                hosted = await tracker._upload_image_file(Meta(category="MUSIC"), cover, client)  # pyright: ignore[reportArgumentType]
                assert hosted == "https://redacted.sh/image/x.png"  # noqa: S101

        asyncio.run(send_batch())
        tracker._finish_image_progress()
    finally:
        clear_progress_callback()

    steps = [(event.get("current"), event.get("total"), event.get("status")) for event in events]
    assert steps == [  # noqa: S101
        (0.0, 4.0, "running"),
        (1.0, 4.0, "running"),
        (2.0, 4.0, "running"),
        (3.0, 4.0, "running"),
        (4.0, 4.0, "running"),
        (4.0, 4.0, "completed"),
    ]
    assert sleeps == [IMAGE_BATCH_PAUSE_SECONDS]  # noqa: S101
    assert events[0]["label"] == "Redacted images"  # noqa: S101
    assert events[0]["detail"] == "This can take a while."  # noqa: S101
    assert events[0]["group"] == "images"  # noqa: S101
    assert events[0]["unit"] == "images"  # noqa: S101


def test_image_batches_post_three_together_then_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        if seconds == 0:
            await real_sleep(0)
            return
        sleeps.append(seconds)

    monkeypatch.setattr(redacted_module.asyncio, "sleep", fake_sleep)
    inflight = 0
    peak = 0

    def make_job():
        async def send() -> None:
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(0)
            inflight -= 1

        return send

    asyncio.run(tracker._run_image_batches([make_job() for _ in range(4)]))

    assert peak == 3  # noqa: S101
    assert sleeps == [IMAGE_BATCH_PAUSE_SECONDS]  # noqa: S101
    assert IMAGE_BATCH_PAUSE_SECONDS == 4  # noqa: S101


def test_image_progress_retry_stays_on_the_same_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    tracker._image_auth = "session-auth"

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(redacted_module.asyncio, "sleep", fake_sleep)
    events: list[dict[str, object]] = []
    set_progress_callback(lambda event: events.append(dict(event)))
    try:
        cover = tmp_path / "cover.png"
        cover.write_bytes(b"\x89PNG")
        client = _ImageClient([_response(429, {"status": "failure", "error": "rate limit"}), _hosted()])
        tracker._begin_image_progress(1)
        hosted = asyncio.run(tracker._upload_image_file(Meta(category="MUSIC"), cover, client))  # pyright: ignore[reportArgumentType]
    finally:
        clear_progress_callback()

    assert hosted == "https://redacted.sh/image/x.png"  # noqa: S101
    assert [event.get("current") for event in events] == [0.0, 1.0]  # noqa: S101
    assert len(client.posts) == 2  # noqa: S101


def test_failed_spectrogram_render_drops_out_of_the_progress_total(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "secret-key"}}})
    tracker._image_auth = "session-auth"

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("sox failed")

    monkeypatch.setattr(redacted_module, "render_spectrogram", boom)
    events: list[dict[str, object]] = []
    set_progress_callback(lambda event: events.append(dict(event)))
    try:
        source = tmp_path / "faixa.flac"
        source.write_bytes(b"fLaC")
        tracker._begin_image_progress(2)

        async def render() -> tuple[str, str]:
            return await tracker._one_spectrogram_pair(
                Meta(category="MUSIC"),  # pyright: ignore[reportArgumentType]
                _ImageClient([]),  # pyright: ignore[reportArgumentType]
                "sox",
                source,
                tmp_path,
                1,
                "Faixa",
                180,
            )

        assert asyncio.run(render()) == ("", "")  # noqa: S101
        assert tracker._image_progress_total == 0  # noqa: S101
        tracker._finish_image_progress()
    finally:
        clear_progress_callback()

    assert events[-1]["status"] == "completed"  # noqa: S101
    assert events[-1]["current"] == 0  # noqa: S101


def test_planned_images_count_the_cover_and_each_plot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(redacted_module.shutil, "which", lambda _name: "/usr/bin/sox")
    long = tmp_path / "long.flac"
    short = tmp_path / "short.flac"
    long.write_bytes(b"fLaC")
    short.write_bytes(b"fLaC")
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"\xff\xd8\xff")
    release = MusicRelease(root=str(tmp_path))
    release.tracks.extend(
        [
            AudioTrack(path=str(long), relative_path="long.flac", format="FLAC", codec="FLAC", duration=180),
            AudioTrack(path=str(short), relative_path="short.flac", format="FLAC", codec="FLAC", duration=0.2),
            AudioTrack(path=str(tmp_path / "gone.flac"), relative_path="gone.flac", format="FLAC", codec="FLAC", duration=180),
        ]
    )
    tracker = Redacted({"TRACKERS": {"REDACTED": {"add_spectrograms": True}}})
    meta = Meta(category="MUSIC", artwork_path=str(cover), artwork_url="https://images.example/cover.jpg", base_dir=str(tmp_path), uuid="album")

    assert tracker._spectrogram_image_count(release) == 3  # noqa: S101
    assert tracker._planned_image_count(meta, release) == 4  # noqa: S101

    site = Meta(category="MUSIC", artwork_url="https://redacted.sh/image/already.jpg", base_dir=str(tmp_path), uuid="album")
    assert tracker._cover_image_count(site) == 0  # noqa: S101
    quiet = Redacted({"TRACKERS": {"REDACTED": {"add_spectrograms": False}}})
    assert quiet._planned_image_count(meta, release) == 1  # noqa: S101


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


def test_spectrogram_bbcode_hides_the_block_and_each_zoom() -> None:
    text = spectrogram_bbcode(
        [
            ("Faixa", "https://redacted.sh/image/full.png", "https://redacted.sh/image/zoom.png"),
            ("Outra", "https://redacted.sh/image/full-2.png", "https://redacted.sh/image/zoom-2.png"),
            ("Sem imagem", "", ""),
        ]
    )

    assert text.startswith("[hide=Spectrograms]\n")  # noqa: S101
    assert text.endswith("\n[/hide]")  # noqa: S101
    assert text.count("[hide=Zoom]") == 2  # noqa: S101
    assert text.count("[/hide]") == 3  # noqa: S101
    assert "[b]Spectrograms[/b]" not in text  # noqa: S101
    assert "[b]Sem imagem[/b]" not in text  # noqa: S101
    assert text.index("[img]https://redacted.sh/image/full.png[/img]") < text.index("[hide=Zoom]")  # noqa: S101
    assert text.index("[hide=Zoom]") < text.index("[img]https://redacted.sh/image/zoom.png[/img]")  # noqa: S101


def test_more_info_links_the_store_and_musicbrainz() -> None:
    release = _release()
    qobuz = "https://www.qobuz.com/us-en/album/acustico-pericles/rzuxbnsv9ql2a"
    release.set_field("store_url", qobuz, MetadataSource.FILE_TAG, 1.0)
    release.external_ids["musicbrainz_release"] = "4c70ff4d-abc3-43c7-ad66-f7aa803521f1"
    release.tracks[0].tags["url"] = ["https://open.qobuz.com/album/rzuxbnsv9ql2a"]
    tracker = Redacted({"TRACKERS": {"REDACTED": {}}})

    release.tracks[0].tags["comment"] = ["https://www.deezer.com/album/1234567"]
    album = tracker._album_description(release)
    torrent = tracker._release_description(release)

    assert album.count("[b]More info:[/b]") == 1  # noqa: S101
    assert album.index("Qobuz") < album.index("Deezer") < album.index("MusicBrainz")  # noqa: S101
    assert "https://i.ibb.co/39M2cBx9/qobuz.png" in album  # noqa: S101
    assert "https://i.ibb.co/zHbd0GMm/deezer.png" in album  # noqa: S101
    assert "https://i.ibb.co/Y42MftV1/musicbrainz.png" in album  # noqa: S101
    assert "ptpimg.me" not in album  # noqa: S101
    assert album.count("[url=") == 3  # noqa: S101
    assert qobuz in album  # noqa: S101
    assert torrent.count("[b]More info:[/b]") == 1  # noqa: S101
    assert "https://i.ibb.co/zHbd0GMm/deezer.png" in torrent  # noqa: S101
    assert torrent.index("Qobuz") < torrent.index("Deezer") < torrent.index("MusicBrainz")  # noqa: S101
    assert qobuz in torrent  # noqa: S101
    assert "Store listing" not in torrent  # noqa: S101


def test_more_info_skips_an_unknown_page() -> None:
    release = _release()
    release.set_field("store_url", "https://example.com/album/1", MetadataSource.FILE_TAG, 1.0)
    release.tracks[0].tags["www"] = ["https://deezer.com/br/album/42"]
    tracker = Redacted({"TRACKERS": {"REDACTED": {}}})
    album = tracker._album_description(release)
    torrent = tracker._release_description(release)

    assert "[b]More info:[/b]" in album  # noqa: S101
    assert "Deezer" in album  # noqa: S101
    assert "example.com" not in album  # noqa: S101
    assert "[b]More info:[/b]" in torrent  # noqa: S101
    assert "[url=https://deezer.com/br/album/42][img]https://i.ibb.co/zHbd0GMm/deezer.png[/img] Deezer[/url]" in torrent  # noqa: S101
    assert "[url=https://example.com/album/1]Store listing[/url]" in torrent  # noqa: S101
    assert "example.com" not in torrent.split("[b]More info:[/b]", 1)[-1]  # noqa: S101


def test_comment_urls_keep_the_torrent_id() -> None:
    client = Clients({"TRACKERS": {}})

    assert client._extract_tracker_ids_from_comment("https://redacted.sh/torrents.php?id=9&torrentid=42") == {"redacted": "42"}  # noqa: S101
    assert client._extract_tracker_ids_from_comment("https://redacted.ch/torrents.php?torrentid=7") == {"redacted": "7"}  # noqa: S101


def test_success_message_names_redacted() -> None:
    meta = Meta(category="MUSIC")
    tracker = Redacted({"TRACKERS": {"REDACTED": {}}})

    accepted = tracker._record_upload_response(meta, {"status": "success", "response": {"torrentId": 7, "groupId": 3}})
    failed = Meta(category="MUSIC")
    rejected = tracker._record_upload_response(failed, {"status": "failure", "error": "nope"})

    assert accepted  # noqa: S101
    assert meta.tracker_status["REDACTED"]["status_message"] == "Upload accepted by Redacted."  # noqa: S101
    assert rejected is False  # noqa: S101
    assert failed.tracker_status["REDACTED"]["status_message"] == "Redacted rejected the upload request: nope"  # noqa: S101


def test_upload_duration_leaves_out_image_hosting(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = Meta(category="MUSIC")
    meta["REDACTED_image_upload_duration"] = 180.0
    monkeypatch.setattr("src.trackerhandle.time.monotonic", lambda: 186.49)

    record_upload_duration(meta, "REDACTED", 0.0)
    longer = Meta(category="MUSIC")
    longer["REDACTED_image_upload_duration"] = 200.0
    record_upload_duration(longer, "REDACTED", 0.0)

    assert meta["REDACTED_upload_duration"] == pytest.approx(6.49)  # noqa: S101
    assert longer["REDACTED_upload_duration"] == 0.0  # noqa: S101


def test_review_fixes_block_real_rule_breaks() -> None:
    raw_aac = _release()
    raw_aac.tracks[0].format = "AAC"
    raw_aac.tracks[0].path = "song.aac"
    assert any(issue.code == "invalid_container" and issue.level.value == "error" for issue in RedactedMusicValidator().validate(raw_aac))  # noqa: S101

    video = _release()
    video.auxiliary.other.append("concert.mkv")
    assert any(issue.code == "banned_sidecar" for issue in RedactedMusicValidator().validate(video))  # noqa: S101

    sampler = _release(release_type="Sampler")
    assert any(issue.code == "release_type" for issue in RedactedMusicValidator().validate(sampler))  # noqa: S101
    with pytest.raises(ValueError, match="Sampler"):
        Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC"), sampler)

    pressed = _release(year="1957", media="CD")
    assert any(issue.code == "edition_year" for issue in RedactedMusicValidator().validate(pressed))  # noqa: S101
    current = _release(year="2018", media="WEB")
    assert not any(issue.code == "edition_year" for issue in RedactedMusicValidator().validate(current))  # noqa: S101

    short = _release(edition="A", release_catalogue_number="1")
    payload = Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC"), short)
    assert payload["remaster_title"] == "A"  # noqa: S101
    assert payload["remaster_catalogue_number"] == "1"  # noqa: S101

    dxd = _release()
    dxd.tracks[0].bit_depth = 24
    dxd.tracks[0].sample_rate = 352_800
    dxd_issues = RedactedMusicValidator().validate(dxd)
    assert any(issue.code == "dxd_rate" and issue.level.value == "warning" for issue in dxd_issues)  # noqa: S101
    assert not any(issue.code == "sample_rate" and issue.level.value == "error" for issue in dxd_issues)  # noqa: S101

    mixed = _release(media="Vinyl")
    mixed.tracks.append(
        AudioTrack(
            path="02.flac",
            relative_path="02.flac",
            format="FLAC",
            codec="FLAC",
            bit_depth=24,
            sample_rate=96_000,
            channels=2,
            title="Outra",
            track_number=2,
        )
    )
    mixed_issues = RedactedMusicValidator().validate(mixed)
    assert any(issue.code == "hybrid_technical" and issue.level.value == "error" for issue in mixed_issues)  # noqa: S101

    alternate = _release()
    alternate.tracks[0].format = "DSD"
    alternate.tracks[0].path = "track.dff"
    alternate.tracks[0].sample_rate = 3_072_000
    bitrate, other, vbr = Redacted({"TRACKERS": {"REDACTED": {}}})._encoding(alternate, "DSD")
    assert (bitrate, other, vbr) == ("DSD64", "", False)  # noqa: S101


def test_inferred_scene_does_not_mark_the_upload() -> None:
    from src.music.prep import _sync_release_to_meta

    release = _release()
    release.set_field("scene", True, MetadataSource.INFERRED, 0.7)
    meta = Meta(category="MUSIC", scene=False)
    _sync_release_to_meta(meta, release)

    assert meta.scene is False  # noqa: S101


def test_claude_review_keeps_the_real_rule_breaks() -> None:
    web_mix = _release(media="WEB", release_year="2018")
    web_mix.tracks.append(
        AudioTrack(path="02.flac", relative_path="02.flac", format="FLAC", codec="FLAC", bit_depth=24, sample_rate=96_000, channels=2, title="Outra", track_number=2)
    )
    web_issues = RedactedMusicValidator().validate(web_mix)
    assert any(issue.code == "hybrid_technical" and issue.level.value == "warning" for issue in web_issues)  # noqa: S101
    assert not any(issue.code == "hybrid_technical" and issue.level.value == "error" for issue in web_issues)  # noqa: S101

    modes = _release(media="CD", release_year="1998")
    modes.tracks[0].format = "MP3"
    modes.tracks[0].path = "01.mp3"
    modes.tracks[0].bitrate = 320_000
    modes.tracks[0].bitrate_mode = "CBR"
    modes.tracks.append(
        AudioTrack(path="02.mp3", relative_path="02.mp3", format="MP3", codec="MP3", bitrate=245_000, bitrate_mode="VBR", sample_rate=44100, channels=2, title="Outra", track_number=2)
    )
    assert any(issue.code == "hybrid_technical" and issue.level.value == "error" for issue in RedactedMusicValidator().validate(modes))  # noqa: S101

    guessed = _release(release_year="2018")
    guessed.set_field("release_label", "Qobuz", MetadataSource.DIRECTORY, 0.9, force=True)
    guessed.set_field("release_catalogue_number", "FLAC 24-96", MetadataSource.DIRECTORY, 0.9, force=True)
    payload = Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC"), guessed)
    assert "remaster_record_label" not in payload  # noqa: S101
    assert "remaster_catalogue_number" not in payload  # noqa: S101

    mqa = _release(release_year="2018")
    mqa.tracks[0].tags = {"MQAENCODER": ["MQAEncode v1.1, 2"]}
    mqa_issues = RedactedMusicValidator().validate(mqa)
    assert any(issue.code == "mqa_edition" and issue.level.value == "error" for issue in mqa_issues)  # noqa: S101
    titled = _release(release_year="2018", edition="MQA")
    titled.tracks[0].tags = {"MQAENCODER": ["MQAEncode v1.1, 2"]}
    assert not any(issue.code == "mqa_edition" for issue in RedactedMusicValidator().validate(titled))  # noqa: S101

    aac = _release(release_year="2018")
    aac.tracks[0].format = "AAC"
    aac.tracks[0].path = "song.m4a"
    aac.tracks[0].bitrate = 180_000
    aac_issues = RedactedMusicValidator().validate(aac)
    assert any(issue.code == "low_bitrate" and issue.level.value == "warning" for issue in aac_issues)  # noqa: S101
    assert not any(issue.code == "low_bitrate" and issue.level.value == "error" for issue in aac_issues)  # noqa: S101

    lossy = _release(release_year="2018")
    lossy.tracks[0].format = "MP3"
    lossy.tracks[0].path = "01.mp3"
    lossy.tracks[0].bit_depth = None
    lossy.tracks[0].bitrate = 320_000
    lossy.tracks[0].bitrate_mode = "CBR"
    assert "?-bit" not in Redacted({"TRACKERS": {"REDACTED": {}}})._release_description(lossy)  # noqa: S101
    assert "320 kbps CBR" in Redacted({"TRACKERS": {"REDACTED": {}}})._release_description(lossy)  # noqa: S101

    preset = _release()
    preset.tracks[0].format = "MP3"
    preset.tracks[0].encoder_settings = "-V 0"
    preset.tracks[0].bitrate_mode = "VBR"
    preset.tracks[0].bitrate = 245_000
    assert Redacted({"TRACKERS": {"REDACTED": {}}})._encoding(preset, "MP3") == ("V0 (VBR)", "", True)  # noqa: S101

    vinyl_dsd = _release(media="Vinyl", release_year="2018")
    vinyl_dsd.tracks[0].format = "DSD"
    vinyl_dsd.tracks[0].path = "track.dsf"
    vinyl_dsd.tracks[0].sample_rate = 2_822_400
    assert any(issue.code == "dsd_media" for issue in RedactedMusicValidator().validate(vinyl_dsd))  # noqa: S101

    sacd = _release(media="SACD", release_year="2018")
    sacd.tracks[0].format = "DSD"
    sacd.tracks[0].path = "track.dsf"
    sacd.tracks[0].sample_rate = 5_644_800
    assert any(issue.code == "dsd_sacd" for issue in RedactedMusicValidator().validate(sacd))  # noqa: S101

    side = _release(release_year="2018")
    side.auxiliary.other.extend(["bonus.wma", "VIDEO_TS/VTS_01_1.VOB", "scene.r00", "Visit site.url"])
    side_issues = RedactedMusicValidator().validate(side)
    assert any(issue.code == "extra_audio" for issue in side_issues)  # noqa: S101
    assert any(issue.code == "banned_sidecar" for issue in side_issues)  # noqa: S101
    assert any(issue.code == "archived_file" for issue in side_issues)  # noqa: S101
    assert any(issue.code == "shortcut_file" and issue.level.value == "warning" for issue in side_issues)  # noqa: S101

    numbers = _release(release_year="2018")
    numbers.tracks[0].relative_path = "01 - a.flac"
    numbers.tracks[0].disc_number = 1
    numbers.tracks.append(
        AudioTrack(path="01 - b.flac", relative_path="01 - b.flac", format="FLAC", codec="FLAC", bit_depth=16, sample_rate=44100, channels=2, disc_number=2, track_number=1, title="B")
    )
    assert any(issue.code == "duplicate_filename_number" for issue in RedactedMusicValidator().validate(numbers))  # noqa: S101

    dated = _release(year="2018", media="WEB", retail_date="2018-08-03")
    assert not any(issue.code == "edition_year_evidence" for issue in RedactedMusicValidator().validate(dated))  # noqa: S101
    bare_cd = _release(year="1995", media="CD")
    assert any(issue.code == "edition_year_evidence" and issue.level.value == "error" for issue in RedactedMusicValidator().validate(bare_cd))  # noqa: S101

    heavy = _release(release_year="2018")
    heavy.tracks[0].embedded_bytes = 1024 * 1024 + 1
    assert any(issue.code == "embedded_art" for issue in RedactedMusicValidator().validate(heavy))  # noqa: S101

    hosted = Redacted({"TRACKERS": {"REDACTED": {}}})
    hosted._hosted_cover = "https://redacted.sh/image/cover.jpg"
    meta = Meta(category="MUSIC", artwork_url="https://images.example/cover.jpg")
    assert hosted.build_upload_payload(meta, _release(release_year="2018"))["image"] == "https://redacted.sh/image/cover.jpg"  # noqa: S101
    assert meta.artwork_url == "https://images.example/cover.jpg"  # noqa: S101


def test_folder_names_do_not_guess_media_from_letters_inside_words() -> None:
    pulp = MusicRelease(root=".")
    MusicReleaseAnalyzer._derive_from_directory(pulp, "PULP - DIFFERENT CLASS (1995)")
    beach = MusicRelease(root=".")
    MusicReleaseAnalyzer._derive_from_directory(beach, "BEACH HOUSE - BLOOM (2012) [FLAC]")
    web = MusicRelease(root=".")
    MusicReleaseAnalyzer._derive_from_directory(web, "Artist - Album (2014) [WEB FLAC]")
    quality = MusicRelease(root=".")
    MusicReleaseAnalyzer._derive_from_directory(quality, "Artist - Album [FLAC 24-96] [Qobuz]")

    assert pulp.get("media") == ""  # noqa: S101
    assert beach.get("media") == ""  # noqa: S101
    assert web.get("media") == "WEB"  # noqa: S101
    assert quality.get("release_catalogue_number") == ""  # noqa: S101
    assert quality.get("release_label") == ""  # noqa: S101


def test_unsplit_flac_with_a_cue_is_a_warning(tmp_path: Path) -> None:
    cue = tmp_path / "album.cue"
    cue.write_text("TRACK 01 AUDIO\nTRACK 02 AUDIO\n", encoding="utf-8")
    release = _release(release_type="Album", release_year="2018")
    release.root = str(tmp_path)
    release.auxiliary.cues.append("album.cue")

    issues = RedactedMusicValidator().validate(release)

    assert any(issue.code == "unsplit_image" and issue.level.value == "warning" for issue in issues)  # noqa: S101
    assert not any(issue.code == "single_track" for issue in issues)  # noqa: S101


def test_loose_file_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    async def allow(_self: Redacted, _meta: Meta) -> bool:
        return True

    monkeypatch.setattr(Redacted, "get_additional_checks", allow)
    release = _release(release_type="Single", release_year="2018")
    meta = Meta(category="MUSIC", isdir=False, debug=True, music_release=release.to_dict())
    tracker = Redacted({"TRACKERS": {"REDACTED": {"api_key": "k", "announce_url": "https://example.invalid/announce"}}})

    assert asyncio.run(tracker.upload(meta)) is False  # noqa: S101
    assert "folder" in meta.tracker_status["REDACTED"]["status_message"]  # noqa: S101


def test_base_url_stays_on_redacted() -> None:
    tracker = Redacted({"TRACKERS": {"REDACTED": {"base_url": "https://not-redacted.example"}}})

    assert tracker.base_url == "https://redacted.sh"  # noqa: S101
