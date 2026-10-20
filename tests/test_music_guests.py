"""Guest performers from file tags, Qobuz credits and Deezer contributors."""

from __future__ import annotations

from src.meta import Meta
from src.music.guests import (
    choose_qobuz_url,
    credits_for_release,
    featured_names,
    format_guest_artists,
    guests_from_tags,
    parse_qobuz_album,
    qobuz_album_id,
    store_tag_guests,
)
from src.music.guests import _choose_deezer_album as choose_deezer_album
from src.music.models import AudioTrack, MetadataSource, MusicRelease
from src.trackers.GAZELLE.redacted import Redacted
from src.uphelper import _music_confirmation_lines

_QOBUZ_PAGE = """
<div class="track" data-track-v2="{&quot;item_name&quot;:&quot;Homem Invis&#237;vel&quot;}">
  <div class="track__item track__item--number"><span>15</span></div>
  <p class="track__info">Braga, Composer - Pericles, MainArtist - Projota, FeaturedArtist</p>
</div>
<div class="track" data-track-v2="{&quot;item_name&quot;:&quot;Na Broderagem&quot;}">
  <div class="track__item track__item--number"><span>17</span></div>
  <p class="track__info">Pericles, MainArtist - Dayane Camargo, Composer</p>
</div>
"""

_SEARCH_PAGE = """
<script type="application/ld+json">
{"@type":"SearchResultsPage","mainEntity":{"@type":"ItemList","itemListElement":[
  {"@type":"ListItem","url":"https://www.qobuz.com/us-en/album/to-achando-que-e-amor-pericles/ybz9neykcs8rc","item":{"name":"Tô Achando Que É Amor","url":"https://www.qobuz.com/us-en/album/to-achando-que-e-amor-pericles/ybz9neykcs8rc"}},
  {"@type":"ListItem","url":"https://www.qobuz.com/us-en/album/to-achando-que-e-amor-ep-1-pericles/phm89ktrvmkoc","item":{"name":"Tô Achando Que É Amor | Ep. 1","url":"https://www.qobuz.com/us-en/album/to-achando-que-e-amor-ep-1-pericles/phm89ktrvmkoc"}}
]}}
</script>
"""


def _release() -> MusicRelease:
    release = MusicRelease(root=".")
    for number, title in ((15, "Homem Invisível"), (17, "Na Broderagem")):
        release.tracks.append(
            AudioTrack(
                path=f"{number}.flac",
                relative_path=f"{number}.flac",
                format="FLAC",
                codec="FLAC",
                title=title,
                artist="Péricles",
                album_artist="Péricles",
                track_number=number,
                tags={"artist": ["Péricles"], "albumartist": ["Péricles"]},
            )
        )
    release.set_field("artist", "Péricles", MetadataSource.FILE_TAG, 1.0)
    release.set_field("artists", ["Péricles"], MetadataSource.FILE_TAG, 1.0)
    release.set_field("album", "Tô Achando Que É Amor", MetadataSource.FILE_TAG, 1.0)
    return release


def test_qobuz_credit_line_keeps_only_the_featured_artist() -> None:
    assert featured_names("Braga, Composer - Pericles, MainArtist - Projota, FeaturedArtist") == ["Projota"]  # noqa: S101
    assert featured_names("Dayane Camargo, Composer - Izaías Marcelo, Producer") == []  # noqa: S101


def test_qobuz_page_matches_the_local_track_number() -> None:
    release = _release()

    guests = credits_for_release(release, parse_qobuz_album(_QOBUZ_PAGE))

    assert guests == [{"name": "Projota", "tracks": [15]}]  # noqa: S101
    assert qobuz_album_id("https://www.qobuz.com/us-en/album/to-achando-que-e-amor-pericles/ybz9neykcs8rc") == "ybz9neykcs8rc"  # noqa: S101


def test_qobuz_search_ignores_the_ep() -> None:
    from src.music.guests import parse_qobuz_search

    url = choose_qobuz_url(parse_qobuz_search(_SEARCH_PAGE), "Péricles", "Tô Achando Que É Amor")

    assert url.endswith("/ybz9neykcs8rc")  # noqa: S101


def test_deezer_album_must_match_the_track_count() -> None:
    albums = [
        {"id": 1, "title": "Tô Achando Que É Amor | Ep. 1", "nb_tracks": 4, "artist": {"name": "Péricles"}},
        {"id": 2, "title": "Tô Achando Que É Amor", "nb_tracks": 17, "artist": {"name": "Péricles"}},
        {"id": 3, "title": "Tô Achando Que É Amor", "nb_tracks": 17, "artist": {"name": "Outra Pessoa"}},
    ]

    assert choose_deezer_album(albums, "Péricles", "Tô Achando Que É Amor", 17) == "2"  # noqa: S101


def test_a_thin_title_match_does_not_import_guests() -> None:
    release = _release()
    release.tracks.append(
        AudioTrack(path="2.flac", relative_path="2.flac", format="FLAC", codec="FLAC", title="Já Tava Bom", track_number=2, tags={"artist": ["Péricles"]})
    )
    remote = [{"title": "Homem Invisível", "featured": ["Projota"]}]

    assert credits_for_release(release, remote) == []  # noqa: S101


def test_file_tags_keep_a_featured_name_and_ignore_a_composer() -> None:
    release = _release()
    release.tracks[0].tags["artist"] = ["Péricles feat. Projota"]
    release.tracks[0].artist = "Péricles feat. Projota"
    release.tracks[0].tags["composer"] = ["Braga"]

    store_tag_guests(release)

    assert release.get("guest_artists") == [{"name": "Projota", "tracks": [15]}]  # noqa: S101
    assert guests_from_tags(_release()) == []  # noqa: S101


def test_redacted_sends_the_guest_with_guest_importance() -> None:
    release = _release()
    release.set_field("guest_artists", [{"name": "Projota", "tracks": [15]}, {"name": "Péricles", "tracks": [15]}], MetadataSource.EXTERNAL, 0.86)
    release.set_field("year", "2020", MetadataSource.FILE_TAG, 1.0)
    release.set_field("media", "WEB", MetadataSource.USER, 1.0)
    release.set_field("release_type", "Album", MetadataSource.EXTERNAL, 0.7)
    payload = Redacted({"TRACKERS": {"REDACTED": {}}}).build_upload_payload(Meta(category="MUSIC", artwork_url="https://images.example/cover.jpg"), release)

    assert payload["artists[]"] == ["Péricles", "Projota"]  # noqa: S101
    assert payload["importance[]"] == [1, 2]  # noqa: S101


def test_confirmation_shows_the_guest_track() -> None:
    release = _release()
    release.set_field("guest_artists", [{"name": "Projota", "tracks": [15]}], MetadataSource.EXTERNAL, 0.86)
    lines = dict(item for item in _music_confirmation_lines(Meta(category="MUSIC", music_release=release.to_dict()), "MISSING") if isinstance(item, tuple))

    assert format_guest_artists(release.get("guest_artists")) == "Projota (15)"  # noqa: S101
    assert "Projota (15)" in lines["Guests"]  # noqa: S101
    assert "external metadata" in lines["Guests"]  # noqa: S101
