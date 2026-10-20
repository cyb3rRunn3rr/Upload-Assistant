"""SoX spectrogram commands used for Redacted music descriptions.

Each audio file gets the full plot and the short zoom from Redacted's spectral
analysis page. The pictures are written outside the album so they stay out of
the torrent.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

FULL_ARGS = ("-x", "3000", "-y", "513", "-z", "120", "-w", "Kaiser")
ZOOM_ARGS = ("-X", "500", "-y", "1025", "-z", "120", "-w", "Kaiser")


def zoom_window(duration: float | None) -> tuple[str, str] | None:
    """Return SoX ``-S`` and ``-d`` values for the zoom plot.

    The usual window is 1:00 for two seconds. A shorter file starts at the
    beginning. A clip under half a second has no useful zoom.
    """
    if duration is None or duration >= 62:
        return ("1:00", "0:02")
    if duration < 0.5:
        return None
    return ("0:00", f"{min(2.0, duration):.2f}")


def sox_spectrogram_command(sox: str, source: Path, output: Path, *, zoom: tuple[str, str] | None = None) -> list[str]:
    """Build one SoX spectrogram command. ``zoom`` selects the 2 second plot."""
    command = [sox, str(source), "-n", "remix", "1", "spectrogram"]
    if zoom is None:
        command.extend(FULL_ARGS)
    else:
        start, length = zoom
        command.extend(ZOOM_ARGS)
        command.extend(("-S", start, "-d", length))
    command.extend(("-o", str(output)))
    return command


def render_spectrogram(sox: str, source: Path, output: Path, *, zoom: tuple[str, str] | None = None) -> None:
    """Run SoX and leave ``output`` on disk."""
    command = sox_spectrogram_command(sox, source, output, zoom=zoom)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Could not run sox for {source.name}: {error}") from error
    if result.returncode != 0 or not output.is_file():
        detail = (result.stderr or result.stdout or "sox produced no image").strip()
        raise RuntimeError(f"sox failed for {source.name}: {detail[:500]}")


def spectrogram_bbcode(rows: list[tuple[str, str, str]]) -> str:
    """BBCode for release descriptions. Each row is title, full URL, zoom URL.

    The plots sit in one hide. Each zoom has its own hide, so a long album
    does not open as a wall of images.
    """
    blocks: list[str] = []
    for title, full_url, zoom_url in rows:
        if not full_url and not zoom_url:
            continue
        lines = [f"[b]{_bbcode_text(title)}[/b]"]
        if full_url:
            lines.append(f"[img]{full_url}[/img]")
        if zoom_url:
            lines.append(f"[hide=Zoom]\n[img]{zoom_url}[/img]\n[/hide]")
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return "[hide=Spectrograms]\n" + "\n\n".join(blocks) + "\n[/hide]"


def output_name(index: int, title: str, kind: str) -> str:
    """Stable file name that cannot escape the output directory."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-._") or "track"
    return f"{index:02d}-{cleaned[:60]}-{kind}.png"


def _bbcode_text(value: str) -> str:
    return value.replace("[", "(").replace("]", ")")
