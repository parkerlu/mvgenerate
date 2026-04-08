import re
from pathlib import Path

# Matches lines that are ONLY a section marker like [Verse], [Chorus 2], [Pre-Chorus], etc.
_SECTION_MARKER = re.compile(r"^\[[\w\s\-]+\]$")


def preprocess_lyrics(raw_text: str) -> list[str]:
    """Clean raw lyrics text: remove section markers, blank lines, trim whitespace."""
    lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION_MARKER.match(stripped):
            continue
        lines.append(stripped)
    return lines


def preprocess_lyrics_file(path: Path) -> list[str]:
    """Read a lyrics file and preprocess it."""
    text = path.read_text(encoding="utf-8")
    return preprocess_lyrics(text)
