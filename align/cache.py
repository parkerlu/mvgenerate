"""Cache for lyrics alignment results."""
import hashlib
import json
from pathlib import Path

from config import TimedLine, TimedWord

CACHE_DIR = Path(".cache/align")


def _cache_key(audio_path: Path, lyrics_lines: list[str]) -> str:
    """Generate cache key from audio file hash + lyrics content."""
    h = hashlib.md5()
    # Hash audio file content
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    # Hash lyrics
    h.update("\n".join(lyrics_lines).encode("utf-8"))
    return h.hexdigest()


def get_cached(audio_path: Path, lyrics_lines: list[str]) -> list[TimedLine] | None:
    """Return cached alignment result, or None if not cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(audio_path, lyrics_lines)
    cache_file = CACHE_DIR / f"{key}.json"

    if not cache_file.exists():
        return None

    data = json.loads(cache_file.read_text("utf-8"))
    return [
        TimedLine(
            text=line["text"],
            start=line["start"],
            end=line["end"],
            words=[TimedWord(w["text"], w["start"], w["end"]) for w in line.get("words", [])],
        )
        for line in data
    ]


def save_cache(audio_path: Path, lyrics_lines: list[str], timed_lines: list[TimedLine]) -> None:
    """Save alignment result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(audio_path, lyrics_lines)
    cache_file = CACHE_DIR / f"{key}.json"

    data = [
        {
            "text": line.text,
            "start": line.start,
            "end": line.end,
            "words": [{"text": w.text, "start": w.start, "end": w.end} for w in line.words],
        }
        for line in timed_lines
    ]
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
