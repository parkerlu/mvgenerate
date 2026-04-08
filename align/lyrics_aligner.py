# align/lyrics_aligner.py
from pathlib import Path
import mlx_whisper

from config import TimedLine, TimedWord

# MLX Whisper model repo (Apple Silicon optimized)
DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


def _detect_language(lyrics_lines: list[str]) -> str:
    """Auto-detect language from lyrics text. Returns 'zh' for Chinese, 'en' for English, etc."""
    text = "".join(lyrics_lines)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "zh"
    if chinese_chars / total_alpha > 0.3:
        return "zh"
    return "en"


def align_lyrics(audio_path: Path, lyrics_lines: list[str], language: str | None = None, model_repo: str = DEFAULT_MODEL) -> list[TimedLine]:
    """
    Align lyrics lines to audio using Whisper word-level timestamps.

    Strategy: Force Whisper to use the correct language for accurate
    character/word-level timestamps, then distribute lyrics lines
    proportionally based on character count.
    """
    if language is None:
        language = _detect_language(lyrics_lines)

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        word_timestamps=True,
        language=language,
    )

    # Collect all word timestamps (we only care about timing, not text)
    word_times: list[tuple[float, float]] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            start = w.get("start", 0.0)
            end = w.get("end", 0.0)
            if end > start:
                word_times.append((start, end))

    if not word_times:
        duration = result.get("duration", 0.0)
        if not duration and result.get("segments"):
            duration = result["segments"][-1].get("end", 0.0)
        return _even_distribute(lyrics_lines, duration)

    # Detect vocal regions: merge word times into continuous singing regions
    # (gap < 1.5s means same region)
    regions: list[tuple[float, float, int, int]] = []  # (start, end, word_start_idx, word_end_idx)
    region_start_idx = 0
    region_start = word_times[0][0]
    region_end = word_times[0][1]

    for i in range(1, len(word_times)):
        if word_times[i][0] - region_end > 1.5:
            # New region
            regions.append((region_start, region_end, region_start_idx, i))
            region_start_idx = i
            region_start = word_times[i][0]
        region_end = word_times[i][1]
    regions.append((region_start, region_end, region_start_idx, len(word_times)))

    # Calculate character count per line
    line_chars = [len(line.replace(" ", "")) for line in lyrics_lines]
    total_chars = sum(line_chars)

    if total_chars == 0:
        return _even_distribute(lyrics_lines, word_times[-1][1])

    # Total number of word slots available
    total_words = len(word_times)

    # Assign word slots to each line proportionally by character count
    timed_lines: list[TimedLine] = []
    word_idx = 0

    for i, line in enumerate(lyrics_lines):
        chars = line_chars[i]
        if chars == 0:
            # Empty line gets a tiny slot
            if timed_lines:
                t = timed_lines[-1].end
            else:
                t = word_times[0][0]
            timed_lines.append(TimedLine(text=line, start=t, end=t + 0.1))
            continue

        # How many word slots this line gets
        remaining_chars = sum(line_chars[i:])
        remaining_words = total_words - word_idx
        word_count = max(1, round(remaining_words * chars / remaining_chars))
        word_count = min(word_count, remaining_words)

        # Get time range from assigned word slots
        start_idx = word_idx
        end_idx = min(word_idx + word_count, total_words) - 1

        line_start = word_times[start_idx][0]
        line_end = word_times[end_idx][1]

        timed_lines.append(TimedLine(text=line, start=line_start, end=line_end))
        word_idx += word_count

    # Handle any remaining unassigned time
    if timed_lines and word_idx < total_words:
        timed_lines[-1].end = word_times[-1][1]

    return timed_lines


def _even_distribute(lyrics_lines: list[str], duration: float) -> list[TimedLine]:
    """Fallback: evenly space lyrics across the audio duration."""
    if not lyrics_lines:
        return []
    line_duration = duration / len(lyrics_lines)
    return [
        TimedLine(
            text=line,
            start=i * line_duration,
            end=(i + 1) * line_duration,
        )
        for i, line in enumerate(lyrics_lines)
    ]
