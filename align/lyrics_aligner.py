# align/lyrics_aligner.py
from pathlib import Path
import mlx_whisper

from config import TimedLine, TimedWord

# MLX Whisper model repo (Apple Silicon optimized)
DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


def align_lyrics(audio_path: Path, lyrics_lines: list[str], model_repo: str = DEFAULT_MODEL) -> list[TimedLine]:
    """
    Align lyrics lines to audio using MLX Whisper (Apple Silicon).

    1. Transcribe the audio with Whisper (word-level timestamps).
    2. Match each provided lyrics line against the transcription.
    3. Return TimedLine objects with start/end times and word timestamps.
    """
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        word_timestamps=True,
    )

    # Collect all transcribed words with timestamps
    transcribed_words: list[TimedWord] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            text = w.get("word", "").strip()
            if text:
                transcribed_words.append(TimedWord(
                    text=text,
                    start=w["start"],
                    end=w["end"],
                ))

    if not transcribed_words:
        # Fallback: evenly distribute lyrics across audio duration
        duration = result.get("duration", 0.0)
        if not duration and "segments" in result and result["segments"]:
            duration = result["segments"][-1].get("end", 0.0)
        return _even_distribute(lyrics_lines, duration)

    # Match lyrics lines to transcribed words using greedy alignment
    timed_lines = _match_lines_to_words(lyrics_lines, transcribed_words)
    return timed_lines


def _match_lines_to_words(lyrics_lines: list[str], words: list[TimedWord]) -> list[TimedLine]:
    """
    Greedy matching: for each lyrics line, find the best matching span
    in the transcribed words by character overlap.
    """
    timed_lines: list[TimedLine] = []
    word_idx = 0

    for line in lyrics_lines:
        line_chars = line.replace(" ", "").lower()
        if not line_chars:
            continue

        best_start_idx = word_idx
        best_score = 0
        best_end_idx = word_idx

        # Sliding window over remaining words
        for start in range(word_idx, len(words)):
            matched_chars = ""
            for end in range(start, min(start + len(line_chars) * 2, len(words))):
                matched_chars += words[end].text.replace(" ", "").lower()
                # Calculate overlap score
                score = _char_overlap(line_chars, matched_chars)
                if score > best_score:
                    best_score = score
                    best_start_idx = start
                    best_end_idx = end + 1
                # Early exit if perfect match
                if score >= len(line_chars):
                    break

        if best_score > 0 and best_start_idx < len(words):
            matched_words = words[best_start_idx:best_end_idx]
            timed_lines.append(TimedLine(
                text=line,
                start=matched_words[0].start,
                end=matched_words[-1].end,
                words=matched_words,
            ))
            word_idx = best_end_idx
        else:
            # No match found — will be filled in by gap filling later
            timed_lines.append(TimedLine(text=line, start=0.0, end=0.0, words=[]))

    # Fill gaps for unmatched lines
    _fill_gaps(timed_lines, words[-1].end if words else 0.0)

    return timed_lines


def _char_overlap(a: str, b: str) -> int:
    """Count matching characters between two strings (order-sensitive)."""
    matches = 0
    b_idx = 0
    for ch in a:
        while b_idx < len(b):
            if b[b_idx] == ch:
                matches += 1
                b_idx += 1
                break
            b_idx += 1
    return matches


def _fill_gaps(lines: list[TimedLine], total_duration: float) -> None:
    """Fill start/end times for lines that had no match, using surrounding lines."""
    for i, line in enumerate(lines):
        if line.start == 0.0 and line.end == 0.0:
            prev_end = lines[i - 1].end if i > 0 else 0.0
            next_start = total_duration
            for j in range(i + 1, len(lines)):
                if lines[j].start > 0:
                    next_start = lines[j].start
                    break
            gap = next_start - prev_end
            line.start = prev_end
            line.end = prev_end + gap


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
