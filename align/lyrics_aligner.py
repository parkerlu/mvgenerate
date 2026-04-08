# align/lyrics_aligner.py
from pathlib import Path
from faster_whisper import WhisperModel

from config import TimedLine, TimedWord


def align_lyrics(audio_path: Path, lyrics_lines: list[str], model_size: str = "large-v3") -> list[TimedLine]:
    """
    Align lyrics lines to audio using Whisper forced alignment.

    1. Transcribe the audio with Whisper (word-level timestamps).
    2. Match each provided lyrics line against the transcription.
    3. Return TimedLine objects with start/end times and word timestamps.
    """
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    segments, _info = model.transcribe(
        str(audio_path),
        language=None,  # auto-detect
        word_timestamps=True,
    )

    # Collect all transcribed words with timestamps
    transcribed_words: list[TimedWord] = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                transcribed_words.append(TimedWord(
                    text=w.word.strip(),
                    start=w.start,
                    end=w.end,
                ))

    if not transcribed_words:
        # Fallback: evenly distribute lyrics across audio duration
        return _even_distribute(lyrics_lines, _info.duration)

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
