"""Orchestrate lyric alignment: Whisper transcribe → DP align → LLM refine."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import mlx_whisper

from config import TimedLine
from align import sequence_aligner, llm_fallback
from align.sequence_aligner import Segment, LineAlignment
from align.llm_fallback import LLMProvider, DeepSeekProvider, ClaudeProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def _detect_language(lyrics_lines: list[str]) -> str:
    text = "".join(lyrics_lines)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "zh"
    if chinese_chars / total_alpha > 0.3:
        return "zh"
    return "en"


def _whisper_to_segments(result: dict) -> list[Segment]:
    segments: list[Segment] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        if text and end > start:
            segments.append(Segment(text=text, start=start, end=end))
    return segments


def _llm_enabled() -> bool:
    return os.environ.get("LYRICS_ALIGNER_LLM_ENABLED", "true").lower() != "false"


def _get_provider() -> LLMProvider | None:
    try:
        name = os.environ.get("LYRICS_ALIGNER_LLM_PROVIDER", "deepseek").lower()
        if name == "claude":
            return ClaudeProvider()
        return DeepSeekProvider()
    except Exception as e:
        logger.warning(f"LLM provider unavailable ({e}), skipping fallback")
        return None


def _emit_timed_lines(
    lyric_lines: list[str],
    alignments: list[LineAlignment],
    segments: list[Segment],
    total_duration: float,
) -> list[TimedLine]:
    n = len(lyric_lines)
    # First pass: resolve matched lines to concrete (start, end) tuples;
    # leave unmatched lines as None placeholders.
    slots: list[tuple[float, float] | None] = [None] * n
    for i in range(n):
        idxs = alignments[i].segment_idxs
        if idxs:
            slots[i] = (
                min(segments[k].start for k in idxs),
                max(segments[k].end for k in idxs),
            )

    # Second pass: distribute contiguous runs of unmatched lines evenly
    # across the gap between surrounding matched lines (or audio bounds).
    i = 0
    while i < n:
        if slots[i] is not None:
            i += 1
            continue
        run_start = i
        while i < n and slots[i] is None:
            i += 1
        run_end = i - 1
        gap_start = slots[run_start - 1][1] if run_start > 0 else 0.0
        gap_end = slots[i][0] if i < n else total_duration
        if gap_end < gap_start:
            gap_end = gap_start
        span = gap_end - gap_start
        count = run_end - run_start + 1
        slot_len = span / count if count > 0 else 0.0
        for k, line_idx in enumerate(range(run_start, run_end + 1)):
            s = gap_start + k * slot_len
            e = s + slot_len
            slots[line_idx] = (s, e)

    # Third pass: final monotonicity clamp as a belt-and-suspenders guard
    # against any upstream inconsistency.
    timed: list[TimedLine] = []
    for i, line in enumerate(lyric_lines):
        start, end = slots[i]  # type: ignore[misc]
        if timed and start < timed[-1].end:
            start = timed[-1].end
            if end < start:
                end = start
        timed.append(TimedLine(text=line, start=start, end=end))
    return timed


def align_lyrics(
    audio_path: Path,
    lyrics_lines: list[str],
    language: str | None = None,
    model_repo: str = DEFAULT_MODEL,
) -> list[TimedLine]:
    """Align user-provided lyric lines to audio timeline.

    Flow:
    1. Whisper transcribes audio → Segment list (text + timing).
    2. Classical DP aligns lyric_lines to segments, produces per-line confidences.
    3. Low-confidence windows (< 0.55) are sent to LLM for semantic refinement.
    4. Time ranges are read from matched segments (LLM never invents timestamps).

    Caller must pre-clean lyrics (strip section markers / blank lines) via
    `align/lyrics_preprocessor.py` — this function assumes `lyrics_lines`
    contains only real lyric lines.
    """
    if not lyrics_lines:
        return []

    if language is None:
        language = _detect_language(lyrics_lines)

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        word_timestamps=True,
        language=language,
    )

    segments = _whisper_to_segments(result)
    total_duration = result.get("duration", 0.0)
    if not total_duration and segments:
        total_duration = segments[-1].end

    if not segments:
        logger.warning("Whisper produced no segments, falling back to even distribution")
        return _even_distribute(lyrics_lines, total_duration)

    alignments, confidences = sequence_aligner.align(lyrics_lines, segments)

    if _llm_enabled() and any(c < 0.55 for c in confidences):
        provider = _get_provider()
        if provider is not None:
            try:
                alignments = llm_fallback.refine(
                    lyrics_lines, segments, alignments, confidences, provider
                )
            except Exception as e:
                logger.warning(f"LLM fallback crashed ({e}), using classical result")

    return _emit_timed_lines(lyrics_lines, alignments, segments, total_duration)


def _even_distribute(lyrics_lines: list[str], duration: float) -> list[TimedLine]:
    if not lyrics_lines:
        return []
    line_duration = duration / len(lyrics_lines) if duration > 0 else 1.0
    return [
        TimedLine(text=line, start=i * line_duration, end=(i + 1) * line_duration)
        for i, line in enumerate(lyrics_lines)
    ]
