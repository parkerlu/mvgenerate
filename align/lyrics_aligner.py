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


def _next_matched_start(
    after_idx: int,
    alignments: list[LineAlignment],
    segments: list[Segment],
    total_duration: float,
) -> float:
    for j in range(after_idx + 1, len(alignments)):
        if alignments[j].segment_idxs:
            return min(segments[k].start for k in alignments[j].segment_idxs)
    return total_duration


def _emit_timed_lines(
    lyric_lines: list[str],
    alignments: list[LineAlignment],
    segments: list[Segment],
    total_duration: float,
) -> list[TimedLine]:
    timed: list[TimedLine] = []
    for i, line in enumerate(lyric_lines):
        a = alignments[i]
        if a.segment_idxs:
            start = min(segments[k].start for k in a.segment_idxs)
            end = max(segments[k].end for k in a.segment_idxs)
        else:
            next_start = _next_matched_start(i, alignments, segments, total_duration)
            prev_end = timed[-1].end if timed else 0.0
            start = prev_end
            end = min(next_start, prev_end + 0.1)
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
