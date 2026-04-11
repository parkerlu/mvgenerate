"""Classical sequence alignment between user lyrics and Whisper segments."""
from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class Segment:
    """A Whisper transcription segment with timing."""
    text: str
    start: float
    end: float


def _normalize(s: str) -> str:
    """Keep only CJK chars and alphanumerics; drop punctuation and whitespace."""
    return ''.join(
        c for c in s
        if c.isalnum() or '\u4e00' <= c <= '\u9fff'
    )


def sim(a: str, b: str) -> float:
    """Character-level similarity in [0.0, 1.0]. Punctuation/whitespace insensitive."""
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


_FILLERS = frozenset("嗯啊哦呀呃喔唉嘿哟噢咦哈")


def skip_cost(segment: Segment) -> float:
    """Cost of skipping a Whisper segment in DP alignment.

    Fillers and short segments cost almost nothing to skip; long segments
    cost more, so the DP strongly prefers matching them to some lyric line.
    Returns a negative number (penalty).
    """
    text = _normalize(segment.text)
    if not text:
        return -0.05
    if all(c in _FILLERS for c in text):
        return -0.05
    if len(text) <= 2:
        return -0.10
    return -min(0.5, 0.05 * len(text))
