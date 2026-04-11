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
