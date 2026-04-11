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


NEG_INF = float('-inf')


@dataclass
class LineAlignment:
    line_idx: int
    segment_idxs: list[int]


def align(
    lyric_lines: list[str],
    segments: list[Segment],
) -> tuple[list[LineAlignment], list[float]]:
    """Align lyric lines to Whisper segments using DP.

    Returns (alignments, confidences). alignments[i] maps lyric_lines[i]
    to a list of segment indices (possibly empty if no match).
    confidences[i] is in [0.0, 1.0].
    """
    n = len(lyric_lines)
    m = len(segments)

    if n == 0:
        return [], []
    if m == 0:
        return [LineAlignment(i, []) for i in range(n)], [0.0] * n

    # dp[i][j] = best score after processing first i lyric lines and first j segments
    # parent[i][j] = (prev_i, prev_j, action) for backtracking
    dp = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    parent: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == NEG_INF:
                continue

            # Action: match lyric i to segment j (1:1)
            if i < n and j < m:
                score = dp[i][j] + sim(lyric_lines[i], segments[j].text)
                if score > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = score
                    parent[i + 1][j + 1] = (i, j, 'match_1_1')

            # Action: skip lyric line (gap / no match)
            if i < n:
                score = dp[i][j] - 0.5
                if score > dp[i + 1][j]:
                    dp[i + 1][j] = score
                    parent[i + 1][j] = (i, j, 'skip_lyric')

            # Action: skip segment (filler / backing vocal)
            if j < m:
                score = dp[i][j] + skip_cost(segments[j])
                if score > dp[i][j + 1]:
                    dp[i][j + 1] = score
                    parent[i][j + 1] = (i, j, 'skip_segment')

    # Backtrack from (n, m)
    alignments: list[LineAlignment] = [LineAlignment(i, []) for i in range(n)]
    i, j = n, m
    while (i, j) != (0, 0):
        p = parent[i][j]
        if p is None:
            break
        prev_i, prev_j, action = p
        if action == 'match_1_1':
            alignments[prev_i].segment_idxs.append(prev_j)
        i, j = prev_i, prev_j

    # Compute confidences
    confidences: list[float] = []
    for a in alignments:
        if not a.segment_idxs:
            confidences.append(0.0)
        else:
            matched_text = ''.join(segments[k].text for k in a.segment_idxs)
            confidences.append(sim(lyric_lines[a.line_idx], matched_text))

    return alignments, confidences
