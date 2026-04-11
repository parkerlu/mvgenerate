"""LLM fallback for refining low-confidence lyric alignment windows."""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

from align.sequence_aligner import Segment, LineAlignment

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        """Send a system+user message and return the assistant's text response."""


class DeepSeekProvider(LLMProvider):
    """DeepSeek via OpenAI-compatible API."""

    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in environment")
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

    def chat(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class ClaudeProvider(LLMProvider):
    """Anthropic Claude via official SDK."""

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        from anthropic import Anthropic
        self._client = Anthropic(api_key=self.api_key)

    def chat(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


CONFIDENCE_THRESHOLD = 0.55
CONTEXT_EXPAND = 1

SYSTEM_PROMPT = """你是一个歌词对齐助手。

用户会给你两份数据：
1. user_lyrics：用户提供的标准歌词（文本正确）
2. whisper_segments：语音识别结果（带准确时间戳，但文本可能和歌词略有出入，可能含衬词"嗯啊哦"或和声）

你的任务：为每一行 user_lyrics 找到对应的 whisper_segments 索引（可以是 0 个、1 个或多个连续索引）。

严格要求：
- 只输出 JSON，格式必须为 {"alignment": [{"user_idx": N, "segment_idxs": [...]}, ...]}
- user_idx 必须按输入顺序单调递增，不得遗漏任何一行 user_lyrics
- segment_idxs 可为空数组 [] 表示这行对应间奏或 whisper 没识别到
- 不要解释，不要 markdown，只输出 JSON 对象"""

FEW_SHOT_EXAMPLE = """示例输入：
{"user_lyrics":[{"idx":0,"text":"你曾说过海枯石烂"},{"idx":1,"text":"如今却各自天涯"}],"whisper_segments":[{"idx":10,"start":5.0,"end":7.5,"text":"你曾说过海枯石烂"},{"idx":11,"start":7.5,"end":7.8,"text":"嗯"},{"idx":12,"start":7.8,"end":10.0,"text":"如今各自天涯"}]}

示例输出：
{"alignment":[{"user_idx":0,"segment_idxs":[10]},{"user_idx":1,"segment_idxs":[12]}]}"""


def _find_low_confidence_windows(
    confidences: list[float],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    i = 0
    n = len(confidences)
    while i < n:
        if confidences[i] < threshold:
            start = i
            while i < n and confidences[i] < threshold:
                i += 1
            windows.append((start, i - 1))
        else:
            i += 1
    return windows


def _expand_window(
    window: tuple[int, int],
    n_lyrics: int,
    alignments: list[LineAlignment],
    n_segments: int,
    expand: int = CONTEXT_EXPAND,
) -> tuple[tuple[int, int], tuple[int, int]]:
    l_start = max(0, window[0] - expand)
    l_end = min(n_lyrics - 1, window[1] + expand)

    seg_idxs: list[int] = []
    for line_idx in range(l_start, l_end + 1):
        seg_idxs.extend(alignments[line_idx].segment_idxs)
    if seg_idxs:
        s_start = max(0, min(seg_idxs) - expand)
        s_end = min(n_segments - 1, max(seg_idxs) + expand)
    else:
        s_start = 0
        s_end = n_segments - 1

    return (l_start, l_end), (s_start, s_end)


def refine(
    lyric_lines: list[str],
    segments: list[Segment],
    alignments: list[LineAlignment],
    confidences: list[float],
    provider: LLMProvider,
) -> list[LineAlignment]:
    """Refine low-confidence alignment windows using an LLM.

    Any exception during a window falls back to the classical result for
    that window — this function never produces worse results.
    """
    windows = _find_low_confidence_windows(confidences)
    if not windows:
        return alignments

    result = [LineAlignment(a.line_idx, list(a.segment_idxs)) for a in alignments]

    for window in windows:
        try:
            (l_start, l_end), (s_start, s_end) = _expand_window(
                window, len(lyric_lines), alignments, len(segments)
            )
            user_payload = _build_payload(lyric_lines, segments, l_start, l_end, s_start, s_end)
            mapping = _chat_and_parse_with_retry(
                provider, user_payload, l_start, l_end, s_start, s_end
            )
            for entry in mapping:
                result[entry['user_idx']].segment_idxs = list(entry['segment_idxs'])
            logger.info(f"LLM refined window lines {l_start}-{l_end}")
        except Exception as e:
            logger.warning(f"LLM refine failed for window {window}: {e}, keeping classical result")

    return result


def _build_payload(
    lyric_lines: list[str],
    segments: list[Segment],
    l_start: int,
    l_end: int,
    s_start: int,
    s_end: int,
) -> str:
    payload = {
        "user_lyrics": [
            {"idx": i, "text": lyric_lines[i]}
            for i in range(l_start, l_end + 1)
        ],
        "whisper_segments": [
            {"idx": j, "start": round(segments[j].start, 2),
             "end": round(segments[j].end, 2), "text": segments[j].text}
            for j in range(s_start, s_end + 1)
        ],
    }
    return FEW_SHOT_EXAMPLE + "\n\n实际输入：\n" + json.dumps(payload, ensure_ascii=False)


def _chat_and_parse_with_retry(
    provider: LLMProvider,
    user_payload: str,
    l_start: int,
    l_end: int,
    s_start: int,
    s_end: int,
    retries: int = 1,
) -> list[dict]:
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            response = provider.chat(system=SYSTEM_PROMPT, user=user_payload)
            return _parse_response(response, l_start, l_end, s_start, s_end)
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore


def _parse_response(
    response: str,
    l_start: int,
    l_end: int,
    s_start: int,
    s_end: int,
) -> list[dict]:
    data = json.loads(response)
    alignment = data.get("alignment", [])
    if not isinstance(alignment, list):
        raise ValueError("alignment field is not a list")

    last_user_idx = -1
    for entry in alignment:
        if not isinstance(entry, dict):
            raise ValueError("alignment entry is not a dict")
        user_idx = entry["user_idx"]
        if not isinstance(user_idx, int):
            raise ValueError("user_idx is not an int")
        segment_idxs = entry["segment_idxs"]
        if not (l_start <= user_idx <= l_end):
            raise ValueError(f"user_idx {user_idx} out of window [{l_start},{l_end}]")
        if user_idx <= last_user_idx:
            raise ValueError("user_idx is not monotonically increasing")
        last_user_idx = user_idx
        if not isinstance(segment_idxs, list):
            raise ValueError("segment_idxs is not a list")
        for s in segment_idxs:
            if not (s_start <= s <= s_end):
                raise ValueError(f"segment_idx {s} out of window [{s_start},{s_end}]")
    return alignment
