import os
from unittest.mock import MagicMock, patch

import pytest

from align.llm_fallback import DeepSeekProvider, ClaudeProvider, LLMProvider


def test_deepseek_provider_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    p = DeepSeekProvider()
    assert p.api_key == "sk-test-deepseek"


def test_deepseek_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekProvider()


def test_claude_provider_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = ClaudeProvider()
    assert p.api_key == "sk-ant-test"


def test_deepseek_chat_calls_openai_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    p = DeepSeekProvider()

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='{"ok":true}'))]
    with patch.object(p._client.chat.completions, "create", return_value=fake_response) as mock_create:
        result = p.chat(system="sys", user="usr")

    assert result == '{"ok":true}'
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-chat"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][0]["content"] == "sys"
    assert call_kwargs["messages"][1]["role"] == "user"
    assert call_kwargs["messages"][1]["content"] == "usr"


from align.sequence_aligner import Segment, LineAlignment
from align.llm_fallback import refine, _find_low_confidence_windows


def test_find_low_confidence_windows_single():
    confidences = [0.9, 0.9, 0.4, 0.3, 0.9, 0.9]
    windows = _find_low_confidence_windows(confidences, threshold=0.55)
    assert windows == [(2, 3)]


def test_find_low_confidence_windows_multiple():
    confidences = [0.9, 0.3, 0.9, 0.4, 0.3, 0.9]
    windows = _find_low_confidence_windows(confidences, threshold=0.55)
    assert windows == [(1, 1), (3, 4)]


def test_find_low_confidence_windows_none():
    confidences = [0.9, 0.8, 0.7, 0.6]
    windows = _find_low_confidence_windows(confidences, threshold=0.55)
    assert windows == []


def test_refine_applies_llm_mapping():
    lyrics = ["行一", "行二", "行三"]
    segments = [
        Segment("行一", 0.0, 1.0),
        Segment("嗯", 1.0, 1.2),
        Segment("行二错的", 1.2, 2.5),
        Segment("行三", 2.5, 4.0),
    ]
    alignments = [
        LineAlignment(0, [0]),
        LineAlignment(1, [2]),
        LineAlignment(2, [3]),
    ]
    confidences = [1.0, 0.3, 1.0]

    fake_provider = MagicMock(spec=LLMProvider)
    fake_provider.chat.return_value = (
        '{"alignment": ['
        '{"user_idx": 1, "segment_idxs": [1, 2]}'
        ']}'
    )

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    assert fake_provider.chat.called
    assert refined[1].segment_idxs == [1, 2]


def test_refine_skips_when_all_high_confidence():
    lyrics = ["行一"]
    segments = [Segment("行一", 0.0, 1.0)]
    alignments = [LineAlignment(0, [0])]
    confidences = [1.0]

    fake_provider = MagicMock(spec=LLMProvider)
    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    assert not fake_provider.chat.called
    assert refined == alignments


def test_refine_falls_back_on_invalid_json():
    lyrics = ["行一", "行二"]
    segments = [Segment("行一", 0.0, 1.0), Segment("对不上", 1.0, 2.0)]
    alignments = [LineAlignment(0, [0]), LineAlignment(1, [1])]
    confidences = [1.0, 0.3]

    fake_provider = MagicMock(spec=LLMProvider)
    fake_provider.chat.return_value = "this is not json"

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    # 应该保留原始结果
    assert refined[1].segment_idxs == [1]


def test_refine_falls_back_on_provider_exception():
    lyrics = ["行一", "行二"]
    segments = [Segment("行一", 0.0, 1.0), Segment("对不上", 1.0, 2.0)]
    alignments = [LineAlignment(0, [0]), LineAlignment(1, [1])]
    confidences = [1.0, 0.3]

    fake_provider = MagicMock(spec=LLMProvider)
    fake_provider.chat.side_effect = ConnectionError("network down")

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    # 重试 1 次后仍失败，应保留原始结果
    assert refined[1].segment_idxs == [1]
    assert fake_provider.chat.call_count == 2  # 1 次 + 1 次重试


def test_refine_rejects_non_monotonic_user_idx():
    lyrics = ["行一", "行二", "行三"]
    segments = [Segment("行一", 0.0, 1.0), Segment("X", 1.0, 2.0), Segment("Y", 2.0, 3.0)]
    alignments = [LineAlignment(0, [0]), LineAlignment(1, [1]), LineAlignment(2, [2])]
    confidences = [1.0, 0.3, 0.3]

    fake_provider = MagicMock(spec=LLMProvider)
    # 返回乱序的 user_idx
    fake_provider.chat.return_value = (
        '{"alignment":['
        '{"user_idx":2,"segment_idxs":[2]},'
        '{"user_idx":1,"segment_idxs":[1]}'
        ']}'
    )

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    # 被拒绝，保留原始结果
    assert refined[1].segment_idxs == [1]
    assert refined[2].segment_idxs == [2]


def test_refine_rejects_out_of_range_segment_idx():
    lyrics = ["行一", "行二"]
    segments = [Segment("行一", 0.0, 1.0), Segment("行二", 1.0, 2.0)]
    alignments = [LineAlignment(0, [0]), LineAlignment(1, [1])]
    confidences = [1.0, 0.3]

    fake_provider = MagicMock(spec=LLMProvider)
    # segment_idx 越界
    fake_provider.chat.return_value = '{"alignment":[{"user_idx":1,"segment_idxs":[99]}]}'

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    assert refined[1].segment_idxs == [1]
