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
