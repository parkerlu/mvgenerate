"""LLM fallback for refining low-confidence lyric alignment windows."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


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
