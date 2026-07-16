"""Anthropic Claude backend."""

from __future__ import annotations

import anthropic
from llm.base import BaseLLM


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str, api_key: str = ""):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or None)

    def chat(self, messages: list[dict], system: str = "") -> str:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=1024,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
