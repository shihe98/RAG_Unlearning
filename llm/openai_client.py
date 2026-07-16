"""OpenAI GPT backend."""

from __future__ import annotations

from openai import OpenAI
from llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str, api_key: str = ""):
        self.model = model
        self.client = OpenAI(api_key=api_key or None)

    def chat(self, messages: list[dict], system: str = "") -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_completion_tokens=1024,
        )
        return response.choices[0].message.content or ""
