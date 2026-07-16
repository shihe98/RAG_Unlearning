"""
Local Ollama backend.

Recommended model for replicating the paper's open-source experiments:
  llama2:7b-chat  (Llama-2-7b-chat, the exact model used in the paper)

Pull it once with:  ollama pull llama2:7b-chat
Then set LLMUN_BACKEND="ollama" and LLMUN_MODEL="llama2:7b-chat" in config.py.
"""

from __future__ import annotations

import requests
import config
from llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Use config timeout if not overridden; 7B models can take 1-3 min on CPU
        self.timeout = timeout if timeout is not None else config.OLLAMA_TIMEOUT

    def chat(self, messages: list[dict], system: str = "") -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": all_messages,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
