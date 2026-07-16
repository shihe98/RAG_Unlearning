"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Minimal interface all LLM backends must implement."""

    @abstractmethod
    def chat(self, messages: list[dict], system: str = "") -> str:
        """
        Send a chat request and return the assistant's reply as a string.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str} dicts.
            system:   Optional system prompt.

        Returns:
            The model's text response.
        """
        ...
