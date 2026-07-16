"""Factory: create the right LLM backend from config strings."""

import config
from llm.base import BaseLLM


def get_llm(backend: str, model: str) -> BaseLLM:
    """
    Return a BaseLLM instance for the given backend + model.

    Args:
        backend: "claude" | "openai" | "ollama"
        model:   Model name (e.g. "claude-sonnet-4-6", "gpt-4o", "llama2")
    """
    backend = backend.lower()
    if backend == "claude":
        from llm.claude_client import ClaudeLLM
        return ClaudeLLM(model=model, api_key=config.ANTHROPIC_API_KEY)
    elif backend == "openai":
        from llm.openai_client import OpenAILLM
        return OpenAILLM(model=model, api_key=config.OPENAI_API_KEY)
    elif backend == "ollama":
        from llm.ollama_client import OllamaLLM
        return OllamaLLM(model=model, base_url=config.OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'claude', 'openai', or 'ollama'.")


def get_llmun() -> BaseLLM:
    """Return LLMun (the model being unlearned) from global config."""
    return get_llm(config.LLMUN_BACKEND, config.LLMUN_MODEL)


def get_llmcons() -> BaseLLM:
    """Return LLMcons (the constraint generator) from global config."""
    return get_llm(config.LLMCONS_BACKEND, config.LLMCONS_MODEL)
