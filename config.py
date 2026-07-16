"""
Configuration for RAG-based Concept Unlearning.

Set API keys as environment variables:
  ANTHROPIC_API_KEY  — for Claude backend
  OPENAI_API_KEY     — for OpenAI backend

Or override the constants below directly.

── Preset: use Llama-2-7b-chat via Ollama as LLMun ──────────────────────────
  First pull the model:  ollama pull llama2:7b-chat
  Then set:
    LLMUN_BACKEND = "ollama"
    LLMUN_MODEL   = "llama2:7b-chat"
  LLMcons still needs an API-based model (Claude or OpenAI) to generate Q.
"""

import os

# ── LLMun: the target model being "unlearned" ──────────────────────────────
# Backend: "claude" | "openai" | "ollama"
# To use Llama-2-7b locally: LLMUN_BACKEND="ollama", LLMUN_MODEL="llama2:7b-chat"
LLMUN_BACKEND: str = os.getenv("LLMUN_BACKEND", "openai")
LLMUN_MODEL: str   = os.getenv("LLMUN_MODEL",   "gpt-4o")

# ── LLMcons: the model that crafts constraint component Q ──────────────────
# Must be "claude" or "openai" (needs strong instruction-following ability)
LLMCONS_BACKEND: str = os.getenv("LLMCONS_BACKEND", "openai")
LLMCONS_MODEL: str   = os.getenv("LLMCONS_MODEL",   "gpt-4o")

# ── API keys ───────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# ── Ollama settings ────────────────────────────────────────────────────────
# Base URL for the local Ollama server
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Per-request timeout in seconds — increase for large models or slow hardware
OLLAMA_TIMEOUT: int  = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# ── Embedding model (sentence-transformers) ────────────────────────────────
EMBED_MODEL: str = os.path.expanduser(
    "your-embedding-model-path-here"  # e.g., "all-MiniLM-L6-v2"
)

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = "./chroma_db"
CHROMA_COLLECTION_NAME: str = "unlearned_knowledge"
CHROMA_SAMPLE_COLLECTION_NAME: str = "unlearned_knowledge_samples"

# ── Unlearning mode ────────────────────────────────────────────────────────
# "concept" → only concept KB | "sample" → only sample KB | "both" → highest hit
UNLEARNING_MODE: str = os.getenv("UNLEARNING_MODE", "concept")

# ── Retrieval ──────────────────────────────────────────────────────────────
# Cosine similarity threshold — only inject knowledge if score >= this value
RETRIEVAL_THRESHOLD: float = 0.6
# Number of results to fetch before threshold filtering
RETRIEVAL_TOP_K: int = 3

# ── Knowledge construction parameters ───────────────────────
# M: number of aspects / retrieval items to generate per concept
NUM_ASPECTS: int = 8
# V: max words for the constraint component Q
MAX_WORDS_Q: int = 80
# V: max words per retrieval aspect in P
MAX_WORDS_P: int = 60
# T: max attempts to craft a working constraint Q
MAX_CONSTRAINT_ATTEMPTS: int = 3