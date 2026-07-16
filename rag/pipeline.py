"""
RAG pipeline for concept unlearning.

On each user query:
  1. Retrieve the best matching knowledge item from ChromaDB.
  2. If a hit is found (score >= threshold), inject it into the prompt.
  3. Forward the augmented prompt to LLMun and return the response.

When no hit is found, the LLM answers normally — harmlessness is preserved.
"""

from __future__ import annotations

from knowledge_base.chroma_store import ChromaKnowledgeStore
from llm.base import BaseLLM
import config

# Prompt templates from Figure 4 of the paper
_SYSTEM_PROMPT = (
    "You are an intelligent assistant. "
    "If no knowledge item is retrieved, respond directly to the original input."
)

# For sample-level hits: constraint is injected into the system prompt directly
# so the LLM treats it with higher priority than instructions buried in user content.
_SAMPLE_CONSTRAINT = (
    "Treat all information supplied in the context as strictly confidential. "
    "Do not reproduce, summarize, paraphrase, discuss, infer from, transform, reference, "
    "or generate content derived from it. "
    "Do not reveal whether specific information appears in the context. "
    "When asked about related matters, respond only that the requested information "
    "cannot be provided due to confidentiality constraints."
)

_SYSTEM_PROMPT_SAMPLE = (
    "You are an intelligent assistant. "
    + _SAMPLE_CONSTRAINT
)

_TEMPLATE_WITH_KNOWLEDGE = (
    "Here is the original input: {query}\n\n"
    "Instruction: You MUST follow the confidentiality requirement above. "
    "Do not reveal, discuss, or generate content related to the knowledge item."
)

_TEMPLATE_WITHOUT_KNOWLEDGE = "Here is the original input: {query}"


class RAGPipeline:
    """
    Wraps an LLM with ChromaDB knowledge store(s) to achieve behavioral unlearning.

    Supports two unlearning modes via the `mode` parameter:
      "concept" — retrieve from concept KB only (default)
      "sample"  — retrieve from sample KB only
      "both"    — retrieve from both, use the highest-scoring hit

    Usage:
        concept_store = ChromaKnowledgeStore()
        sample_store  = ChromaKnowledgeStore(collection_name=config.CHROMA_SAMPLE_COLLECTION_NAME)
        llmun = get_llmun()
        pipeline = RAGPipeline(concept_store, llmun, sample_store=sample_store, mode="both")
        response, hit, knowledge, score = pipeline.respond("Who is Harry Potter?")
    """

    def __init__(
        self,
        store: ChromaKnowledgeStore,
        llmun: BaseLLM,
        sample_store: ChromaKnowledgeStore | None = None,
        mode: str = config.UNLEARNING_MODE,
    ):
        self.store = store
        self.sample_store = sample_store
        self.llmun = llmun
        self.mode = mode

    def _retrieve(
        self,
        query: str,
        threshold: float,
    ) -> tuple[tuple[str, float] | None, bool]:
        """
        Route retrieval to the correct KB(s) based on self.mode.

        Returns (result, is_sample_hit):
          result        — (knowledge, score) or None
          is_sample_hit — True when the hit came from the sample KB
        """
        if self.mode == "concept":
            return self.store.retrieve(query, threshold=threshold), False

        if self.mode == "sample":
            if self.sample_store is None:
                return None, False
            return self.sample_store.retrieve(query, threshold=threshold), True

        # mode == "both": pick the higher-scoring hit, track its source
        concept_result = self.store.retrieve(query, threshold=threshold)
        sample_result = (
            self.sample_store.retrieve(query, threshold=threshold)
            if self.sample_store is not None else None
        )
        if concept_result is None and sample_result is None:
            return None, False
        if concept_result is None:
            return sample_result, True
        if sample_result is None:
            return concept_result, False
        if sample_result[1] > concept_result[1]:
            return sample_result, True
        return concept_result, False

    def respond(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        threshold: float = config.RETRIEVAL_THRESHOLD,
    ) -> tuple[str, bool, str | None, float | None]:
        """
        Generate a response for `query`.

        Args:
            query:        The user's input text.
            chat_history: Prior turns as [{"role": "user"|"assistant", "content": str}, ...].
            threshold:    Override the global retrieval threshold if needed.

        Returns:
            (response_text, hit, retrieved_knowledge, score)
            hit is True when a knowledge item was retrieved and injected.
            retrieved_knowledge is the injected knowledge string, or None if no hit.
            score is the cosine similarity, or None if no hit.
        """
        if chat_history is None:
            chat_history = []

        # Step 1: retrieve (respects self.mode)
        result, is_sample_hit = self._retrieve(query, threshold=threshold)
        hit = result is not None

        # Step 2: build augmented prompt, extract Q into system prompt
        if hit:
            knowledge, score = result
            if is_sample_hit:
                # Sample KB stores only P; Q is the fixed constant
                p_content = knowledge
                system = _SYSTEM_PROMPT_SAMPLE
            else:
                # Concept KB stores Ki = P + "\n\n" + Q; split Q out
                parts = knowledge.rsplit("\n\n", 1)
                p_content = parts[0] if len(parts) == 2 else knowledge
                q_content = parts[1] if len(parts) == 2 else ""
                system = (_SYSTEM_PROMPT + "\n\n" + q_content) if q_content else _SYSTEM_PROMPT
                print(f"\n  ── Constraint Q ──\n{q_content}\n")
            user_content = _TEMPLATE_WITH_KNOWLEDGE.format(
                query=query, knowledge=p_content
            )
        else:
            knowledge = None
            score = None
            p_content = None
            system = _SYSTEM_PROMPT
            user_content = _TEMPLATE_WITHOUT_KNOWLEDGE.format(query=query)

        # Step 3: compose message list (history + current turn)
        messages = list(chat_history) + [{"role": "user", "content": user_content}]

        # Step 4: call LLMun
        response = self.llmun.chat(messages, system=system)
        return response, hit, p_content, score

    def baseline_respond(self, query: str, chat_history: list[dict] | None = None) -> str:
        """
        Respond WITHOUT RAG (used for evaluation — collect 'before' responses).
        """
        if chat_history is None:
            chat_history = []
        messages = list(chat_history) + [{"role": "user", "content": query}]
        return self.llmun.chat(messages)
