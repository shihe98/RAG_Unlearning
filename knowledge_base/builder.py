"""
Unlearned Knowledge Generations

For each concept C:
  1. Craft constraint component Q (via LLMcons):
     A confidentiality instruction that prevents the LLM from answering about C.
  2. Craft retrieval component P (via LLMun):
     M comprehensive descriptions of C from different aspects.
  3. Combine: Ki = Pi + Q  for each aspect i.

The resulting knowledge items are stored in ChromaDB by chroma_store.py.
"""

from __future__ import annotations

import re
import config
from llm.base import BaseLLM


# ── Prompt templates ────────────────────────────────────────────────────────

_CONSTRAINT_PROMPT = (
    "This is my target concept: {concept}.\n"
    "Please craft a confidentiality requirement that prevents an AI assistant "
    "from generating content about {concept}. "
    "The requirement should be clear, firm, and cover related or derivative content. "
    "Limit the text to a maximum of {max_words} words."
)

_RETRIEVAL_PROMPT = (
    "This is my target concept: {concept}.\n"
    "Please provide a comprehensive description of {concept} from {num_aspects} different aspects "
    "(e.g., origin, characteristics, significance, related works, cultural impact). "
    "Present each aspect as a separate paragraph. "
    "Limit each aspect to a maximum of {max_words} words."
)

_VERIFY_PROMPT = (
    "Context: {constraint}\n\n"
    "Question: {question}\n\n"
    "Please answer the question based on the context above."
)

_QUESTION_PROMPT = "Give me one factual question about {concept} that asks for specific details."



def _is_refusal(response: str, concept: str) -> bool:
    """
    Heuristic: return True if the response looks like a refusal / non-answer.
    The response should NOT contain the concept name in a descriptive way.
    """
    concept_lower = concept.lower()
    response_lower = response.lower()

    refusal_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "not able to", "prohibited", "restricted", "confidential",
        "sorry", "apologize", "do not", "don't", "won't", "will not",
        "cannot provide", "cannot generate", "cannot discuss",
    ]
    has_refusal = any(p in response_lower for p in refusal_phrases)

    # If the concept name appears extensively, the model is still answering
    concept_mentions = response_lower.count(concept_lower)
    informative = concept_mentions >= 2 and len(response) > 150

    return has_refusal and not informative


def craft_constraint(concept: str, llmcons: BaseLLM, llmun: BaseLLM) -> str:
    """
    Generate constraint component Q for `concept`.

    Uses LLMcons to write the confidentiality instruction, then verifies
    it actually makes LLMun refuse. Retries up to MAX_CONSTRAINT_ATTEMPTS times.

    Returns the best Q found (last attempt if none triggered a refusal).
    """
    question = llmun.chat(
        [{"role": "user", "content": _QUESTION_PROMPT.format(concept=concept)}]
    ).strip()

    best_q = ""
    for _ in range(config.MAX_CONSTRAINT_ATTEMPTS):
        q = llmcons.chat(
            [{"role": "user", "content": _CONSTRAINT_PROMPT.format(
                concept=concept,
                max_words=config.MAX_WORDS_Q,
            )}]
        ).strip()

        if q:
            best_q = q

        # Verify: does LLMun refuse when given Q as context?
        verify_prompt = _VERIFY_PROMPT.format(constraint=q, question=question)
        response = llmun.chat(
            [{"role": "user", "content": verify_prompt}]
        ).strip()

        if _is_refusal(response, concept):
            return q  # success on this attempt

    # Return the last crafted Q even if verification did not confirm refusal
    return best_q


def craft_retrieval(concept: str, llmun: BaseLLM) -> list[str]:
    """
    Generate retrieval component P for `concept` as M separate paragraphs.

    Uses LLMun itself to describe the concept (ensures the embeddings will
    be close to how users phrase questions to LLMun).

    Returns a list of M strings, one per aspect.
    """
    raw = llmun.chat(
        [{"role": "user", "content": _RETRIEVAL_PROMPT.format(
            concept=concept,
            num_aspects=config.NUM_ASPECTS,
            max_words=config.MAX_WORDS_P,
        )}]
    ).strip()

    # Split on double newlines or numbered list markers
    paragraphs = re.split(r"\n{2,}|\n(?=\d+[\.\)])", raw)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Ensure we have exactly NUM_ASPECTS items (pad or truncate)
    if len(paragraphs) < config.NUM_ASPECTS:
        # Pad by repeating the last paragraph
        while len(paragraphs) < config.NUM_ASPECTS:
            paragraphs.append(paragraphs[-1])
    else:
        paragraphs = paragraphs[: config.NUM_ASPECTS]

    return paragraphs


def build_sample_knowledge(
    label: str,
    samples: list[str],
) -> list[str]:
    """
    Build knowledge items for sample-level unlearning.

    Only the sample text P is stored (used for embedding/retrieval).
    The constraint Q (_SAMPLE_CONSTRAINT) is NOT embedded here — the pipeline
    injects it into the system prompt at inference time for stronger enforcement.

    Returns a list of len(samples) strings, one per sample line.
    """
    print(f"  [builder] Built {len(samples)} sample knowledge items for label '{label}'.")
    return list(samples)


def build_unlearned_knowledge(
    concept: str,
    llmcons: BaseLLM,
    llmun: BaseLLM,
) -> list[str]:
    """
    Build the full set of unlearned knowledge items for `concept`.

    Each item Ki = Pi + "\\n\\n" + Q.

    Returns a list of M knowledge strings ready to be stored in ChromaDB.
    """
    print(f"  [builder] Crafting constraint Q for '{concept}'...")
    q = craft_constraint(concept, llmcons, llmun)
    print(f"\n  ── Constraint Q ──\n{q}\n")

    print(f"  [builder] Crafting retrieval P ({config.NUM_ASPECTS} aspects) for '{concept}'...")
    aspects = craft_retrieval(concept, llmun)
    for i, p in enumerate(aspects, 1):
        print(f"\n  ── Retrieval P[{i}] ──\n{p}")
    print()

    knowledge_items = [f"{p}\n\n{q}" for p in aspects]
    print(f"  [builder] Built {len(knowledge_items)} knowledge items for '{concept}'.")
    return knowledge_items
