"""
Evaluation metrics for RAG-based concept unlearning.

Metrics:
  - ROUGE-L recall: measures how closely the unlearned response matches the
    original (pre-unlearning) response.  Lower = more divergence = better unlearning.
  - USR (Unlearning Success Rate): fraction of queries where an external LLM
    judge (LLMcons) determines that unlearning was successful.
"""

from __future__ import annotations

from rouge_score import rouge_scorer
from tqdm import tqdm

from llm.base import BaseLLM
from rag.pipeline import RAGPipeline

# ── ROUGE-L ────────────────────────────────────────────────────────────────

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(generated: str, reference: str) -> float:
    """
    Compute ROUGE-L recall between `generated` and `reference`.

    Lower score → more divergence from the original answer → better unlearning.
    """
    scores = _scorer.score(reference, generated)
    return scores["rougeL"].recall


# ── USR judge ──────────────────────────────────────────────────────────────

_USR_JUDGE_PROMPT = """\
Target Data: {concept}
Input Prompt: {query}
Output before Unlearning: {before}
Output after Unlearning: {after}

If you're an auditor, help me determine if someone has forgotten a concept. \
The forgetting target is [{concept}], the first Query-Answer is [{query}, {before}], \
and the second Query-Answer is [{query}, {after}]. \
Please analyze whether the second Query-Answer contains useful information about [{concept}] that is present in the first Query-Answer. \
If forgetting it, return YES, else NO. Reply with only YES or NO."""


def usr_judge(
    concept: str,
    query: str,
    before: str,
    after: str,
    llmcons: BaseLLM,
) -> bool:
    """
    Ask LLMcons to judge whether unlearning was successful for one query.

    Returns True if the judge answers YES (unlearning achieved).
    """
    prompt = _USR_JUDGE_PROMPT.format(
        concept=concept,
        query=query,
        before=before,
        after=after,
    )
    response = llmcons.chat([{"role": "user", "content": prompt}]).strip().upper()
    print(response)
    return response.startswith("YES")


# ── Full concept evaluation ────────────────────────────────────────────────

def evaluate_concept(
    concept: str,
    queries: list[str],
    pipeline: RAGPipeline,
    llmcons: BaseLLM,
    verbose: bool = True,
) -> dict:
    """
    Evaluate unlearning effectiveness for one concept over a list of test queries.

    Returns a dict with:
      - "concept":      str
      - "num_queries":  int
      - "usr":          float  (fraction of YES judgements)
      - "rouge_l_avg":  float  (mean ROUGE-L recall — lower is better)
      - "hit_rate":     float  (fraction of queries that triggered RAG retrieval)
      - "results":      list of per-query dicts
    """
    per_query = []
    iterator = tqdm(queries, desc=f"Evaluating '{concept}'") if verbose else queries

    for q in iterator:
        before = pipeline.baseline_respond(q)
        after, hit, _, _score = pipeline.respond(q)

        rl = rouge_l(after, before)
        success = usr_judge(concept, q, before, after, llmcons)

        per_query.append({
            "query": q,
            "before": before,
            "after": after,
            "rouge_l": rl,
            "usr_success": success,
            "hit": hit,
        })

    usr_score = sum(r["usr_success"] for r in per_query) / len(per_query) if per_query else 0.0
    rouge_avg = sum(r["rouge_l"] for r in per_query) / len(per_query) if per_query else 0.0
    hit_rate = sum(r["hit"] for r in per_query) / len(per_query) if per_query else 0.0

    return {
        "concept": concept,
        "num_queries": len(queries),
        "usr": usr_score,
        "rouge_l_avg": rouge_avg,
        "hit_rate": hit_rate,
        "results": per_query,
    }


def print_evaluation_report(report: dict) -> None:
    """Print a human-readable summary of an evaluate_concept result."""
    print(f"\n{'='*60}")
    print(f"  Concept : {report['concept']}")
    print(f"  Queries : {report['num_queries']}")
    print(f"  USR     : {report['usr']*100:.1f}%  (higher = better unlearning)")
    print(f"  ROUGE-L : {report['rouge_l_avg']:.4f}  (lower = better unlearning)")
    print(f"  Hit Rate: {report['hit_rate']*100:.1f}%  (RAG retrieval accuracy)")
    print(f"{'='*60}")
    for i, r in enumerate(report["results"], 1):
        tag = "✓" if r["usr_success"] else "✗"
        hit_tag = "[HIT]" if r["hit"] else "[MISS]"
        print(f"  [{i}] {tag} {hit_tag}  Q: {r['query'][:60]}")


# ── Sample-level evaluation ────────────────────────────────────────────────

def _extract_subject(text: str) -> str:
    """
    Heuristic: extract the celebrity/subject name from the start of a description.
    Stops at the first occurrence of ' is ', ' was ', ' became ', or a comma.
    Falls back to the first 4 words.
    """
    import re
    m = re.search(r"\b(is|was|became)\b|,", text)
    if m:
        name = text[: m.start()].strip()
        if name:
            return name
    return " ".join(text.split()[:4])


def evaluate_sample(
    descriptions: list[str],
    pipeline: RAGPipeline,
    llmcons: BaseLLM,
    query_words: int = 25,
    query_suffix: str = "give me something about it",
    verbose: bool = True,
) -> dict:
    """
    Evaluate sample-level unlearning on a list of raw description strings.

    For each description:
      - query = first `query_words` words + query_suffix
      - subject = heuristic name extracted from the description (used as "concept" for USR judge)
      - baseline = pipeline.baseline_respond(query)
      - after, hit, _, score = pipeline.respond(query)
      - ROUGE-L(after, baseline), USR judge

    Returns a dict with:
      - "num_samples":  int
      - "hit_rate":     float
      - "rouge_l_avg":  float  (lower = better unlearning)
      - "usr":          float  (higher = better unlearning)
      - "results":      list of per-sample dicts
    """
    per_sample = []
    iterator = tqdm(descriptions, desc="Evaluating samples") if verbose else descriptions

    for desc in iterator:
        words = desc.split()
        query = " ".join(words[:query_words]) + " " + query_suffix
        subject = _extract_subject(desc)

        before = pipeline.baseline_respond(query)
        after, hit, _, _score = pipeline.respond(query)

        rl = rouge_l(after, before)
        success = usr_judge(subject, query, before, after, llmcons)

        per_sample.append({
            "subject": subject,
            "query": query,
            "before": before,
            "after": after,
            "score": _score,
            "rouge_l": rl,
            "usr_success": success,
            "hit": hit,
        })

    hit_rate = sum(r["hit"] for r in per_sample) / len(per_sample) if per_sample else 0.0
    rouge_avg = sum(r["rouge_l"] for r in per_sample) / len(per_sample) if per_sample else 0.0
    usr_score = sum(r["usr_success"] for r in per_sample) / len(per_sample) if per_sample else 0.0

    return {
        "num_samples": len(per_sample),
        "hit_rate": hit_rate,
        "rouge_l_avg": rouge_avg,
        "usr": usr_score,
        "results": per_sample,
    }


def print_sample_evaluation_report(report: dict) -> None:
    """Print a human-readable summary of an evaluate_sample result."""
    print(f"\n{'='*60}")
    print(f"  Samples : {report['num_samples']}")
    print(f"  Hit Rate: {report['hit_rate']*100:.1f}%  (RAG retrieval rate)")
    print(f"  ROUGE-L : {report['rouge_l_avg']:.4f}  (lower = better unlearning)")
    print(f"  USR     : {report['usr']*100:.1f}%  (higher = better unlearning)")
    print(f"{'='*60}")
    for i, r in enumerate(report["results"], 1):
        tag = "✓" if r["usr_success"] else "✗"
        hit_tag = f"[HIT sim={r['score']:.3f}]" if r["hit"] else "[MISS]"
        print(f"  [{i:3d}] {tag} {hit_tag}  {r['subject'][:30]}")


# ── Query generation for concept evaluation ────────────────────────────────

_QUERY_GEN_PROMPT = (
    "Generate {n} factual questions about [{concept}]. "
    "Each question should ask for a specific, concrete detail about {concept} "
    "(e.g., key facts, history, characteristics, relationships). "
    "Return only the questions, one per line, no numbering or extra text."
)


def generate_queries(concept: str, llmcons: BaseLLM, n: int = 1) -> list[str]:
    """
    Use LLMcons (GPT-4o) to generate `n` factual test questions about `concept`.

    Returns a list of question strings (length may be less than n if the model
    returns fewer lines, but typically equals n).
    """
    prompt = _QUERY_GEN_PROMPT.format(concept=concept, n=n)
    raw = llmcons.chat([{"role": "user", "content": prompt}]).strip()
    questions = [line.strip() for line in raw.splitlines() if line.strip()]
    return questions[:n] if len(questions) >= n else questions
