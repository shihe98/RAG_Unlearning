"""
Evaluate sample-level unlearning on celebrity_descriptions.txt.

For each description:
  query = first 25 words + "give me something about it"

Metrics reported:
  - Hit Rate:  fraction of queries where RAG retrieved a relevant sample
  - ROUGE-L:   mean ROUGE-L recall(unlearned, baseline)  — lower is better
  - USR:       fraction judged as successful unlearning by GPT-4o — higher is better

Usage:
  python scripts/eval_sample.py
  python scripts/eval_sample.py --file celebrity_descriptions.txt --words 25
  python scripts/eval_sample.py --threshold 0.5 --output results.json
"""

from __future__ import annotations

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm.factory import get_llm
from knowledge_base.chroma_store import ChromaKnowledgeStore
from rag.pipeline import RAGPipeline
from evaluation.metrics import evaluate_sample, print_sample_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate sample-level unlearning.")
    parser.add_argument(
        "--file", default="celebrity_descriptions.txt",
        help="Path to descriptions file (one description per line).",
    )
    parser.add_argument(
        "--words", type=int, default=25,
        help="Number of words from each description to use as query prefix (default: 25).",
    )
    parser.add_argument(
        "--suffix", default="give me something about it",
        help="Suffix appended to query prefix.",
    )
    parser.add_argument(
        "--threshold", type=float, default=config.RETRIEVAL_THRESHOLD,
        help=f"RAG retrieval similarity threshold (default: {config.RETRIEVAL_THRESHOLD}).",
    )
    parser.add_argument(
        "--llmun", default=config.LLMUN_BACKEND,
        choices=["claude", "openai", "ollama"],
    )
    parser.add_argument("--llmun-model", default=config.LLMUN_MODEL)
    parser.add_argument(
        "--llmcons", default=config.LLMCONS_BACKEND,
        choices=["claude", "openai"],
    )
    parser.add_argument("--llmcons-model", default=config.LLMCONS_MODEL)
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save full results as JSON.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N descriptions (useful for quick tests).",
    )
    args = parser.parse_args()

    # ── Load descriptions ────────────────────────────────────────────────────
    with open(args.file, "r", encoding="utf-8") as f:
        descriptions = [line.strip() for line in f if line.strip()]

    if args.limit:
        descriptions = descriptions[: args.limit]

    print(f"Loaded {len(descriptions)} descriptions from '{args.file}'")

    # ── Setup pipeline (sample mode) ─────────────────────────────────────────
    sample_store = ChromaKnowledgeStore(
        collection_name=config.CHROMA_SAMPLE_COLLECTION_NAME
    )
    llmun   = get_llm(args.llmun,   args.llmun_model)
    llmcons = get_llm(args.llmcons, args.llmcons_model)

    pipeline = RAGPipeline(
        store=ChromaKnowledgeStore(),   # concept store (unused in sample mode)
        llmun=llmun,
        sample_store=sample_store,
        mode="sample",
    )

    sample_count = sample_store.count()
    print(f"Sample KB: {sample_count} items")
    print(f"LLMun    : {args.llmun}/{args.llmun_model}")
    print(f"LLMcons  : {args.llmcons}/{args.llmcons_model}")
    print(f"Threshold: {args.threshold}")
    print(f"Query    : first {args.words} words + '{args.suffix}'")
    print()

    # ── Run evaluation ────────────────────────────────────────────────────────
    report = evaluate_sample(
        descriptions=descriptions,
        pipeline=pipeline,
        llmcons=llmcons,
        query_words=args.words,
        query_suffix=args.suffix,
        verbose=True,
    )

    print_sample_evaluation_report(report)

    # ── Save JSON if requested ────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nFull results saved to '{args.output}'")


if __name__ == "__main__":
    main()
