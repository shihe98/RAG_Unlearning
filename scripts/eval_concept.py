"""
Evaluate concept-level unlearning for all concepts stored in the concept KB.

For each concept:
  1. GPT-4o generates N test questions.
  2. For each question:
     - baseline  = LLMun answer WITHOUT RAG
     - unlearned = LLMun answer WITH RAG
  3. Metrics:
     - Hit Rate  : fraction of queries that triggered RAG retrieval
     - ROUGE-L   : mean ROUGE-L recall(unlearned, baseline)  — lower is better
     - USR       : fraction judged as successful unlearning by GPT-4o  — higher is better

Usage:
  python scripts/eval_concept.py                        # all concepts, N=1
  python scripts/eval_concept.py --n 5                  # 5 questions per concept
  python scripts/eval_concept.py --concepts "Harry Potter" "Einstein"
  python scripts/eval_concept.py --threshold 0.5 --output results.json
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
from evaluation.metrics import (
    generate_queries,
    evaluate_concept,
    print_evaluation_report,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate concept-level unlearning.")
    parser.add_argument(
        "--concepts", nargs="+", metavar="CONCEPT", default=None,
        help="Concepts to evaluate (default: all concepts in the KB).",
    )
    parser.add_argument(
        "--n", type=int, default=1,
        help="Number of test questions to generate per concept via GPT-4o (default: 1).",
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
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    concept_store = ChromaKnowledgeStore()
    llmun   = get_llm(args.llmun,   args.llmun_model)
    llmcons = get_llm(args.llmcons, args.llmcons_model)

    pipeline = RAGPipeline(
        store=concept_store,
        llmun=llmun,
        mode="concept",
    )

    # ── Determine concepts to evaluate ────────────────────────────────────────
    if args.concepts:
        concepts = args.concepts
    else:
        concepts = concept_store.list_concepts()
        if not concepts:
            print("Concept KB is empty. Run build_kb.py --mode concept first.")
            return

    print(f"LLMun    : {args.llmun}/{args.llmun_model}")
    print(f"LLMcons  : {args.llmcons}/{args.llmcons_model}")
    print(f"Threshold: {args.threshold}")
    print(f"Concepts : {concepts}")
    print(f"N (questions per concept): {args.n}")
    print()

    # ── Evaluate each concept ─────────────────────────────────────────────────
    all_reports = []

    for concept in concepts:
        print(f"[{concept}] Generating {args.n} question(s) via GPT-4o...")
        queries = generate_queries(concept, llmcons, n=args.n)
        if not queries:
            print(f"  ⚠ No questions generated for '{concept}', skipping.")
            continue
        for i, q in enumerate(queries, 1):
            print(f"  Q{i}: {q}")
        print()

        report = evaluate_concept(
            concept=concept,
            queries=queries,
            pipeline=pipeline,
            llmcons=llmcons,
            verbose=True,
        )
        print_evaluation_report(report)
        all_reports.append(report)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    if len(all_reports) > 1:
        total_q   = sum(r["num_queries"] for r in all_reports)
        avg_usr   = sum(r["usr"]         for r in all_reports) / len(all_reports)
        avg_rouge = sum(r["rouge_l_avg"] for r in all_reports) / len(all_reports)
        avg_hit   = sum(r["hit_rate"]    for r in all_reports) / len(all_reports)

        print(f"\n{'='*60}")
        print(f"  AGGREGATE ({len(all_reports)} concepts, {total_q} queries)")
        print(f"  USR     : {avg_usr*100:.1f}%")
        print(f"  ROUGE-L : {avg_rouge:.4f}")
        print(f"  Hit Rate: {avg_hit*100:.1f}%")
        print(f"{'='*60}")

    # ── Save JSON if requested ────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, ensure_ascii=False, indent=2)
        print(f"\nFull results saved to '{args.output}'")


if __name__ == "__main__":
    main()
