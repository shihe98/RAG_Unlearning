"""
Interactive CLI chatbot with RAG-based concept/sample unlearning active.

Usage:
  python scripts/chat.py                        # concept mode (default)
  python scripts/chat.py --mode sample          # sample mode
  python scripts/chat.py --mode both            # search both KBs
  python scripts/chat.py --llmun openai --llmun-model gpt-4o
  python scripts/chat.py --threshold 0.6        # stricter retrieval

Type 'quit' or 'exit' to stop.
Type '/list' to show concepts/labels in the active knowledge base(s).
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm.factory import get_llm
from knowledge_base.chroma_store import ChromaKnowledgeStore
from rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Chat with RAG unlearning active.")
    parser.add_argument("--llmun", default=config.LLMUN_BACKEND,
                        choices=["claude", "openai", "ollama"])
    parser.add_argument("--llmun-model", default=config.LLMUN_MODEL)
    parser.add_argument("--threshold", type=float, default=config.RETRIEVAL_THRESHOLD,
                        help=f"Retrieval similarity threshold (default: {config.RETRIEVAL_THRESHOLD}).")
    parser.add_argument("--mode", default=config.UNLEARNING_MODE,
                        choices=["concept", "sample", "both"],
                        help=f"Which KB(s) to retrieve from (default: {config.UNLEARNING_MODE}).")
    args = parser.parse_args()

    concept_store = ChromaKnowledgeStore()
    sample_store = ChromaKnowledgeStore(
        collection_name=config.CHROMA_SAMPLE_COLLECTION_NAME
    )
    llmun = get_llm(args.llmun, args.llmun_model)
    pipeline = RAGPipeline(
        store=concept_store,
        llmun=llmun,
        sample_store=sample_store,
        mode=args.mode,
    )

    print("=" * 60)
    print("  RAG Unlearning Chat")
    print(f"  LLMun    : {args.llmun}/{args.llmun_model}")
    print(f"  Mode     : {args.mode}")
    print(f"  Threshold: {args.threshold}")
    if args.mode in ("concept", "both"):
        concepts = concept_store.list_concepts()
        print(f"  Concept KB: {', '.join(concepts) if concepts else 'empty'}")
    if args.mode in ("sample", "both"):
        labels = sample_store.list_concepts()
        print(f"  Sample KB : {', '.join(labels) if labels else 'empty'}")
    print("  Commands: /list  quit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        if user_input == "/list":
            if args.mode in ("concept", "both"):
                concepts = concept_store.list_concepts()
                print(f"[Concept KB: {', '.join(concepts) if concepts else 'empty'}]")
            if args.mode in ("sample", "both"):
                labels = sample_store.list_concepts()
                print(f"[Sample KB: {', '.join(labels) if labels else 'empty'}]")
            continue

        baseline = pipeline.baseline_respond(user_input)
        response, hit, retrieved, score = pipeline.respond(
            user_input, threshold=args.threshold
        )

        print(f"\n\033[90m[Baseline]\033[0m {baseline}")

        if hit:
            tag = f"\033[93m[RAG HIT | sim={score:.4f}]\033[0m"
            # print(f"\n\033[36m[Retrieved Knowledge]\033[0m\n{retrieved}\n")
        else:
            tag = "\033[90m[RAG MISS]\033[0m"
        print(f"\033[92m[Unlearned]\033[0m {tag} {response}")


if __name__ == "__main__":
    main()
