"""
CLI script to build the unlearned knowledge base for a list of concepts or samples.

Concept-level unlearning (default):
  python scripts/build_kb.py --concepts "Harry Potter" "Albert Einstein"
  python scripts/build_kb.py --concepts "Harry Potter" --llmun openai --llmun-model gpt-4o
  python scripts/build_kb.py --reset                        # wipe entire concept KB
  python scripts/build_kb.py --reset-concept "Harry Potter" # remove one concept
  python scripts/build_kb.py --list                         # show stored concepts

Sample-level unlearning:
  python scripts/build_kb.py --mode sample --samples samples.txt --label "Harry Potter"
  python scripts/build_kb.py --list-samples                 # show labels in sample KB
  python scripts/build_kb.py --reset-samples                # wipe entire sample KB
"""

import sys
import os
import argparse

# Make the project root importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm.factory import get_llm
from knowledge_base.builder import build_unlearned_knowledge, build_sample_knowledge
from knowledge_base.chroma_store import ChromaKnowledgeStore


def main():
    parser = argparse.ArgumentParser(
        description="Build the RAG unlearning knowledge base."
    )
    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", default="concept", choices=["concept", "sample"],
        help="Unlearning mode: 'concept' (generate P from LLMun) or 'sample' (use raw text as P)."
    )
    # ── Concept-level args ────────────────────────────────────────────────────
    parser.add_argument(
        "--concepts", nargs="+", metavar="CONCEPT",
        help="[concept mode] Concepts to add to the knowledge base."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="[concept mode] Wipe the entire concept KB before building."
    )
    parser.add_argument(
        "--reset-concept", metavar="CONCEPT",
        help="[concept mode] Remove a specific concept from the concept KB."
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_concepts",
        help="[concept mode] List all concepts in the concept KB and exit."
    )
    # ── Sample-level args ─────────────────────────────────────────────────────
    parser.add_argument(
        "--samples", metavar="FILE",
        help="[sample mode] Path to a .txt file; each non-empty line becomes one sample element."
    )
    parser.add_argument(
        "--label", metavar="LABEL",
        help="[sample mode] Label for these samples (used to generate Q and as KB metadata)."
    )
    parser.add_argument(
        "--reset-samples", action="store_true",
        help="[sample mode] Wipe the entire sample KB."
    )
    parser.add_argument(
        "--list-samples", action="store_true",
        help="[sample mode] List all labels in the sample KB and exit."
    )
    # ── Shared LLM args ───────────────────────────────────────────────────────
    parser.add_argument(
        "--llmun", default=config.LLMUN_BACKEND,
        choices=["claude", "openai", "ollama"],
        help=f"LLMun backend (default: {config.LLMUN_BACKEND})."
    )
    parser.add_argument(
        "--llmun-model", default=config.LLMUN_MODEL,
        help=f"LLMun model name (default: {config.LLMUN_MODEL})."
    )
    parser.add_argument(
        "--llmcons", default=config.LLMCONS_BACKEND,
        choices=["claude", "openai"],
        help=f"LLMcons backend for crafting Q (default: {config.LLMCONS_BACKEND})."
    )
    parser.add_argument(
        "--llmcons-model", default=config.LLMCONS_MODEL,
        help=f"LLMcons model name (default: {config.LLMCONS_MODEL})."
    )
    args = parser.parse_args()

    concept_store = ChromaKnowledgeStore()
    sample_store = ChromaKnowledgeStore(
        collection_name=config.CHROMA_SAMPLE_COLLECTION_NAME
    )

    # ── Concept KB list / reset ───────────────────────────────────────────────
    if args.list_concepts:
        concepts = concept_store.list_concepts()
        if concepts:
            print(f"Concepts in concept KB ({concept_store.count()} items total):")
            for c in concepts:
                print(f"  • {c}")
        else:
            print("Concept KB is empty.")
        return

    if args.reset:
        concept_store.reset()
        print("Concept KB wiped.")

    if args.reset_concept:
        concept_store.reset(concept=args.reset_concept)
        print(f"Removed entries for concept: '{args.reset_concept}'.")

    # ── Sample KB list / reset ────────────────────────────────────────────────
    if args.list_samples:
        labels = sample_store.list_concepts()
        if labels:
            print(f"Labels in sample KB ({sample_store.count()} items total):")
            for lb in labels:
                print(f"  • {lb}")
        else:
            print("Sample KB is empty.")
        return

    if args.reset_samples:
        sample_store.reset()
        print("Sample KB wiped.")

    # ── Build ─────────────────────────────────────────────────────────────────
    needs_llm = args.concepts or args.samples
    if not needs_llm:
        parser.print_help()
        return

    print(f"LLMun  : {args.llmun} / {args.llmun_model}")
    print(f"LLMcons: {args.llmcons} / {args.llmcons_model}")
    print()

    llmun = get_llm(args.llmun, args.llmun_model)
    llmcons = get_llm(args.llmcons, args.llmcons_model)

    # ── Concept mode ──────────────────────────────────────────────────────────
    if args.mode == "concept":
        if not args.concepts:
            parser.error("--concepts is required when --mode concept")
        for concept in args.concepts:
            print(f"Building concept knowledge for: '{concept}'")
            items = build_unlearned_knowledge(concept, llmcons=llmcons, llmun=llmun)
            concept_store.add_knowledge(concept, items)
            print(f"  ✓ Stored {len(items)} items for '{concept}' in concept KB.\n")
        print(f"Done. Concept KB now contains {concept_store.count()} items "
              f"across {len(concept_store.list_concepts())} concept(s).")

    # ── Sample mode ───────────────────────────────────────────────────────────
    elif args.mode == "sample":
        if not args.samples:
            parser.error("--samples is required when --mode sample")
        if not args.label:
            parser.error("--label is required when --mode sample")

        with open(args.samples, "r", encoding="utf-8") as f:
            raw_samples = [line.strip() for line in f if line.strip()]

        print(f"Building sample knowledge for label: '{args.label}' ({len(raw_samples)} sample(s))")
        items = build_sample_knowledge(
            label=args.label,
            samples=raw_samples,
        )

        # Each sample gets its own auto-incremented ID as concept key
        start_idx = sample_store.count()
        for i, item in enumerate(items):
            sample_store.add_knowledge(str(start_idx + i), [item])
        print(f"  ✓ Stored {len(items)} items (IDs {start_idx}~{start_idx+len(items)-1}) in sample KB.\n")
        print(f"Done. Sample KB now contains {sample_store.count()} items "
              f"across {sample_store.count()} sample(s).")


if __name__ == "__main__":
    main()
