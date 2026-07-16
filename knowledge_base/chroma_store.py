"""
ChromaDB-backed vector store for unlearned knowledge items.

Each document stored here is one knowledge item Ki = Pi + Q for some concept.
Metadata field "concept" allows per-concept reset.

Similarity is cosine distance (ChromaDB default with sentence-transformers).
A retrieval is considered a "hit" only when the score exceeds RETRIEVAL_THRESHOLD.
"""

from __future__ import annotations

import uuid
import chromadb
from chromadb.utils import embedding_functions

import config


class ChromaKnowledgeStore:
    def __init__(
        self,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
        embed_model: str = config.EMBED_MODEL,
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────────

    def add_knowledge(self, concept: str, knowledge_items: list[str]) -> None:
        """
        Store all knowledge items for a concept.
        Each item gets a unique ID.  Existing items for the same concept are
        NOT removed — call reset(concept) first if you want a clean slate.
        """
        ids = [str(uuid.uuid4()) for _ in knowledge_items]
        metadatas = [{"concept": concept} for _ in knowledge_items]
        self._collection.add(
            documents=knowledge_items,
            ids=ids,
            metadatas=metadatas,
        )

    # ── Read ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        threshold: float = config.RETRIEVAL_THRESHOLD,
    ) -> tuple[str, float] | None:
        """
        Return the best matching knowledge item and its similarity score,
        or None if no item exceeds `threshold`.

        ChromaDB returns *distances* in cosine space (0 = identical, 2 = opposite).
        We convert to similarity = 1 - distance/2  so that 1.0 = perfect match.
        """
        count = self._collection.count()
        if count == 0:
            return None

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "distances"],
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        # Convert cosine distance → similarity, keep best
        best_doc, best_sim = None, -1.0
        for doc, dist in zip(documents, distances):
            sim = 1.0 - dist / 2.0
            if sim > best_sim:
                best_doc, best_sim = doc, sim

        if best_sim >= threshold:
            return best_doc, best_sim
        return None

    def count(self) -> int:
        return self._collection.count()

    def list_concepts(self) -> list[str]:
        """Return sorted list of distinct concept names in the store."""
        if self._collection.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        concepts = {m["concept"] for m in results["metadatas"]}
        return sorted(concepts)

    # ── Delete ─────────────────────────────────────────────────────────────

    def reset(self, concept: str | None = None) -> None:
        """
        Remove items from the store.

        Args:
            concept: If given, remove only items for that concept.
                     If None, wipe the entire collection.
        """
        if concept is None:
            # Delete and recreate the collection
            self._client.delete_collection(self._collection.name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection.name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            results = self._collection.get(
                where={"concept": concept},
                include=["metadatas"],
            )
            ids = results["ids"]
            if ids:
                self._collection.delete(ids=ids)
