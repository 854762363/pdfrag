"""
Retrieval module — vector search (Chroma) + BM25 keyword search + RRF fusion.

Architecture:
  vector_search(query_vec, collection, k) → results
  BM25Retriever.build(chunks) → index → .search(query, k) → results
  rrf_fusion(vec_results, bm25_results, k) → merged results
"""

from __future__ import annotations

from typing import Any

from src.chunker import Chunk


def vector_search(
    query_embedding: list[float],
    collection: Any,  # Chroma Collection
    k: int = 5,
) -> list[dict[str, Any]]:
    """Search Chroma collection with a query embedding.

    Args:
        query_embedding: Query vector as list of floats.
        collection: Chroma collection object.
        k: Number of results to return.

    Returns:
        List of result dicts with keys: id, text, metadata, score.
    """
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict[str, Any]] = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for i in range(len(ids)):
        # Convert distance to similarity score (Chroma returns L2/cosine distance)
        dist = distances[i] if i < len(distances) else 1.0
        score = 1.0 / (1.0 + dist)  # normalize to [0, 1]

        results.append({
            "id": ids[i] if i < len(ids) else "",
            "text": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
            "score": score,
        })

    return results


def _jieba_tokenize(text: str) -> list[str]:
    """Tokenize Chinese text with jieba for BM25 keyword search."""
    try:
        import jieba
        return [w.strip() for w in jieba.cut(text) if w.strip()]
    except ImportError:
        # Fallback: character-level (poor but functional)
        return list(text)


class BM25Retriever:
    """BM25 keyword-based retriever for hybrid search."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._index: Any = None

    def build(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        self._chunks = chunks
        if not chunks:
            self._index = None
            return

        from rank_bm25 import BM25Okapi

        # Word-level tokenization with jieba for Chinese
        tokenized = [_jieba_tokenize(c.text) for c in chunks]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search BM25 index.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of result dicts with keys: id, text, metadata, score.
        """
        if self._index is None or not self._chunks:
            return []

        tokenized_query = _jieba_tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        indexed = [(scores[i], i) for i in range(len(scores))]
        indexed.sort(key=lambda x: x[0], reverse=True)

        # Normalize scores
        max_score = indexed[0][0] if indexed else 1.0
        results: list[dict[str, Any]] = []
        for score, idx in indexed[:k]:
            chunk = self._chunks[idx]
            results.append({
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "section_path": chunk.section_path,
                    "document_name": chunk.document_name,
                },
                "score": score / max_score if max_score > 0 else 0.0,
            })

        return results


def rrf_fusion(
    vec_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 5,
    vec_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Fuse vector and BM25 results using Reciprocal Rank Fusion (RRF).

    Args:
        vec_results: Results from vector search.
        bm25_results: Results from BM25 search.
        k: Number of final results to return.
        vec_weight: Weight for vector results (default 0.7).
        bm25_weight: Weight for BM25 results (default 0.3).

    Returns:
        Merged and re-ranked list of result dicts.
    """
    # Build id → result lookup
    id_map: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}

    for rank, r in enumerate(vec_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1.0 / (60 + rank)
        id_map[rid] = r

    for rank, r in enumerate(bm25_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1.0 / (60 + rank)
        if rid not in id_map:
            id_map[rid] = r

    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged = [id_map[rid] for rid, _ in ranked[:k]]
    return merged
