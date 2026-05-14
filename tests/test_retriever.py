"""
Tests for src.retriever — vector + BM25 hybrid retrieval.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.chunker import Chunk


# ---- helpers ----

def make_chunk(text: str, chunk_id: str = "1", page: int = 1,
               section: str = "", doc: str = "test") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        document_name=doc,
        page_number=page,
        section_path=section,
    )


# ---- Tests ----

class TestVectorSearch:
    """Test Chroma-based vector search."""

    @pytest.fixture
    def mock_collection(self) -> MagicMock:
        """Mock Chroma collection."""
        col = MagicMock()
        col.query.return_value = {
            "ids": [["c1", "c2"]],
            "documents": [["文本一", "文本二"]],
            "metadatas": [[
                {"chunk_id": "c1", "page_number": 1, "section_path": "第一章"},
                {"chunk_id": "c2", "page_number": 2, "section_path": "第二章"},
            ]],
            "distances": [[0.1, 0.3]],
        }
        return col

    def test_vector_search_returns_correct_count(self, mock_collection: MagicMock) -> None:
        """vector_search returns k results."""
        from src.retriever import vector_search

        query_vec = np.random.randn(512).astype(np.float32).tolist()

        results = vector_search(query_vec, mock_collection, k=5)

        assert len(results) == 2  # mock returns 2
        mock_collection.query.assert_called_once()

    def test_vector_search_result_structure(self, mock_collection: MagicMock) -> None:
        """Each result has id, text, metadata, score."""
        from src.retriever import vector_search

        query_vec = np.random.randn(512).astype(np.float32).tolist()

        results = vector_search(query_vec, mock_collection, k=5)

        for r in results:
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r
            assert 0 <= r["score"] <= 1


class TestBM25Search:
    """Test BM25 keyword search."""

    def test_bm25_build_and_search(self) -> None:
        """BM25 index can be built from chunks and searched."""
        from src.retriever import BM25Retriever

        chunks = [
            make_chunk("深度学习是机器学习的一个分支", chunk_id="1"),
            make_chunk("Python是一种编程语言", chunk_id="2"),
            make_chunk("机器学习需要大量数据", chunk_id="3"),
        ]

        bm25 = BM25Retriever()
        bm25.build(chunks)

        results = bm25.search("机器学习", k=2)

        assert len(results) == 2
        # Chunks about ML should score higher
        assert results[0]["id"] in ("1", "3")

    def test_bm25_empty_chunks(self) -> None:
        """Building BM25 with empty chunks is safe."""
        from src.retriever import BM25Retriever

        bm25 = BM25Retriever()
        bm25.build([])

        results = bm25.search("query", k=3)
        assert results == []

    def test_bm25_result_structure(self) -> None:
        """Each BM25 result has expected fields."""
        from src.retriever import BM25Retriever

        chunks = [make_chunk("测试文本", chunk_id="1")]
        bm25 = BM25Retriever()
        bm25.build(chunks)

        results = bm25.search("测试", k=1)

        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["text"] == "测试文本"
        assert "score" in results[0]


class TestHybridSearch:
    """Test RRF fusion of vector + BM25 results."""

    def test_rrf_fusion_combines_results(self) -> None:
        """RRF fusion produces a merged ranked list."""
        from src.retriever import rrf_fusion

        vec_results = [
            {"id": "a", "text": "A", "score": 0.9},
            {"id": "b", "text": "B", "score": 0.7},
            {"id": "c", "text": "C", "score": 0.5},
        ]
        bm25_results = [
            {"id": "b", "text": "B", "score": 0.8},
            {"id": "c", "text": "C", "score": 0.6},
            {"id": "d", "text": "D", "score": 0.4},
        ]

        merged = rrf_fusion(vec_results, bm25_results, k=3)

        assert len(merged) <= 3
        # b appears in both → should rank high
        assert merged[0]["id"] == "b"

    def test_rrf_fusion_dedup(self) -> None:
        """RRF fusion deduplicates by id."""
        from src.retriever import rrf_fusion

        vec = [{"id": "x", "text": "X", "score": 1.0}]
        bm25 = [{"id": "x", "text": "X", "score": 1.0}]

        merged = rrf_fusion(vec, bm25, k=5)

        assert len(merged) == 1

    def test_rrf_empty_inputs(self) -> None:
        """RRF with empty inputs is safe."""
        from src.retriever import rrf_fusion

        merged = rrf_fusion([], [], k=5)
        assert merged == []
