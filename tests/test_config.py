"""
Tests for app.config module.
"""

from __future__ import annotations

import os

import pytest

from src.config import Settings


class TestSettings:
    """Test configuration loading and defaults."""

    def test_default_values(self) -> None:
        """Verify all settings have sensible defaults."""
        s = Settings()

        assert s.pdf_dpi == 300
        assert s.pdf_max_pages == 200
        assert s.pdf_max_file_size_mb == 50
        assert s.ocr_lang == "ch"
        assert s.ocr_use_angle_cls is True

        assert s.chunk_size == 512
        assert s.chunk_overlap == 64
        assert s.chunk_min_size == 50

        assert s.embedding_model == "BAAI/bge-small-zh-v1.5"
        assert s.embedding_dim == 512
        assert s.embedding_batch_size == 32

        assert s.retrieval_top_k == 5
        assert s.retrieval_vector_weight == 0.7
        assert s.retrieval_bm25_weight == 0.3

        assert s.llm_model == "deepseek-chat"
        assert s.llm_temperature == 0.3
        assert s.llm_max_tokens == 1024

        assert s.conversation_max_history == 10

        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.debug is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify environment variables override defaults."""
        monkeypatch.setenv("CHUNK_SIZE", "1024")
        monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
        monkeypatch.setenv("DEBUG", "true")

        s = Settings()

        assert s.chunk_size == 1024
        assert s.llm_model == "deepseek-reasoner"
        assert s.debug is True
        # Non-overridden values stay at default
        assert s.pdf_dpi == 300

    def test_path_resolution(self) -> None:
        """Verify Path fields default to relative paths."""
        s = Settings()
        assert str(s.data_dir) == "data"
        assert str(s.chroma_dir) == "data/chroma"

    def test_immutability_after_creation(self) -> None:
        """Settings should be frozen/hashable after creation."""
        s = Settings()
        # Should be able to access all fields without error
        _ = s.model_dump()
