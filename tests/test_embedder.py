"""
Tests for src.embedder — text embedding with sentence-transformers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbedder:
    """Test Embedder class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Mock sentence-transformers model returning fixed embeddings."""
        mock = MagicMock()
        mock.encode.return_value = np.random.randn(3, 512).astype(np.float32)
        return mock

    @pytest.fixture
    def embedder(self, mock_model: MagicMock):
        """Create Embedder with mocked model."""
        from src.embedder import Embedder
        emb = Embedder.__new__(Embedder)
        emb.model = mock_model
        emb.model_name = "mock-model"
        emb.dim = 512
        return emb

    def test_embed_batch_returns_correct_shape(self, embedder) -> None:
        """embed_batch returns array of shape (N, embedding_dim)."""
        texts = ["第一段文本", "第二段文本", "第三段文本"]
        result = embedder.embed_batch(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 512)
        assert result.dtype == np.float32

    def test_embed_query_returns_vector(self, embedder) -> None:
        """embed_query returns 1D vector of dim embedding_dim."""
        embedder.model.encode.return_value = np.random.randn(1, 512).astype(np.float32)

        result = embedder.embed_query("一个问题")

        assert isinstance(result, np.ndarray)
        assert result.shape == (512,)
        assert result.dtype == np.float32

    def test_embed_batch_empty_list(self, embedder) -> None:
        """Embedding empty list returns empty array."""
        result = embedder.embed_batch([])
        assert result.shape == (0, 512)

    def test_embed_query_empty_string_raises(self, embedder) -> None:
        """Embedding empty string raises ValueError."""
        from src.embedder import Embedder
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_query("")

    def test_lazy_model_loading(self) -> None:
        """Model is loaded lazily on first use, not at __init__."""
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            from src.embedder import Embedder
            emb = Embedder(model_name="test-model")
            mock_st.assert_not_called()
            emb._load_model()
            mock_st.assert_called_once_with("test-model", device="cpu")
