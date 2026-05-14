"""
Text embedding module — wraps sentence-transformers for chunk vectorization.

Architecture:
  Embedder(model_name) → lazy-load SentenceTransformer
  embed_batch(texts) → ndarray (N, dim)
  embed_query(text) → ndarray (dim,)
  store_chunks() → Chroma collection (future: integrate with retriever)
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Text embedder using sentence-transformers.

    Lazy-loads the model on first use to avoid startup delay.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.dim = 512  # bge-small-zh-v1.5 output dimension

    def _load_model(self) -> None:
        """Load the sentence-transformers model (lazy)."""
        if self._model is not None:
            return
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def model(self):
        """Access the underlying model, loading if necessary."""
        self._load_model()
        return self._model

    @model.setter
    def model(self, value):
        """Allow injecting mock model for testing."""
        self._model = value

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            ndarray of shape (len(texts), dim) with float32 dtype.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text.

        Args:
            text: Query string.

        Returns:
            1D ndarray of shape (dim,) with float32 dtype.

        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Query text cannot be empty")
        result = self.model.encode([text], convert_to_numpy=True)
        return result[0]
