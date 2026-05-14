"""
Application configuration via environment variables.
All settings have sensible defaults for development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Paths ----
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma")
    cache_dir: Path = Path("data/cache")

    # ---- PDF Processing ----
    pdf_dpi: int = 300
    pdf_max_pages: int = 200
    pdf_max_file_size_mb: int = 50
    ocr_lang: Literal["ch", "en", "ch_en"] = "ch"
    ocr_use_angle_cls: bool = True

    # ---- Chunking ----
    chunk_size: int = 512          # max chars per chunk
    chunk_overlap: int = 64        # overlap between adjacent chunks
    chunk_min_size: int = 50       # discard chunks shorter than this

    # ---- Embeddings ----
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_batch_size: int = 32

    # ---- Vector DB ----
    chroma_collection_name: str = "pdfrag_docs"

    # ---- Retrieval ----
    retrieval_top_k: int = 5
    retrieval_vector_weight: float = 0.7   # vs BM25 weight
    retrieval_bm25_weight: float = 0.3

    # ---- LLM ----
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"  # DeepSeek-V3, latest chat model
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # ---- Conversation ----
    conversation_max_history: int = 10

    # ---- Server ----
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # ---- Rate Limiting ----
    rate_limit_chat_per_minute: int = 20
    rate_limit_upload_per_minute: int = 3

    # ---- Logging ----
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Singleton
settings = Settings()
