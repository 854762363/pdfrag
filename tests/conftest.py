"""
Pytest configuration and shared fixtures for PDFRAG tests.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Temporary data directory that cleans up after the test."""
    with tempfile.TemporaryDirectory(prefix="pdfrag_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_pdf_dir(temp_data_dir: Path) -> Path:
    """Temporary directory for test PDF uploads."""
    uploads = temp_data_dir / "uploads"
    uploads.mkdir(exist_ok=True)
    return uploads


@pytest.fixture(autouse=True)
def isolate_env() -> Generator[None, None, None]:
    """Isolate environment variables during tests."""
    original = os.environ.copy()
    yield
    # Restore
    os.environ.clear()
    os.environ.update(original)


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration overrides for testing."""
    return {
        "pdf_dpi": 150,
        "chunk_size": 256,
        "retrieval_top_k": 3,
        "llm_model": "deepseek-chat",
    }
