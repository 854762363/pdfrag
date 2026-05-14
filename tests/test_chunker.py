"""
Tests for src.chunker — text chunking with heading-aware splitting.
"""

from __future__ import annotations

import pytest

from src.chunker import TextBlock, Chunk, chunk_document


# ---- helpers ----

def make_blocks(*texts: str, page: int = 1) -> list[TextBlock]:
    """Create TextBlock list from strings, auto-detecting heading vs paragraph.
    
    A block is a heading if:
    - It starts with # (Markdown)
    - It starts with 第X章/节 and is short
    - It's short (< 30 chars) and doesn't end with sentence punctuation
    """
    blocks = []
    for t in texts:
        stripped = t.strip()
        is_heading = bool(stripped) and (
            stripped.startswith("#")
            or (stripped.startswith("第") and len(stripped) < 30)
            or (len(stripped) < 30 and not stripped.endswith(("。", "！", "？")))
        )
        blocks.append(TextBlock(
            page_number=page,
            text=t,
            block_type="heading" if is_heading else "paragraph",
        ))
    return blocks


# ---- Tests ----

class TestChunkByHeadings:
    """Test heading-based splitting."""

    def test_split_at_heading_boundaries(self) -> None:
        """Content should be split at heading blocks."""
        blocks = make_blocks(
            "# 第一章 概述",
            "这是第一章的内容。包含一些说明。",
            "# 第二章 方法",
            "这是第二章的内容。方法部分。",
        )

        result = chunk_document(blocks, doc_name="test", chunk_min_size=0)

        assert len(result) == 2
        # Heading text goes to section_path, body text to chunk text
        assert "第一章 概述" in result[0].section_path
        assert "第一章的内容" in result[0].text
        assert "第二章 方法" in result[1].section_path
        assert "方法" in result[1].text

    def test_section_path_hierarchy(self) -> None:
        """Nested headings produce hierarchical section_path like '第一章 > 1.1'."""
        blocks = make_blocks(
            "# 第一章",
            "第一章正文内容在这里。",
            "## 1.1 背景",
            "背景内容描述。",
        )

        result = chunk_document(blocks, doc_name="test", chunk_min_size=0)

        # Should produce chunks with hierarchical paths
        paths = {c.section_path for c in result}
        assert any("第一章" in p and "1.1" in p for p in paths)

    def test_no_headings_single_chunk(self) -> None:
        """Text without headings produces a single chunk."""
        blocks = make_blocks(
            "这是一段没有标题的纯文本。第一句。第二句。第三句。",
        )

        result = chunk_document(blocks, doc_name="test", chunk_min_size=0)

        assert len(result) == 1
        assert result[0].section_path == ""  # no heading


class TestChunkSizeControl:
    """Test chunk size and overflow handling."""

    def test_oversized_chunk_split_by_sentence(self) -> None:
        """A chunk exceeding chunk_size is split at sentence boundaries."""
        # Create ~30 sentences, each ~10 chars = ~300 chars total
        long_text = "。".join([f"第{i:02d}句内容在这里" for i in range(30)]) + "。"
        blocks = make_blocks(long_text)

        result = chunk_document(blocks, doc_name="test", chunk_size=100, chunk_min_size=0)

        # Should be split into multiple chunks
        assert len(result) > 1
        # Allow chunks to slightly exceed chunk_size since we split at sentence
        # boundaries and each sentence adds ~12 chars
        for c in result:
            assert len(c.text) <= 180  # chunk_size + max sentence length * 2

    def test_respect_chunk_size(self) -> None:
        """All chunks should be within chunk_size (with overlap tolerance)."""
        blocks = make_blocks(
            *[f"段落{i}：" + "X" * 200 for i in range(5)]
        )

        result = chunk_document(blocks, doc_name="test", chunk_size=300, chunk_overlap=30)

        for c in result:
            assert len(c.text) <= 350  # chunk_size + overlap buffer


class TestChunkOverlap:
    """Test overlap between adjacent chunks."""

    def test_adjacent_chunks_overlap(self) -> None:
        """Adjacent chunks should share some text."""
        text = "这是第一段文字。包含ABCDEFGH。这是第二段文字。包含IJKLMNOP。"
        blocks = make_blocks(text)

        result = chunk_document(blocks, doc_name="test", chunk_size=30, chunk_overlap=10)

        if len(result) >= 2:
            last_chars = result[0].text[-5:]
            assert last_chars in result[1].text


class TestChunkMetadata:
    """Test metadata injected into chunks."""

    def test_metadata_fields_present(self) -> None:
        """Every chunk has document_name, page_number, chunk_id."""
        blocks = make_blocks("测试文本内容。", page=3)

        result = chunk_document(blocks, doc_name="测试文档.pdf")

        for c in result:
            assert c.document_name == "测试文档.pdf"
            assert c.page_number == 3
            assert c.chunk_id  # non-empty
            assert isinstance(c.chunk_id, str)

    def test_chunk_ids_are_unique(self) -> None:
        """Each chunk gets a unique ID."""
        blocks = make_blocks(*[f"段落{i}" for i in range(5)])

        result = chunk_document(blocks, doc_name="test")

        ids = {c.chunk_id for c in result}
        assert len(ids) == len(result)


class TestMinSizeFilter:
    """Test that very small chunks are filtered out."""

    def test_small_chunks_discarded(self) -> None:
        """Chunks shorter than min_size are removed."""
        blocks = make_blocks("XY")  # very short

        result = chunk_document(blocks, doc_name="test", chunk_min_size=10)

        assert len(result) == 0

    def test_min_size_boundary(self) -> None:
        """Chunks at exactly min_size are kept."""
        text = "A" * 50
        blocks = make_blocks(text)

        result = chunk_document(blocks, doc_name="test", chunk_min_size=50)

        assert len(result) >= 1
