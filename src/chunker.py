"""
Text chunking module — heading-aware recursive splitting with overlap.

Strategy:
  1. Detect heading blocks → build section hierarchy
  2. Split at heading boundaries
  3. For oversized chunks, split at sentence boundaries (。！？)
  4. Add overlap between adjacent chunks
  5. Inject metadata (doc_name, page, section_path, chunk_id)
  6. Filter out chunks shorter than min_size
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    """A block of text extracted from a PDF page."""
    page_number: int
    text: str
    block_type: str = "paragraph"  # "heading" | "paragraph" | "table"
    font_size: float = 0.0


@dataclass
class Chunk:
    """A text chunk with metadata for retrieval."""
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    section_path: str = ""   # e.g. "第一章 > 1.1 > 1.1.2"
    start_char: int = 0
    end_char: int = 0


# Heading detection pattern:
# - Markdown headings (#, ##, ###...)
# - Chinese chapter/section patterns (第X章, 第X节, X., X.X, 一、...)
HEADING_PATTERN = re.compile(
    r"^(#{1,6}\s)"               # Markdown headings
    r"|^(第[一二三四五六七八九十\d]+[章节条款])"  # Chinese chapters
    r"|^([\d]+[\.\、])"           # Numbered headings
    r"|^([一二三四五六七八九十]+[\.\、])"  # Chinese numbered
)

SENTENCE_END = re.compile(r"[。！？；\n]")


def _is_heading(block: TextBlock) -> bool:
    """Check if a block looks like a heading."""
    if block.block_type == "heading":
        return True
    text = block.text.strip()
    if not text:
        return False
    if HEADING_PATTERN.match(text):
        return True
    # Short lines without sentence-ending punctuation are likely headings
    if len(text) < 30 and not SENTENCE_END.search(text):
        return True
    return False


def _split_sentences(text: str, max_chars: int) -> list[str]:
    """Split text into chunks at sentence boundaries, respecting max_chars."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    # Split by sentence boundaries
    parts = SENTENCE_END.split(text)
    for part in parts:
        if not part:
            continue
        candidate = current + part + "。"

        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = part + "。"
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    # If any chunk is still > max_chars, force-split by char
    final: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            # Force split at midpoint
            for i in range(0, len(c), max_chars):
                final.append(c[i:i + max_chars])

    return final


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlapping text between adjacent chunks."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = result[-1]
        curr = chunks[i]
        if len(prev) > overlap:
            overlap_text = prev[-overlap:]
        else:
            overlap_text = prev
        result.append(overlap_text + curr)

    return result


def chunk_document(
    blocks: list[TextBlock],
    doc_name: str = "unknown",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    chunk_min_size: int = 50,
) -> list[Chunk]:
    """Split a document into chunks with heading awareness.

    Args:
        blocks: List of TextBlocks from PDF parsing.
        doc_name: Document filename for metadata.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters between adjacent chunks.
        chunk_min_size: Discard chunks shorter than this.

    Returns:
        List of Chunk objects ready for embedding.
    """
    if not blocks:
        return []

    # Step 1: Build section hierarchy from headings
    sections: list[tuple[str, list[TextBlock]]] = []
    section_stack: list[str] = []  # hierarchical path stack
    current_blocks: list[TextBlock] = []

    for block in blocks:
        if _is_heading(block):
            # Save previous section if not empty
            if current_blocks:
                sections.append((" > ".join(section_stack), current_blocks))
                current_blocks = []

            heading_text = block.text.strip().lstrip("#").strip()

            # Determine heading level for hierarchy
            level = 1
            if block.text.strip().startswith("#"):
                level = len(block.text) - len(block.text.lstrip("#"))
            # Adjust stack to current level
            while len(section_stack) >= level:
                section_stack.pop()
            section_stack.append(heading_text)
        else:
            current_blocks.append(block)

    # Don't forget the last section
    if current_blocks or not sections:
        sections.append((" > ".join(section_stack) if section_stack else "", current_blocks))

    # Handle case where sections is empty (shouldn't happen but be safe)
    if not sections:
        return []

    # Step 2: Create chunks from each section
    chunks: list[Chunk] = []

    for section_path, section_blocks in sections:
        # Combine all text in this section
        section_text = "".join(b.text for b in section_blocks)
        if not section_text.strip():
            continue

        # Determine page number (use the first block's page)
        page = section_blocks[0].page_number if section_blocks else 1

        # Split into sentence-level chunks
        raw_chunks = _split_sentences(section_text, chunk_size)

        # Add overlap
        final_texts = _add_overlap(raw_chunks, chunk_overlap)

        for text in final_texts:
            text = text.strip()
            if len(text) < chunk_min_size:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4())[:8],
                text=text,
                document_name=doc_name,
                page_number=page,
                section_path=section_path,
            ))

    return chunks
