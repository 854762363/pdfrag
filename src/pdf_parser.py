"""
PDF parsing module — pdf2image + PaddleOCR PP-StructureV3 for Chinese documents.

Architecture:
  PDF file → pdf2image (per-page PNG) → PaddleOCR PPStructureV3
  → structured text blocks (TextBlock list with page/type/text)

PPStructureV3 handles: layout analysis, table extraction (SLANet), OCR (PP-OCRv5).
All in one pipeline optimized for Chinese documents.
"""

from __future__ import annotations

# Must be set before any PaddlePaddle import — CPU oneDNN backend
# doesn't support PIR ArrayAttribute<DoubleAttribute> conversion.
import os
os.environ["FLAGS_use_mkldnn"] = "0"

from pathlib import Path

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError


class PdfParserError(Exception):
    """Raised when PDF parsing fails."""


def get_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF file."""
    if not pdf_path.exists():
        raise PdfParserError(f"PDF file not found: {pdf_path}")

    try:
        from pdf2image.pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path)
        return info["Pages"]
    except Exception as e:
        raise PdfParserError(f"Failed to get page count for {pdf_path}: {e}") from e


def convert_pdf_to_images(
    pdf_path: Path,
    dpi: int = 300,
    output_dir: Path | None = None,
    fmt: str = "png",
) -> list[Path]:
    """Convert a PDF file to a list of images (one per page)."""
    if not pdf_path.exists():
        raise PdfParserError(f"PDF file not found: {pdf_path}")

    if output_dir is None:
        output_dir = pdf_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = convert_from_path(pdf_path, dpi=dpi, fmt=fmt)
    except Exception as e:
        raise PdfParserError(f"Convert failed for {pdf_path}: {e}") from e

    image_paths: list[Path] = []
    for i, img in enumerate(images):
        out_path = output_dir / f"{pdf_path.stem}_page_{i + 1}.{fmt}"
        img.save(str(out_path))
        image_paths.append(out_path)

    return image_paths


def parse_pdf_to_text(
    pdf_path: Path,
    dpi: int = 300,
    lang: str = "ch",
) -> list[dict]:
    """Parse a PDF with PaddleOCR PPStructureV3 for full layout+OCR.

    Uses PPStructureV3 pipeline: layout detection → table extraction → OCR.
    Optimized for Chinese documents with tables and mixed layouts.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Image resolution for pdf2image conversion.
        lang: OCR language ('ch', 'en', 'ch_en').

    Returns:
        List of dicts, each page: {
            "page": int,
            "blocks": [{"type": "text"|"table"|"formula", "text": "...", "bbox": [...]}, ...]
        }
    """
    if not pdf_path.exists():
        raise PdfParserError(f"PDF file not found: {pdf_path}")

    # Step 1: Convert PDF pages to images
    import tempfile
    with tempfile.TemporaryDirectory(prefix="pdfrag_ocr_") as tmpdir:
        tmp = Path(tmpdir)
        images = convert_pdf_to_images(pdf_path, dpi=dpi, output_dir=tmp)

        # Step 2: PaddleOCR PPStructureV3
        try:
            from paddleocr import PPStructureV3
        except ImportError:
            raise PdfParserError(
                "PaddleOCR not installed. Run: pip install paddlepaddle paddleocr"
            )

        engine = PPStructureV3(lang=lang)

        # Batch predict all pages
        image_paths = [str(p) for p in images]
        page_results = engine.predict(image_paths)

        results: list[dict] = []
        for i, page_result in enumerate(page_results):
            # LayoutParsingResultV2 is dict-like
            parsing_res_list = page_result.get(
                "parsing_res_list", page_result["parsing_res_list"]
            )

            blocks = []
            for block in parsing_res_list:
                # LayoutBlock: .label, .content, .bbox
                label = block.label
                content = (block.content or "").strip()
                bbox = list(map(int, block.bbox))

                if not content:
                    continue

                # Post-process OCR text
                content = _clean_ocr_text(content)

                # Filter noise blocks
                if _is_noise_block(content, label):
                    continue

                # Map PPStructureV3 labels to simplified type
                type_map = {
                    "text": "text",
                    "header": "text",
                    "footer": "text",
                    "reference": "text",
                    "footnote": "text",
                    "table": "table",
                    "formula": "formula",
                    "image": "image",
                    "chart": "chart",
                    "seal": "seal",
                }
                region_type = type_map.get(label, "text")

                blocks.append({
                    "type": region_type,
                    "text": content,
                    "bbox": bbox,
                })

            # Deduplicate near-duplicate consecutive blocks
            blocks = _deduplicate_blocks(blocks)

            results.append({"page": i + 1, "blocks": blocks})

        return results


# ---- OCR Text Post-processing ----


def _clean_ocr_text(text: str) -> str:
    """Clean up common OCR artifacts in Chinese text.

    - Normalize whitespace (collapse spaces, excessive newlines)
    - Remove lines that are purely punctuation/symbols
    - Fix common OCR confusion pairs for technical standards
    """
    import re

    # Collapse multiple spaces to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Remove lines that are entirely special characters
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines with no CJK/word/digit content
        if not re.search(r"[\u4e00-\u9fff\w\d]", stripped):
            if len(stripped) <= 3:
                continue
        cleaned.append(line)
    text = "\n".join(cleaned)

    return text.strip()


def _is_noise_block(text: str, label: str) -> bool:
    """Check if a block is likely noise (not meaningful content)."""
    stripped = text.strip()

    # Empty
    if not stripped:
        return True

    # Pure single punctuation
    if stripped in {".", ",", "。", "，", "、"}:
        return True

    # Image/seal blocks with no useful text
    if label in ("image", "seal") and len(stripped) < 5:
        return True

    return False


def _deduplicate_blocks(blocks: list[dict]) -> list[dict]:
    """Remove near-duplicate consecutive blocks."""
    if len(blocks) <= 1:
        return blocks

    result = [blocks[0]]
    for block in blocks[1:]:
        prev = result[-1]
        # Same type, very similar text → skip
        if block["type"] == prev["type"]:
            shorter = min(block["text"], prev["text"], key=len)
            longer = max(block["text"], prev["text"], key=len)
            if shorter and longer:
                ratio = len(shorter) / len(longer) if longer else 0
                if ratio > 0.85 and shorter in longer:
                    continue
            if block["text"] == prev["text"]:
                continue
        result.append(block)

    return result
