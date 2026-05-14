"""
Tests for src.pdf_parser — PDF to image conversion.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---- helpers ----

def make_minimal_pdf(path: Path, pages: int = 1) -> Path:
    """Create a minimal valid PDF with *pages* blank pages."""
    objects: list[str] = []
    offsets: list[int] = []

    # Object 1: Catalog
    objects.append("1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n")
    # Object 2: Pages
    kids = " ".join(f"{i} 0 R" for i in range(3, 3 + pages))
    objects.append(f"2 0 obj<</Type/Pages/Kids[{kids}]/Count {pages}>>endobj\n")
    # Page objects
    for i in range(pages):
        objects.append(
            f"{3 + i} 0 obj<</Type/Page/MediaBox[0 0 612 792]"
            f"/Parent 2 0 R/Resources<<>>>>endobj\n"
        )

    # Compute offsets
    header = b"%PDF-1.4\n"
    body = b""
    for obj in objects:
        offsets.append(len(header) + len(body))
        body += obj.encode()

    xref_offset = len(header) + len(body)
    xref = b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode()

    path.write_bytes(header + body + xref + trailer)
    return path


def make_corrupt_pdf(path: Path) -> Path:
    """Create a file that is NOT a valid PDF (garbage bytes)."""
    path.write_bytes(b"this is not a pdf file\n")
    return path


# ---- Tests ----

class TestConvertPdfToImages:
    """Test convert_pdf_to_images function."""

    def test_returns_list_of_paths(self, tmp_path: Path) -> None:
        """Convert a single-page PDF and get back a list of image Paths."""
        from src.pdf_parser import convert_pdf_to_images

        pdf_path = make_minimal_pdf(tmp_path / "test.pdf", pages=1)

        result = convert_pdf_to_images(pdf_path, dpi=72)

        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(p, Path) for p in result)
        assert result[0].exists()

    def test_multi_page_conversion(self, tmp_path: Path) -> None:
        """Convert a 3-page PDF and get back 3 images."""
        from src.pdf_parser import convert_pdf_to_images

        pdf_path = make_minimal_pdf(tmp_path / "test.pdf", pages=3)

        result = convert_pdf_to_images(pdf_path, dpi=72)

        assert len(result) == 3
        for i, p in enumerate(result):
            assert p.exists()
            assert p.suffix == ".png"
            assert f"_page_{i + 1}" in p.name

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        """Output images go to the specified directory."""
        from src.pdf_parser import convert_pdf_to_images

        pdf_path = make_minimal_pdf(tmp_path / "test.pdf", pages=1)
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        result = convert_pdf_to_images(pdf_path, dpi=72, output_dir=out_dir)

        assert len(result) == 1
        assert result[0].parent == out_dir

    def test_custom_format(self, tmp_path: Path) -> None:
        """Output images in JPEG format when fmt='jpeg'."""
        from src.pdf_parser import convert_pdf_to_images

        pdf_path = make_minimal_pdf(tmp_path / "test.pdf", pages=1)

        result = convert_pdf_to_images(pdf_path, dpi=72, fmt="jpeg")

        assert result[0].suffix == ".jpeg"


class TestConvertPdfToImagesErrors:
    """Test error handling in convert_pdf_to_images."""

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Converting a nonexistent file should raise PdfParserError."""
        from src.pdf_parser import convert_pdf_to_images, PdfParserError

        bad_path = tmp_path / "does_not_exist.pdf"

        with pytest.raises(PdfParserError, match="not found"):
            convert_pdf_to_images(bad_path)

    def test_corrupt_pdf_raises(self, tmp_path: Path) -> None:
        """Converting a corrupt/non-PDF file should raise PdfParserError."""
        from src.pdf_parser import convert_pdf_to_images, PdfParserError

        corrupt = make_corrupt_pdf(tmp_path / "corrupt.pdf")

        with pytest.raises(PdfParserError, match="[Cc]onvert"):
            convert_pdf_to_images(corrupt)


class TestGetPageCount:
    """Test get_page_count function."""

    def test_single_page(self, tmp_path: Path) -> None:
        """get_page_count returns 1 for a 1-page PDF."""
        from src.pdf_parser import get_page_count

        pdf_path = make_minimal_pdf(tmp_path / "test.pdf", pages=1)
        assert get_page_count(pdf_path) == 1

    def test_multi_page(self, tmp_path: Path) -> None:
        """get_page_count returns correct count for multi-page PDF."""
        from src.pdf_parser import get_page_count

        pdf_path = make_minimal_pdf(tmp_path / "test.pdf", pages=5)
        assert get_page_count(pdf_path) == 5

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """get_page_count should raise PdfParserError for nonexistent files."""
        from src.pdf_parser import get_page_count, PdfParserError

        with pytest.raises(PdfParserError, match="not found"):
            get_page_count(tmp_path / "missing.pdf")
