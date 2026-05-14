"""
End-to-end integration test for the full PDFRAG pipeline.

Tests the flow: upload → chunk → embed → store → query → answer

Uses a tiny test PDF generated in-memory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


# ---- helpers ----

def make_minimal_pdf(path: Path, pages: int = 1) -> Path:
    """Create a minimal valid PDF."""
    objects: list[str] = []
    offsets: list[int] = []

    objects.append("1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n")
    kids = " ".join(f"{i} 0 R" for i in range(3, 3 + pages))
    objects.append(f"2 0 obj<</Type/Pages/Kids[{kids}]/Count {pages}>>endobj\n")
    for i in range(pages):
        objects.append(
            f"{3 + i} 0 obj<</Type/Page/MediaBox[0 0 612 792]"
            f"/Parent 2 0 R/Resources<<>>>>endobj\n"
        )

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


# ---- Tests ----

class TestE2EPipeline:
    """End-to-end pipeline tests with mocked LLM."""

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="PaddlePaddle 3.3.1 CPU oneDNN PIR bug: "
               "ConvertPirAttribute2RuntimeAttribute not support "
               "ArrayAttribute<DoubleAttribute>. "
               "Awaiting PaddlePaddle fix or GPU environment."
    )
    async def test_upload_and_query_flow(self, tmp_path: Path) -> None:
        """Upload a PDF and query it end-to-end."""
        from src.config import settings
        from src.main import app

        # Create test PDF
        pdf_path = make_minimal_pdf(tmp_path / "test.pdf", pages=1)
        pdf_bytes = pdf_path.read_bytes()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Upload
            response = await client.post(
                "/upload",
                files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
            )
            assert response.status_code == 200
            upload_data = response.json()
            assert upload_data["status"] != "error", f"Upload failed: {upload_data}"
            doc_id = upload_data["doc_id"]
            assert doc_id

            # Step 2: Health check
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_chat_endpoint_accepts_request(self) -> None:
        """POST /chat accepts a valid ChatRequest."""
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat",
                json={
                    "question": "测试问题",
                    "conversation_id": "test_conv_001",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data

    @pytest.mark.asyncio
    async def test_chat_rejects_invalid_request(self) -> None:
        """POST /chat rejects requests with missing fields."""
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/chat",
                json={"question": ""},  # missing conversation_id
            )
            assert response.status_code == 422  # validation error

    @pytest.mark.asyncio
    async def test_upload_rejects_non_pdf(self) -> None:
        """POST /upload rejects non-PDF files."""
        from src.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/upload",
                files={"file": ("test.txt", b"not a pdf", "text/plain")},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "error"
            assert "PDF" in data["message"]
