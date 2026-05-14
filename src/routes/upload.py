"""
Upload route — POST /upload for PDF document ingestion.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from src.config import settings
from src.models import UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF document for processing.

    The file is saved, processed through the pipeline (OCR → chunk → embed),
    and a doc_id is returned for querying.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return UploadResponse(
            doc_id="",
            filename=file.filename or "unknown",
            status="error",
            message="Only PDF files are accepted",
        )

    # Check file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.pdf_max_file_size_mb:
        return UploadResponse(
            doc_id="",
            filename=file.filename,
            status="error",
            message=f"File too large ({size_mb:.1f}MB, max {settings.pdf_max_file_size_mb}MB)",
        )

    # Save file
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / file.filename
    pdf_path.write_bytes(contents)

    # Process through pipeline
    from src.pipeline import Pipeline
    pipeline = Pipeline()

    try:
        result = pipeline.process_document(pdf_path)
        return UploadResponse(**result)
    except Exception as e:
        return UploadResponse(
            doc_id="",
            filename=file.filename,
            status="error",
            message=str(e),
        )
