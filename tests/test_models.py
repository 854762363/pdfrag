"""
Tests for app.models — validate Pydantic request/response schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    ChatRequest,
    ChatResponse,
    Source,
    UploadResponse,
    StatusResponse,
    ErrorResponse,
    DocumentInfo,
    DocumentListResponse,
    DeleteResponse,
)


class TestChatRequest:
    def test_valid_request(self) -> None:
        req = ChatRequest(question="什么是RAG？", conversation_id="user_001")
        assert req.question == "什么是RAG？"
        assert req.conversation_id == "user_001"
        assert req.doc_id is None

    def test_with_doc_id(self) -> None:
        req = ChatRequest(
            question="最大并发数？",
            conversation_id="user_001",
            doc_id="doc_abc"
        )
        assert req.doc_id == "doc_abc"

    def test_empty_question_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(question="", conversation_id="user_001")

    def test_empty_conversation_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(question="Hello", conversation_id="")

    def test_long_question_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(question="x" * 2001, conversation_id="user_001")

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(question="Hello")  # type: ignore[call-arg]


class TestChatResponse:
    def test_full_response(self) -> None:
        resp = ChatResponse(
            answer="最大并发数为5000 QPS。",
            sources=[
                Source(page=3, section="性能规格",
                       content_preview="并发数 5000 QPS...", score=0.92)
            ],
            confidence=0.89,
            conversation_id="user_001",
        )
        assert resp.answer
        assert len(resp.sources) == 1
        assert resp.sources[0].page == 3
        assert resp.confidence == 0.89

    def test_empty_sources_allowed(self) -> None:
        resp = ChatResponse(answer="未找到相关信息。", confidence=0.0)
        assert resp.sources == []
        assert resp.conversation_id == ""


class TestStatusResponse:
    def test_processing_status(self) -> None:
        resp = StatusResponse(
            doc_id="doc_123",
            filename="report.pdf",
            status="processing",
            progress={"current": 5, "total": 20, "stage": "ocr"},
        )
        assert resp.status == "processing"
        assert resp.progress["stage"] == "ocr"

    def test_error_status(self) -> None:
        resp = StatusResponse(
            doc_id="doc_123",
            filename="broken.pdf",
            status="error",
            error="PDF is encrypted",
        )
        assert resp.status == "error"
        assert resp.error == "PDF is encrypted"


class TestErrorResponse:
    def test_error_response(self) -> None:
        resp = ErrorResponse(error="doc_not_found", detail="Document doc_xyz not found")
        assert resp.error == "doc_not_found"
