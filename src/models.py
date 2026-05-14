"""
Pydantic data models for API requests and responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---- Request Models ----

class ChatRequest(BaseModel):
    """POST /chat request body."""
    question: str = Field(..., min_length=1, max_length=2000,
                          description="User's natural-language question")
    conversation_id: str = Field(..., min_length=1, max_length=128,
                                  description="Conversation session identifier")
    doc_id: str | None = Field(default=None, max_length=128,
                                 description="Optional specific document ID to query")


class UploadResponse(BaseModel):
    """POST /upload response."""
    doc_id: str
    filename: str
    pages: int = 0
    status: str = "processing"
    message: str = ""


class StatusResponse(BaseModel):
    """GET /status/{doc_id} response."""
    doc_id: str
    filename: str
    status: str               # "processing" | "ready" | "error"
    progress: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Source(BaseModel):
    """A single source reference in an answer."""
    page: int
    section: str = ""
    content_preview: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    """POST /chat response."""
    answer: str
    sources: list[Source] = Field(default_factory=list)
    confidence: float = 0.0
    conversation_id: str = ""


class DocumentInfo(BaseModel):
    """Document summary for listing."""
    doc_id: str
    filename: str
    pages: int = 0
    chunks: int = 0
    status: str = "unknown"
    created_at: str = ""


class DocumentListResponse(BaseModel):
    """GET /documents response."""
    documents: list[DocumentInfo] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    """DELETE /documents/{doc_id} response."""
    doc_id: str
    deleted: bool = False


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""


# ---- Evaluation Models ----


class EvalMetrics(BaseModel):
    """Per-question evaluation metrics."""
    qa_id: str
    question: str
    ground_truth: str
    llm_answer: str
    keyword_recall: float = 0.0       # 0-1, ground-truth keyword hit rate
    semantic_similarity: float = 0.0  # 0-1, BGE embedding cosine similarity
    source_page_match: bool = False   # Whether LLM cited correct page range
    confidence: float = 0.0           # LLM self-reported confidence
    passed: bool = False              # Overall pass/fail


class EvalReport(BaseModel):
    """Full evaluation report."""
    document: str = ""
    total_questions: int = 0
    passed: int = 0
    failed: int = 0
    avg_keyword_recall: float = 0.0
    avg_semantic_similarity: float = 0.0
    avg_confidence: float = 0.0
    results: list[EvalMetrics] = Field(default_factory=list)
    summary: str = ""
