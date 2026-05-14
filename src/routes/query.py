"""
Query route — POST /chat for document Q&A.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.models import ChatRequest, ChatResponse
from src.pipeline import Pipeline

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about uploaded documents.

    Uses RAG pipeline: embed query → retrieve chunks → LLM answer.
    """
    pipeline = Pipeline()

    try:
        result = pipeline.query(
            question=request.question,
            doc_id=request.doc_id,
            conversation_id=request.conversation_id,
        )
        return ChatResponse(**result)
    except Exception as e:
        return ChatResponse(
            answer=f"查询失败：{e}",
            sources=[],
            confidence=0.0,
            conversation_id=request.conversation_id,
        )
