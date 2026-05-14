"""
Pipeline orchestrator — coordinates the full RAG pipeline.

Upload flow:
  PDF → pdf_parser → chunker → embedder → Chroma

Query flow:
  Question → embedder → retriever → llm → answer
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from src.chunker import Chunk, TextBlock, chunk_document
from src.config import settings
from src.embedder import Embedder
from src.llm import LLMClient
from src.pdf_parser import PdfParserError, get_page_count, parse_pdf_to_text
from src.retriever import BM25Retriever, rrf_fusion, vector_search


class Pipeline:
    """Full RAG pipeline orchestrator."""

    def __init__(self):
        self.embedder = Embedder(
            model_name=settings.embedding_model,
        )
        self.llm = LLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        self._chroma_client: Any = None
        self._conversations: dict[str, list[dict[str, str]]] = {}
        # BM25: store chunks per doc_id for hybrid search
        self._bm25_indexes: dict[str, BM25Retriever] = {}
        self._doc_chunks: dict[str, list[Chunk]] = {}

    @property
    def chroma(self):
        if self._chroma_client is None:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(settings.chroma_dir)
            )
        return self._chroma_client

    def process_document(self, pdf_path: Path) -> dict[str, Any]:
        """Process a PDF through the full pipeline.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict with doc_id, filename, pages, chunks, status.
        """
        doc_id = str(uuid.uuid4())[:12]
        filename = pdf_path.name

        # 1. Get page count
        pages = get_page_count(pdf_path)
        if pages > settings.pdf_max_pages:
            raise PdfParserError(
                f"PDF has {pages} pages, max allowed is {settings.pdf_max_pages}"
            )

        # 2. Parse PDF with PaddleOCR PP-Structure
        ocr_result = parse_pdf_to_text(
            pdf_path,
            dpi=settings.pdf_dpi,
            lang=settings.ocr_lang,
        )

        # 3. Convert OCR result to TextBlocks
        blocks: list[TextBlock] = []
        for page_data in ocr_result:
            page_num = page_data["page"]
            for region in page_data["blocks"]:
                blocks.append(TextBlock(
                    page_number=page_num,
                    text=region["text"],
                    block_type=region.get("type", "paragraph"),
                ))

        # 4. Chunk
        chunks = chunk_document(
            blocks,
            doc_name=filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunk_min_size=settings.chunk_min_size,
        )

        if not chunks:
            raise PdfParserError("No text chunks were produced from the document")

        # 5. Build BM25 keyword index for hybrid search
        bm25 = BM25Retriever()
        bm25.build(chunks)
        self._bm25_indexes[doc_id] = bm25
        self._doc_chunks[doc_id] = chunks

        # 6. Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        # 7. Store in Chroma
        collection = self.chroma.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        collection.add(
            ids=[f"{doc_id}_{c.chunk_id}" for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {
                    "doc_id": doc_id,
                    "chunk_id": c.chunk_id,
                    "page_number": c.page_number,
                    "section_path": c.section_path,
                    "document_name": c.document_name,
                }
                for c in chunks
            ],
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "pages": pages,
            "chunks": len(chunks),
            "status": "ready",
        }

    def query(
        self,
        question: str,
        doc_id: str | None = None,
        conversation_id: str = "default",
    ) -> dict[str, Any]:
        """Answer a question using the RAG pipeline.

        Args:
            question: User's question.
            doc_id: Optional document ID to restrict search.
            conversation_id: Conversation session ID.

        Returns:
            Dict with answer, sources, confidence.
        """
        # 1. Get or create collection
        collection = self.chroma.get_or_create_collection(
            name=settings.chroma_collection_name,
        )

        # 2. Embed query
        query_vec = self.embedder.embed_query(question)

        # 3. Vector search
        where_filter = None
        if doc_id:
            where_filter = {"doc_id": doc_id}

        vec_results = vector_search(
            query_vec.tolist(),
            collection,
            k=settings.retrieval_top_k,
        )

        # 4. BM25 keyword search
        bm25_results: list[dict[str, Any]] = []
        if doc_id and doc_id in self._bm25_indexes:
            # Search within the specific document
            bm25 = self._bm25_indexes[doc_id]
            bm25_results = bm25.search(question, k=settings.retrieval_top_k)
        elif self._bm25_indexes:
            # Search across all documents
            all_bm25: list[dict[str, Any]] = []
            for _doc_id, bm25 in self._bm25_indexes.items():
                all_bm25.extend(bm25.search(question, k=settings.retrieval_top_k // 2))
            # Sort by score descending
            all_bm25.sort(key=lambda x: x.get("score", 0), reverse=True)
            bm25_results = all_bm25[:settings.retrieval_top_k]

        # 5. RRF fusion
        merged = rrf_fusion(
            vec_results,
            bm25_results,
            k=settings.retrieval_top_k,
        )

        # 6. LLM answer
        history = self._conversations.get(conversation_id, [])
        response = self.llm.ask(question, merged, history=history)

        # 7. Save conversation history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response.answer})
        if len(history) > settings.conversation_max_history * 2:
            history = history[-settings.conversation_max_history * 2:]
        self._conversations[conversation_id] = history

        return response.model_dump()
