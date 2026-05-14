"""
LLM client module — DeepSeek Chat API wrapper for document Q&A.

Architecture:
  LLMClient(api_key, base_url, model) → .ask(question, context, history) → ChatResponse
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.models import ChatResponse, Source


SYSTEM_PROMPT = """你是一个专业的技术标准文档问答助手。根据提供的文档片段回答用户问题。

规则：
1. 如果文档中有相关信息，请基于文档内容准确回答，并引用来源（页码）。
2. 如果文档中没有相关信息，请如实说明"文档中未找到相关信息"。
3. 回答要简洁、准确，不要编造内容。
4. 使用中文回答。
5. 对于数值型问题（如尺寸、公差、材料参数），请精确引用文档中的数值和单位。
6. 对于表格中的数据，请正确解读并引用。
7. 如果涉及多个条件或分类，请逐条列举。

技术标准文档特点：
- 包含精确的数值要求（如尺寸、公差、材料参数）
- 使用标准编号引用（如 GB/T 1184、GB/T 11334）
- 分类/等级/条件较多
- 数字、单位、符号必须准确"""


class LLMClient:
    """DeepSeek Chat API client for RAG Q&A."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            import httpx
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.Client(
                    timeout=httpx.Timeout(120.0, connect=30.0),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                ),
            )
        return self._client

    @client.setter
    def client(self, value: OpenAI) -> None:
        self._client = value

    def _reset_client(self) -> None:
        """Reset the HTTP client to recover from connection timeouts."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Client already closed or in bad state
        self._client = None

    def _build_prompt(
        self,
        question: str,
        context: list[dict[str, Any]],
    ) -> str:
        """Build the user prompt with context snippets."""
        if not context:
            return question

        parts = ["根据以下文档片段回答问题：\n"]
        for i, item in enumerate(context):
            meta = item.get("metadata", {})
            src = (
                f"[来源: {meta.get('document_name', 'unknown')}, "
                f"章节: {meta.get('section_path', '-')}, "
                f"页码: {meta.get('page_number', '?')}]"
            )
            parts.append(f"--- 片段 {i + 1} {src} ---")
            parts.append(item.get("text", ""))

        parts.append(f"\n问题: {question}")
        parts.append("\n请基于以上文档内容回答，并引用来源。")
        return "\n".join(parts)

    def ask(
        self,
        question: str,
        context: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> ChatResponse:
        """Answer a question based on retrieved context.

        Args:
            question: User's question.
            context: Retrieved chunks from retriever.
            history: Previous conversation turns.

        Returns:
            ChatResponse with answer and sources.
        """
        import time

        user_prompt = self._build_prompt(question, context)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_prompt})

        last_error: Exception | None = None

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content or ""

                # Build sources from context
                sources = []
                for item in context:
                    meta = item.get("metadata", {})
                    sources.append(Source(
                        page=meta.get("page_number", 0),
                        section=meta.get("section_path", ""),
                        content_preview=item.get("text", "")[:100],
                        score=item.get("score", 0.0),
                    ))

                return ChatResponse(
                    answer=answer,
                    sources=sources,
                    confidence=0.8 if context else 0.0,
                )

            except Exception as e:
                last_error = e
                err_msg = str(e)
                is_conn_error = (
                    "closed" in err_msg.lower()
                    or "connection" in err_msg.lower()
                    or "timeout" in err_msg.lower()
                )

                if is_conn_error and attempt < 2:
                    print(f"  ⚠️  LLM connection error (attempt {attempt + 1}/3), "
                          f"resetting client and retrying...")
                    self._reset_client()
                    time.sleep(1.0 * (attempt + 1))  # backoff: 1s, 2s
                    continue
                break

        return ChatResponse(
            answer=f"抱歉，回答生成失败：{last_error}",
            sources=[],
            confidence=0.0,
        )
