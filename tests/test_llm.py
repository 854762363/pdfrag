"""
Tests for src.llm — DeepSeek LLM client.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models import ChatResponse


class TestLLMClient:
    """Test LLMClient class."""

    @pytest.fixture
    def client(self):
        """Create LLMClient with mock settings."""
        from src.llm import LLMClient
        return LLMClient(
            api_key="sk-test",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=1024,
        )

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch("src.llm.OpenAI") as mock:
            client = MagicMock()
            mock.return_value = client
            completion = MagicMock()
            completion.choices = [
                MagicMock(message=MagicMock(content="这是测试答案。"))
            ]
            client.chat.completions.create.return_value = completion
            yield client

    def test_build_prompt_includes_context(self, client) -> None:
        """Prompt should include document context snippets."""
        context = [
            {"id": "c1", "text": "深度学习是AI的分支。", "metadata": {
                "page_number": 3, "section_path": "第一章", "document_name": "test.pdf"
            }},
            {"id": "c2", "text": "Transformer架构于2017年提出。", "metadata": {
                "page_number": 5, "section_path": "第二章", "document_name": "test.pdf"
            }},
        ]

        prompt = client._build_prompt("什么是深度学习？", context)

        assert "深度学习是AI的分支" in prompt
        assert "Transformer架构" in prompt
        assert "什么是深度学习？" in prompt
        assert "test.pdf" in prompt
        assert "第一章" in prompt

    def test_ask_returns_chat_response(self, client, mock_openai) -> None:
        """ask() returns a ChatResponse with answer and sources."""
        context = [
            {"id": "c1", "text": "参考内容。", "metadata": {
                "page_number": 1, "section_path": "概述", "document_name": "test.pdf"
            }},
        ]

        result = client.ask("问题？", context, history=[])

        assert isinstance(result, ChatResponse)
        assert result.answer == "这是测试答案。"
        assert len(result.sources) > 0
        assert result.sources[0].page == 1

    def test_ask_empty_context(self, client, mock_openai) -> None:
        """ask() works with empty context."""
        result = client.ask("问题？", [], history=[])
        assert isinstance(result, ChatResponse)

    def test_ask_includes_history(self, client, mock_openai) -> None:
        """ask() includes conversation history in prompt."""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
        ]

        result = client.ask("继续", [], history=history)
        assert isinstance(result, ChatResponse)
        # Verify history was passed to API
        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert any(m["role"] == "assistant" for m in messages)
        assert any(m["content"] == "你好" for m in messages)

    def test_api_error_returns_error_response(self, client, mock_openai) -> None:
        """API errors return ChatResponse with error message."""
        mock_openai.chat.completions.create.side_effect = Exception("API Error")

        result = client.ask("问题？", [], history=[])

        assert isinstance(result, ChatResponse)
        assert "错误" in result.answer or "error" in result.answer.lower()
