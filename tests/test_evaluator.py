"""
Tests for src.evaluator — Q&A answer quality evaluation.
"""

from __future__ import annotations

import pytest


class TestKeywordMetrics:
    """Test keyword recall and precision computation."""

    def test_keyword_recall_perfect_match(self) -> None:
        """Perfect keyword overlap → high recall."""
        from src.evaluator import compute_keyword_recall

        score = compute_keyword_recall(
            ground_truth="键的抗拉强度应大于等于 590 MPa",
            llm_answer="根据标准，键的抗拉强度应大于等于 590 MPa",
        )
        assert score >= 0.6, f"Expected >= 0.6, got {score}"

    def test_keyword_recall_no_match(self) -> None:
        """Completely unrelated → low recall."""
        from src.evaluator import compute_keyword_recall

        score = compute_keyword_recall(
            ground_truth="键的抗拉强度应大于等于 590 MPa",
            llm_answer="今天天气很好适合出去玩",
        )
        assert score < 0.2, f"Expected < 0.2, got {score}"

    def test_keyword_recall_partial_match(self) -> None:
        """Partial keyword overlap → moderate recall."""
        from src.evaluator import compute_keyword_recall

        score = compute_keyword_recall(
            ground_truth="键表面不允许有裂纹、浮锈、氧化皮和毛刺",
            llm_answer="键表面不允许有裂纹和毛刺",
        )
        assert 0.3 < score < 0.8, f"Expected 0.3~0.8, got {score}"

    def test_keyword_recall_handles_numbers(self) -> None:
        """Numbers in answer should be recognized as keywords."""
        from src.evaluator import compute_keyword_recall

        score = compute_keyword_recall(
            ground_truth="抗拉强度应大于等于 590 MPa",
            llm_answer="590 MPa",
        )
        assert score > 0, f"Expected > 0, got {score}"


class TestSemanticSimilarity:
    """Test semantic similarity computation with BGE embeddings."""

    def test_similarity_same_meaning(self) -> None:
        """Same meaning, different wording → high similarity."""
        from src.evaluator import compute_semantic_similarity

        sim = compute_semantic_similarity(
            "键表面不允许有裂纹、浮锈、氧化皮和毛刺",
            "标准规定键的表面不应出现裂纹、浮锈、氧化皮以及毛刺等缺陷",
        )
        assert sim > 0.7, f"Expected > 0.7, got {sim}"

    def test_similarity_unrelated(self) -> None:
        """Completely different topics → low similarity."""
        from src.evaluator import compute_semantic_similarity

        sim = compute_semantic_similarity(
            "键的抗拉强度应大于等于 590 MPa",
            "今天天气很好适合出去玩",
        )
        assert sim < 0.5, f"Expected < 0.5, got {sim}"

    def test_similarity_identical(self) -> None:
        """Identical texts → near 1.0 similarity."""
        from src.evaluator import compute_semantic_similarity

        text = "键表面不允许有裂纹、浮锈、氧化皮和毛刺"
        sim = compute_semantic_similarity(text, text)
        assert sim > 0.95, f"Expected > 0.95, got {sim}"

    def test_similarity_returns_float_between_0_and_1(self) -> None:
        """All similarity scores must be in [0, 1]."""
        from src.evaluator import compute_semantic_similarity

        sim = compute_semantic_similarity("测试文本A", "测试文本B")
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0, f"Out of range: {sim}"


class TestEvaluateSingle:
    """Test single QA evaluation."""

    @pytest.fixture
    def sample_qa(self):
        from src.qa_parser import QAPair
        return QAPair(
            id="1",
            category="测试类",
            question="测试问题？",
            answer="这是标准答案。",
        )

    @pytest.fixture
    def sample_sources(self):
        from src.models import Source
        return [
            Source(page=1, section="第一章", content_preview="这是...", score=0.9),
        ]

    def test_evaluate_single_returns_metrics(self, sample_qa, sample_sources) -> None:
        """Should return EvalMetrics with all fields populated."""
        from src.evaluator import evaluate_single
        from src.models import EvalMetrics

        result = evaluate_single(
            qa_pair=sample_qa,
            llm_answer="这是来自LLM的答案。",
            sources=sample_sources,
        )
        assert isinstance(result, EvalMetrics)
        assert result.qa_id == "1"
        assert result.question == "测试问题？"
        assert result.ground_truth == "这是标准答案。"
        assert result.llm_answer == "这是来自LLM的答案。"
        assert 0.0 <= result.keyword_recall <= 1.0
        assert hasattr(result, "semantic_similarity")

    def test_evaluate_single_passed_threshold(self, sample_qa, sample_sources) -> None:
        """A good answer should pass the evaluation threshold."""
        from src.evaluator import evaluate_single

        result = evaluate_single(
            qa_pair=sample_qa,
            llm_answer="这是标准答案。",  # matches ground truth
            sources=sample_sources,
        )
        assert result.passed, f"Expected passed=True, got {result}"

    def test_evaluate_single_fails_bad_answer(self, sample_qa, sample_sources) -> None:
        """A completely wrong answer should fail."""
        from src.evaluator import evaluate_single

        result = evaluate_single(
            qa_pair=sample_qa,
            llm_answer="我无法回答这个问题。",
            sources=sample_sources,
        )
        assert not result.passed, f"Expected passed=False for bad answer"
