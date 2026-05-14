"""
Tests for src.qa_parser — parse qa_pairs.md into structured QAPair objects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# RED — these tests will fail until qa_parser is implemented


class TestParseQAPairs:
    """Test qa_pairs.md parsing."""

    @pytest.fixture
    def qa_pairs_path(self) -> Path:
        path = Path(__file__).parent / "qa_pairs.md"
        assert path.exists(), f"qa_pairs.md not found at {path}"
        return path

    def test_parse_extracts_all_12_questions(self, qa_pairs_path: Path) -> None:
        """Should extract all 12 QA pairs from qa_pairs.md."""
        from src.qa_parser import parse_qa_pairs

        pairs = parse_qa_pairs(qa_pairs_path)

        assert len(pairs) == 12, f"Expected 12 pairs, got {len(pairs)}"

    def test_parse_returns_structured_data(self, qa_pairs_path: Path) -> None:
        """Each pair should have id, category, question, answer."""
        from src.qa_parser import parse_qa_pairs

        pairs = parse_qa_pairs(qa_pairs_path)

        # Check first pair
        first = pairs[0]
        assert first.id == "1"
        assert first.category == "基本属性"
        assert "这个标准适用于哪些键" in first.question
        assert "除花键外" in first.answer

    def test_all_pairs_have_non_empty_fields(self, qa_pairs_path: Path) -> None:
        """Every pair must have non-empty id, category, question, answer."""
        from src.qa_parser import parse_qa_pairs

        pairs = parse_qa_pairs(qa_pairs_path)

        for p in pairs:
            assert p.id, f"Empty id in pair: {p}"
            assert p.category, f"Empty category in pair {p.id}"
            assert p.question.strip(), f"Empty question in pair {p.id}"
            assert p.answer.strip(), f"Empty answer in pair {p.id}"

    def test_ids_are_unique(self, qa_pairs_path: Path) -> None:
        """Each QA pair should have a unique ID."""
        from src.qa_parser import parse_qa_pairs

        pairs = parse_qa_pairs(qa_pairs_path)

        ids = [p.id for p in pairs]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_parse_nonexistent_file_raises(self) -> None:
        """Parsing a nonexistent file should raise FileNotFoundError."""
        from src.qa_parser import parse_qa_pairs

        with pytest.raises(FileNotFoundError):
            parse_qa_pairs(Path("nonexistent.md"))

    def test_categories_match_source(self, qa_pairs_path: Path) -> None:
        """Categories should match those defined in qa_pairs.md."""
        from src.qa_parser import parse_qa_pairs

        pairs = parse_qa_pairs(qa_pairs_path)

        expected_categories = {
            "基本属性", "材料要求", "表面质量", "平键要求",
            "半圆键", "平行度要求", "楔键角度", "毛刺处理",
            "验收抽样", "包装要求", "防锈要求", "新旧标准变化",
        }
        actual = {p.category for p in pairs}
        assert actual == expected_categories, f"Mismatch:\n  got: {actual}\n  expected: {expected_categories}"

    def test_specific_answer_content(self, qa_pairs_path: Path) -> None:
        """Spot-check specific answers for correctness."""
        from src.qa_parser import parse_qa_pairs

        pairs = parse_qa_pairs(qa_pairs_path)
        lookup = {p.id: p for p in pairs}

        # Q2: 材料要求
        assert "590 MPa" in lookup["2"].answer
        # Q3: 表面质量
        assert "裂纹" in lookup["3"].answer
        assert "浮锈" in lookup["3"].answer
        # Q11: 防锈要求
        assert "一年" in lookup["11"].answer
