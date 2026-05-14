"""
QA pairs parser — parse qa_pairs.md into structured QAPair objects.

Format:
    ### N. Category
    Q: question text
    A: answer text (may span multiple lines until next ### or EOF)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QAPair:
    """A single question-answer pair with metadata."""
    id: str
    category: str
    question: str
    answer: str


# Pattern: "### N. Category" where N is the ID number
HEADING_PATTERN = re.compile(r"^###\s+(\d+)\.\s+(.+)$")
# Pattern: "Q: text" or "A: text"
QA_LINE = re.compile(r"^([QA]):\s*(.*)$")


def parse_qa_pairs(filepath: Path) -> list[QAPair]:
    """Parse a qa_pairs.md file into a list of QAPair objects.

    Args:
        filepath: Path to the markdown file.

    Returns:
        List of QAPair objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"QA pairs file not found: {filepath}")

    text = filepath.read_text(encoding="utf-8")

    pairs: list[QAPair] = []
    current_id = ""
    current_category = ""
    current_question = ""
    current_answer = ""
    in_answer = False

    for line in text.split("\n"):
        stripped = line.strip()

        # Check for heading: "### N. Category"
        heading_match = HEADING_PATTERN.match(stripped)
        if heading_match:
            # Save previous pair
            if current_id and current_question:
                pairs.append(QAPair(
                    id=current_id,
                    category=current_category,
                    question=current_question.strip(),
                    answer=current_answer.strip(),
                ))

            current_id = heading_match.group(1)
            current_category = heading_match.group(2).strip()
            current_question = ""
            current_answer = ""
            in_answer = False
            continue

        # Check for Q/A line
        qa_match = QA_LINE.match(stripped)
        if qa_match:
            marker = qa_match.group(1)
            content = qa_match.group(2)

            if marker == "Q":
                current_question = content
                in_answer = False
            elif marker == "A":
                current_answer = content
                in_answer = True
            continue

        # Multi-line answer continuation
        if in_answer and stripped and not stripped.startswith("#"):
            current_answer += "\n" + stripped

    # Don't forget the last pair
    if current_id and current_question:
        pairs.append(QAPair(
            id=current_id,
            category=current_category,
            question=current_question.strip(),
            answer=current_answer.strip(),
        ))

    return pairs
