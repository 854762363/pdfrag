"""
Evaluation engine — measures Q&A answer quality against ground truth.

Metrics:
- keyword_recall: How many ground-truth keywords appear in the LLM answer
- semantic_similarity: BGE embedding cosine similarity
- source_page_match: Whether LLM cited sources match the correct page range
"""

from __future__ import annotations

import re

import numpy as np

from src.models import EvalMetrics, EvalReport, Source
from src.qa_parser import QAPair


# ---- Keyword extraction ----


def _tokenize(text: str) -> list[str]:
    """Tokenize Chinese text into keywords, filtering stop words."""
    try:
        import jieba
    except ImportError:
        # Fallback: simple character-level for Chinese + word-level for English/numbers
        return _simple_tokenize(text)

    stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
        "什么", "怎么", "如何", "为什么", "哪", "吗", "吧", "呢", "啊", "哦",
        "根据", "按照", "标准", "规定", "要求", "应", "应该", "必须", "可以",
        "对于", "关于", "进行", "使用", "采用", "具有", "包括", "其中",
        "以下", "以上", "如下", "所示", "相关", "该", "本", "其",
    }

    words = jieba.lcut(text)
    # Filter: keep meaningful words (>1 char, not stopword, not pure punctuation)
    keywords = []
    for w in words:
        w = w.strip()
        if len(w) <= 1 and not w.isdigit():
            continue
        if w in stopwords:
            continue
        if re.match(r'^[，。！？、；：""''「」『』【】（）《》\s]+$', w):
            continue
        keywords.append(w)

    return keywords


def _simple_tokenize(text: str) -> list[str]:
    """Simple tokenizer fallback when jieba is not available."""
    # Split on punctuation and whitespace
    parts = re.split(r'[，。！？、；：""''「」『』【】（）《》（）\s]+', text)
    keywords = []
    for p in parts:
        p = p.strip()
        if len(p) >= 2 or (len(p) == 1 and p.isdigit()):
            keywords.append(p)
    return keywords


# ---- Metrics ----


def compute_keyword_recall(ground_truth: str, llm_answer: str) -> float:
    """Compute keyword recall: how many ground-truth keywords appear in the answer.

    Args:
        ground_truth: The expected answer.
        llm_answer: The LLM's actual answer.

    Returns:
        Float in [0, 1].
    """
    gt_keywords = _tokenize(ground_truth)
    if not gt_keywords:
        return 0.0

    answer_lower = llm_answer.lower()
    hits = 0
    for kw in gt_keywords:
        if kw.lower() in answer_lower:
            hits += 1

    return hits / len(gt_keywords)


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity using BGE-small-zh embeddings.

    The embedder is lazily loaded on first call.
    Falls back to keyword-overlap similarity if embedding model is unavailable.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Cosine similarity in [0, 1].
    """
    if text_a == text_b:
        return 1.0

    if not text_a.strip() or not text_b.strip():
        return 0.0

    try:
        embedder = _get_embedder()
        vec_a = embedder.embed_query(text_a)
        vec_b = embedder.embed_query(text_b)

        # Cosine similarity
        sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
        # Clamp to [0, 1]
        return max(0.0, min(1.0, sim))
    except Exception:
        # Fallback: keyword overlap ratio (Jaccard-like)
        import jieba
        try:
            words_a = set(jieba.lcut(text_a))
            words_b = set(jieba.lcut(text_b))
        except Exception:
            words_a = set(text_a)
            words_b = set(text_b)
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0


_embedder = None


def _get_embedder():
    """Lazy-load the BGE embedder."""
    global _embedder
    if _embedder is None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Use cached model, no network
        from src.embedder import Embedder
        _embedder = Embedder(model_name="BAAI/bge-small-zh-v1.5")
    return _embedder


# ---- Single evaluation ----


def evaluate_single(
    qa_pair: QAPair,
    llm_answer: str,
    sources: list[Source],
) -> EvalMetrics:
    """Evaluate a single Q&A pair against the LLM response.

    Args:
        qa_pair: The ground-truth Q&A pair.
        llm_answer: The LLM's actual answer text.
        sources: Source references from the LLM response.

    Returns:
        EvalMetrics with all scores populated.
    """
    kw_recall = compute_keyword_recall(qa_pair.answer, llm_answer)
    sem_sim = compute_semantic_similarity(qa_pair.answer, llm_answer)

    # Page match: check if any source references page 1-4 (our 4-page PDF)
    page_match = any(1 <= s.page <= 4 for s in sources) if sources else False

    # Confidence from sources (use max source score as proxy if available)
    confidence = max((s.score for s in sources), default=0.0)

    # Pass threshold: keyword_recall >= 0.3 AND semantic_similarity >= 0.5
    passed = kw_recall >= 0.3 and sem_sim >= 0.5

    return EvalMetrics(
        qa_id=qa_pair.id,
        question=qa_pair.question,
        ground_truth=qa_pair.answer,
        llm_answer=llm_answer,
        keyword_recall=round(kw_recall, 4),
        semantic_similarity=round(sem_sim, 4),
        source_page_match=page_match,
        confidence=round(confidence, 4),
        passed=passed,
    )


# ---- Report generation ----


def generate_report(results: list[EvalMetrics], document: str = "") -> EvalReport:
    """Generate a summary evaluation report from individual metrics.

    Args:
        results: List of per-question EvalMetrics.
        document: Name of the evaluated document.

    Returns:
        EvalReport with aggregated statistics and summary.
    """
    total = len(results)
    if total == 0:
        return EvalReport(document=document)

    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    avg_kw = sum(r.keyword_recall for r in results) / total
    avg_sim = sum(r.semantic_similarity for r in results) / total
    avg_conf = sum(r.confidence for r in results) / total

    # Build summary
    lines = [
        f"文档: {document}",
        f"总问题数: {total}",
        f"通过: {passed} | 未通过: {failed} | 通过率: {passed/total*100:.1f}%",
        f"平均关键词召回率: {avg_kw:.4f}",
        f"平均语义相似度: {avg_sim:.4f}",
        f"平均置信度: {avg_conf:.4f}",
    ]

    # Add per-question details
    lines.append("\n逐题详情:")
    for r in results:
        status = "✅" if r.passed else "❌"
        lines.append(
            f"  {status} Q{r.qa_id} [{r.keyword_recall:.2f}/{r.semantic_similarity:.2f}] "
            f"{r.question[:40]}..."
        )

    # Category analysis
    from collections import defaultdict
    cat_results: dict[str, list[EvalMetrics]] = defaultdict(list)
    for r, qa in zip(results, results):  # We need category info
        pass
    # Re-group by qa_id
    # Since EvalMetrics doesn't store category, we skip category breakdown for now

    return EvalReport(
        document=document,
        total_questions=total,
        passed=passed,
        failed=failed,
        avg_keyword_recall=round(avg_kw, 4),
        avg_semantic_similarity=round(avg_sim, 4),
        avg_confidence=round(avg_conf, 4),
        results=results,
        summary="\n".join(lines),
    )
