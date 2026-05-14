# PDFRAG 问答效果全流程评估 — 实现计划

> **For Hermes:** 使用 TDD + subagent-driven-development 按任务执行。

**Goal:** 用真实 PDF（GBT 1568-2008 键 技术条件）驱动全流程，对 12 组问答对评估 LLM 回答质量，输出可量化的评估报告。

**Architecture:** 
- 解析 `tests/qa_pairs.md` → 提取 12 组 Q&A 作为 ground truth
- 将 PDF 通过 pipeline 索引到 Chroma（如已索引则跳过 OCR 直接用）
- 对每组 Q 调用 `pipeline.query()` 获取 LLM 回答
- 评估维度：关键词命中率、语义相似度、来源引用正确性、confidence 分布
- 输出结构化 JSON 报告 + 可读摘要

**Tech Stack:** Python, pytest, sentence-transformers (用于语义相似度评估), DeepSeek Chat API, ChromaDB, PaddleOCR（仅首次索引时）

---

## 现状分析

### 现有测试的问题
| 测试文件 | 问题 |
|---------|------|
| `test_llm.py` | 全 mock，从未真实调用 DeepSeek API，测的是 mock 行为不是回答质量 |
| `test_e2e.py` | `make_minimal_pdf` 生成空 PDF（无文本），上传后 chunk=0 直接失败，从未走到 LLM 回答环节 |
| `test_retriever.py` | mock Chroma collection，未测真实检索效果 |
| 整体 | 0 个测试验证过「LLM 回答是否与文档内容一致」 |

### 已有资源
- 真实测试 PDF：`tests/GBT 1568-2008 键 技术条件.pdf`（4 页中文标准文档）
- 问答对：`tests/qa_pairs.md`（12 组人工编写的 Q&A ground truth）
- API Key：DeepSeek `sk-be7...57f6` 已配置在 `.env`
- Chroma 目录存在但需确认是否已有索引数据

---

## 任务分解

### Phase 1: 环境准备与数据确认（不动代码）

#### Task 1: 确认 Chroma 索引状态
**Objective:** 检查 Chroma 中是否已有 GBT 1568-2008 的索引数据

**Step 1:** 运行检查脚本
```
cd /mnt/d/project/llm/hermes/pdfrag && .venv/bin/python -c "
import chromadb
from pathlib import Path
client = chromadb.PersistentClient(path='data/chroma')
for c in client.list_collections():
    print(c.name, c.count())
    if c.count() > 0:
        sample = c.get(limit=1, include=['metadatas'])
        print('  doc names:', set(m.get('document_name','?') for m in sample['metadatas']))
"
```

**预期输出:** `pdfrag_docs` collection 存在，可能已有 0 或 N 条记录。

**决策点:**
- 如已有 GBT 1568-2008 的 chunk（可从 metadata.document_name 判断）→ 跳过 Task 2
- 如无 → 执行 Task 2 做 OCR 索引

#### Task 2: [条件执行] 索引 GBT 1568-2008 PDF
**Objective:** 将真实 PDF 通过完整 pipeline（OCR→分块→向量化）索引到 Chroma

**命令:**
```bash
cd /mnt/d/project/llm/hermes/pdfrag && .venv/bin/python -c "
from pathlib import Path
from src.pipeline import Pipeline

p = Pipeline()
result = p.process_document(Path('tests/GBT 1568-2008 键 技术条件.pdf'))
print(result)
"
```

**预估时间:** PaddleOCR 4 页 ≈ 2-3 分钟
**预期输出:** `{'doc_id': '...', 'filename': '...', 'pages': 4, 'chunks': N, 'status': 'ready'}`

---

### Phase 2: 问答对解析器（TDD）

#### Task 3: 编写 qa_pairs.md 解析器测试
**Objective:** 测试能从 markdown 文件中提取结构化 Q&A

**Files:**
- Create: `tests/test_qa_parser.py`
- Create: `src/qa_parser.py`

**Step 1: 写 RED 测试**
```python
def test_parse_qa_pairs_extracts_all_questions():
    pairs = parse_qa_pairs("tests/qa_pairs.md")
    assert len(pairs) == 12
    assert pairs[0].question == "这个标准适用于哪些键？"
    assert "除花键外" in pairs[0].answer

def test_parse_qa_pairs_returns_structured_data():
    pairs = parse_qa_pairs("tests/qa_pairs.md")
    for p in pairs:
        assert p.id  # "1", "2", ...
        assert p.category  # "基本属性", "材料要求", ...
        assert len(p.question) > 0
        assert len(p.answer) > 0

def test_parse_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_qa_pairs("nonexistent.md")
```

**Step 2:** 验证 RED — 所有 test 因 `parse_qa_pairs` 不存在而 ImportError

**Step 3:** 实现 `src/qa_parser.py`:
- 用正则解析 markdown，匹配 `### N. 分类` → `Q:` → `A:` 模式
- 返回 `list[QAPair]` dataclass

**Step 4:** 验证 GREEN — `pytest tests/test_qa_parser.py -v`

**Step 5:** Commit

---

### Phase 3: 评估引擎（TDD）

#### Task 4: 定义评估指标数据模型
**Objective:** 定义评估结果的 Pydantic 模型

**Files:**
- Modify: `src/models.py`
- Create: `tests/test_eval_models.py`

**新增模型:**
```python
class EvalMetrics(BaseModel):
    """Per-question evaluation metrics."""
    qa_id: str
    question: str
    ground_truth: str
    llm_answer: str
    keyword_recall: float        # 0-1, how many ground-truth keywords appear in answer
    keyword_precision: float     # 0-1, how many answer keywords come from ground truth
    semantic_similarity: float   # 0-1, embedding cosine similarity
    source_page_match: bool      # LLM 引用的页码是否在正确答案页码范围内
    confidence: float            # LLM 自报的 confidence
    passed: bool                 # keyword_recall >= 0.3 AND semantic_similarity >= 0.5

class EvalReport(BaseModel):
    """Full evaluation report."""
    document: str
    total_questions: int
    passed: int
    failed: int
    avg_keyword_recall: float
    avg_semantic_similarity: float
    avg_confidence: float
    results: list[EvalMetrics]
    summary: str  # 人类可读的摘要
```

#### Task 5: 编写评估器核心逻辑测试
**Objective:** 测试评估函数能正确计算各项指标

**Files:**
- Create: `tests/test_evaluator.py`
- Create: `src/evaluator.py`

**Step 1: RED 测试**
```python
def test_keyword_recall_perfect_match():
    """完全匹配时 recall=1.0"""
    result = compute_keyword_recall(
        ground_truth="键的抗拉强度应大于等于 590 MPa",
        llm_answer="根据标准，键的抗拉强度应大于等于 590 MPa"
    )
    assert result >= 0.7  # 允许分词差异

def test_keyword_recall_no_match():
    """完全不相关时 recall=0"""
    result = compute_keyword_recall(
        ground_truth="键的抗拉强度应大于等于 590 MPa",
        llm_answer="今天天气很好"
    )
    assert result < 0.2

def test_semantic_similarity_same_meaning():
    """语义相同但措辞不同 → 高相似度"""
    sim = compute_semantic_similarity(
        "键表面不允许有裂纹、浮锈、氧化皮和毛刺",
        "标准规定键的表面不应出现裂纹、浮锈、氧化皮以及毛刺等缺陷"
    )
    assert sim > 0.7

def test_evaluate_single_qa_returns_metrics():
    """单条 Q&A 评估返回完整 EvalMetrics"""
    metrics = evaluate_single(
        qa_pair=QAPair(id="1", category="基本属性", 
                       question="这个标准适用于哪些键？",
                       answer="适用于除花键外的各种键"),
        llm_answer="本标准适用于除花键外的各种键。",
        sources=[Source(page=1, section="1 范围", ...)]
    )
    assert isinstance(metrics, EvalMetrics)
    assert metrics.keyword_recall > 0.5
```

实现 evaluate_single() 的算法：
- **关键词召回**: 对 ground truth 做 jieba 分词，统计有多少词出现在 LLM 回答中
- **语义相似度**: 用 sentence-transformers 对两段文字分别 encode，算 cosine
- **来源页码**: 检查 sources 的 page 是否在合理范围

#### Task 6: 实现评估器
**Objective:** 实现 `src/evaluator.py` 并通过所有测试

**Step:** 按 TDD 逐个 test 实现函数

---

### Phase 4: 全流程集成评估（TDD）

#### Task 7: 编写全流程评估的集成测试
**Objective:** 端到端测试：PDF→pipeline→12 Q→评估→报告

**Files:**
- Create: `tests/test_evaluation.py`

```python
@pytest.mark.slow  # 需要真实 API 和 OCR
class TestFullEvaluation:
    
    def test_evaluate_all_qa_pairs(self):
        """对 12 组 Q&A 做全流程评估"""
        # 1. 加载 Q&A pairs
        pairs = parse_qa_pairs("tests/qa_pairs.md")
        assert len(pairs) == 12
        
        # 2. 初始化 pipeline（复用已有 Chroma 索引）
        pipeline = Pipeline()
        
        # 3. 逐条查询并评估
        results = []
        for qa in pairs:
            response = pipeline.query(qa.question)
            metrics = evaluate_single(qa, response.answer, response.sources)
            results.append(metrics)
        
        # 4. 生成报告
        report = generate_report(results, document="GBT 1568-2008")
        assert report.total_questions == 12
        
        # 5. 基准要求（至少 50% 通过）
        assert report.passed >= 6, f"Only {report.passed}/12 passed"

def test_evaluation_report_saved_to_file(self, tmp_path):
    """评估报告可以序列化为 JSON"""
    ...

def test_report_generates_readable_summary(self):
    """报告包含中文可读摘要"""
    ...
```

#### Task 8: 运行全流程评估
**Objective:** 执行完整评估，生成报告

**命令:**
```bash
cd /mnt/d/project/llm/hermes/pdfrag
.venv/bin/pytest tests/test_evaluation.py::TestFullEvaluation -v -s --timeout=600
```

**预期输出:**
```
EvalReport:
  Document: GBT 1568-2008 键 技术条件
  Passed: X/12
  Avg Keyword Recall: 0.XX
  Avg Semantic Similarity: 0.XX
  Avg Confidence: 0.XX
```

---

### Phase 5: 报告与总结

#### Task 9: 输出评估报告
**Objective:** 生成人类可读的评估摘要

**输出内容:**
1. 整体指标（通过率、平均召回率、语义相似度、置信度）
2. 逐题详情（哪些题过了、哪些题没过、为什么）
3. 问题分析（哪类问题容易失败：数字型？定义型？推理型？）
4. 改进建议

---

## 执行顺序

```
Task 1 (检查 Chroma) → Task 2 (可能需要的 OCR 索引)
    ↓
Task 3 (QA 解析器 TDD)
    ↓
Task 4-6 (评估器 TDD)
    ↓
Task 7-8 (集成测试 + 真实运行)
    ↓
Task 9 (报告输出)
```

## 关键决策点

1. **OCR 是否重跑**: 如果 Chroma 已有索引 → 跳过；如果没有 → 需跑 4 页 PaddleOCR（~2-3 分钟）
2. **评估阈值**: `keyword_recall >= 0.3 AND semantic_similarity >= 0.5` 暂定，根据实际结果可调整
3. **关键词提取**: 使用 jieba 分词 + 停用词过滤，不做复杂 NLP
4. **语义相似度模型**: 复用现有的 BGE-small-zh（已安装），避免额外下载

## 潜在风险

- PaddleOCR 在 WSL 下可能因缺少系统依赖（poppler）失败 → 用纯文本回退方案
- DeepSeek API 限流 → 12 题之间加 1s 间隔
- BGE 模型加载吃内存 → WSL 环境 OK，384 维小模型
