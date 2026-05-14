# PDFRAG — Part3+4 合并实施计划

> **For Hermes:** chunker → embedder → retriever，TDD 每步 RED→GREEN。
> 依赖安装后台进行，编码前台并行。

**Goal:** 实现文本分块(chunker) + 向量嵌入(embedder) + 多路检索(retriever)

**Architecture:** 
- chunker: 纯 Python，标题层级递归切分 + 句号二次切分 + 元数据注入
- embedder: 封装 sentence-transformers，batch 编码，写入 Chroma
- retriever: Chroma 向量检索 + BM25 关键词 + RRF 融合

**Tech Stack:** sentence-transformers, chromadb, rank-bm25

---

## Phase 0: 依赖安装（后台）

```bash
.venv/bin/pip install chromadb sentence-transformers rank-bm25
```

## Phase 1: chunker.py（纯 Python，无外部依赖）

### C1: RED — test_chunk_by_headings
- 输入：带 Markdown 标题的文本列表
- 输出：按标题切分的 chunk 列表，含 section_path 元数据

### C2: GREEN — 实现 split_by_headings

### C3: RED — test_chunk_by_sentence_overflow
- 长段落超过 chunk_size → 按句号二次切分

### C4: GREEN — 实现 split_by_sentences

### C5: RED — test_chunk_overlap
- 相邻 chunk 有 overlap 字符重叠

### C6: GREEN — 实现 overlap 机制

### C7: RED — test_chunk_metadata
- 每个 chunk 含 document_name, page_number, section_path, chunk_id

### C8: GREEN — 实现 metadata 注入

### C9: RED — test_chunk_min_size_filter
- 短于 min_size 的 chunk 被丢弃

### C10: GREEN — 实现 min_size 过滤

### C11: RED — test_chunk_pipeline
- 端到端：结构化文本列表 → chunk 列表（集成测试）

### C12: GREEN — 实现主入口 chunk_document()

### C13: 全量测试

## Phase 2: embedder.py（依赖 sentence-transformers）

### E1: RED — test_embed_batch_returns_correct_shape
- mock 模型，验证输出 (N, 384)

### E2: GREEN — 实现 Embedder 类 + embed_batch()

### E3: RED — test_embed_query
- 单文本嵌入，返回 (384,) 向量

### E4: GREEN — 实现 embed_query()

### E5: RED — test_store_chunks_creates_collection
- 将 chunk 写入 Chroma collection

### E6: GREEN — 实现 store_chunks()

## Phase 3: retriever.py（依赖 chromadb）

### R1: RED — test_vector_search_returns_top_k
- 向量检索返回 k 个结果，含 score

### R2: GREEN — 实现 vector_search()

### R3: RED — test_bm25_search
- BM25 关键词检索

### R4: GREEN — 实现 bm25_search()

### R5: RED — test_hybrid_search_rrf
- 向量+BM25 RRF 融合

### R6: GREEN — 实现 hybrid_search()

### R7: 全量回归测试

---

## 最终测试覆盖

| 模块 | 预计测试数 |
|------|:--:|
| chunker | 8 |
| embedder | 5 |
| retriever | 5 |
| 存量 | 26 |
| **总计** | **~44** |
