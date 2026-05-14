# PDFRAG - PDF 文档问答系统开发计划

## 1. 项目概述

基于 RAG（检索增强生成）的 PDF 文档智能问答 Demo。上传 PDF → 解析文本 → 分块向量化 → 检索 → LLM 生成答案。

**运行环境**: WSL (无 GPU), Python 3.12, Docker(Windows宿主机)

---

## 2. 架构设计

```
用户 ──HTTP──▶ FastAPI ──▶ PDF解析模块 ──▶ 文本分块模块 ──▶ 向量化模块 ──▶ ChromaDB
                  ▲                                                              │
                  │                                                              ▼
                  └──────── LLM 问答模块 ◀──── 检索模块 ◀─────────────────────────┘
```

### 数据流

```
PDF 上传 → 临时存储 → pdf2image 转图片 → 版面分析 + OCR → 结构化文本
→ 目录解析 + 递归分块 → bge-small-zh 嵌入 → Chroma 存储

用户提问 → 向量检索 Top-K → 拼接上下文 → LLM 生成答案 → 返回(答案 + 来源)
```

---

## 3. 技术选型（无 GPU 约束）— 已确认

| 组件 | 选型 | 理由 |
|------|------|------|
| **PDF→图片** | pdf2image (poppler) | 稳定，支持所有 PDF |
| **版面分析** | PaddleOCR PP-Structure | ✅ 有中文版面模型(CDLA)，表格结构提取(SLANet) |
| **OCR** | PaddleOCR PP-OCRv4 (server) | ✅ 中文95%+准确率，CPU 2-5秒/页，模型仅200MB |
| **文本分块** | 自研递归分块器 + Contextual Retrieval | 目录感知 + 元数据 + BM25关键词索引 |
| **向量模型** | BAAI/bge-small-zh (384维) | 小模型，CPU 友好，中文效果好 |
| **向量数据库** | ChromaDB (本地模式) | 轻量，Python 原生，持久化 |
| **混合检索** | 向量(BGE) + BM25 + RRF融合 | 语义+关键词双通路 |
| **LLM** | DeepSeek API (已配) | 中文能力强，API 调用 |
| **多轮对话** | 内存 LRU + Prompt 拼接 | conversation_id 索引，最近10轮 |
| **Web框架** | FastAPI + uvicorn + Jinja2 | 异步，API+页面同服务 |
| **容器化** | Docker (Windows Docker Desktop) | 一键部署 |

### 3.1 PaddleOCR vs Surya 调研结论

针对**中文扫描版PDF（多栏混合排版+表格）** 场景：

| 维度 | PaddleOCR PP-OCRv4 | Surya (v0.10) | 胜出 |
|------|-------------------|---------------|:--:|
| 中文识别准确率 | **95%+** （17M+中文字符训练） | 70-85%（通用多语言模型） | 🏆 Paddle |
| 表格结构提取 | **✅ 完整单元格级**（HTML/Excel输出） | ❌ 仅检测区域框 | 🏆 Paddle |
| CPU速度 | **2-5秒/页**（MKLDNN优化） | 30-120秒/页（GPU优先） | 🏆 Paddle |
| 模型大小 | **~200MB**（server版） | ~1.5GB | 🏆 Paddle |
| 多栏阅读顺序 | 基础支持 | ✅ 专用阅读顺序模型 | Surya |
| 安装复杂度 | pip install paddlepaddle + paddleocr | pip install surya-ocr | Surya |
| API 简洁度 | 功能丰富但 API 复杂 | 极简 Python API | Surya |

**最终方案**: PaddleOCR PP-Structure 一站式流水线（版面→表格→OCR），无需 Surya。阅读顺序通过 PP-Structure 布局结果 + 坐标排序解决。

---

## 4. 项目结构

```
/mnt/d/project/llm/hermes/pdfrag/
├── README.md                   # 项目文档
├── Dockerfile                  # 容器构建
├── docker-compose.yml          # 一键启动
├── requirements.txt            # Python 依赖
├── plan.md                     # 本文件
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置管理
│   ├── models.py               # Pydantic 数据模型
│   ├── pdf_parser.py           # PDF 解析（pdf2image + OCR）
│   ├── chunker.py              # 文本分块（目录感知）
│   ├── embedder.py             # 向量嵌入（bge-small-zh）
│   ├── retriever.py            # 向量检索（Chroma）
│   ├── llm.py                  # LLM 问答（DeepSeek API）
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py           # 上传接口
│   │   └── query.py            # 问答接口
│   └── static/
│       └── index.html          # 简易前端页面
└── data/                       # 运行时数据目录
    ├── uploads/                # 上传的 PDF
    ├── images/                 # 转出的图片（临时）
    ├── chroma/                 # Chroma 持久化
    └── cache/                  # 模型缓存
```

---

## 5. 核心模块设计

### 5.1 PDF 解析模块 (`src/pdf_parser.py`)

```
输入: PDF 文件路径
流程:
  1. pdf2image 将每页转 PNG (300 DPI)
  2. PaddleOCR 版面分析：识别文本区域、表格、图片区域
  3. PaddleOCR 文字识别：对文本区域逐行 OCR
  4. 输出结构化文本：{page, block_type, text, bbox}
```

**关键点**:
- 首次运行会自动下载 PaddleOCR 模型 (~100MB)，需网络
- CPU 模式下每页约 5-15 秒，提示用户大文件需等待
- 保留页面号和坐标信息

### 5.2 文本分块模块 (`src/chunker.py`)

```
输入: 结构化文本列表
流程:
  1. 提取目录结构（基于字体大小/缩进/关键词匹配）
  2. 按章节递归切分（一级标题 → 二级标题 → 段落）
  3. 每个 chunk 注入元数据：
     - document_name: 文档名
     - page_number: 页码
     - section_path: 目录路径（如 "第三章 > 3.1 > 3.1.2"）
     - chunk_id: 唯一标识
     - timestamp: 处理时间
  4. chunk 大小控制在 256-512 token（中文约 500-1000 字）
```

**分块策略**:
- 首选按标题层级切分
- 段落过长时按句号/换行二次切分
- 相邻块保留 10% 重叠

### 5.3 向量嵌入模块 (`src/embedder.py`)

```
模型: BAAI/bge-small-zh (HuggingFace)
维度: 384
批处理: batch_size=32
流程:
  1. 加载 sentence-transformers 模型
  2. 对所有 chunk 批量编码
  3. 存入 Chroma collection
```

### 5.4 检索模块 (`src/retriever.py`)

```
流程:
  1. 对用户 query 进行向量编码
  2. Chroma similarity_search(query_embedding, k=5)
  3. 返回 top-K chunks（含元数据）
```

### 5.5 LLM 问答模块 (`src/llm.py`)

```
模型: deepseek-chat (通过已配置的 DEEPSEEK_API_KEY)
流程:
  1. 拼接系统提示 + 检索上下文 + 用户问题
  2. 调用 DeepSeek Chat API
  3. 解析返回：答案 + 引用标注

Prompt 模板:
  你是一个文档问答助手。根据以下文档片段回答问题。
  如果文档中没有相关信息，请如实说明。
  
  文档片段：
  [来源: {doc_name}, 章节: {section}, 页码: {page}]
  {chunk_text}
  ...
  
  问题: {question}
  
  请给出答案并引用来源。
```

### 5.6 FastAPI 服务 (`app/main.py`)

```
接口:
  GET  /              → 简易 Web UI
  POST /upload        → 上传 PDF，返回文档 ID 和处理状态
  GET  /status/{id}   → 查询处理进度
  POST /query         → 问答，返回 {answer, sources}

数据模型:
  UploadResponse: {doc_id, filename, pages, status}
  QueryRequest:   {doc_id, question}
  QueryResponse:  {answer, sources: [{page, section, text, score}]}
```

---

## 6. API 设计

### POST /upload

```
请求: multipart/form-data, file=xxx.pdf
响应: {
  "doc_id": "abc123",
  "filename": "报告.pdf",
  "pages": 15,
  "status": "processing",
  "message": "文档正在处理中，请稍后查询状态"
}
```

### GET /status/{doc_id}

```
响应: {
  "doc_id": "abc123",
  "status": "ready|processing|error",
  "progress": "5/15 pages",
  "error": null
}
```

### POST /query

```
请求: {
  "doc_id": "abc123",
  "question": "这份报告的主要结论是什么？"
}
响应: {
  "answer": "报告的主要结论是...",
  "sources": [
    {"page": 3, "section": "摘要", "text": "...", "score": 0.92},
    {"page": 12, "section": "结论", "text": "...", "score": 0.87}
  ]
}
```

---

## 7. 依赖清单

```
# requirements.txt
fastapi==0.115.*
uvicorn[standard]==0.34.*
python-multipart==0.0.*
pdf2image==1.17.*
paddlepaddle==3.0.*        # CPU 版
paddleocr==2.9.*
sentence-transformers==3.4.*
chromadb==0.5.*
openai==1.70.*             # DeepSeek 兼容 OpenAI API
pydantic==2.*
python-dotenv==1.*
Pillow==11.*
```

---

## 8. Docker 构建

### Dockerfile

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/uploads data/images data/chroma data/cache
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
services:
  pdfrag:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./app:/app/app
      - ./core:/app/core
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
```

---

## 9. 实现步骤

| 步骤 | 内容 | 预计耗时 |
|:---|:---|:---|
| 1 | 创建项目骨架，安装依赖 | 10 min |
| 2 | 实现 `src/pdf_parser.py`（PDF→图片→OCR） | 30 min |
| 3 | 实现 `src/chunker.py`（目录感知分块） | 20 min |
| 4 | 实现 `src/embedder.py` + `src/retriever.py` | 20 min |
| 5 | 实现 `src/llm.py`（DeepSeek 问答） | 15 min |
| 6 | 实现 FastAPI 路由和服务 | 20 min |
| 7 | 编写 Dockerfile + docker-compose | 10 min |
| 8 | 测试端到端流程 | 15 min |
| 9 | Git 初始化 + GitHub 推送 + README | 10 min |

---

## 10. 关键约束与应对

| 约束 | 影响 | 应对 |
|------|------|------|
| **无 GPU** | OCR/Embedding 慢 | bge-small-zh 极轻量；PaddleOCR CPU 模式可接受 |
| **WSL** | Docker 需 Windows 端运行 | Dockerfile 在 Windows Docker Desktop 构建 |
| **中文为主** | 英文 OCR/分块策略不适用 | 全链路中文优化 |
| **网络依赖** | 模型下载/API调用需网络 | 首次启动自动下载模型；LLM 走 API |

---

*计划创建于 2026-05-11*
