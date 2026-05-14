# PDFRAG — PDF 文档智能问答系统

基于 RAG（检索增强生成）的 PDF 文档智能问答 Demo。上传 PDF → 解析文本 → 分块向量化 → 检索 → LLM 生成答案。

## 特性

- **📄 PDF 解析** — pdf2image + PaddleOCR，支持中文扫描版 PDF
- **🔍 混合检索** — 向量检索（BGE）+ BM25 关键词检索 + RRF 融合
- **🧠 智能分块** — 目录感知递归分块，保留文档结构
- **💬 DeepSeek 问答** — 基于检索上下文的 LLM 答案生成
- **🚀 FastAPI 服务** — RESTful API + 简易 Web UI
- **🐳 Docker 部署** — 一键容器化部署

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI + uvicorn + Jinja2 |
| PDF 处理 | pdf2image + PaddleOCR PP-Structure |
| 文本分块 | 自研递归分块器（目录感知） |
| 向量嵌入 | BAAI/bge-small-zh-v1.5 (512维) |
| 向量数据库 | ChromaDB (本地模式) |
| 混合检索 | BGE 向量 + BM25 + RRF 融合 |
| LLM | DeepSeek Chat API |
| 容器化 | Docker + docker-compose |

## 快速开始

### 环境要求

- Python 3.12+
- poppler-utils（PDF 转图片）
- DeepSeek API Key

### 安装

```bash
# 1. 克隆项目
cd pdfrag

# 2. 安装系统依赖
sudo apt-get install -y poppler-utils

# 3. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 4. 安装 Python 依赖
pip install -r requirements.txt

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 运行

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 http://localhost:8000 打开 Web UI。

### Docker 部署

```bash
docker-compose up -d
```

## 项目结构

```
pdfrag/
├── src/                       # 源代码
│   ├── main.py                # FastAPI 入口
│   ├── config.py              # 配置管理（环境变量）
│   ├── models.py              # Pydantic 数据模型
│   ├── pdf_parser.py          # PDF 解析（pdf2image + OCR）
│   ├── chunker.py             # 文本分块（目录感知）
│   ├── embedder.py            # 向量嵌入
│   ├── retriever.py           # 向量检索
│   └── llm.py                 # LLM 问答
├── tests/                     # 测试
├── data/                      # 运行时数据
│   ├── uploads/               # 上传的 PDF
│   ├── images/                # 转出的图片
│   ├── chroma/                # Chroma 持久化
│   └── cache/                 # 模型缓存
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── plan.md                    # 开发计划
└── README.md
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web UI 界面 |
| GET | `/health` | 健康检查 |
| POST | `/upload` | 上传 PDF 文档 |
| GET | `/status/{doc_id}` | 查询处理状态 |
| POST | `/query` | 文档问答 |

### 问答示例

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "这份合同的有效期是多久？",
    "conversation_id": "user_001",
    "doc_id": "doc_abc123"
  }'
```

响应：

```json
{
  "answer": "根据合同第三条，有效期为2024年1月1日至2025年12月31日。",
  "sources": [
    {
      "page": 2,
      "section": "第三条 合同期限",
      "content_preview": "本合同有效期自2024年1月1日起...",
      "score": 0.92
    }
  ],
  "confidence": 0.89,
  "conversation_id": "user_001"
}
```

## 配置说明

通过环境变量或 `.env` 文件配置，详见 `src/config.py`：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | - | DeepSeek API Key |
| `LLM_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 向量模型 |
| `PDF_DPI` | `300` | PDF 转图片 DPI |
| `CHUNK_SIZE` | `512` | 分块大小（字符） |
| `RETRIEVAL_TOP_K` | `5` | 检索返回数量 |

## 开发

```bash
# 运行测试
pytest tests/ -v

# 代码覆盖率
pytest tests/ --cov=src --cov-report=html
```

## 许可

MIT
