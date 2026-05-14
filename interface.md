# PDFRAG API 接口文档

**Base URL**: `http://localhost:8000`

---

## 1. POST /chat

文档问答接口（核心接口）。接收用户问题，返回基于文档检索的答案。

### 请求

```http
POST /chat
Content-Type: application/json
```

```json
{
  "question": "这个产品的最大并发数是多少？",
  "conversation_id": "user_123",
  "doc_id": "doc_abc456"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `question` | string | ✅ | 用户自然语言问题 |
| `conversation_id` | string | ✅ | 会话ID，用于多轮对话记忆 |
| `doc_id` | string | ❌ | 指定文档ID；不传则检索所有已处理的文档 |

### 响应

**200 OK**

```json
{
  "answer": "根据产品手册第3页，该产品的最大并发数为 5000 QPS。",
  "sources": [
    {
      "page": 3,
      "section": "2.1 性能规格",
      "content_preview": "性能规格：最大并发数 5000 QPS，响应时间 ≤ 100ms...",
      "score": 0.92
    }
  ],
  "confidence": 0.89,
  "conversation_id": "user_123"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | 基于检索文档生成的回答 |
| `sources` | array | 引用来源列表 |
| `sources[].page` | int | 来源页码 |
| `sources[].section` | string | 来源章节路径 |
| `sources[].content_preview` | string | 来源内容摘要（前100字） |
| `sources[].score` | float | 检索相似度分数 (0-1) |
| `confidence` | float | 答案置信度 (0-1) |
| `conversation_id` | string | 回显会话ID |

### 错误响应

**400 Bad Request** — 缺少必填字段
```json
{
  "error": "validation_error",
  "detail": "question is required"
}
```

**404 Not Found** — 文档未处理
```json
{
  "error": "doc_not_found",
  "detail": "Document doc_abc456 not found or not yet processed"
}
```

**500 Internal Server Error**
```json
{
  "error": "internal_error",
  "detail": "LLM API call failed: timeout"
}
```

---

## 2. POST /upload

上传 PDF 文档进行解析和索引。

### 请求

```http
POST /upload
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `file` | file | ✅ | PDF 文件（最大 50MB） |

### 响应

**200 OK**

```json
{
  "doc_id": "doc_20260511_a1b2c3",
  "filename": "产品手册v3.pdf",
  "pages": 42,
  "status": "processing",
  "message": "文档已接收，正在解析中。预计需要 5-10 分钟。"
}
```

### 状态说明

| status | 说明 |
|--------|------|
| `processing` | 正在解析/分块/向量化 |
| `ready` | 处理完成，可问答 |
| `error` | 处理失败，查看 error 字段 |

---

## 3. GET /status/{doc_id}

查询文档处理状态。

### 请求

```http
GET /status/doc_20260511_a1b2c3
```

### 响应

```json
{
  "doc_id": "doc_20260511_a1b2c3",
  "filename": "产品手册v3.pdf",
  "status": "processing",
  "progress": {
    "current": 15,
    "total": 42,
    "stage": "ocr"
  },
  "error": null
}
```

`stage` 可能值: `ocr` | `chunking` | `embedding` | `done`

---

## 4. GET /documents

列出所有已处理的文档。

```json
{
  "documents": [
    {
      "doc_id": "doc_20260511_a1b2c3",
      "filename": "产品手册v3.pdf",
      "pages": 42,
      "chunks": 156,
      "status": "ready",
      "created_at": "2026-05-11T12:00:00Z"
    }
  ]
}
```

---

## 5. DELETE /documents/{doc_id}

删除文档及其所有索引数据。

```json
{
  "doc_id": "doc_20260511_a1b2c3",
  "deleted": true
}
```

---

## 通用说明

### 多轮对话

系统保留每个 `conversation_id` 最近 10 轮对话历史，自动传入 LLM 以保持上下文连续性。

### 混合检索

检索阶段结合了：
- **向量检索**（语义匹配，权重 0.7）
- **关键词检索**（BM25 精确匹配，权重 0.3）

最终结果经过 RRF (Reciprocal Rank Fusion) 融合排序。

### 内容安全

系统会检查用户输入，拒绝包含恶意内容的问题，返回 422。

### 限流

- 每个 IP 每分钟最多 20 次 `/chat` 请求
- `/upload` 每 IP 每分钟最多 3 次
