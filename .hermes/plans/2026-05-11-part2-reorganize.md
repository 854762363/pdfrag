# PDFRAG — Reorganize + Part2 实施计划

> **For Hermes:** 逐任务执行，严格 TDD。每步 RED→GREEN→REFACTOR。

**Goal:** 将项目结构从 app/ + core/ 迁移到 src/，然后实现 Part2 PDF 解析模块。

**Architecture:** 扁平 src/ 目录，config/models/main 直接放在 src/ 下。pdf_parser 先用 pdf2image 转图片，接口预留 OCR 集成。

**Tech Stack:** FastAPI, Pydantic, pdf2image, poppler-utils, pytest-asyncio

---

## Phase A: 项目结构重组

### Task A1: 创建 src/ 目录并迁移 config.py

**Objective:** 将 app/config.py 迁移到 src/config.py，保留内容不变

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`（从 app/config.py 复制）
- Modify: `tests/test_config.py`（更新 import 路径）
- Delete: `app/config.py`（最后统一清理）

**Step 1: 创建 src/__init__.py**
**Step 2: 复制 app/config.py → src/config.py**
**Step 3: 更新 tests/test_config.py：`from app.config` → `from src.config`**
**Step 4: 运行 `pytest tests/test_config.py -v`，预期全部 4 个通过**

### Task A2: 迁移 models.py

**Objective:** 将 app/models.py 迁移到 src/models.py

**Files:**
- Create: `src/models.py`
- Modify: `tests/test_models.py`（import 路径）
- Modify: `src/config.py`（无引用，仅确认）

**Step 1: 复制 app/models.py → src/models.py**
**Step 2: 更新 tests/test_models.py：`from app.models` → `from src.models`**
**Step 3: 运行 `pytest tests/test_models.py -v`，预期全部 11 个通过**

### Task A3: 迁移 main.py 并更新内部 import

**Objective:** 将 app/main.py 迁移到 src/main.py，更新其内部 import

**Files:**
- Create: `src/main.py`
- Modify: `tests/test_app.py`（import 路径）

**Step 1: 复制 app/main.py → src/main.py，将 `from app.config` 改为 `from src.config`**
**Step 2: 更新 tests/test_app.py：`from app.main` → `from src.main`**
**Step 3: 运行 `pytest tests/test_app.py -v`，预期全部 2 个通过**

### Task A4: 清理旧目录

**Objective:** 删除 app/ 和 core/ 目录

**Step 1: 删除 app/ 目录所有文件**
**Step 2: 删除 core/ 目录所有文件**
**Step 3: 运行 `pytest tests/ -v`，预期全部 17 个通过**

### Task A5: 更新 plan.md 中的项目结构说明

**Objective:** 更新 plan.md，将 app/ → src/，core/ → src/ 的路径反映到文档

**Step 1:** patch plan.md，替换路径引用

### Task A6: 编写 README.md

**Objective:** 根据 plan.md 编写完整项目 README

**内容:** 项目简介、特性、技术栈、快速开始、项目结构、API 端点、配置说明、许可

---

## Phase B: Part2 — PDF 解析模块

### Task B1: 安装 pdf2image 依赖

**Objective:** 安装 poppler-utils（pdf2image 系统依赖）

```bash
sudo apt-get update && sudo apt-get install -y poppler-utils
```

**验证:** `pdftoppm -v` 输出版本号

### Task B2: 写 RED 测试 — test_convert_pdf_to_images

**Objective:** 写一个会失败的测试：测试单页 PDF 转图片

**Files:**
- Create: `tests/test_pdf_parser.py`

**测试 1: test_convert_pdf_to_images_returns_list_of_paths**
- 创建单页 PDF → 调用 pdf_parser.convert_pdf_to_images() → 返回 Path 列表
- 预期失败：pdf_parser 模块不存在

### Task B3: GREEN — 实现 convert_pdf_to_images 最小代码

**Objective:** 实现 pdf2image.convert_from_path 的最小包装

**Files:**
- Create: `src/pdf_parser.py`

### Task B4: 写 RED 测试 — test_convert_empty_pdf_raises

**测试 2: test_convert_empty_pdf_raises**
- 空 PDF（0页）→ 应抛出 PdfParserError

### Task B5: GREEN — 实现空 PDF 校验

### Task B6: 写 RED 测试 — test_convert_nonexistent_file_raises

### Task B7: GREEN — 文件存在性校验

### Task B8: 写 RED 测试 — test_convert_encrypted_pdf_raises

### Task B9: GREEN — 加密 PDF 检测

### Task B10: 写 RED 测试 — test_get_page_count

### Task B11: GREEN — 实现 get_page_count

### Task B12: REFACTOR — 抽取 PdfParserError 异常类，整理代码

### Task B13: 全量测试验证

运行 `pytest tests/ -v`，预期全部通过

---

## 执行顺序

Phase A (A1→A6) → Phase B (B1→B13)

## 最终结构

```
pdfrag/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── main.py
│   └── pdf_parser.py       ← Part2 新增
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_app.py
│   └── test_pdf_parser.py  ← Part2 新增
├── data/
├── README.md
├── requirements.txt
├── Dockerfile
├── plan.md
└── .gitignore
```
