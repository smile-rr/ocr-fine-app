# PDF 抽取 + Chunking + Contextual Retrieval —— 2026 最佳实践

> 本文档定义项目的新一代 PDF → RAG 流水线。现有 `src/pdf_utils.py` (pdfplumber) 和 `src/rag.py` (每行单独 chunk) 的局限在讨论里已经明确：丢文字、跨页不合并、表格行 chunk 语义稀薄。本文给出基于 **Docling + MinerU + Contextual Retrieval** 的替代方案。

## 设计决策（最终）

| 维度 | 选择 | 理由 |
|---|---|---|
| 文本层 PDF 引擎 | **Docling** (IBM) | 布局感知、自动 OCR fallback、输出结构化 `DoclingDocument`、LlamaIndex/LangChain 原生集成 |
| 扫描件/复杂 PDF 引擎 | **MinerU** (OpenDataLab) | 中文 PDF 之王、84 语言 OCR、跨页表格内置合并、对金融年报这种影印件最强 |
| 自动路由 | 按首页文本层字节数 | 有文字 → Docling；无文字 → MinerU |
| 跨页合并 | Azure 启发式 + 必要时 LLM 兜底 | bbox 位置 + 列结构匹配 + header 识别 |
| Chunking | **多粒度**（table 汇总 + row with context） | 表格单行脱离上下文召回差 |
| Contextual Retrieval | **开启**（Anthropic 2024 方法） | 召回提升 35%+，成本可控 |
| ColPali/ColQwen | **不启用** | 存储成本 10×，本项目不值得 |

---

## 整体架构

```
                    PDF 输入
                       │
                       ▼
            ┌──────────────────────┐
            │  router.detect()     │  ← 用 PyMuPDF 快速探测首页文字
            │                      │
            └──────┬───────────────┘
                   │
          ┌────────┴────────┐
     有文字                  纯扫描
          ▼                   ▼
    ┌─────────┐         ┌──────────┐
    │ Docling │         │  MinerU  │
    │ engine  │         │  engine  │
    └────┬────┘         └────┬─────┘
         │                   │
         └─────┬─────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  Normalized DocElement 流    │ ← 统一 schema
    │  type: heading/paragraph/    │
    │        table/list/footnote   │
    │  text / page / bbox / data   │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │  merge_cross_page_tables()  │  ← Azure 启发式 + LLM 兜底
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │  chunking.chunk_document()   │
    │   ├── paragraph → 语义切/滑窗  │
    │   ├── table     → 多粒度       │
    │   │    ├─ table_summary       │
    │   │    └─ table_row × N       │
    │   └── list      → 整体保留     │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │  contextual.enrich()         │  ← Anthropic Contextual Retrieval
    │  为每个 chunk 用小模型生成     │
    │  50 字上下文，prepend 后再 embed│
    └────────────┬────────────────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │  ChromaDB (复用现有向量库)    │
    │  - documents: 原始 chunk 文本  │
    │  - embeddings: [context]+chunk │
    │  - metadata: type/page/table_id│
    └─────────────────────────────┘
```

---

## 组件 1：引擎与路由

### Docling（文本层 PDF）

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

pipeline_options = PdfPipelineOptions(
    do_ocr=True,                     # 遇扫描页自动走 OCR
    do_table_structure=True,         # 识别表格结构（行列/合并单元格）
)
pipeline_options.table_structure_options.do_cell_matching = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
result = converter.convert("/path/to/file.pdf")
doc = result.document   # DoclingDocument

# 遍历结构化元素（保留阅读顺序）
for item, _level in doc.iterate_items():
    label = item.label   # "text" / "table" / "list" / "picture" / ...
    text = item.text
    # bbox: item.prov[0].bbox  (x0,y0,x1,y1)
    # page: item.prov[0].page_no
```

**优点**：
- 输出树状结构（`DoclingDocument`），有 heading hierarchy
- 表格有完整 `TableItem`，含合并单元格信息
- 自动识别页眉页脚并标注（可过滤）
- `export_to_markdown()` 一键输出

**装**：`pip install docling`（首次会下 2GB 模型到 `~/.cache/huggingface`）

### MinerU（扫描件 / 中文复杂版式）

MinerU 的 Python API 随版本变动较多。**2025 年后推荐用 CLI + 读 JSON 输出**，更稳：

```python
import subprocess
import json
from pathlib import Path

def mineru_extract(pdf_path: Path, out_dir: Path) -> dict:
    """调 mineru CLI，返回它输出的结构化 JSON。"""
    subprocess.run(
        ["mineru", "-p", str(pdf_path), "-o", str(out_dir), "-m", "auto"],
        check=True,
    )
    # MinerU 输出 <stem>_content_list.json (结构化) 和 <stem>.md
    stem = pdf_path.stem
    content_json = out_dir / stem / "auto" / f"{stem}_content_list.json"
    return json.loads(content_json.read_text(encoding="utf-8"))
```

**装**：`pip install -U "mineru[core]"`（2GB 左右）+ 首次调用会下 layout 模型

### 路由逻辑

```python
def detect_pdf_type(pdf_path: Path) -> str:
    """返回 'text' / 'scanned'"""
    import fitz
    doc = fitz.open(pdf_path)
    # 前 3 页的文字字节总数；< 100 字节 = 扫描件
    total_text = sum(len(doc[i].get_text().strip()) for i in range(min(3, len(doc))))
    doc.close()
    return "text" if total_text > 100 else "scanned"
```

当两个引擎结果都不确定时，**Docling 优先**（它的 fallback OCR 已经处理大部分扫描情况）。MinerU 仅用于 Docling 明显失败（表格乱/中文识别差）的场景。

---

## 组件 2：跨页表格合并

基于 Azure Document Intelligence 公开方案，加 LLM 兜底。

### 启发式规则（无 LLM 的 95% 情况）

```python
def should_merge(prev: DocElement, curr: DocElement, page_height: float = 842) -> bool:
    """两个相邻 table 元素是否应当合并（跨页续接）。"""
    if curr.type != "table" or prev.type != "table":
        return False
    # 1. 必须相邻页
    if curr.page != prev.page + 1:
        return False
    # 2. 列数匹配
    if prev.data.shape[1] != curr.data.shape[1]:
        return False
    # 3. 位置启发式: prev 贴近页底, curr 贴近页顶
    if prev.bbox and curr.bbox:
        prev_bottom_gap = page_height - prev.bbox[3]
        curr_top_gap = curr.bbox[1]
        if prev_bottom_gap > 80 or curr_top_gap > 80:
            return False
    return True


def merge_tables(tables: list[DocElement]) -> DocElement:
    """合并 2+ 个 table，处理重复 header。"""
    import pandas as pd
    merged_df = tables[0].data.copy()
    for t in tables[1:]:
        df = t.data.copy()
        # 如果后表第一行和前表 header 完全一致，drop 掉（重复表头）
        if df.shape[1] == merged_df.shape[1] and \
           list(df.iloc[0].astype(str).str.strip()) == list(merged_df.columns.astype(str).str.strip()):
            df = df.iloc[1:]
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    result = DocElement(**tables[0].__dict__)
    result.data = merged_df
    result.cross_page = True
    result.pages = [t.page for t in tables]
    return result
```

### LLM 兜底（5% 候选）

规则判定"可能但不确定"时（如页内有多个表格、列宽不一但列数一样），走一次小模型验证：

```python
VERIFY_PROMPT = """判断以下两个表格片段是否是同一张跨页表格的上下部分。
只回答 YES 或 NO。

片段 A (第{pa}页 底部):
{a}

片段 B (第{pb}页 顶部):
{b}
"""
```

用 Qwen2.5-0.5B-Instruct / haiku / gpt-4o-mini 都够，成本 <$0.01/文档。

---

## 组件 3：多粒度 Chunking

### 表格 —— 两层 chunk

```python
def chunk_table(elem: DocElement, doc_id: str) -> list[Chunk]:
    df = elem.data
    title = elem.title or f"表格 (p{elem.page})"

    out = []

    # Level 1: 表格汇总（整张表一个 chunk）
    headers = list(df.columns)
    out.append(Chunk(
        text=(
            f"【{title}】\n"
            f"列: {' / '.join(str(h) for h in headers)}\n"
            f"共 {len(df)} 行。\n"
            + (f"前 3 行:\n{df.head(3).to_markdown(index=False)}" if len(df) > 0 else "")
        ),
        type="table_summary",
        metadata={
            "doc_id": doc_id, "page": elem.page, "table_id": elem.id,
            "n_rows": len(df), "n_cols": len(df.columns),
            "cross_page": elem.cross_page,
        },
    ))

    # Level 2: 每行一个 chunk（但前缀带表名 + 列名 → 自带上下文）
    for ri, row in df.iterrows():
        kv = " | ".join(f"{c}: {row[c]}" for c in df.columns)
        out.append(Chunk(
            text=f"[{title} · 第{ri+1}行] {kv}",
            type="table_row",
            metadata={
                "doc_id": doc_id, "page": elem.page, "table_id": elem.id,
                "row_idx": int(ri),
            },
        ))

    return out
```

**对比当前代码**（`src/rag.py:75`）：
- 现在：只有一层，每行 `[doc | p1 | r3] col: val | ...`
- 现在缺：表格标题、列名语义、整表摘要用于"这张表在讲啥"查询

### 段落 —— 语义切分或滑窗

```python
def chunk_paragraph(text: str, doc_id: str, page: int,
                    max_tokens: int = 512, overlap: int = 50) -> list[Chunk]:
    """简单版：按句号切到 max_tokens，带 overlap"""
    import re
    sentences = re.split(r"(?<=[。.！!？?])\s+", text)
    chunks, buf = [], []
    buf_len = 0
    for s in sentences:
        if buf_len + len(s) > max_tokens * 2 and buf:   # 粗估 token
            chunks.append("".join(buf))
            # overlap：保留最后若干字符
            tail = "".join(buf)[-overlap:]
            buf, buf_len = [tail, s], len(tail) + len(s)
        else:
            buf.append(s); buf_len += len(s)
    if buf:
        chunks.append("".join(buf))

    return [Chunk(text=c, type="paragraph",
                  metadata={"doc_id": doc_id, "page": page})
            for c in chunks]
```

生产版可以换成 **semantic chunking**（sentence embedding 相似度突变处切），但对大多数场景滑窗够用，收益有限。

### 列表 / 代码块 / 脚注

- **列表**：整块一个 chunk（不要按 item 切散）
- **代码块**：整块一个 chunk（绝不切）
- **脚注**：挂到引用段落的 metadata 里，不单独 chunk

---

## 组件 4：Contextual Retrieval（Anthropic 2024）

### 原理

每个 chunk 前面 prepend 一段**自动生成的上下文摘要**（50 字），embed 时用「context + chunk」，但保留原始 chunk 文本给 LLM 最终生成用。

```
原始 chunk:  "2023 年营收 120 亿，同比增长 20%。"
生成 context: "苹果 2023 财年 Q4 财报表格"
embed text:  "[苹果 2023 财年 Q4 财报表格] 2023 年营收 120 亿，同比增长 20%。"
```

检索时查询 embedding 更容易命中，因为 chunk 带了文档级线索。Anthropic 官方数字：**召回 @20 提升 35%**。

### Prompt 模板

```python
CONTEXT_PROMPT = """<document>
{document}
</document>

这是上面文档的一个片段：
<chunk>
{chunk}
</chunk>

请用不超过 50 字的一句话，描述这个片段在整份文档里的位置和主题，便于检索时用。
只输出描述，不要其他内容。
"""
```

### 成本控制：prompt caching + 文档截断

对每个 chunk 都调一次 LLM 看起来很贵，但有两个优化：

1. **Anthropic prompt caching** —— 文档部分 prefix 缓存，第 2 次调用起只扣输出。100-chunk 文档用 Haiku ≈ $0.02
2. **文档截断**：`document` 超过 20K token 就只喂开头 + 结尾（chunk 在中间的情况稀少）

本项目**本地模式**建议：
```python
# 配合本地 Ollama 免费跑
OLLAMA_BASE = "http://localhost:11434/v1"
context_model = "qwen2.5:0.5b-instruct-q4_K_M"
```

### 何时开 / 关

| 场景 | 开 Contextual Retrieval? |
|---|---|
| 长文档（年报/白皮书/合同） | ✅ 必开 |
| 单页短文档（发票/表单） | ❌ 不必要 |
| 表格行 chunk | ✅ 有效（表名就是天然 context） |
| 已有 BM25 混合检索 | ⚠️ 收益递减，加 reranker 更值 |

本项目默认：**对长文档的 paragraph 和 table_row chunk 开**，对 table_summary chunk 不开（它本身就是 summary）。

---

## 组件 5：与现有系统集成

### 不破坏老代码的迁移路径

```
src/
├── pdf_utils.py          # ⚠️ 保留（streamlit 还在用）
├── rag.py                # ⚠️ 保留（streamlit 还在用）
├── extract.py            # ⭐ 新：docling + mineru + 跨页合并
├── chunking.py           # ⭐ 新：多粒度 + Contextual Retrieval
└── rag_v2.py             # ⭐ 新：编排入口
```

### 使用示例

```python
from src.rag_v2 import Pipeline, OllamaContextGen

pipe = Pipeline(
    # 引擎配置
    engine="auto",                   # auto / docling / mineru
    # Chunking 配置
    max_chunk_tokens=512,
    table_multigranularity=True,
    # Contextual Retrieval
    context_gen=OllamaContextGen(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:0.5b-instruct-q4_K_M",
    ),
    # 向量库
    chroma_dir="./chroma_db",
    embed_model="BAAI/bge-small-zh-v1.5",
)

# 摄入
report = pipe.ingest_pdf(
    pdf_path="data/samples/apple_2023_q4.pdf",
    doc_id="apple-q4",
)
print(f"extracted {report.n_elements} elements, "
      f"merged {report.n_merged_tables} cross-page tables, "
      f"produced {report.n_chunks} chunks")

# 检索
hits = pipe.search("2023 年营收最高是哪一年？", top_k=5)
```

### Streamlit UI 可选接入

在 `app/streamlit_app.py` Tab 1 里加一个 checkbox：
```python
use_v2 = st.checkbox("使用 V2 抽取（Docling + 多粒度 chunking + Contextual Retrieval）",
                      value=False)
if use_v2:
    from src.rag_v2 import Pipeline
    pipe = Pipeline(...)
    pipe.ingest_pdf(pdf_path, doc_id)
else:
    # 老逻辑
    extract_tables(pdf_path)
    ...
```

不强制切换，方便 A/B 对比。

---

## 性能与成本估算

对 1 份 30 页的年报（文字型 PDF）：

| 阶段 | V1 (pdfplumber) | V2 (Docling) |
|---|---|---|
| 抽取时间 | ~5s | ~30s（含 table structure 模型推理） |
| 元素数 | 只有表格（~10） | 全文（~200） |
| Chunk 数 | ~50 (每行一个) | ~150 (多粒度) |
| Contextual 调用 | 0 | 150 次（Ollama 本地约 2 分钟）/ Haiku $0.02 |
| 向量库存储 | ~50 × 384 dim | ~150 × 384 dim |
| 检索召回 Recall@5 | ~60% | ~85% |
| 答案准确率 | 基线 | +30-40% |

**什么时候值得上 V2**：长文档（>10 页）、有表格和正文混排、中文年报。**什么时候 V1 够用**：单表格 CSV 化、快速 demo、短发票。

---

## 下一步

- [x] 文档（本文件）
- [x] `src/extract.py` —— Docling + MinerU 双引擎 + 跨页合并
- [x] `src/chunking.py` —— 多粒度 + Contextual Retrieval
- [x] `src/rag_v2.py` —— 编排 Pipeline 类
- [x] `pyproject.toml` 加 `[project.optional-dependencies].extract` —— `uv sync --extra extract`
- [x] `app/streamlit_app.py` 集成（新增「🔬 抽取引擎对比」Tab）
- [x] 更新 `PROJECT_MAP.md`
- [ ] `tests/test_extract.py` —— 简单冒烟测试 (TODO)

## 安装

```bash
# 项目根目录
uv sync --extra extract

# 首次会下 ~2GB 的 docling + mineru layout 模型到 ~/.cache/huggingface
# 想加速（可选）: export HF_TOKEN="hf_xxx"  或  hf auth login
```

## 跑

```bash
# 单文件测试（不开 Contextual Retrieval）
uv run python -m src.rag_v2 data/samples/apple_2023_q4.pdf "2023 年营收?"

# UI
uv run streamlit run app/streamlit_app.py
# sidebar 勾选 "⭐ V2: Docling / MinerU"，上传 PDF 看 Tab 3 "🔬 抽取引擎对比"
```
