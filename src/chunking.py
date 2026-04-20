"""多粒度 Chunking + Contextual Retrieval.

消费 src/extract.py 输出的 DocElement 流，生成 Chunk 流入向量库。

核心差别 vs src/rag.py 的老 chunk_table:
  - 表格: summary chunk + row chunks (每行带表名 + 列名)
  - 段落: 按句切到 max_tokens, 带 overlap
  - Contextual Retrieval: 为每个 chunk 用小 LLM 生成 50 字上下文, prepend 后再 embed

文档:
    docs/pdf-extraction-2026.md
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .extract import DocElement, ElementType

logger = logging.getLogger(__name__)


# ============================================================
# Chunk schema —— chunking 输出、embedding 输入
# ============================================================
@dataclass
class Chunk:
    """Embedding 用的最小单元。

    text:       给 LLM 最终生成用的原文（裸 chunk）
    embed_text: 给 embedder 用的增强文本（= [context] + text 当 contextual 开启时）
                当 contextual 关闭时, embed_text = text
    type:       paragraph / table_summary / table_row / list / heading
    metadata:   doc_id / page / table_id / row_idx / cross_page ...
    """
    text: str
    type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embed_text: str = ""             # 若为空则 fallback 到 text
    context: str = ""                # contextual retrieval 生成的 50 字上下文

    def __post_init__(self):
        if not self.embed_text:
            self.embed_text = self.text


# ============================================================
# 表格 chunking —— 多粒度
# ============================================================
def chunk_table(elem: DocElement, doc_id: str,
                table_seq: int = 0, max_row_chunks: int = 500) -> list[Chunk]:
    """把一个表格 DocElement 切成:
       - 1 个 table_summary chunk (整表一条, 用于"这张表在说啥"召回)
       - N 个 table_row chunks (每行一条, 前缀带表名 + 列名)
    """
    chunks: list[Chunk] = []
    if elem.data is None or elem.data.empty:
        # 没有 DataFrame, 当成纯 markdown 整块存
        chunks.append(Chunk(
            text=elem.text, type="table_summary",
            metadata={"doc_id": doc_id, "page": elem.page,
                      "table_id": elem.id, "cross_page": elem.cross_page},
        ))
        return chunks

    df = elem.data
    title = elem.title or f"表格#{table_seq} (p{elem.page})"
    headers = [str(c).strip() for c in df.columns]

    # Level 1: 表格汇总
    preview = df.head(3).to_markdown(index=False) if len(df) > 0 else ""
    summary_text = (
        f"【{title}】\n"
        f"列名: {' | '.join(headers)}\n"
        f"共 {len(df)} 行 × {len(headers)} 列"
        + (f"{'（跨页表格）' if elem.cross_page else ''}")
        + (f"\n\n前 3 行预览:\n{preview}" if preview else "")
    )
    chunks.append(Chunk(
        text=summary_text, type="table_summary",
        metadata={
            "doc_id": doc_id, "page": elem.page, "table_id": elem.id,
            "table_seq": table_seq, "n_rows": len(df), "n_cols": len(headers),
            "cross_page": elem.cross_page,
            "pages": elem.pages if elem.cross_page else [elem.page],
        },
    ))

    # Level 2: 逐行
    for ri, row in df.iterrows():
        if ri >= max_row_chunks:
            logger.warning(f"table {elem.id} 超过 {max_row_chunks} 行, 截断")
            break
        kv = " | ".join(f"{h}: {row[df.columns[i]]}" for i, h in enumerate(headers))
        chunks.append(Chunk(
            text=f"[{title} · 第{ri+1}行] {kv}",
            type="table_row",
            metadata={
                "doc_id": doc_id, "page": elem.page,
                "table_id": elem.id, "table_seq": table_seq,
                "row_idx": int(ri),
                "cross_page": elem.cross_page,
            },
        ))

    return chunks


# ============================================================
# 段落 / 列表 chunking
# ============================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[。．.！!？?\n])\s*")


def _approx_tokens(text: str) -> int:
    """粗估 token 数 (汉字 1:1, 英文词 0.75)."""
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - chinese
    return chinese + other // 4


def chunk_paragraph(elem: DocElement, doc_id: str,
                    max_tokens: int = 512, overlap_chars: int = 80) -> list[Chunk]:
    """按句边界切到 max_tokens, 带 overlap."""
    text = elem.text.strip()
    if not text:
        return []
    # 短段落直接返回
    if _approx_tokens(text) <= max_tokens:
        return [Chunk(
            text=text, type="paragraph",
            metadata={"doc_id": doc_id, "page": elem.page, "elem_id": elem.id},
        )]

    sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        s_tokens = _approx_tokens(s)
        if buf_tokens + s_tokens > max_tokens and buf:
            chunk_text = "".join(buf).strip()
            chunks.append(Chunk(
                text=chunk_text, type="paragraph",
                metadata={"doc_id": doc_id, "page": elem.page, "elem_id": elem.id},
            ))
            # overlap: 保留尾部若干字符
            tail = chunk_text[-overlap_chars:] if overlap_chars > 0 else ""
            buf = [tail, s] if tail else [s]
            buf_tokens = _approx_tokens(tail) + s_tokens
        else:
            buf.append(s)
            buf_tokens += s_tokens
    if buf:
        chunks.append(Chunk(
            text="".join(buf).strip(), type="paragraph",
            metadata={"doc_id": doc_id, "page": elem.page, "elem_id": elem.id},
        ))
    return chunks


def chunk_list(elem: DocElement, doc_id: str) -> list[Chunk]:
    """列表整体一个 chunk (不按 item 拆散, 保持语义完整)."""
    text = elem.text.strip()
    if not text:
        return []
    return [Chunk(
        text=text, type="list",
        metadata={"doc_id": doc_id, "page": elem.page, "elem_id": elem.id},
    )]


# ============================================================
# 编排: 把一份 DocElement 流切成 Chunk 流
# ============================================================
def chunk_document(
    elements: list[DocElement],
    doc_id: str,
    max_paragraph_tokens: int = 512,
    overlap_chars: int = 80,
    include_headings: bool = False,
) -> list[Chunk]:
    """把 DocElement 流转成 Chunk 流.

    include_headings: 是否把标题当 chunk (默认 False, 把标题作为下一段的 context 前缀更好)
    """
    chunks: list[Chunk] = []
    table_seq = 0
    current_section: str = ""

    for elem in elements:
        if elem.type == ElementType.HEADING:
            current_section = elem.text.strip()
            if include_headings:
                chunks.append(Chunk(
                    text=f"{'#' * (elem.level or 2)} {elem.text}",
                    type="heading",
                    metadata={"doc_id": doc_id, "page": elem.page,
                              "level": elem.level},
                ))
            continue

        if elem.type == ElementType.TABLE:
            new_chunks = chunk_table(elem, doc_id, table_seq=table_seq)
            # 注入 section 信息到 metadata
            for ch in new_chunks:
                ch.metadata["section"] = current_section
            chunks.extend(new_chunks)
            table_seq += 1

        elif elem.type == ElementType.PARAGRAPH:
            new_chunks = chunk_paragraph(elem, doc_id,
                                          max_tokens=max_paragraph_tokens,
                                          overlap_chars=overlap_chars)
            for ch in new_chunks:
                ch.metadata["section"] = current_section
            chunks.extend(new_chunks)

        elif elem.type == ElementType.LIST:
            new_chunks = chunk_list(elem, doc_id)
            for ch in new_chunks:
                ch.metadata["section"] = current_section
            chunks.extend(new_chunks)

        # FOOTNOTE / CAPTION / FIGURE / FORMULA: 目前统一当 paragraph 处理
        elif elem.type in (ElementType.FOOTNOTE, ElementType.CAPTION,
                           ElementType.FORMULA):
            if elem.text.strip():
                chunks.append(Chunk(
                    text=elem.text, type=elem.type.value,
                    metadata={"doc_id": doc_id, "page": elem.page,
                              "section": current_section},
                ))
        # FIGURE: 默认跳过（无 OCR 的纯图片）

    return chunks


# ============================================================
# Contextual Retrieval (Anthropic 2024)
# ============================================================
CONTEXT_PROMPT = """<document>
{document}
</document>

以下是上面这份文档的一个片段:
<chunk>
{chunk}
</chunk>

请用不超过 50 字的一句话，描述这个片段在整份文档里的主题与位置，方便后续检索。
只输出描述本身，不要任何前缀/引号/解释。
"""


ContextFn = Callable[[str, str], str]
"""签名: (document_text, chunk_text) -> context_string (<=50字)"""


class OllamaContextGen:
    """基于本地 Ollama 的 context 生成器 (零成本，Mac 上能跑)。

    用法:
        cg = OllamaContextGen(model="qwen2.5:0.5b-instruct-q4_K_M")
        ctx = cg(full_doc, chunk_text)
    """
    def __init__(self,
                 base_url: str = "http://localhost:11434/v1",
                 api_key: str = "ollama",
                 model: str = "qwen2.5:0.5b-instruct-q4_K_M",
                 max_doc_chars: int = 20000,
                 max_ctx_chars: int = 200,
                 timeout: float = 30.0):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("需要 pip install openai") from e
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_doc_chars = max_doc_chars
        self.max_ctx_chars = max_ctx_chars

    def __call__(self, document: str, chunk: str) -> str:
        doc = self._truncate(document)
        prompt = CONTEXT_PROMPT.format(document=doc, chunk=chunk)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            ctx = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"context gen failed: {e}")
            return ""
        # 清理换行 + 裁长
        ctx = re.sub(r"\s+", " ", ctx)
        return ctx[: self.max_ctx_chars]

    def _truncate(self, document: str) -> str:
        if len(document) <= self.max_doc_chars:
            return document
        half = self.max_doc_chars // 2
        return (document[:half] + "\n...[中间省略]...\n" + document[-half:])


class OpenAIContextGen(OllamaContextGen):
    """对接 Azure OpenAI / OpenAI 官网的变体。

    用法:
        cg = OpenAIContextGen(
            base_url="https://ocr.openai.azure.com/openai/deployments/gpt-4o",
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            model="gpt-4o",     # 这里其实是 deployment name (Azure) 或 model name (OpenAI)
        )
    """
    pass  # 完全一样，只是 base_url + api_key 换


# ============================================================
# 应用 contextual retrieval 到一批 chunks
# ============================================================
def enrich_with_context(
    chunks: list[Chunk],
    full_doc_text: str,
    context_fn: ContextFn,
    skip_types: tuple[str, ...] = ("table_summary", "heading"),
    show_progress: bool = True,
) -> list[Chunk]:
    """为每个 chunk 生成 context 并拼到 embed_text.

    skip_types: 跳过这些类型（table_summary 本身就是 summary，heading 是 metadata）
    """
    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        if ch.type in skip_types:
            continue
        try:
            ctx = context_fn(full_doc_text, ch.text)
        except Exception as e:
            logger.warning(f"context_fn error on chunk {i}: {e}")
            ctx = ""
        ch.context = ctx
        if ctx:
            ch.embed_text = f"[{ctx}] {ch.text}"
        if show_progress and i % 20 == 0:
            logger.info(f"contextual retrieval: {i}/{total}")
    return chunks


def build_full_doc_text(elements: list[DocElement]) -> str:
    """把 DocElement 流拼成完整的文档文本 (给 context_fn 用)."""
    parts: list[str] = []
    for el in elements:
        if el.type == ElementType.HEADING:
            parts.append(f"{'#' * (el.level or 2)} {el.text}")
        elif el.type == ElementType.TABLE:
            parts.append(el.text)          # markdown 表格
        elif el.text.strip():
            parts.append(el.text)
    return "\n\n".join(parts)


# ============================================================
# 调试
# ============================================================
if __name__ == "__main__":
    import sys
    from .extract import extract_document

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m src.chunking <pdf>")
        sys.exit(1)

    elements = extract_document(sys.argv[1])
    chunks = chunk_document(elements, doc_id="test")
    print(f"\n{len(chunks)} chunks:")
    from collections import Counter
    for k, v in Counter(c.type for c in chunks).items():
        print(f"  {k}: {v}")
    print("\n=== sample ===")
    for ch in chunks[:5]:
        preview = ch.text[:120].replace("\n", " ")
        print(f"[{ch.type}] {preview}")
