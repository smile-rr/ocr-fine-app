"""V2 RAG Pipeline —— 把 extract + chunking + contextual retrieval + 向量库串起来.

和老 src/rag.py 并存, 不替换.
老 API (TableVectorStore / parse_markdown_table) 保持不变, streamlit_app.py 照常能跑.

用法:
    from src.rag_v2 import Pipeline, OllamaContextGen

    pipe = Pipeline(
        engine="docling",
        contextual=True,
        context_gen=OllamaContextGen(model="qwen2.5:0.5b-instruct-q4_K_M"),
    )
    report = pipe.ingest_pdf("foo.pdf", doc_id="foo")
    hits = pipe.search("问题", top_k=5)

文档:
    docs/pdf-extraction-2026.md
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as C
from .chunking import (
    Chunk, ContextFn, OllamaContextGen, OpenAIContextGen,
    build_full_doc_text, chunk_document, enrich_with_context,
)
from .extract import DocElement, ElementType, extract_document

logger = logging.getLogger(__name__)


# ============================================================
# IngestReport: 摄入一份文档后返回的统计（给 Streamlit 展示）
# ============================================================
@dataclass
class IngestReport:
    doc_id: str
    engine_used: str                     # docling / mineru
    n_elements: int
    n_tables: int
    n_merged_cross_page: int             # 跨页表格合并数量
    n_chunks: int
    chunks_by_type: dict[str, int] = field(default_factory=dict)
    contextual_enriched: bool = False
    elapsed_extract_ms: float = 0.0
    elapsed_chunk_ms: float = 0.0
    elapsed_context_ms: float = 0.0
    elapsed_embed_ms: float = 0.0
    elements_preview: list[dict] = field(default_factory=list)


# ============================================================
# Pipeline
# ============================================================
class Pipeline:
    """V2 摄入 + 检索流水线。"""

    def __init__(
        self,
        *,
        # 抽取
        engine: str = "docling",           # docling / mineru / pymupdf4llm / pdfplumber / vision_llm:*
        merge_tables: bool = True,
        # Chunking
        max_paragraph_tokens: int = 512,
        overlap_chars: int = 80,
        # Contextual Retrieval
        contextual: bool = True,
        context_gen: ContextFn | None = None,
        # 向量库
        chroma_dir: str | Path = C.CHROMA_DIR,
        embed_model: str = C.EMBED_MODEL,
        collection_name: str = "tables_v2",     # 与 v1 分开，避免互相污染
    ):
        self.engine = engine
        self.merge_tables = merge_tables
        self.max_paragraph_tokens = max_paragraph_tokens
        self.overlap_chars = overlap_chars
        self.contextual = contextual
        self.context_gen = context_gen
        self.collection_name = collection_name

        # 懒加载 embedder + chroma（首次 ingest/search 时再开）
        self._embedder = None
        self._collection = None
        self._chroma_dir = Path(chroma_dir)
        self._embed_model = embed_model

    # ---------- 懒加载 ----------
    def _lazy_init(self):
        if self._collection is not None:
            return
        import chromadb
        from sentence_transformers import SentenceTransformer

        logger.info(f"loading embedder {self._embed_model}")
        self._embedder = SentenceTransformer(self._embed_model)
        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._collection = client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"chroma collection ready: {self.collection_name} "
                    f"({self._collection.count()} items)")

    # ---------- 主摄入 ----------
    def ingest_pdf(self, pdf_path: str | Path, doc_id: str) -> IngestReport:
        pdf_path = Path(pdf_path)
        self._lazy_init()

        # 1. 抽取（engine 显式由 caller 指定，不再 auto）
        t0 = time.time()
        engine_used = self.engine
        elements = extract_document(
            pdf_path, engine=engine_used, merge_tables=self.merge_tables,
        )
        elapsed_extract = (time.time() - t0) * 1000

        # 2. 统计跨页合并次数
        n_merged = sum(
            max(0, len(e.pages) - 1)
            for e in elements
            if e.type == ElementType.TABLE and e.cross_page
        )
        n_tables = sum(1 for e in elements if e.type == ElementType.TABLE)

        # 3. Chunking
        t1 = time.time()
        chunks = chunk_document(
            elements, doc_id=doc_id,
            max_paragraph_tokens=self.max_paragraph_tokens,
            overlap_chars=self.overlap_chars,
        )
        elapsed_chunk = (time.time() - t1) * 1000

        # 4. Contextual Retrieval (可选)
        elapsed_context = 0.0
        enriched = False
        if self.contextual and self.context_gen is not None and chunks:
            t2 = time.time()
            full_doc = build_full_doc_text(elements)
            chunks = enrich_with_context(
                chunks, full_doc_text=full_doc, context_fn=self.context_gen,
            )
            elapsed_context = (time.time() - t2) * 1000
            enriched = True

        # 5. Embed + 存入 Chroma
        t3 = time.time()
        if chunks:
            self._embed_and_add(chunks, doc_id)
        elapsed_embed = (time.time() - t3) * 1000

        # 6. 报告
        from collections import Counter
        chunks_by_type = dict(Counter(c.type for c in chunks))

        preview = []
        for el in elements[:8]:
            preview.append({
                "page": el.page, "type": el.type.value,
                "text": el.text[:120].replace("\n", " "),
                "cross_page": el.cross_page,
            })

        return IngestReport(
            doc_id=doc_id, engine_used=engine_used,
            n_elements=len(elements), n_tables=n_tables,
            n_merged_cross_page=n_merged,
            n_chunks=len(chunks), chunks_by_type=chunks_by_type,
            contextual_enriched=enriched,
            elapsed_extract_ms=elapsed_extract, elapsed_chunk_ms=elapsed_chunk,
            elapsed_context_ms=elapsed_context, elapsed_embed_ms=elapsed_embed,
            elements_preview=preview,
        )

    # ---------- Embed + 存 ----------
    def _embed_and_add(self, chunks: list[Chunk], doc_id: str):
        ids, docs, metas, embed_inputs = [], [], [], []
        for i, ch in enumerate(chunks):
            ids.append(f"{doc_id}_{ch.type}_{i}_{hash(ch.text) & 0xFFFFFFFF:08x}")
            docs.append(ch.text)                # 原文（给 LLM 用）
            meta = {**ch.metadata, "type": ch.type}
            # Chroma metadata 只接受标量; list/dict 要序列化
            meta = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                    for k, v in meta.items() if v is not None}
            if ch.context:
                meta["context"] = ch.context
            metas.append(meta)
            embed_inputs.append(ch.embed_text)  # 可能带 [context] prefix

        logger.info(f"embedding {len(chunks)} chunks")
        embs = self._embedder.encode(embed_inputs, normalize_embeddings=True,
                                      show_progress_bar=False).tolist()
        self._collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 5,
               doc_filter: str | None = None,
               type_filter: str | None = None) -> list[dict]:
        self._lazy_init()
        q_emb = self._embedder.encode([query], normalize_embeddings=True).tolist()

        where: dict[str, Any] | None = None
        conds = []
        if doc_filter:
            conds.append({"doc_id": doc_filter})
        if type_filter:
            conds.append({"type": type_filter})
        if len(conds) == 1:
            where = conds[0]
        elif len(conds) > 1:
            where = {"$and": conds}

        res = self._collection.query(
            query_embeddings=q_emb, n_results=top_k, where=where,
        )
        out = []
        for doc, dist, meta, _id in zip(
            res["documents"][0], res["distances"][0],
            res["metadatas"][0], res["ids"][0],
        ):
            out.append({
                "id": _id, "text": doc, "score": 1 - dist, "metadata": meta,
            })
        return out

    # ---------- 工具 ----------
    def count(self) -> int:
        self._lazy_init()
        return self._collection.count()

    def clear(self):
        """清空本 collection（调试/重建用）。"""
        self._lazy_init()
        # Chroma 不支持 "delete all"，重建 collection 最快
        import chromadb
        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        client.delete_collection(self.collection_name)
        self._collection = client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )


# ============================================================
# 调试
# ============================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m src.rag_v2 <pdf> [query]")
        sys.exit(1)

    # 不开 contextual 先看基础流程
    pipe = Pipeline(contextual=False)
    report = pipe.ingest_pdf(sys.argv[1], doc_id="smoke")
    print(f"\n=== INGEST REPORT ===")
    print(f"  engine:              {report.engine_used}")
    print(f"  elements:            {report.n_elements}")
    print(f"  tables:              {report.n_tables}")
    print(f"  cross_page merged:   {report.n_merged_cross_page}")
    print(f"  chunks:              {report.n_chunks}")
    print(f"  chunks_by_type:      {report.chunks_by_type}")
    print(f"  extract:             {report.elapsed_extract_ms:.0f}ms")
    print(f"  chunk:               {report.elapsed_chunk_ms:.0f}ms")
    print(f"  embed:               {report.elapsed_embed_ms:.0f}ms")

    if len(sys.argv) > 2:
        q = sys.argv[2]
        print(f"\n=== SEARCH '{q}' ===")
        for h in pipe.search(q, top_k=5):
            print(f"[{h['score']:.2f}] ({h['metadata'].get('type')}) {h['text'][:120]}")
