"""表格清洗 + ChromaDB 向量库 + RAG 检索。"""
from __future__ import annotations
from pathlib import Path
import logging
import re
from io import StringIO

import pandas as pd

from . import config as C

logger = logging.getLogger(__name__)


# ---------- 表格清洗 ----------

def parse_markdown_table(md: str) -> pd.DataFrame:
    """Markdown 表格 → DataFrame。

    对空列名 / 重复列名自动加后缀（pandas 不支持 dup 列做列选择，
    否则 `df[col]` 会返回 DataFrame 而不是 Series，下游逻辑会崩）。
    """
    lines = [l for l in md.strip().splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return pd.DataFrame()
    header = [c.strip() for c in lines[0].strip("|").split("|")]

    # dedup：空 → col_i；重复 → name__2, name__3 ...
    seen: dict[str, int] = {}
    unique: list[str] = []
    for i, h in enumerate(header):
        name = h or f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}__{seen[name]}"
        else:
            seen[name] = 1
        unique.append(name)

    rows = []
    for line in lines[2:]:  # 跳过分隔线
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(unique):
            rows.append(cells)
    return pd.DataFrame(rows, columns=unique)


def normalize_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """去千分位、转全角数字、百分比归一。"""
    df = df.copy()
    for col in df.columns:
        s = df[col].astype(str)
        s = s.str.translate(str.maketrans("０１２３４５６７８９．，", "0123456789.,"))
        s = s.str.replace(",", "", regex=False)
        df[col] = s
    return df


# ---------- 向量库 ----------

class TableVectorStore:
    def __init__(self, persist_dir: str | Path = C.CHROMA_DIR,
                 embed_model: str = C.EMBED_MODEL):
        import chromadb
        from sentence_transformers import SentenceTransformer

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            "tables", metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer(embed_model)

    def chunk_table(self, df: pd.DataFrame, doc_id: str, page: int) -> list[dict]:
        """每行一个 chunk：'[doc | p1 | r3] 列A: 值 | 列B: 值'。"""
        chunks = []
        for ri, row in df.iterrows():
            kv = " | ".join(f"{c}: {row[c]}" for c in df.columns)
            chunks.append({
                "id": f"{doc_id}_p{page}_r{ri}",
                "text": f"[{doc_id} | page {page} | row {ri+1}] {kv}",
                "metadata": {"doc_id": doc_id, "page": page, "row": int(ri)},
            })
        return chunks

    def add(self, df: pd.DataFrame, doc_id: str, page: int) -> int:
        chunks = self.chunk_table(df, doc_id, page)
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embs = self.encoder.encode(texts, normalize_embeddings=True).tolist()
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embs,
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = C.TOP_K,
               doc_filter: str | None = None) -> list[dict]:
        q = self.encoder.encode([query], normalize_embeddings=True).tolist()
        where = {"doc_id": doc_filter} if doc_filter else None
        res = self.collection.query(query_embeddings=q, n_results=top_k, where=where)
        out = []
        for doc, score, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
            out.append({"text": doc, "score": 1 - score, "metadata": meta})
        return out

    def count(self) -> int:
        return self.collection.count()


# ---------- Prompt 组装 ----------

RAG_SYSTEM = """你是一个精确的数据分析助手。规则：
1. 只基于给定的表格数据回答，不要引入外部知识
2. 答案必须包含具体数值，并注明数据来源（文档/页/行）
3. 数据不足时直接说明"数据不足"，不要猜测
4. 涉及计算时展示计算过程"""


def build_rag_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context = "\n".join(f"- {c['text']} (score={c['score']:.2f})" for c in chunks)
    user = f"检索到的表格数据：\n{context}\n\n问题：{query}"
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": user},
    ]
