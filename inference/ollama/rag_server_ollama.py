"""参考版 RAG API（业务层 + Ollama 推理分离的企业级模式）。

和 src/serve/api.py 对比：
  - src/serve/api.py: FastAPI 进程里直接 load transformers 模型，推理靠 model.generate()
  - 本文件:          FastAPI 只做业务编排，推理通过 HTTP 调独立的 Ollama server

启动:
    1. ollama serve &          # :11434，后台常驻
    2. ollama pull qwen2.5:0.5b-instruct-q4_K_M
    3. uv run python inference/ollama/rag_server_ollama.py     # :8001

测试:
    curl -X POST http://localhost:8001/query \
        -H 'Content-Type: application/json' \
        -d '{"question":"2024 年净利润是多少？","top_k":3}'

注意：
  - 本服务跑在 :8001（主服务在 :8000），不冲突，可同时启动对比
  - 共用项目 chroma_db/（需要先跑 /ingest_markdown 或 Streamlit 灌过数据）
  - embedder 仍在本进程里跑（轻量，没必要再开一个 server；要极致可用 text-embeddings-inference）
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer

from src import config as C
from src.rag import RAG_SYSTEM


# ---------- 配置 ----------
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b-instruct-q4_K_M")
EMBED_MODEL = os.environ.get("EMBED_MODEL", C.EMBED_MODEL)
CHROMA_DIR = os.environ.get("CHROMA_DIR", str(C.CHROMA_DIR))


# ---------- 初始化 ----------
app = FastAPI(title="OCR-Fine-App RAG (Ollama backend)", version="0.1.0")

llm = OpenAI(base_url=OLLAMA_BASE, api_key="ollama")
encoder = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    "tables", metadata={"hnsw:space": "cosine"}
)


# ---------- Schemas ----------
class QueryIn(BaseModel):
    question: str
    top_k: int = 5
    doc_filter: str | None = None


class RAGOut(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_ms: float
    generation_ms: float
    model: str


# ---------- Endpoints ----------
@app.get("/health")
def health():
    try:
        models = llm.models.list()
        ollama_ok = any(m.id == OLLAMA_MODEL for m in models.data)
    except Exception as e:
        ollama_ok = False
        models = str(e)
    return {
        "status": "ok",
        "backend": "ollama",
        "ollama_base": OLLAMA_BASE,
        "ollama_model": OLLAMA_MODEL,
        "ollama_reachable": ollama_ok,
        "vector_count": collection.count(),
    }


@app.post("/query", response_model=RAGOut)
def query(req: QueryIn):
    # 1. 检索
    t0 = time.time()
    q_emb = encoder.encode([req.question], normalize_embeddings=True).tolist()
    where = {"doc_id": req.doc_filter} if req.doc_filter else None
    res = collection.query(
        query_embeddings=q_emb, n_results=req.top_k, where=where
    )
    hits = [
        {"text": d, "score": 1 - s, "metadata": m}
        for d, s, m in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]
    retr_ms = (time.time() - t0) * 1000

    if not hits:
        raise HTTPException(404, "no context found in vector store")

    # 2. 组 prompt
    ctx = "\n".join(f"- {h['text']} (score={h['score']:.2f})" for h in hits)
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"检索到的表格数据：\n{ctx}\n\n问题：{req.question}"},
    ]

    # 3. 调 Ollama (OpenAI 兼容)
    t1 = time.time()
    try:
        resp = llm.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
    except Exception as e:
        raise HTTPException(502, f"Ollama upstream failed: {e}")
    gen_ms = (time.time() - t1) * 1000

    return RAGOut(
        answer=resp.choices[0].message.content,
        sources=hits,
        retrieval_ms=retr_ms,
        generation_ms=gen_ms,
        model=OLLAMA_MODEL,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
