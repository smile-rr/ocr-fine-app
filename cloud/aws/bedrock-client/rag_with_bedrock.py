"""把 Bedrock 接到本项目的 RAG pipeline。

和 inference/ollama/rag_server_ollama.py 是对偶：
  - 本地开发: Ollama
  - 云生产:   Bedrock

启动:
    uv run python cloud/aws/bedrock-client/rag_with_bedrock.py
    curl -X POST http://localhost:8002/query \
        -d '{"question":"..."}' -H 'Content-Type: application/json'
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import boto3
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src import config as C
from src.rag import RAG_SYSTEM


REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
)


app = FastAPI(title="OCR-Fine-App RAG (Bedrock backend)", version="0.1.0")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
encoder = SentenceTransformer(os.environ.get("EMBED_MODEL", C.EMBED_MODEL))
chroma_client = chromadb.PersistentClient(path=os.environ.get("CHROMA_DIR", str(C.CHROMA_DIR)))
collection = chroma_client.get_or_create_collection(
    "tables", metadata={"hnsw:space": "cosine"}
)


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "bedrock",
        "region": REGION,
        "model": MODEL_ID,
        "vector_count": collection.count(),
    }


@app.post("/query", response_model=RAGOut)
def query(req: QueryIn):
    # 1. 检索（本地 embedder + chroma，同 ollama 版）
    t0 = time.time()
    q_emb = encoder.encode([req.question], normalize_embeddings=True).tolist()
    where = {"doc_id": req.doc_filter} if req.doc_filter else None
    res = collection.query(query_embeddings=q_emb, n_results=req.top_k, where=where)
    hits = [
        {"text": d, "score": 1 - s, "metadata": m}
        for d, s, m in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]
    retr_ms = (time.time() - t0) * 1000

    if not hits:
        raise HTTPException(404, "no context found in vector store")

    # 2. 组 prompt
    ctx = "\n".join(f"- {h['text']} (score={h['score']:.2f})" for h in hits)
    user_msg = f"检索到的表格数据：\n{ctx}\n\n问题：{req.question}"

    # 3. 调 Bedrock Converse API
    t1 = time.time()
    try:
        resp = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": RAG_SYSTEM}],
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.1},
        )
    except Exception as e:
        raise HTTPException(502, f"Bedrock upstream failed: {e}")
    gen_ms = (time.time() - t1) * 1000

    return RAGOut(
        answer=resp["output"]["message"]["content"][0]["text"],
        sources=hits,
        retrieval_ms=retr_ms,
        generation_ms=gen_ms,
        model=MODEL_ID,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
