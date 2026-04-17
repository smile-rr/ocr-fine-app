"""FastAPI 推理服务（Docker 部署用）。

跟 src/infer.py 不同：本服务强制用 HF transformers，读 fused 后的模型，
这样同一镜像能跑在 Linux/CPU/CUDA 环境，不依赖 MLX。

环境变量：
    STAGE1_MODEL_PATH   默认 /app/models/stage1_fused   (VLM)
    STAGE2_MODEL_PATH   默认 /app/models/stage2_fused   (LLM)
    EMBED_MODEL         默认 BAAI/bge-small-zh-v1.5
    CHROMA_DIR          默认 /app/chroma_db
    DEVICE              默认 cpu；可设 cuda / mps
    MAX_TOKENS          默认 512
    ENABLE_STAGE1       默认 1；0 则不加载 VLM（节省内存）
"""
from __future__ import annotations
import gc
import hashlib
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("serve")


# ---------------- 配置 ----------------
STAGE1_PATH = os.environ.get("STAGE1_MODEL_PATH", "/app/models/stage1_fused")
STAGE2_PATH = os.environ.get("STAGE2_MODEL_PATH", "/app/models/stage2_fused")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
CHROMA_DIR  = os.environ.get("CHROMA_DIR", "/app/chroma_db")
DEVICE      = os.environ.get("DEVICE", "cpu")
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "512"))
ENABLE_S1   = os.environ.get("ENABLE_STAGE1", "1") == "1"
ADMIN_KEY   = os.environ.get("ADMIN_API_KEY", "")
AUTO_RELOAD = os.environ.get("AUTO_RELOAD", "0") == "1"


# ---------------- 全局状态 ----------------
# 用锁保护 STATE 的读写，保证热加载时的原子切换
_lock = threading.RLock()

STATE: dict[str, Any] = {
    "stage1": None,    # (model, processor)
    "stage2": None,    # (model, tokenizer)
    "embed": None,
    "chroma": None,
    "versions": {"stage1": None, "stage2": None},  # 记录当前版本 fingerprint
    "loaded_at": {"stage1": None, "stage2": None},
}


def _dir_fingerprint(path: str | Path) -> str | None:
    """取目录下所有 .safetensors / .bin 的 mtime+size 做 hash，当版本号用。"""
    p = Path(path)
    if not p.exists():
        return None
    items = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".json"):
            st = f.stat()
            items.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha1("|".join(items).encode()).hexdigest()[:12] if items else None


def _load_stage2(path: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    path = path or STAGE2_PATH
    if not Path(path).exists():
        raise RuntimeError(f"Stage2 model not found at {path}")
    log.info(f"loading Stage2 LLM from {path} on {DEVICE}")
    tok = AutoTokenizer.from_pretrained(path)
    dtype = torch.float16 if DEVICE != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    model.to(DEVICE).eval()
    return model, tok


def _load_stage1(path: str = None):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    import torch
    path = path or STAGE1_PATH
    if not Path(path).exists():
        raise RuntimeError(f"Stage1 model not found at {path}")
    log.info(f"loading Stage1 VLM from {path} on {DEVICE}")
    processor = AutoProcessor.from_pretrained(path)
    dtype = torch.float16 if DEVICE != "cpu" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype)
    model.to(DEVICE).eval()
    return model, processor


def _swap_model(stage: int, new_obj: tuple, path: str):
    """原子替换：锁内换引用，锁外释放旧模型。"""
    key = f"stage{stage}"
    with _lock:
        old = STATE[key]
        STATE[key] = new_obj
        STATE["versions"][key] = _dir_fingerprint(path)
        STATE["loaded_at"][key] = time.time()
    # 锁外做清理，避免阻塞其他请求
    del old
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_embed_and_chroma():
    from sentence_transformers import SentenceTransformer
    import chromadb
    log.info(f"loading embed {EMBED_MODEL}")
    embed = SentenceTransformer(EMBED_MODEL)
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_or_create_collection("tables", metadata={"hnsw:space": "cosine"})
    return embed, coll


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 懒加载：启动时只拉 embed + stage2，stage1 首次请求再加载
    try:
        STATE["stage2"] = _load_stage2()
        STATE["versions"]["stage2"] = _dir_fingerprint(STAGE2_PATH)
        STATE["loaded_at"]["stage2"] = time.time()
    except Exception as e:
        log.error(f"stage2 load failed: {e}")
    try:
        STATE["embed"], STATE["chroma"] = _load_embed_and_chroma()
    except Exception as e:
        log.error(f"embed/chroma load failed: {e}")

    observer = None
    if AUTO_RELOAD:
        observer = _start_watcher()

    yield

    if observer:
        observer.stop()
        observer.join(timeout=5)
    STATE.clear()


def _start_watcher():
    """用 watchdog 监听 models/ 目录，检测到变化触发 reload。"""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        log.warning("watchdog 未安装，跳过 AUTO_RELOAD")
        return None

    class Handler(FileSystemEventHandler):
        def __init__(self):
            self._last: dict[int, float] = {}

        def on_any_event(self, event):
            if event.is_directory:
                return
            p = Path(event.src_path)
            stage = 1 if str(STAGE1_PATH) in str(p) else (2 if str(STAGE2_PATH) in str(p) else None)
            if stage is None:
                return
            # 防抖：2s 内只触发一次
            now = time.time()
            if now - self._last.get(stage, 0) < 2.0:
                return
            self._last[stage] = now
            # 给 IO 一点时间再加载
            threading.Timer(3.0, lambda: _reload_if_changed(stage)).start()

    obs = Observer()
    handler = Handler()
    for p in (STAGE1_PATH, STAGE2_PATH):
        if Path(p).parent.exists():
            obs.schedule(handler, str(Path(p).parent), recursive=True)
    obs.daemon = True
    obs.start()
    log.info(f"🔁 AUTO_RELOAD enabled; watching {Path(STAGE1_PATH).parent}")
    return obs


def _reload_if_changed(stage: int):
    """只在 fingerprint 变化时才 reload，避免无谓的重载。"""
    path = STAGE1_PATH if stage == 1 else STAGE2_PATH
    new_fp = _dir_fingerprint(path)
    cur_fp = STATE["versions"].get(f"stage{stage}")
    if new_fp and new_fp != cur_fp:
        log.info(f"🔁 stage{stage} changed: {cur_fp} -> {new_fp}, reloading...")
        try:
            loader = _load_stage1 if stage == 1 else _load_stage2
            new_obj = loader(path)
            _swap_model(stage, new_obj, path)
            log.info(f"✅ stage{stage} hot-reloaded")
        except Exception as e:
            log.error(f"❌ stage{stage} reload failed: {e}；保留旧模型")


app = FastAPI(title="OCR-Fine-App Inference API", version="0.1.0", lifespan=lifespan)


# ---------------- Schemas ----------------
class QueryIn(BaseModel):
    question: str
    top_k: int = 5
    doc_filter: str | None = None


class RAGOut(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_ms: float
    generation_ms: float


class ReloadIn(BaseModel):
    stage: int                          # 1 或 2
    path: str | None = None             # 覆盖默认路径（可选）
    force: bool = False                 # True 忽略 fingerprint 强制重载


def _require_admin(x_admin_key: str | None):
    if not ADMIN_KEY:
        return  # 未设 key 则端点开放（dev 友好）
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(401, "invalid admin key")


# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "stage1_loaded": STATE["stage1"] is not None,
        "stage2_loaded": STATE["stage2"] is not None,
        "embed_loaded": STATE["embed"] is not None,
        "vector_count": STATE["chroma"].count() if STATE["chroma"] else 0,
        "device": DEVICE,
        "versions": STATE["versions"],
        "loaded_at": STATE["loaded_at"],
        "auto_reload": AUTO_RELOAD,
    }


@app.post("/admin/reload")
def admin_reload(req: ReloadIn, x_admin_key: str | None = Header(default=None)):
    """热加载：原子替换模型，不中断服务。

    Header: X-Admin-Key: <ADMIN_API_KEY>
    """
    _require_admin(x_admin_key)
    if req.stage not in (1, 2):
        raise HTTPException(400, "stage must be 1 or 2")

    path = req.path or (STAGE1_PATH if req.stage == 1 else STAGE2_PATH)
    new_fp = _dir_fingerprint(path)
    cur_fp = STATE["versions"].get(f"stage{req.stage}")
    if not req.force and new_fp == cur_fp and cur_fp is not None:
        return {"status": "unchanged", "version": cur_fp, "message": "加 force=true 强制重载"}

    t0 = time.time()
    try:
        loader = _load_stage1 if req.stage == 1 else _load_stage2
        new_obj = loader(path)
    except Exception as e:
        log.exception("reload failed")
        raise HTTPException(500, f"reload failed (旧模型保留): {e}")

    _swap_model(req.stage, new_obj, path)
    return {
        "status": "reloaded",
        "stage": req.stage,
        "path": str(path),
        "old_version": cur_fp,
        "new_version": STATE["versions"][f"stage{req.stage}"],
        "elapsed_s": round(time.time() - t0, 2),
    }


@app.post("/admin/unload")
def admin_unload(stage: int, x_admin_key: str | None = Header(default=None)):
    """卸载一个 stage 的模型，释放内存（下次请求会懒加载）。"""
    _require_admin(x_admin_key)
    if stage not in (1, 2):
        raise HTTPException(400, "stage must be 1 or 2")
    key = f"stage{stage}"
    with _lock:
        old = STATE[key]
        STATE[key] = None
        STATE["versions"][key] = None
    del old
    gc.collect()
    return {"status": "unloaded", "stage": stage}


@app.post("/extract")
async def extract(file: UploadFile = File(...), doc_id: str | None = Form(None)):
    """上传 PDF 或图片 → VLM 抽表 → Markdown。"""
    if not ENABLE_S1:
        raise HTTPException(503, "Stage1 VLM disabled by ENABLE_STAGE1=0")
    with _lock:
        if STATE["stage1"] is None:
            STATE["stage1"] = _load_stage1()
            STATE["versions"]["stage1"] = _dir_fingerprint(STAGE1_PATH)
            STATE["loaded_at"]["stage1"] = time.time()
        model, processor = STATE["stage1"]   # 抓住当前引用

    from PIL import Image
    import torch

    content = await file.read()
    filename = file.filename or "upload"
    doc_id = doc_id or Path(filename).stem

    # PDF → 页图
    images: list[Image.Image] = []
    if filename.lower().endswith(".pdf"):
        import fitz
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    else:
        images.append(Image.open(io.BytesIO(content)))

    prompt = "请提取图中所有表格，以标准 Markdown 格式输出。如无表格输出 '无表格'。"
    out_tables = []
    for page_no, img in enumerate(images, 1):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        md = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)[0]
        out_tables.append({"page": page_no, "markdown": md})

        # 自动入向量库
        if STATE["chroma"] and STATE["embed"] and "|" in md:
            _ingest_markdown_table(md, doc_id, page_no)

    return {"doc_id": doc_id, "n_pages": len(images), "tables": out_tables}


def _ingest_markdown_table(md: str, doc_id: str, page: int):
    """简化版：把 md 每非 header 行作为 chunk 入库。"""
    from src.rag import parse_markdown_table
    df = parse_markdown_table(md)
    if df.empty:
        return
    chunks, ids, metas = [], [], []
    for ri, row in df.iterrows():
        kv = " | ".join(f"{c}: {row[c]}" for c in df.columns)
        chunks.append(f"[{doc_id} | page {page} | row {ri+1}] {kv}")
        ids.append(f"{doc_id}_p{page}_r{ri}")
        metas.append({"doc_id": doc_id, "page": page, "row": int(ri)})
    embs = STATE["embed"].encode(chunks, normalize_embeddings=True).tolist()
    STATE["chroma"].add(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)


@app.post("/query", response_model=RAGOut)
def query(req: QueryIn):
    """RAG 检索 + LLM 回答。"""
    with _lock:
        if STATE["stage2"] is None or STATE["embed"] is None:
            raise HTTPException(503, "LLM or embedder not loaded")
        model, tok = STATE["stage2"]   # 抓住引用后释放锁，热加载时旧模型仍可用

    # 检索
    t0 = time.time()
    q_emb = STATE["embed"].encode([req.question], normalize_embeddings=True).tolist()
    where = {"doc_id": req.doc_filter} if req.doc_filter else None
    res = STATE["chroma"].query(query_embeddings=q_emb, n_results=req.top_k, where=where)
    hits = [
        {"text": d, "score": 1 - s, "metadata": m}
        for d, s, m in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]
    retr_ms = (time.time() - t0) * 1000

    # 组 prompt + 生成
    import torch
    from src.rag import RAG_SYSTEM
    ctx = "\n".join(f"- {h['text']} (score={h['score']:.2f})" for h in hits)
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"检索到的表格数据：\n{ctx}\n\n问题：{req.question}"},
    ]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    answer = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    gen_ms = (time.time() - t1) * 1000

    return RAGOut(answer=answer, sources=hits, retrieval_ms=retr_ms, generation_ms=gen_ms)


@app.post("/ingest_markdown")
def ingest_markdown(doc_id: str = Form(...), page: int = Form(1), markdown: str = Form(...)):
    """直接入一段 markdown 表格（跳过 VLM，适合已用 pdfplumber 拿到结构化表）。"""
    if STATE["chroma"] is None or STATE["embed"] is None:
        raise HTTPException(503, "embedder not loaded")
    _ingest_markdown_table(markdown, doc_id, page)
    return {"status": "ok", "vector_count": STATE["chroma"].count()}
