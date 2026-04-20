"""最小 FastAPI + HF transformers 推理服务.

和项目主服务 src/serve/api.py 的区别:
    - 这里只有 /chat 端点，纯教学用（没 RAG 没向量库）
    - 端口 :8004，主 API 在 :8000，可同时启动对比

跑:
    uvicorn serve_fastapi:app --host 0.0.0.0 --port 8004
    # 或
    python serve_fastapi.py

测试:
    curl -X POST http://localhost:8004/chat \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"你好"}]}'

为什么**不要**用这个跑生产:
    - 单请求串行（没 continuous batching）
    - 无并发保护（model.generate 不是线程安全）
    - 无请求队列（打爆就 OOM）
    → 生产直接用 vLLM / TGI / Ollama
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "../../models/stage2_fused")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))


# ============================================================
# 全局状态 —— 简单起见用 module-level 变量
# 生产上应封装成 class 或用 lifespan 管理
# ============================================================
STATE: dict[str, Any] = {"model": None, "tokenizer": None}
MODEL_LOCK = Lock()     # generate 不是线程安全，必须加锁串行


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时清理."""
    print(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()
    STATE["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if STATE["tokenizer"].pad_token is None:
        STATE["tokenizer"].pad_token = STATE["tokenizer"].eos_token

    STATE["model"] = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    STATE["model"].eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    yield

    # cleanup
    STATE["model"] = None
    STATE["tokenizer"] = None


app = FastAPI(title="HF Transformers Inference", lifespan=lifespan)


# ============================================================
# Schemas
# ============================================================
class Message(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Message]
    max_tokens: int = MAX_TOKENS
    temperature: float = 0.1
    do_sample: bool = False


class ChatOut(BaseModel):
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: float


# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "model_path": MODEL_PATH,
        "device": str(STATE["model"].device) if STATE["model"] else None,
    }


@app.post("/chat", response_model=ChatOut)
async def chat(req: ChatIn):
    if STATE["model"] is None:
        raise HTTPException(503, "model not loaded")

    t0 = time.time()
    tokenizer = STATE["tokenizer"]

    # 构 prompt
    messages = [m.model_dump() for m in req.messages]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(STATE["model"].device)
    prompt_tokens = inputs.input_ids.shape[1]

    # ⭐ 关键：model.generate 在 asyncio 里要放到 executor，不然阻塞事件循环
    def _generate():
        with MODEL_LOCK:              # 串行保护
            with torch.no_grad():
                return STATE["model"].generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    do_sample=req.do_sample,
                    temperature=req.temperature if req.do_sample else 1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

    loop = asyncio.get_event_loop()
    output_ids = await loop.run_in_executor(None, _generate)

    # 切掉 input 部分
    gen_ids = output_ids[0, prompt_tokens:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    completion_tokens = gen_ids.shape[0]

    return ChatOut(
        response=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_time_ms=(time.time() - t0) * 1000,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
