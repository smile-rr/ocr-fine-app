"""流式输出 —— 像 ChatGPT 那样一个字一个字往外吐。

原理:
    generate() 默认是"全部生成完才返回"的阻塞调用。
    TextIteratorStreamer 把 token 发到一个队列，主线程 for 循环取。
    因为 generate 会阻塞线程，所以要另起线程跑 generate，主线程消费队列。

跑:
    python streaming.py
"""
from __future__ import annotations

import os
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_PATH = os.environ.get("MODEL_PATH", "../../models/stage2_fused")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {"role": "user", "content": "用 150 字介绍 RAG 的核心思想和典型架构。"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # ==================== ⭐ 流式输出的核心 ====================
    # TextIteratorStreamer: generate 每产生一个 token 就 push 到这个 streamer
    # - skip_prompt=True:   不把 input 部分 echo 出来
    # - skip_special_tokens=True:  过滤 <|im_end|> 之类
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0,              # 取 token 的超时
    )

    # generation kwargs 单独包一下，给后面 Thread 用
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 另起线程跑 generate（它是阻塞的）
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 主线程从 streamer 拿 token 块（一般是一段中文 / 一小段英文）
    print("Response: ", end="", flush=True)
    for token_chunk in streamer:
        print(token_chunk, end="", flush=True)
    print()                       # 最后换行

    thread.join()


if __name__ == "__main__":
    main()


# ============================================================
# 生产中怎么做 SSE (Server-Sent Events) 流式 HTTP 返回
# ============================================================
# FastAPI + StreamingResponse:
#
#   from fastapi.responses import StreamingResponse
#
#   @app.post("/stream")
#   def stream_gen(req: QueryIn):
#       streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, ...)
#       Thread(target=model.generate, kwargs={..., "streamer": streamer}).start()
#
#       def sse():
#           for chunk in streamer:
#               yield f"data: {json.dumps({'token': chunk})}\n\n"
#           yield "data: [DONE]\n\n"
#
#       return StreamingResponse(sse(), media_type="text/event-stream")
#
# 客户端用 EventSource 接。OpenAI SDK stream=True 底下就是这个。
