"""`pipeline()` —— 5 行代码跑 HuggingFace 推理的最快方式.

什么时候用:
    - 快速验证模型能用
    - Colab / Notebook demo
    - 批处理离线任务

什么时候不用:
    - 生产 (没法精细控制 generation 参数 / device_map)
    - 需要流式 (pipeline 不支持 streaming)
    - 需要 chat template (pipeline 对 chat 支持最近才完善)

跑:
    python pipeline_api.py
"""
from __future__ import annotations

import os

import torch
from transformers import pipeline

MODEL_PATH = os.environ.get("MODEL_PATH", "../../models/stage2_fused")


def demo_text_generation():
    """文本生成 pipeline —— 最常见."""
    pipe = pipeline(
        task="text-generation",
        model=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # chat 格式（transformers 4.45+ 自动处理 chat template）
    messages = [
        {"role": "system", "content": "你是数据分析助手。"},
        {"role": "user", "content": "什么是 Continuous Batching？用 50 字回答。"},
    ]

    # do_sample=False 等价于 greedy；生产建议 False 为复现
    out = pipe(
        messages,
        max_new_tokens=200,
        do_sample=False,
    )
    print("=== text-generation ===")
    print(out[0]["generated_text"][-1])   # 最后一轮 assistant 的回复


def demo_batch():
    """Pipeline 也支持批处理 —— 注意 batch_size 参数."""
    pipe = pipeline(
        task="text-generation",
        model=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    inputs = [
        [{"role": "user", "content": "2022+2023="}],
        [{"role": "user", "content": "中国首都"}],
        [{"role": "user", "content": "Python 怎么读 JSONL"}],
    ]

    # batch_size 别开太大，OOM 会很快
    outputs = pipe(inputs, max_new_tokens=50, batch_size=2, do_sample=False)
    print("\n=== batch ===")
    for i, out in enumerate(outputs):
        print(f"[{i}] {out[0]['generated_text'][-1]['content'][:80]}")


def demo_other_tasks():
    """pipeline 其它 task（本项目没直接用，但要知道能做啥）."""
    print("\n=== 其它可用 pipeline task ===")
    print("""
    pipeline('text-generation', ...)     # 本文件演示的
    pipeline('summarization', ...)
    pipeline('translation', ...)
    pipeline('question-answering', ...)
    pipeline('token-classification', ...)  # NER
    pipeline('feature-extraction', ...)    # embedding (不如 sentence-transformers)
    pipeline('text-classification', ...)   # 分类
    pipeline('zero-shot-classification', ...)
    pipeline('fill-mask', ...)             # BERT-style MLM
    pipeline('image-to-text', ...)         # VLM caption
    pipeline('automatic-speech-recognition', ...)  # Whisper
    """)


if __name__ == "__main__":
    demo_text_generation()
    demo_batch()
    demo_other_tasks()
