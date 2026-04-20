"""最基础的 HuggingFace 推理 —— 面试白板题模板。

流程: 加载 tokenizer → 加载 model → apply chat template → generate → decode

跑:
    python basic_generate.py

前置:
    models/stage2_fused/ 存在（跑过 setup_demo_models.sh 或微调合并过）
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 配置 —— 改这里切不同模型 / 本地 vs HF Hub
# ============================================================
# 本项目合并后的模型路径（本地）
MODEL_PATH = os.environ.get("MODEL_PATH", "../../models/stage2_fused")
# 或直接用 HF Hub 上的模型（会自动下载）
# MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # ==================== 1. Tokenizer ====================
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # generate 时用左 padding（右 padding 会把 attention 搞乱）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==================== 2. Model ====================
    # torch_dtype: bf16 比 fp16 对 Ampere+ 卡更友好（H100 上差别更大）
    # device_map="auto": 单卡 = cuda:0；多卡自动切分
    # attn_implementation: sdpa 兼容性最好；flash_attention_2 要装 flash-attn
    print(f"Loading model on {DEVICE}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()                  # 推理模式（关 dropout 等）
    print(f"  loaded in {time.time()-t0:.1f}s")

    # ==================== 3. 构造对话 ====================
    messages = [
        {"role": "system", "content": "你是精确的数据分析助手，回答要简洁。"},
        {
            "role": "user",
            "content": (
                "表格：\n"
                "| 年份 | 营收(亿) | 净利润(亿) |\n"
                "|---|---|---|\n"
                "| 2022 | 100 | 15 |\n"
                "| 2023 | 120 | 18 |\n"
                "| 2024 | 135 | 22 |\n\n"
                "问题：净利润增长最快的是哪一年？"
            ),
        },
    ]

    # ⭐ apply_chat_template 关键点
    #   - tokenize=False 返回字符串，方便打印
    #   - add_generation_prompt=True 加上 "<|im_start|>assistant\n"，让模型接着生成
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"\n--- prompt ---\n{prompt_text}\n")

    # ==================== 4. Tokenize ====================
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    print(f"input tokens: {inputs.input_ids.shape[1]}")

    # ==================== 5. Generate ====================
    t1 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,              # greedy，可复现
            # 采样模式：
            # do_sample=True,
            # temperature=0.7,
            # top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,               # 推理必开，10× 加速
        )
    gen_time = time.time() - t1

    # ==================== 6. 解码（注意切掉 input 部分）====================
    # generate 返回的是 [prompt_ids..., generated_ids...]
    # 要切出生成的部分
    generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ==================== 7. 汇报 ====================
    n_gen = generated_ids.shape[0]
    print(f"\n--- response ({n_gen} tokens, {gen_time:.2f}s, "
          f"{n_gen/gen_time:.1f} tok/s) ---")
    print(response)


if __name__ == "__main__":
    main()
