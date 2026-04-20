"""手动 Batch 推理 —— 为什么 transformers 的 batch 远不如 vLLM.

场景: 一次处理 N 个请求，看 batch size 对吞吐的影响。

关键认识:
    - transformers 的 batch 是 "全部填充到同一长度同时 forward"
    - 短请求被长请求拖累（必须等最长的完成）
    - KV cache 按 batch 整体分配，短请求浪费
    → 这就是 vLLM continuous batching 要解决的问题

跑:
    python batched.py
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "../../models/stage2_fused")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"                # ⭐ batch 推理必须左 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ==================== 准备一批不同长度的请求 ====================
    prompts_raw = [
        [{"role": "user", "content": "你好"}],                                         # 短
        [{"role": "user", "content": "用一句话解释 RAG"}],                             # 中
        [{"role": "user", "content": "介绍 PagedAttention 的原理和它解决什么问题"}],    # 长
        [{"role": "user", "content": "为什么 LoRA 只训低秩矩阵就够了？"}],              # 中
    ]
    prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]

    # ==================== ⭐ 关键: 一次 tokenize 整个 batch + padding ====================
    # padding=True 自动填短序列到最长；左 padding（上面设的）
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    print(f"batch size: {len(prompts)}")
    print(f"padded input shape: {inputs.input_ids.shape}")
    print(f"  -> 最短 prompt 也占用了最长的长度（浪费的 token 位置不计 attention）")

    # ==================== 生成 ====================
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0

    # ==================== 解码（每个样本单独切） ====================
    # output_ids shape: [batch, padded_input_len + max_new_tokens]
    # 每个样本的 input 长度不同（左 padding），要逐个切
    input_lens = [len(ids) for ids in inputs.input_ids]
    print(f"\n--- results (batch forward {elapsed:.2f}s) ---")
    total_gen_tokens = 0
    for i, (prompt, out_ids, in_len) in enumerate(
        zip(prompts_raw, output_ids, input_lens)
    ):
        gen_ids = out_ids[in_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        total_gen_tokens += (gen_ids != tokenizer.pad_token_id).sum().item()
        print(f"\n[{i}] Q: {prompt[0]['content']}")
        print(f"    A: {gen_text.strip()}")

    throughput = total_gen_tokens / elapsed
    print(f"\n总吞吐: {total_gen_tokens} tokens / {elapsed:.2f}s = {throughput:.1f} tok/s")

    # ==================== 对比演示 ====================
    print("\n" + "=" * 60)
    print("为什么 vLLM 更快（理论）:")
    print("=" * 60)
    print(
        "  transformers 的 batch 必须等所有样本到同一长度才能继续 forward，\n"
        "  最短的请求要等最长的请求；而且 KV cache 按 padded 长度整体分配，\n"
        "  短请求浪费的 KV 内存不能给新来的请求用。\n"
        "\n"
        "  vLLM 的 continuous batching:\n"
        "    - 每生成 1 个 token 就重新决定 batch 组成\n"
        "    - 完成的请求立即退出，新请求立即加入\n"
        "    - KV cache 按 block（16/32 token）分页管理，按需分配\n"
        "    - 结果: 同样硬件吞吐 10-30×"
    )


if __name__ == "__main__":
    main()
