"""LoRA adapter 推理的 3 种姿势。

1. 挂 adapter 推理（慢一点，但灵活 —— 可以热切换）
2. 合并 adapter 到 base（推理和没挂 adapter 一样快，但失去灵活性）
3. 多 adapter 路由（一个 base 对应多个任务的 adapter）

跑:
    python with_adapter.py
"""
from __future__ import annotations

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_PATH = os.environ.get(
    "ADAPTER_PATH",
    "../../finetuning/huggingface/outputs/stage2_lora_hf/final",
)


def demo_mount_adapter():
    """方式 1: 挂 adapter —— base + adapter 分开存，推理时合并 forward."""
    print("=" * 60)
    print("方式 1: 挂 adapter (PeftModel.from_pretrained)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ⭐ 关键: 加载 adapter
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    _quick_test(model, tokenizer, "base + adapter")


def demo_merge_adapter():
    """方式 2: 合并 adapter —— 变成一个普通模型，推理速度等于 base."""
    print("\n" + "=" * 60)
    print("方式 2: 合并 adapter (merge_and_unload)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    peft_model = PeftModel.from_pretrained(base, ADAPTER_PATH)

    # ⭐ 合并 —— W_new = W_base + (alpha/r) * B @ A
    # 合并后返回普通的 AutoModelForCausalLM，peft API 全部消失
    merged = peft_model.merge_and_unload()
    merged.eval()

    # 可以保存为新 base 给 vLLM / Docker 加载
    # merged.save_pretrained("../../models/stage2_fused")
    # tokenizer.save_pretrained("../../models/stage2_fused")

    _quick_test(merged, tokenizer, "merged (= pure base after merge)")


def demo_multi_adapter_routing():
    """方式 3: 多 adapter —— 一个 base 挂多个 adapter, 运行时按 name 切."""
    print("\n" + "=" * 60)
    print("方式 3: 多 adapter 路由 (set_adapter)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # 先挂第一个
    model = PeftModel.from_pretrained(base, ADAPTER_PATH, adapter_name="table_qa")

    # 加载更多（如果有不同任务的 adapter）
    # model.load_adapter(".../stage2_translate", adapter_name="translation")
    # model.load_adapter(".../stage2_classify", adapter_name="classify")

    # 切换 adapter
    model.set_adapter("table_qa")
    print(f"active adapter: {model.active_adapter}")

    _quick_test(model, tokenizer, "routed to 'table_qa'")
    print("\n生产上这个模式被 vLLM S-LoRA 抽象成内建功能，见 inference/vllm/")


def _quick_test(model, tokenizer, label: str):
    """三个 demo 共用的快速生成测试."""
    messages = [
        {"role": "user", "content": "表格：2023年营收 120亿。问：2023年营收？"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[{label}] {resp.strip()}")


if __name__ == "__main__":
    if not os.path.isdir(ADAPTER_PATH):
        print(f"⚠️  Adapter 目录不存在: {ADAPTER_PATH}")
        print("   请先跑过 finetuning/huggingface/train_lora.py 或改 ADAPTER_PATH")
        print("   这里只演示方式 2 (合并不挂 adapter 也不会出错)")
    else:
        demo_mount_adapter()
        demo_merge_adapter()
        demo_multi_adapter_routing()
