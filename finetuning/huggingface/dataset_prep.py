"""HF datasets + tokenizer 数据流水线 —— 所有 train_*.py 共用。

做的事（按顺序）：
    1. datasets.load_dataset 读项目 data/stage2_train/*.jsonl（alpaca 格式）
    2. 把 alpaca 三元组转成 chat messages
    3. 应用 tokenizer.apply_chat_template 生成训练字符串
    4. tokenize 成 input_ids / attention_mask / labels
    5. （SFT-only）label 中把 prompt 部分设 -100，只算 response 的 loss

训练用 alpaca 格式:
    {"instruction": "...", "input": "...", "output": "..."}
输出训练字段:
    {"input_ids": [...], "attention_mask": [...], "labels": [...]}
"""
from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import MAX_SEQ_LENGTH, MODEL_ID, TRAIN_FILE, VAL_FILE


# ============================================================
# Step 1 & 2: 读 JSONL 并转成 messages 列表
# ============================================================
def load_raw_dataset() -> DatasetDict:
    """返回 HF DatasetDict{"train", "validation"}。"""
    ds = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VAL_FILE),
        },
    )
    return ds


def alpaca_to_messages(example: dict) -> dict:
    """alpaca 三元组 → chat messages 列表（sharegpt 风格）。"""
    user_content = example["instruction"]
    if example.get("input"):
        user_content = f"{example['instruction']}\n\n{example['input']}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


# ============================================================
# Step 3 & 4: 应用 chat template + tokenize
# ============================================================
def build_tokenizer(model_id: str = MODEL_ID) -> PreTrainedTokenizerBase:
    """标准流程：加载 tokenizer + 补 pad_token（Qwen/Llama 默认无）。"""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # 很多模型没 pad_token（Qwen/Llama3 的原始 tokenizer），训练会报错
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 右 padding（CausalLM 训练标准；左 padding 是 generate 时用）
    tok.padding_side = "right"
    return tok


def format_and_tokenize(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_SEQ_LENGTH,
) -> dict:
    """应用 chat template → tokenize。

    返回的 dict 能直接喂给 Trainer：input_ids / attention_mask。
    这里还没处理 labels（prompt mask）—— 见 mask_prompt_in_labels()。
    """
    # apply_chat_template 自动插 <|im_start|> <|im_end|> 等特殊 token
    # add_generation_prompt=False 因为我们训练完整对话（含 assistant 回复）
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,           # DataCollator 里再 padding
        return_tensors=None,
    )
    return tokens


def mask_prompt_in_labels(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_SEQ_LENGTH,
) -> dict:
    """生成 labels，并把 prompt 部分设为 -100（不算 loss）。

    这是 SFT 的标准做法：我们只想让模型学"怎么生成 assistant 回复"，
    不想浪费 loss 在预测 user 的输入上。

    实现：分两次 tokenize —— 一次只 user 部分，一次 user+assistant 全文。
    前 N 个 token 的 labels 设 -100（N = 只 user 部分的长度）。
    """
    user_msg = [example["messages"][0]]              # 只 user
    full_msg = example["messages"]                   # user + assistant

    prompt_text = tokenizer.apply_chat_template(
        user_msg, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_msg, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )["input_ids"]

    labels = list(full_ids)
    # prompt 部分 → -100 (CrossEntropyLoss 的 ignore_index)
    mask_len = min(len(prompt_ids), len(labels))
    for i in range(mask_len):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ============================================================
# 一站式入口
# ============================================================
def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_SEQ_LENGTH,
    mask_prompt: bool = True,
) -> DatasetDict:
    """读取 + 格式化 + tokenize + (可选) mask prompt。返回的 ds 可直接给 Trainer。

    mask_prompt=True  → 生成 labels，prompt 部分 -100 (标准 SFT)
    mask_prompt=False → labels = input_ids (续写式训练, 如 continued pretraining)
    """
    ds = load_raw_dataset()
    ds = ds.map(alpaca_to_messages)

    if mask_prompt:
        ds = ds.map(
            lambda ex: mask_prompt_in_labels(ex, tokenizer, max_length),
            remove_columns=ds["train"].column_names,
        )
    else:
        ds = ds.map(
            lambda ex: format_and_tokenize(ex, tokenizer, max_length),
            remove_columns=ds["train"].column_names,
        )
        # 非 mask 模式下让 labels = input_ids
        ds = ds.map(lambda ex: {**ex, "labels": ex["input_ids"]})

    return ds


# ============================================================
# 调试 —— 直接 python dataset_prep.py 跑可以看处理结果
# ============================================================
if __name__ == "__main__":
    tok = build_tokenizer()
    ds = prepare_dataset(tok)
    print(ds)
    print("\n=== First example ===")
    ex = ds["train"][0]
    print(f"input_ids[:30]:    {ex['input_ids'][:30]}")
    print(f"labels[:30]:       {ex['labels'][:30]}  (-100 = prompt mask)")
    print(f"len(input_ids):    {len(ex['input_ids'])}")
    print(f"len mask_prompt:   {sum(1 for x in ex['labels'] if x == -100)}")
    print(f"\n--- decoded ---\n{tok.decode(ex['input_ids'])[:500]}...")
