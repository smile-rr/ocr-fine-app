"""TRL SFTTrainer —— HF 官方现代推荐，LLaMA-Factory 的内核就是它。

和 train_lora.py 对比 SFTTrainer 帮你省的代码：
    ❌ 不需要手写 mask_prompt_in_labels (SFT 自动做)
    ❌ 不需要手动 apply_chat_template (dataset_text_field 或 formatting_func)
    ❌ 不需要手动 DataCollator (SFT 自带合适的)
    ✅ 内建 packing (多个短样本拼到 max_length, 吞吐 2-3×)
    ✅ 内建 DPO/KTO/ORPO 入口 (只换 Trainer 类名即可)

跑:
    python train_sft_trl.py
"""
from __future__ import annotations

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import config


def alpaca_to_messages(example: dict) -> dict:
    """和 dataset_prep.py 的函数一样，这里 inline 展示 SFTTrainer 只需要 messages 字段。"""
    user = example["instruction"]
    if example.get("input"):
        user = f"{example['instruction']}\n\n{example['input']}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def main():
    # ==================== 1. Tokenizer ====================
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==================== 2. Dataset ====================
    # SFTTrainer 能直接吃 "messages" 列（会自动 apply_chat_template）
    ds = load_dataset(
        "json",
        data_files={"train": str(config.TRAIN_FILE), "validation": str(config.VAL_FILE)},
    )
    ds = ds.map(alpaca_to_messages)
    # SFTTrainer 看到 messages 字段就自动处理模板 + mask prompt

    # ==================== 3. Model ====================
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16 if config.USE_BF16 else torch.float16,
        attn_implementation=config.ATTN_IMPL,
        trust_remote_code=True,
    )

    # ==================== 4. LoRA 配置直接传给 SFTTrainer ====================
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    # ==================== 5. SFTConfig（继承 TrainingArguments + SFT 特有字段） ====================
    sft_args = SFTConfig(
        output_dir=str(config.OUTPUT_DIR) + "_trl",
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=config.USE_BF16,
        fp16=config.USE_FP16,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        report_to=["tensorboard"],
        seed=42,

        # ⭐ SFT 特有
        max_seq_length=config.MAX_SEQ_LENGTH,
        packing=True,                           # 多样本拼接，吞吐 2-3×
        # 它会自动读 messages 字段应用 chat_template，不用手写
    )

    # ==================== 6. SFTTrainer ====================
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        peft_config=peft_config,                # 传 LoRA 配置，SFTTrainer 自动 get_peft_model
    )

    # ==================== 7. Train ====================
    print("\n" + "=" * 60)
    print("Start SFT training (TRL)...")
    print("=" * 60)
    trainer.train()

    # ==================== 8. 保存 ====================
    final_dir = config.OUTPUT_DIR.parent / "stage2_trl_sft" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n✅ Done. Adapter saved to: {final_dir}")


# ============================================================
# BONUS: 同一个 SFTTrainer 怎么换成 DPO 训练
# ============================================================
# 就换个 Trainer 类名 + 准备偏好数据（prompt + chosen + rejected）:
#
#   from trl import DPOConfig, DPOTrainer
#
#   dpo_args = DPOConfig(output_dir="./dpo_out", num_train_epochs=1, ...)
#   trainer = DPOTrainer(
#       model=model,                       # SFT 后的模型做 init policy
#       ref_model=None,                    # None = 自动用 frozen copy of model
#       args=dpo_args,
#       train_dataset=preference_dataset,  # {prompt, chosen, rejected}
#       tokenizer=tokenizer,
#       peft_config=peft_config,
#   )
#   trainer.train()
#
# KTO / ORPO / SimPO 类似，只换类名。TRL 统一了接口。


if __name__ == "__main__":
    main()
