"""QLoRA 微调 —— LoRA + 4-bit NF4 量化 base.

显存需求: 0.5B ≈ 4GB, 7B ≈ 10GB（T4/3060 能训 7B 就靠这个）
对比 LLaMA-Factory: configs/stage2_qwen25_qlora.yaml 的完整版本

和 train_lora.py 的**唯一差别**:
    1. BitsAndBytesConfig 传给 from_pretrained
    2. 调 prepare_model_for_kbit_training() 让 4-bit 模型能正确反向传播
    （其它代码 100% 一样，便于对照学习）

跑:
    # 仅 Linux + CUDA
    python train_qlora.py
"""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

import config
from dataset_prep import build_tokenizer, prepare_dataset


def main():
    # ==================== 1. Tokenizer ====================
    tokenizer = build_tokenizer()

    # ==================== 2. Dataset ====================
    print("Loading dataset...")
    ds = prepare_dataset(tokenizer, mask_prompt=True)
    print(f"  train: {len(ds['train'])}  val: {len(ds['validation'])}")

    # ==================== 3. ⭐ BitsAndBytes 4-bit 量化配置 ====================
    # QLoRA 的核心：base 权重 4-bit NF4 存，forward 时反量化到 bf16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                              # 4-bit 模式
        bnb_4bit_quant_type="nf4",                      # NF4 比 FP4 精度好
        bnb_4bit_compute_dtype=torch.bfloat16,          # forward/backward 用 bf16
        bnb_4bit_use_double_quant=True,                 # 对量化常数再量化，省 0.4 bit/param
    )

    # ==================== 4. Model (with 4-bit) ====================
    print(f"\nLoading base model in 4-bit: {config.MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,                 # ⭐ 关键一行
        attn_implementation=config.ATTN_IMPL,
        trust_remote_code=True,
        device_map="auto",                              # 4-bit 模型必须指定 device_map
    )

    # ==================== 5. ⭐ 准备 4-bit 模型能训练 ====================
    # 这一步做了几件事:
    #   - 冻结 base 权重（4-bit 不能训）
    #   - 把 layer norm 转 fp32（数值稳定）
    #   - 开 gradient_checkpointing
    #   - 开 input_require_grads (配合 gradient_checkpointing)
    # 不调这个函数直接挂 LoRA 会导致梯度不回传
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # ==================== 6. LoRA（和 train_lora.py 完全一样） ====================
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ==================== 7. TrainingArguments ====================
    args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR) + "_qlora",
        overwrite_output_dir=True,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=config.USE_BF16,
        fp16=config.USE_FP16,
        # ⭐ QLoRA 推荐用 paged_adamw (8-bit Adam + CPU offload, 省显存)
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,                              # QLoRA 原论文建议 0.3
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=42,
        dataloader_pin_memory=False,
    )

    # ==================== 8. Collator / Trainer / Train ====================
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    print("\n" + "=" * 60)
    print("Start QLoRA training...")
    print("=" * 60)
    trainer.train()

    # ==================== 9. 保存 ====================
    final_dir = config.OUTPUT_DIR.parent / "stage2_qlora_hf" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n✅ Done. Adapter saved to: {final_dir}")

    print("\n⚠️  QLoRA 合并注意事项:")
    print("    直接 merge_and_unload() 会把 adapter 量化掉（精度丢失）")
    print("    正确姿势: 加载 base 时用 fp16/bf16（不量化），再 merge")
    print("    from transformers import AutoModelForCausalLM")
    print(f"    base = AutoModelForCausalLM.from_pretrained('{config.MODEL_ID}', torch_dtype='bfloat16')")
    print(f"    peft = PeftModel.from_pretrained(base, '{final_dir}')")
    print("    merged = peft.merge_and_unload()")


if __name__ == "__main__":
    main()
