"""纯 LoRA 微调 —— Trainer + peft.LoraConfig (无量化).

显存需求: 0.5B ≈ 8GB, 7B ≈ 20GB
对比 LLaMA-Factory: configs/stage2_qwen25_qlora.yaml 去掉 quantization_bit 的版本

跑:
    source .venv/bin/activate
    python train_lora.py
"""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
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

    # ==================== 3. Model ====================
    print(f"\nLoading base model: {config.MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16 if config.USE_BF16 else torch.float16,
        attn_implementation=config.ATTN_IMPL,
        trust_remote_code=True,
        # device_map="auto",  # 多卡时启用；单卡不用
    )

    # 训练前准备（gradient_checkpointing 要在 peft 之前开）
    model.gradient_checkpointing_enable()
    # 下面这行必须，否则 gradient_checkpointing + LoRA 组合有梯度断流的坑
    model.enable_input_require_grads()

    # ==================== 4. LoRA 挂载 ====================
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
    # 预期输出: trainable params: ~1M / all params: ~500M (0.5B 模型, r=8)
    #          比例 0.2% —— 这就是 LoRA 的魔法

    # ==================== 5. TrainingArguments ====================
    args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        # 精度
        bf16=config.USE_BF16,
        fp16=config.USE_FP16,
        # 优化器
        optim="adamw_torch",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        # 日志 / 保存
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 其它
        report_to=["tensorboard"],     # tensorboard --logdir outputs
        seed=42,
        # 禁用 pin_memory 在某些环境下报 "pin_memory is not supported"
        dataloader_pin_memory=False,
    )

    # ==================== 6. Data Collator ====================
    # DataCollatorForSeq2Seq 会把 batch 内序列 pad 到相同长度
    # label_pad_token_id=-100 让 padding 位置不算 loss
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,          # 凑 8 的倍数，利用 Tensor Core
    )

    # ==================== 7. Trainer ====================
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # ==================== 8. 开练 ====================
    print("\n" + "=" * 60)
    print("Start training...")
    print("=" * 60)
    trainer.train()

    # ==================== 9. 保存最终 adapter ====================
    final_dir = config.OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))   # 只保存 adapter weights + config
    tokenizer.save_pretrained(str(final_dir))

    print(f"\n✅ Done. Adapter saved to: {final_dir}")
    print("\n下一步合并到 base:")
    print("  from peft import PeftModel")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{config.MODEL_ID}')")
    print(f"  model = PeftModel.from_pretrained(base, '{final_dir}')")
    print("  merged = model.merge_and_unload()")
    print("  merged.save_pretrained('../../models/stage2_fused')")


if __name__ == "__main__":
    main()
