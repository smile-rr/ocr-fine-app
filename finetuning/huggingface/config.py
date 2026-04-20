"""共享配置 —— 所有 train_*.py 都 import 这里。"""
from __future__ import annotations
from pathlib import Path

# ============================================================
# 路径（相对项目根解析）
# ============================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stage2_train"
OUTPUT_DIR = HERE / "outputs" / "stage2_lora_hf"

TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

# ============================================================
# 模型
# ============================================================
# 0.5B 适合 Colab T4 / 笔记本 RTX 3060；7B 需要 A10G/A100
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # 换大的取消注释

# ============================================================
# 训练超参
# ============================================================
MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8                    # 有效 batch = 2 × 8 = 16
LEARNING_RATE = 2e-4              # LoRA 标准值（比全参数微调大 10×）
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# ============================================================
# LoRA 超参
# ============================================================
LORA_R = 8
LORA_ALPHA = 16                   # 惯例 = 2 × r
LORA_DROPOUT = 0.05

# target_modules: 覆盖越全越好
# "all-linear" 是 peft>=0.10 支持的快捷方式，自动挂所有 linear 层
# 或显式列出：
LORA_TARGET_MODULES = "all-linear"
# LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
#                        "gate_proj", "up_proj", "down_proj"]

# ============================================================
# 精度 / 设备
# ============================================================
# A100/H100/3090+ 用 bf16；T4/V100 用 fp16
USE_BF16 = True
USE_FP16 = False

# flash-attn 装不上就改 "sdpa"
ATTN_IMPL = "sdpa"  # "flash_attention_2" / "sdpa" / "eager"
