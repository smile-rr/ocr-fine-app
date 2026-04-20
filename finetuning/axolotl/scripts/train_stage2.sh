#!/usr/bin/env bash
# Stage 2 QLoRA (Axolotl)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/axolotl"
source .venv/bin/activate

export HF_HUB_ENABLE_HF_TRANSFER=1

CONFIG="$HERE/configs/stage2_qwen25_qlora.yml"
# 切 LoRA 非量化版: CONFIG="$HERE/configs/stage2_qwen25_lora_full.yml"

mkdir -p "$HERE/logs"
accelerate launch -m axolotl.cli.train "$CONFIG" \
    2>&1 | tee "$HERE/logs/stage2_$(date +%Y%m%d_%H%M%S).log"

echo "✅ 训练完成 -> $HERE/axolotl/outputs/stage2_qlora/"
echo "下一步: bash scripts/merge_stage2.sh"
