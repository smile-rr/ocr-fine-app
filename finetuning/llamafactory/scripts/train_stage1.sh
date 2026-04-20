#!/usr/bin/env bash
# Stage 1 VLM QLoRA 训练
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/LLaMA-Factory"
source .venv/bin/activate

# 国内加速（可选）
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$HERE/logs"
llamafactory-cli train "$HERE/configs/stage1_qwen2vl_qlora.yaml" \
    2>&1 | tee "$HERE/logs/stage1_$(date +%Y%m%d_%H%M%S).log"

echo "✅ Stage 1 训练完成 -> $HERE/outputs/stage1_lora/"
echo "下一步: bash scripts/merge_stage1.sh"
