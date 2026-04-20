#!/usr/bin/env bash
# Stage 2 LLM QLoRA 训练
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/LLaMA-Factory"
source .venv/bin/activate

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$HERE/logs"
llamafactory-cli train "$HERE/configs/stage2_qwen25_qlora.yaml" \
    2>&1 | tee "$HERE/logs/stage2_$(date +%Y%m%d_%H%M%S).log"

echo "✅ Stage 2 训练完成 -> $HERE/outputs/stage2_lora/"
echo "下一步: bash scripts/merge_stage2.sh"
