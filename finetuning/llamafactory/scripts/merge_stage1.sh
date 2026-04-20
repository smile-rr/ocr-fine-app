#!/usr/bin/env bash
# 合并 Stage 1 LoRA adapter 到 base → HF 格式（供 Docker API / vLLM 加载）
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$HERE/LLaMA-Factory"
source .venv/bin/activate

OUT="$PROJECT_ROOT/models/stage1_fused"
mkdir -p "$OUT"

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --adapter_name_or_path "$HERE/outputs/stage1_lora" \
    --template qwen2_vl \
    --finetuning_type lora \
    --visual_inputs true \
    --export_dir "$OUT" \
    --export_size 4 \
    --export_legacy_format false \
    --trust_remote_code true

echo "✅ 合并完成 -> $OUT"
echo "Docker 里 STAGE1_MODEL_PATH 会自动指到这里"
