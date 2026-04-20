#!/usr/bin/env bash
# 合并 Stage 2 LoRA adapter → HF 格式
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$HERE/LLaMA-Factory"
source .venv/bin/activate

OUT="$PROJECT_ROOT/models/stage2_fused"
mkdir -p "$OUT"

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --adapter_name_or_path "$HERE/outputs/stage2_lora" \
    --template qwen \
    --finetuning_type lora \
    --export_dir "$OUT" \
    --export_size 4 \
    --export_legacy_format false \
    --trust_remote_code true

echo "✅ 合并完成 -> $OUT"
echo "热加载到运行中的 Docker:"
echo "  curl -X POST http://localhost:8000/admin/reload \\"
echo "    -H 'X-Admin-Key: change-me-in-prod' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"stage\":2,\"force\":true}'"
