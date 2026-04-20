#!/usr/bin/env bash
# 合并 Axolotl 训练出的 LoRA adapter → HF 格式
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$HERE/axolotl"
source .venv/bin/activate

CONFIG="$HERE/configs/stage2_qwen25_qlora.yml"
ADAPTER_DIR="outputs/stage2_qlora"   # 与 yaml 里的 output_dir 对齐

python -m axolotl.cli.merge_lora "$CONFIG" \
    --lora_model_dir "$ADAPTER_DIR"

# axolotl 合并默认输出到 $ADAPTER_DIR/merged/
MERGED="$HERE/axolotl/$ADAPTER_DIR/merged"
DEST="$PROJECT_ROOT/models/stage2_fused"

if [ -d "$MERGED" ]; then
    rm -rf "$DEST"
    mkdir -p "$DEST"
    cp -r "$MERGED"/* "$DEST/"
    echo "✅ 合并完成并复制到 $DEST"
    echo "热加载到运行中的 Docker:"
    echo "  curl -X POST http://localhost:8000/admin/reload \\"
    echo "    -H 'X-Admin-Key: change-me-in-prod' \\"
    echo "    -d '{\"stage\":2,\"force\":true}'"
else
    echo "❌ 没找到合并后的目录 $MERGED"
    exit 1
fi
