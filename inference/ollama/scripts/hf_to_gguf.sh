#!/usr/bin/env bash
# 把 HF 格式模型转成 GGUF 供 Ollama/llama.cpp 使用
# 用法: bash inference/ollama/scripts/hf_to_gguf.sh [model_dir]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC_MODEL="${1:-$PROJECT_ROOT/models/stage2_fused}"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$SRC_MODEL" ]; then
    echo "❌ 源模型目录不存在: $SRC_MODEL"
    echo "先跑: bash scripts/setup_demo_models.sh"
    exit 1
fi

TMP="$OUT_DIR/.llamacpp_tmp"
mkdir -p "$TMP"
cd "$TMP"

echo "=== 1. Clone llama.cpp (转换工具 + 量化工具) ==="
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

echo
echo "=== 2. 装转换脚本依赖 ==="
pip install -r requirements/requirements-convert_hf_to_gguf.txt

echo
echo "=== 3. 编译 llama-quantize ==="
if [ ! -f "build/bin/llama-quantize" ]; then
    cmake -B build -DLLAMA_METAL=ON    # Mac Metal 加速
    cmake --build build --config Release --target llama-quantize -j
fi

echo
echo "=== 4. HF → GGUF (fp16) ==="
python convert_hf_to_gguf.py "$SRC_MODEL" \
    --outfile "$OUT_DIR/stage2-fp16.gguf" \
    --outtype f16

echo
echo "=== 5. GGUF fp16 → Q4_K_M (4-bit 量化) ==="
./build/bin/llama-quantize \
    "$OUT_DIR/stage2-fp16.gguf" \
    "$OUT_DIR/stage2-q4_k_m.gguf" \
    Q4_K_M

echo
echo "=== ✅ 完成 ==="
ls -lh "$OUT_DIR"/*.gguf
echo
echo "下一步:"
echo "  cd $OUT_DIR"
echo "  ollama create ocr-stage2 -f Modelfile"
echo "  ollama run ocr-stage2"
