#!/usr/bin/env bash
# 安装 LLaMA-Factory 到独立目录 (不污染项目主 venv)
# 用法: bash finetuning/llamafactory/scripts/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

echo "=== 1. Clone LLaMA-Factory ==="
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
else
    echo "LLaMA-Factory/ 已存在，跳过 clone"
fi

echo
echo "=== 2. 建独立 venv ==="
cd LLaMA-Factory
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo
echo "=== 3. 安装 LLaMA-Factory (含 bitsandbytes for QLoRA) ==="
pip install --upgrade pip
pip install -e ".[torch,metrics,bitsandbytes]"

# 可选：flash-attn (CUDA 才有用；装不上就 fallback 到 sdpa)
if python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo
    echo "=== 4. 尝试装 flash-attn (CUDA 环境) ==="
    pip install flash-attn --no-build-isolation || \
        echo "⚠️  flash-attn 装失败；训练 yaml 里把 flash_attn: fa2 改成 sdpa"
else
    echo "检测到无 CUDA，跳过 flash-attn"
fi

echo
echo "=== 5. 注册数据集 ==="
mkdir -p data
if [ ! -L "data/dataset_info.json" ] && [ -f "data/dataset_info.json" ]; then
    mv data/dataset_info.json data/dataset_info.json.orig
    echo "备份原 dataset_info.json -> dataset_info.json.orig"
fi
ln -sf "$HERE/configs/dataset_info.json" data/dataset_info.json
echo "软链接: LLaMA-Factory/data/dataset_info.json -> configs/dataset_info.json"

echo
echo "=== ✅ 安装完成 ==="
echo "下一步:"
echo "  1. 生成训练数据:  cd ../../../  &&  uv run python scripts/prepare_stage2.py"
echo "  2. 启动训练:      bash finetuning/llamafactory/scripts/train_stage2.sh"
