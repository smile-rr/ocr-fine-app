#!/usr/bin/env bash
# 安装 Axolotl 到独立目录
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

echo "=== 1. Clone Axolotl ==="
if [ ! -d "axolotl" ]; then
    git clone --depth 1 https://github.com/OpenAccess-AI-Collective/axolotl.git
else
    echo "axolotl/ 已存在，跳过 clone"
fi

echo
echo "=== 2. 建独立 venv ==="
cd axolotl
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo
echo "=== 3. 安装 Axolotl ==="
pip install --upgrade pip
# packaging 要先装，axolotl 的 setup.py 会用
pip install packaging ninja

# 根据 CUDA 情况装，默认 CUDA 12.1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 || \
    pip install torch==2.3.1
pip install -e '.[flash-attn,deepspeed]' || \
    pip install -e '.'  # flash-attn 装失败就只装主包

echo
echo "=== 4. 初始化 accelerate 配置 ==="
accelerate config default  # 默认单机单卡；多卡跑 accelerate config 交互式配

echo
echo "=== ✅ 安装完成 ==="
echo "下一步:"
echo "  1. 生成训练数据:  cd ../../../  &&  uv run python scripts/prepare_stage2.py"
echo "  2. 启动训练:      bash finetuning/axolotl/scripts/train_stage2.sh"
