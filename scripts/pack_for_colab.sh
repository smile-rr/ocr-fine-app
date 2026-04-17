#!/usr/bin/env bash
# 为 Colab 训练打包本地已准备好的数据。
#
# 用法：
#   bash scripts/pack_for_colab.sh        # 两个 zip 都生成
#   bash scripts/pack_for_colab.sh stage1 # 只 Stage 1
#   bash scripts/pack_for_colab.sh stage2 # 只 Stage 2
#
# 产物（放项目根目录）：
#   stage1_colab.zip  ~ 80–160 MB  · jsonl + 训练图片
#   stage2_colab.zip  ~ 1–5  MB   · 纯 jsonl
#
# 上传到 Colab：runtime → Files → 拖进去
set -e

cd "$(dirname "$0")/.."
STAGE="${1:-all}"

pack_stage1() {
    if [ ! -f "data/stage1_train/train.jsonl" ]; then
        echo "❌ data/stage1_train/train.jsonl 不存在；先跑 prepare_stage1.py"
        return 1
    fi
    echo "📦 packing stage1 ..."
    rm -f stage1_colab.zip
    zip -q -r stage1_colab.zip \
        data/stage1_train \
        data/stage1_images
    ls -lh stage1_colab.zip
}

pack_stage2() {
    if [ ! -f "data/stage2_train/train.jsonl" ]; then
        echo "❌ data/stage2_train/train.jsonl 不存在；先跑 prepare_stage2.py"
        return 1
    fi
    echo "📦 packing stage2 ..."
    rm -f stage2_colab.zip
    zip -q -r stage2_colab.zip data/stage2_train
    ls -lh stage2_colab.zip
}

case "$STAGE" in
    stage1) pack_stage1 ;;
    stage2) pack_stage2 ;;
    all)    pack_stage1 && pack_stage2 ;;
    *) echo "unknown: $STAGE (want: stage1 | stage2 | all)"; exit 1 ;;
esac

echo "✅ done. 接下来打开 notebooks/0{2,3}b_*_colab.ipynb 照做。"
