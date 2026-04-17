#!/usr/bin/env bash
# 下载 HF base 模型到 models/stageN_fused/，跳过训练直接跑 Docker API。
#
# 用法：
#   bash scripts/setup_demo_models.sh           # 只下 stage2_fused (~1GB)
#   bash scripts/setup_demo_models.sh --stage1  # 加下 stage1_fused VLM (~4GB)
#   bash scripts/setup_demo_models.sh --v2      # 加下 stage2_fused_v2 (热加载用, ~3GB)
#   bash scripts/setup_demo_models.sh --all     # 全部
#
# v1 / v2 选型：
#   v1 (stage2_fused)      = Qwen/Qwen2.5-0.5B-Instruct     快 · 答案简短
#   v2 (stage2_fused_v2)   = Qwen/Qwen2.5-1.5B-Instruct     稍慢 · 答案更完整
#   热加载时同一问题能看出明显区别，用来验证 /admin/reload
set -e
cd "$(dirname "$0")/.."

STAGE1=0; V2=0
for arg; do
    case "$arg" in
        --stage1) STAGE1=1 ;;
        --v2)     V2=1 ;;
        --all)    STAGE1=1; V2=1 ;;
        -h|--help)
            sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

export STAGE1 V2
# hf_transfer 并发加速（可选）
export HF_HUB_ENABLE_HF_TRANSFER=1
: "${HF_ENDPOINT:=https://huggingface.co}"
echo "HF_ENDPOINT=$HF_ENDPOINT  HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"

uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download

# 只要 safetensors，跳过 .bin / original / 演示视频等，能省 30%+ 下载量
IGNORE = ["*.bin", "*.pt", "*.gguf", "original/*", "*.mp4", "*.png"]

def pull(repo, dst, desc):
    print(f"\n↓ {desc}\n  repo: {repo}\n  dst : {dst}")
    snapshot_download(repo_id=repo, local_dir=dst, ignore_patterns=IGNORE)
    print("  ✓ done")

# v1 —— 默认必下：Docker 启动时 stage2 是 eager load，必须存在
pull("Qwen/Qwen2.5-0.5B-Instruct",
     "models/stage2_fused",
     "Stage 2 v1 · 表格 QA LLM (0.5B)")

if os.environ.get("STAGE1") == "1":
    pull("Qwen/Qwen2-VL-2B-Instruct",
         "models/stage1_fused",
         "Stage 1 · 表格抽取 VLM (2B, ~4GB)")

if os.environ.get("V2") == "1":
    pull("Qwen/Qwen2.5-1.5B-Instruct",
         "models/stage2_fused_v2",
         "Stage 2 v2 · 热加载演示 (1.5B, ~3GB)")
PY

echo
echo "✅ 模型就位。目录："
ls -d models/*/ 2>/dev/null || true
echo
echo "👉 下一步：docker compose up --build"
