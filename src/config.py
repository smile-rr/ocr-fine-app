"""全局路径与默认配置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HF_CACHE = DATA_DIR / "hf_cache"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
CHROMA_DIR = ROOT / "chroma_db"
LOGS_DIR = ROOT / "logs"

for p in [DATA_DIR, RAW_DIR, HF_CACHE, SAMPLES_DIR, MODELS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# 模型选择（MacBook 友好 0.5B / 2B 级别）
STAGE1_VLM_HF = "Qwen/Qwen2-VL-2B-Instruct"            # Stage 1 VLM 表格提取
STAGE1_VLM_MLX = "mlx-community/Qwen2-VL-2B-Instruct-4bit"  # MLX 4bit 版
STAGE2_LLM_HF = "Qwen/Qwen2.5-0.5B-Instruct"          # Stage 2 QA LLM
STAGE2_LLM_MLX = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 轻量中文 embedding (~95MB)

# Stage 2 v2（演示热加载 / 同任务不同模型对比；不需要微调）
STAGE2_LLM_MLX_V2 = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
STAGE2_LLM_HF_V2  = "Qwen/Qwen2.5-1.5B-Instruct"

# LoRA 产物路径
STAGE1_ADAPTER = MODELS_DIR / "stage1_adapter"
STAGE2_ADAPTER = MODELS_DIR / "stage2_adapter"

# RAG 参数
TOP_K = 5
MAX_CTX_TOKENS = 2048
