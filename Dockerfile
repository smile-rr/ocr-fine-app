# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------
# OCR-Fine-App 推理服务
# 默认 CPU 镜像（可在 x86_64 / arm64 跑）。
# 若要 CUDA：base FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
# -----------------------------------------------------------

# -------- Stage 1: 装依赖 --------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（pdfplumber 需要 libraries）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        && rm -rf /var/lib/apt/lists/*

# uv 提供最快的依赖装载
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./

# 容器内不需要 mlx* / streamlit / jupyter，生成精简 requirements
RUN python -c "\
import tomllib, pathlib;\
data = tomllib.loads(pathlib.Path('pyproject.toml').read_text());\
deps = data['project']['dependencies'];\
excluded = ('mlx', 'streamlit', 'jupyter', 'ipywidgets', 'matplotlib');\
keep = [d for d in deps if not d.startswith(excluded)];\
keep += ['torch>=2.1.0'];\
pathlib.Path('req-docker.txt').write_text('\n'.join(keep))"

RUN uv pip install --system --no-cache-dir -r req-docker.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/ \
    torch

RUN uv pip install --system --no-cache-dir -r req-docker.txt

# -------- Stage 2: 运行时镜像 --------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    STAGE1_MODEL_PATH=/app/models/stage1_fused \
    STAGE2_MODEL_PATH=/app/models/stage2_fused \
    CHROMA_DIR=/app/chroma_db \
    DEVICE=cpu \
    ENABLE_STAGE1=1 \
    MAX_TOKENS=512

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app app

WORKDIR /app

# 从 builder 拷贝已装好的 site-packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 拷贝源码（模型通过 volume 挂载，不打进 image）
COPY src/ /app/src/
COPY pyproject.toml /app/

RUN mkdir -p /app/models /app/chroma_db /app/.cache \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)"

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
