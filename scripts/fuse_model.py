"""合并 LoRA adapter 到 base，导出 HuggingFace 格式供 Docker/transformers 使用。

用法：
    # 合并 Stage 2 (LLM)
    uv run python scripts/fuse_model.py --stage 2

    # 合并 Stage 1 (VLM)
    uv run python scripts/fuse_model.py --stage 1

    # 全部合并
    uv run python scripts/fuse_model.py --stage all

输出：
    models/stage1_fused/   ← HF 格式，可被 transformers.from_pretrained 加载
    models/stage2_fused/
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C


def fuse_stage2(save_hf: bool = True):
    """用 mlx_lm.fuse 合并 Stage2 LoRA -> HF 格式。"""
    out = C.MODELS_DIR / "stage2_fused"
    cmd = [
        "python", "-m", "mlx_lm.fuse",
        "--model", C.STAGE2_LLM_MLX,
        "--adapter-path", str(C.STAGE2_ADAPTER),
        "--save-path", str(out),
    ]
    if save_hf:
        # 导出 HF 格式（mlx_lm.fuse 最新版默认就是 HF 格式；加 --export-gguf 可出 gguf）
        cmd += ["--de-quantize"]  # dequantize 才能被 transformers 直接读
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"✅ stage2 fused -> {out}")


def fuse_stage1(save_hf: bool = True):
    """用 mlx_vlm.fuse 合并 Stage1 LoRA。"""
    out = C.MODELS_DIR / "stage1_fused"
    cmd = [
        "python", "-m", "mlx_vlm.fuse",
        "--model", C.STAGE1_VLM_MLX,
        "--adapter-path", str(C.STAGE1_ADAPTER),
        "--save-path", str(out),
    ]
    if save_hf:
        cmd += ["--de-quantize"]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"✅ stage1 fused -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    args = p.parse_args()

    if args.stage in ("1", "all"):
        if not C.STAGE1_ADAPTER.exists():
            print(f"⚠️  {C.STAGE1_ADAPTER} 不存在，跳过 Stage 1")
        else:
            fuse_stage1()
    if args.stage in ("2", "all"):
        if not C.STAGE2_ADAPTER.exists():
            print(f"⚠️  {C.STAGE2_ADAPTER} 不存在，跳过 Stage 2")
        else:
            fuse_stage2()


if __name__ == "__main__":
    main()
