"""Stage 1 训练集构建：image + HTML → image + Markdown (sharegpt 格式)。

数据来源（已配对好的 parquet 数据集）：
- `data/raw/pubtabnet_sample/`  —— apoidea/pubtabnet-html     (学术论文表)
- `data/raw/fintabnet_sample/`  —— ds4sd/FinTabNet_OTSL       (金融表)

两个都有 `image` + `html`，HTML 统一转成 Markdown 当 assistant 答案。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from datasets import load_from_disk
from src import config as C
from src.data import (
    build_stage1_samples, html_table_to_markdown, save_jsonl, split_train_val,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prepare_stage1")


# 候选字段名（两个数据集的 html 字段名略有差异；ds4sd 还提供 html_restored）
HTML_FIELDS = ("html_table", "html", "html_restored")


def _extract_image_and_md(row: dict) -> tuple[object | None, str]:
    img = row.get("image") or row.get("png")
    html = None
    for f in HTML_FIELDS:
        if row.get(f):
            html = row[f]
            break
    md = html_table_to_markdown(html) if html else ""
    return img, md


def _iter_source(name: str, dirname: str):
    src = C.RAW_DIR / dirname
    if not src.exists():
        log.info(f"skip {name}: {src} 不存在")
        return
    ds = load_from_disk(str(src))
    log.info(f"loaded {name}: {len(ds)} rows · features={list(ds.features)}")
    yield from ((name, i, row) for i, row in enumerate(ds))


def main():
    out_img_dir = C.DATA_DIR / "stage1_images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    image_paths, markdowns = [], []
    total_seen = 0

    for src_name, dirname in [("pubtabnet", "pubtabnet_sample"),
                              ("fintabnet", "fintabnet_sample")]:
        for _, idx, row in _iter_source(src_name, dirname):
            total_seen += 1
            img, md = _extract_image_and_md(row)
            if img is None or not md:
                continue
            img_path = out_img_dir / f"{src_name}_{idx:05d}.png"
            try:
                img.save(img_path)
            except Exception as e:
                log.warning(f"  skip {src_name}#{idx}: image save failed: {e}")
                continue
            image_paths.append(str(img_path.relative_to(C.ROOT)))
            markdowns.append(md)

    log.info(f"built {len(image_paths)} / {total_seen} samples (kept / seen)")
    if not image_paths:
        log.error("没拿到任何有效样本。先跑 scripts/download_data.py")
        return

    samples = build_stage1_samples(image_paths, markdowns)
    train, val = split_train_val(samples, val_ratio=0.1)
    out_dir = C.DATA_DIR / "stage1_train"
    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val,   out_dir / "val.jsonl")
    log.info(f"saved train={len(train)} val={len(val)} -> {out_dir}")


if __name__ == "__main__":
    main()
