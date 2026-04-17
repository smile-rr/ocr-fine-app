"""Stage 2 QA 训练集构建。

主来源：`data/raw/fintabnet_sample/`（ds4sd/FinTabNet_OTSL）——
    把 html → markdown，再用模板生成 QA。

可选来源：`data/raw/comtqa_sample/`（ByteDance/ComTQA）——
    ComTQA 只给 `image_name + question + answer`（表格本身不在 row 里），
    想用必须按 image_name 和 PubTabNet/FinTabNet 做 join。默认跳过，
    等你需要真实 QA 时再单独接。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import random
from datasets import load_from_disk
from src import config as C
from src.data import (
    build_stage2_samples, html_table_to_markdown, save_jsonl, split_train_val,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prepare_stage2")


HTML_FIELDS = ("html_table", "html", "html_restored")


def _row_to_md(row: dict) -> str:
    """从任意 row（PubTabNet/FinTabNet_OTSL）取 html 并转 markdown。"""
    for f in HTML_FIELDS:
        if row.get(f):
            return html_table_to_markdown(row[f])
    return row.get("markdown") or row.get("table") or ""


TEMPLATE_Q = [
    ("{col} 最大的是哪一行？",
     lambda df, col: df.loc[df[col].astype(str).map(_as_num).idxmax()]
                     if _numeric(df, col) else None),
    ("{col} 最小的是哪一行？",
     lambda df, col: df.loc[df[col].astype(str).map(_as_num).idxmin()]
                     if _numeric(df, col) else None),
]


def _as_num(s):
    try:
        return float(str(s).replace(",", "").replace("%", ""))
    except Exception:
        return float("nan")


def _numeric(df, col) -> bool:
    vals = df[col].astype(str).map(_as_num)
    return vals.notna().sum() >= max(2, len(df) // 2)


def template_qa_from_table(table_md: str, n: int = 3) -> list[dict]:
    from src.rag import parse_markdown_table
    df = parse_markdown_table(table_md)
    if df.empty or len(df) < 2:
        return []
    pairs = []
    rng = random.Random(42)
    for col in rng.sample(list(df.columns), min(len(df.columns), n)):
        for q_tpl, fn in TEMPLATE_Q:
            try:
                row = fn(df, col)
            except Exception:
                # 坏表、全 NaN、仍有 dup 列等：跳过这个 (col, template)
                continue
            if row is None:
                continue
            q = q_tpl.format(col=col)
            a = f"{col} 最值在第 {row.name + 1} 行，值为 {row[col]}。"
            pairs.append({"table_md": table_md, "question": q, "answer": a})
    return pairs


def from_paired_dataset(dirname: str) -> list[dict]:
    """对 PubTabNet / FinTabNet_OTSL 这类 image+html 数据集，
    取 html 转 markdown 后，跑模板 QA 生成训练对。"""
    src = C.RAW_DIR / dirname
    if not src.exists():
        return []
    ds = load_from_disk(str(src))
    log.info(f"loaded {dirname}: {len(ds)} rows")
    out = []
    for row in ds:
        md = _row_to_md(row)
        if md:
            out.extend(template_qa_from_table(md))
    return out


def from_comtqa() -> list[dict]:
    """ComTQA 只有 image_name + question + answer，没有表格内容本身，
    单独使用无法训练。需要按 image_name 和 PubTabNet / FinTabNet 做 join
    才能还原表格 markdown——留作 TODO，默认返回空。"""
    src = C.RAW_DIR / "comtqa_sample"
    if not src.exists():
        return []
    log.info("ComTQA 已下载，但需要 image_name join 才能用；本脚本默认跳过")
    return []


def main():
    qa: list[dict] = []

    # 主来源：FinTabNet_OTSL 的 html 模板生成
    qa.extend(from_paired_dataset("fintabnet_sample"))
    # 备用：PubTabNet 的 html 也可以（量更大，非金融域）
    if len(qa) < 100:
        qa.extend(from_paired_dataset("pubtabnet_sample"))
    # 可选：ComTQA（默认跳过）
    qa.extend(from_comtqa())

    log.info(f"total QA pairs: {len(qa)}")
    if not qa:
        log.error("没生成任何 QA。先跑 scripts/download_data.py 下 fintabnet/pubtabnet")
        return

    samples = build_stage2_samples(qa)
    train, val = split_train_val(samples, val_ratio=0.1)
    out = C.DATA_DIR / "stage2_train"
    save_jsonl(train, out / "train.jsonl")
    save_jsonl(val,   out / "val.jsonl")
    log.info(f"saved train={len(train)} val={len(val)} -> {out}")


if __name__ == "__main__":
    main()
