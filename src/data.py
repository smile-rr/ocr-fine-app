"""HuggingFace 数据集采样 + Stage1/Stage2 训练集构建。"""
from __future__ import annotations
from pathlib import Path
import json
import logging

from . import config as C

logger = logging.getLogger(__name__)


def download_hf_dataset(repo_id: str, split: str = "train",
                        n: int | None = 500, save_to: str | Path | None = None):
    """下载 HF 数据集子集，保存到 disk。"""
    from datasets import load_dataset
    slice_spec = f"{split}[:{n}]" if n else split
    ds = load_dataset(repo_id, split=slice_spec, cache_dir=str(C.HF_CACHE))
    if save_to:
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_to))
        logger.info(f"saved {len(ds)} rows to {save_to}")
    return ds


def html_table_to_markdown(html) -> str:
    """HTML 表格 → Markdown（无 lxml 依赖）。

    支持两种输入形式：
    - str：完整 HTML，如 `apoidea/pubtabnet-html` 的 `html` 字段
    - list[str]：HTML token 列表，如 `ds4sd/FinTabNet_OTSL` 的 `html` 字段
    """
    import re
    from html import unescape

    if isinstance(html, list):
        html = "".join(html)
    if not isinstance(html, str) or not html.strip():
        return ""
    html = unescape(html)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)
    md_rows = []
    for r in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, flags=re.S | re.I)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if cells:
            md_rows.append(cells)
    if len(md_rows) < 2:
        return ""
    header = md_rows[0]
    body = md_rows[1:]
    width = len(header)
    line1 = "| " + " | ".join(header) + " |"
    sep = "|" + "|".join(["---"] * width) + "|"
    body_lines = [
        "| " + " | ".join((r[:width] + [""] * (width - len(r)))) + " |"
        for r in body
    ]
    return "\n".join([line1, sep] + body_lines)


STAGE1_INSTRUCTION = "请提取图中所有表格，以标准 Markdown 格式输出。如果没有表格，输出 '无表格'。"


def build_stage1_samples(image_paths: list[str], markdowns: list[str]) -> list[dict]:
    """构建 Stage1 (VLM) 训练样本：sharegpt 风格。"""
    assert len(image_paths) == len(markdowns)
    out = []
    for img, md in zip(image_paths, markdowns):
        out.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": STAGE1_INSTRUCTION},
                ]},
                {"role": "assistant", "content": md},
            ]
        })
    return out


def build_stage2_samples(qa_pairs: list[dict]) -> list[dict]:
    """构建 Stage2 (LLM) 训练样本：alpaca 风格。

    输入每条：{"table_md": str, "question": str, "answer": str}
    """
    out = []
    for row in qa_pairs:
        out.append({
            "instruction": "基于以下表格数据回答问题，引用具体数值与来源行。",
            "input": f"表格：\n{row['table_md']}\n\n问题：{row['question']}",
            "output": row["answer"],
        })
    return out


def save_jsonl(rows: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def split_train_val(rows: list[dict], val_ratio: float = 0.1, seed: int = 42):
    import random
    rng = random.Random(seed)
    rows = rows.copy()
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]
