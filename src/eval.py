"""评估指标：TEDS / EM / F1 / ROUGE-L。"""
from __future__ import annotations
import re
from collections import Counter

import editdistance
from rouge_score import rouge_scorer

from .rag import parse_markdown_table


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p_toks = list(_normalize(pred))  # 字符级，适合中英混合
    g_toks = list(_normalize(gold))
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def rouge_l(pred: str, gold: str) -> float:
    return _rouge.score(gold or "", pred or "")["rougeL"].fmeasure


def teds(pred_md: str, gold_md: str) -> float:
    """简化 TEDS：序列化成 (row, col, value) 三元组后做编辑距离。"""
    def serialize(md: str) -> list[str]:
        df = parse_markdown_table(md)
        toks = []
        for ri, row in df.iterrows():
            for col in df.columns:
                toks.append(f"{ri}|{col}|{_normalize(str(row[col]))}")
        return toks

    p, g = serialize(pred_md), serialize(gold_md)
    if not g:
        return 0.0
    dist = editdistance.eval(p, g)
    denom = max(len(p), len(g))
    return 1 - dist / denom if denom else 0.0


def cell_f1(pred_md: str, gold_md: str) -> dict:
    """单元格级 P/R/F1。"""
    def cells(md: str) -> set[tuple]:
        df = parse_markdown_table(md)
        s = set()
        for ri, row in df.iterrows():
            for col in df.columns:
                s.add((ri, col, _normalize(str(row[col]))))
        return s

    p, g = cells(pred_md), cells(gold_md)
    if not p and not g:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not p or not g:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}
