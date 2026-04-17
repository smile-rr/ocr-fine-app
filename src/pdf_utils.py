"""PDF → 图片 + 表格 ground truth 抽取（pdfplumber）。"""
from __future__ import annotations
from pathlib import Path
import logging

import fitz  # pymupdf
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 150,
                  max_size: tuple[int, int] = (1344, 1344)) -> list[Path]:
    """每页渲染为 PNG。返回图片路径列表。"""
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    paths: list[Path] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img.thumbnail(max_size, Image.LANCZOS)
        p = out_dir / f"{pdf_path.stem}_p{i+1:03d}.png"
        img.save(p)
        paths.append(p)
    doc.close()
    return paths


def rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """简单 markdown 表格渲染。"""
    if not headers:
        return ""
    head = "| " + " | ".join(h or "" for h in headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join((c or "") for c in r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def extract_tables(pdf_path: str | Path) -> list[dict]:
    """用 pdfplumber 抽取所有表格。返回 [{page, index, headers, rows, markdown}]。"""
    pdf_path = Path(pdf_path)
    out: list[dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pi, page in enumerate(pdf.pages, 1):
                try:
                    tables = page.extract_tables() or []
                except Exception as e:
                    logger.warning(f"{pdf_path.name} page {pi} extract failed: {e}")
                    continue
                for ti, t in enumerate(tables):
                    if not t or len(t) < 2:
                        continue
                    headers, rows = t[0], t[1:]
                    headers = [(h or "").strip() for h in headers]
                    rows = [[(c or "").strip() for c in r] for r in rows]
                    out.append({
                        "page": pi,
                        "index": ti,
                        "headers": headers,
                        "rows": rows,
                        "markdown": rows_to_markdown(headers, rows),
                    })
    except Exception as e:
        logger.error(f"open {pdf_path} failed: {e}")
    return out
