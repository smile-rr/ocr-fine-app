"""新一代 PDF 抽取 —— Docling (文本) + MinerU (扫描) 双引擎 + 跨页表格合并.

和老的 src/pdf_utils.py 并存，不动老代码。

核心流程:
    extract_document(pdf_path, engine="auto")
    ├── detect_pdf_type  → text / scanned
    ├── engine.extract   → list[DocElement] (统一 schema)
    └── merge_cross_page_tables

返回的 DocElement 流可以直接喂给 chunking.py。

装依赖:
    uv sync --extra extract

文档:
    docs/pdf-extraction-2026.md
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 统一 schema —— Docling / MinerU 结果都归一到这个
# ============================================================
class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    OTHER = "other"


@dataclass
class DocElement:
    """统一的文档元素。引擎无关。

    Attributes:
        type: 元素类型
        text: Markdown 文本（表格是 markdown 表格字符串）
        page: 页码 (1-based)
        bbox: (x0, y0, x1, y1) 页面内坐标, None 表示引擎未提供
        level: 仅 heading 用 (1-6)
        id: 元素唯一 ID (同一 doc 内稳定)
        title: 仅 table 用，表格标题/caption
        data: 仅 table 用，pandas.DataFrame
        cross_page: 是否经过跨页合并
        pages: 跨页合并时的原始页码列表
        metadata: 扩展字段（引擎特定的）
    """
    type: ElementType
    text: str
    page: int
    bbox: tuple[float, float, float, float] | None = None
    level: int | None = None
    id: str = ""
    title: str | None = None
    data: pd.DataFrame | None = None
    cross_page: bool = False
    pages: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 路由：探测文本层
# ============================================================
def detect_pdf_type(pdf_path: str | Path, sample_pages: int = 3,
                    min_text_bytes: int = 100) -> str:
    """返回 "text" 或 "scanned"。

    策略：取前 N 页的可提取文字字节总数，< 阈值视为扫描件。
    这是最便宜、最准的启发式；复杂情况（混合 PDF）走 Docling 即可自动 fallback OCR。
    """
    pdf_path = Path(pdf_path)
    try:
        doc = fitz.open(pdf_path)
        n = min(sample_pages, len(doc))
        total_text = sum(len(doc[i].get_text().strip()) for i in range(n))
        doc.close()
        return "text" if total_text >= min_text_bytes else "scanned"
    except Exception as e:
        logger.warning(f"detect_pdf_type failed for {pdf_path}: {e}; 默认走 docling")
        return "text"


# ============================================================
# Docling 引擎
# ============================================================
def extract_with_docling(pdf_path: str | Path) -> list[DocElement]:
    """用 Docling 抽取，返回 DocElement 流。

    Docling 输出 DoclingDocument，含 heading hierarchy、表格结构、bbox。
    这里把它归一到我们的 schema。
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as e:
        raise RuntimeError(
            "docling 未安装。运行: uv sync --extra extract"
        ) from e

    pdf_path = Path(pdf_path)

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,                     # 扫描页自动 OCR
        do_table_structure=True,         # 识别表格行列
    )
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    logger.info(f"docling extracting {pdf_path.name}")
    result = converter.convert(str(pdf_path))
    doc = result.document

    elements: list[DocElement] = []
    for idx, (item, _level) in enumerate(doc.iterate_items()):
        el = _docling_item_to_element(item, idx)
        if el is not None:
            elements.append(el)

    logger.info(f"docling -> {len(elements)} elements")
    return elements


def _docling_item_to_element(item: Any, idx: int) -> DocElement | None:
    """把 Docling 的一个 item 转成统一 DocElement。"""
    label = str(getattr(item, "label", "")).lower()
    text = (getattr(item, "text", "") or "").strip()

    # 页码和 bbox（Docling 把 provenance 放在 .prov 列表里）
    page, bbox = 1, None
    prov = getattr(item, "prov", None) or []
    if prov:
        page = int(getattr(prov[0], "page_no", 1))
        b = getattr(prov[0], "bbox", None)
        if b is not None:
            bbox = (
                float(getattr(b, "l", 0)), float(getattr(b, "t", 0)),
                float(getattr(b, "r", 0)), float(getattr(b, "b", 0)),
            )

    # 表格单独处理
    if "table" in label:
        try:
            df = item.export_to_dataframe()
        except Exception:
            df = None
        md = ""
        try:
            md = item.export_to_markdown()
        except Exception:
            pass
        return DocElement(
            type=ElementType.TABLE, text=md, page=page, bbox=bbox,
            id=f"table_{idx}", data=df,
        )

    # 标题
    if "header" in label or "title" in label or "section_header" in label:
        level = getattr(item, "level", None) or 2
        return DocElement(
            type=ElementType.HEADING, text=text, page=page, bbox=bbox,
            level=int(level), id=f"h_{idx}",
        )

    # 列表
    if "list" in label:
        return DocElement(
            type=ElementType.LIST, text=text, page=page, bbox=bbox, id=f"list_{idx}",
        )

    if "footnote" in label:
        return DocElement(
            type=ElementType.FOOTNOTE, text=text, page=page, bbox=bbox, id=f"fn_{idx}",
        )

    if "caption" in label:
        return DocElement(
            type=ElementType.CAPTION, text=text, page=page, bbox=bbox, id=f"cap_{idx}",
        )

    if "picture" in label or "figure" in label:
        return DocElement(
            type=ElementType.FIGURE, text=text, page=page, bbox=bbox, id=f"fig_{idx}",
        )

    if "formula" in label:
        return DocElement(
            type=ElementType.FORMULA, text=text, page=page, bbox=bbox, id=f"eq_{idx}",
        )

    # 过滤掉空白/页眉页脚（Docling 会打 page_header / page_footer 标签，先略过）
    if not text or "page_header" in label or "page_footer" in label:
        return None

    return DocElement(
        type=ElementType.PARAGRAPH, text=text, page=page, bbox=bbox, id=f"p_{idx}",
    )


# ============================================================
# MinerU 引擎（通过 CLI）
# ============================================================
def extract_with_mineru(pdf_path: str | Path,
                        work_dir: str | Path | None = None) -> list[DocElement]:
    """用 MinerU CLI 抽取，读它的 JSON 输出归一化。

    MinerU 2.x 稳定 CLI: `mineru -p <pdf> -o <out_dir> -m auto`
    输出: <out_dir>/<stem>/auto/<stem>_content_list.json
    """
    pdf_path = Path(pdf_path)

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mineru_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"mineru extracting {pdf_path.name} -> {work_dir}")
    try:
        subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(work_dir), "-m", "auto"],
            check=True, capture_output=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "mineru CLI 未找到。运行: uv sync --extra extract"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"mineru failed: {e.stderr.decode(errors='replace')[:500]}"
        ) from e

    # 读输出
    content_json = work_dir / pdf_path.stem / "auto" / f"{pdf_path.stem}_content_list.json"
    if not content_json.exists():
        # 不同版本路径略有差别，兜底搜
        candidates = list(work_dir.rglob("*_content_list.json"))
        if not candidates:
            raise RuntimeError(f"mineru 没生成 content_list.json in {work_dir}")
        content_json = candidates[0]

    raw = json.loads(content_json.read_text(encoding="utf-8"))

    elements: list[DocElement] = []
    for idx, item in enumerate(raw):
        el = _mineru_item_to_element(item, idx)
        if el is not None:
            elements.append(el)

    logger.info(f"mineru -> {len(elements)} elements")
    return elements


def _mineru_item_to_element(item: dict, idx: int) -> DocElement | None:
    """把 MinerU content_list.json 的一项转成 DocElement。

    MinerU JSON 字段约定（2024-2025）:
        type: "text" / "title" / "table" / "image" / "equation"
        text: 文本内容
        page_idx: 0-based
        bbox: [x0,y0,x1,y1]  (可能不存在)
        text_level: 对 title 是 1~6
    表格还会有 table_body / table_caption / html 等字段。
    """
    ty = item.get("type", "text")
    page = int(item.get("page_idx", 0)) + 1
    text = (item.get("text") or item.get("text_body") or "").strip()

    bbox = None
    raw_bbox = item.get("bbox")
    if raw_bbox and len(raw_bbox) == 4:
        bbox = tuple(float(x) for x in raw_bbox)

    if ty == "table":
        # 优先用 HTML → DataFrame；fallback markdown
        html = item.get("table_body") or item.get("html")
        md = item.get("table_body_md") or item.get("markdown", "")
        df = None
        if html:
            try:
                df = pd.read_html(html)[0]
                if md == "":
                    md = df.to_markdown(index=False)
            except Exception:
                df = None
        title = item.get("table_caption")
        return DocElement(
            type=ElementType.TABLE, text=md, page=page, bbox=bbox,
            id=f"table_{idx}", title=title, data=df,
        )

    if ty == "title":
        level = int(item.get("text_level", 2))
        return DocElement(
            type=ElementType.HEADING, text=text, page=page, bbox=bbox,
            level=level, id=f"h_{idx}",
        )

    if ty == "equation":
        return DocElement(
            type=ElementType.FORMULA, text=text, page=page, bbox=bbox, id=f"eq_{idx}",
        )

    if ty == "image":
        caption = item.get("image_caption", "")
        return DocElement(
            type=ElementType.FIGURE, text=caption, page=page, bbox=bbox,
            id=f"fig_{idx}",
        )

    if not text:
        return None

    return DocElement(
        type=ElementType.PARAGRAPH, text=text, page=page, bbox=bbox, id=f"p_{idx}",
    )


# ============================================================
# 跨页表格合并（Azure Document Intelligence 启发式）
# ============================================================
# A4 / Letter 默认页高（PDF 坐标系 72dpi 下）；需要精确的话从 fitz 拿
DEFAULT_PAGE_HEIGHT = 842.0


def merge_cross_page_tables(
    elements: list[DocElement],
    page_height: float = DEFAULT_PAGE_HEIGHT,
    edge_threshold: float = 80.0,
    llm_verify: callable | None = None,
) -> list[DocElement]:
    """合并跨页的续接表格。

    Args:
        elements: DocElement 流（按阅读顺序）
        page_height: PDF 页高（坐标系 pt）
        edge_threshold: prev 底部到页底 / curr 顶部到页顶 的最大距离 (pt)
        llm_verify: 可选，对"可能但不确定"的候选调一次 LLM
                    签名: (prev_md, curr_md, prev_page, curr_page) -> bool

    返回一个新列表，跨页的表格合并为一个 DocElement（cross_page=True）。
    """
    out: list[DocElement] = []
    i = 0
    while i < len(elements):
        el = elements[i]
        if el.type != ElementType.TABLE:
            out.append(el)
            i += 1
            continue

        # 向前贪心聚合相邻页的表格
        cluster = [el]
        j = i + 1
        while j < len(elements):
            nxt = _next_table(elements, j)
            if nxt is None:
                break
            nxt_el, nxt_idx = nxt
            if not _is_continuation(cluster[-1], nxt_el, page_height, edge_threshold):
                break
            # 确信级别：列数相同 + bbox 吻合 → 直接合
            # 不确信（列数相同但 bbox 不齐） → 走 LLM
            if llm_verify is not None and not _strong_match(cluster[-1], nxt_el, page_height):
                try:
                    if not llm_verify(
                        cluster[-1].text, nxt_el.text, cluster[-1].page, nxt_el.page
                    ):
                        break
                except Exception as e:
                    logger.warning(f"llm_verify failed: {e}; 默认不合并")
                    break

            # 中间若有非表格元素（罕见，如页眉），跳过
            # 真实情况下 page_header/footer 已被 docling 过滤；这里保底
            cluster.append(nxt_el)
            i_before = i
            i = nxt_idx + 1
            j = i

        if len(cluster) > 1:
            out.append(_combine_tables(cluster))
            continue

        out.append(el)
        i += 1

    return out


def _next_table(elements: list[DocElement], start: int) -> tuple[DocElement, int] | None:
    """从 start 开始找下一个 table 元素，允许跳过中间非表格元素。"""
    for k in range(start, len(elements)):
        if elements[k].type == ElementType.TABLE:
            return elements[k], k
        # 遇到标题/段落就停（说明进入新 section 了）
        if elements[k].type in (ElementType.HEADING, ElementType.PARAGRAPH):
            return None
    return None


def _is_continuation(prev: DocElement, curr: DocElement,
                     page_height: float, edge_threshold: float) -> bool:
    """两个表格是否构成跨页续接？"""
    if curr.page != prev.page + 1:
        return False
    if prev.data is None or curr.data is None:
        return False
    if prev.data.shape[1] != curr.data.shape[1]:
        return False
    if prev.bbox and curr.bbox:
        prev_bottom_gap = page_height - prev.bbox[3]
        curr_top_gap = curr.bbox[1]
        if prev_bottom_gap > edge_threshold or curr_top_gap > edge_threshold:
            return False
    return True


def _strong_match(prev: DocElement, curr: DocElement, page_height: float) -> bool:
    """确信级别：列数同 + bbox 齐 + header 模式一致."""
    if prev.data is None or curr.data is None:
        return False
    if prev.data.shape[1] != curr.data.shape[1]:
        return False
    if not (prev.bbox and curr.bbox):
        return False
    prev_bottom_gap = page_height - prev.bbox[3]
    curr_top_gap = curr.bbox[1]
    return prev_bottom_gap < 30 and curr_top_gap < 30


def _combine_tables(cluster: list[DocElement]) -> DocElement:
    """把 2+ 个表格合并成一个。

    - 数据层: pd.concat 行方向堆叠
    - 识别重复表头（后表第一行 == 前表 header），drop 掉
    - cross_page=True，pages 记录原始页码
    """
    base = cluster[0]
    merged_df = base.data.copy() if base.data is not None else pd.DataFrame()

    pages = [base.page]
    for t in cluster[1:]:
        pages.append(t.page)
        if t.data is None or t.data.empty:
            continue
        df = t.data.copy()
        # 去重复表头
        if df.shape[1] == merged_df.shape[1] and len(df) > 0:
            first_row = df.iloc[0].astype(str).str.strip().tolist()
            header = [str(c).strip() for c in merged_df.columns]
            if first_row == header:
                df = df.iloc[1:]
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    # 重新生成 markdown
    try:
        md = merged_df.to_markdown(index=False)
    except Exception:
        md = "\n\n".join(t.text for t in cluster)

    return DocElement(
        type=ElementType.TABLE,
        text=md,
        page=base.page,              # 以起始页为"主页码"
        bbox=base.bbox,
        id=f"{base.id}_merged",
        title=base.title,
        data=merged_df,
        cross_page=True,
        pages=pages,
        metadata={"n_fragments": len(cluster)},
    )


# ============================================================
# 主入口
# ============================================================
def extract_document(
    pdf_path: str | Path,
    engine: str = "auto",
    merge_tables: bool = True,
    llm_verify: callable | None = None,
) -> list[DocElement]:
    """统一入口：路由 → 抽取 → 跨页合并 → 返回 DocElement 流。

    Args:
        pdf_path: PDF 路径
        engine: "auto" | "docling" | "mineru"
        merge_tables: 是否做跨页表格合并
        llm_verify: 可选的 LLM 跨页验证函数

    Returns:
        list[DocElement]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    # 路由
    if engine == "auto":
        kind = detect_pdf_type(pdf_path)
        engine = "docling" if kind == "text" else "mineru"
        logger.info(f"auto-detected: {kind} -> using {engine}")

    # 抽取
    if engine == "docling":
        elements = extract_with_docling(pdf_path)
    elif engine == "mineru":
        elements = extract_with_mineru(pdf_path)
    else:
        raise ValueError(f"unknown engine: {engine}")

    # 跨页合并
    if merge_tables:
        before = sum(1 for e in elements if e.type == ElementType.TABLE)
        elements = merge_cross_page_tables(elements, llm_verify=llm_verify)
        after = sum(1 for e in elements if e.type == ElementType.TABLE)
        if before != after:
            logger.info(f"cross-page merge: {before} -> {after} tables")

    return elements


# ============================================================
# 调试: python -m src.extract path/to/file.pdf
# ============================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m src.extract <pdf>")
        sys.exit(1)

    elements = extract_document(sys.argv[1])
    print(f"\n=== {len(elements)} elements ===")
    counts: dict[str, int] = {}
    for el in elements:
        counts[el.type.value] = counts.get(el.type.value, 0) + 1
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("\n=== first 5 elements ===")
    for el in elements[:5]:
        preview = el.text[:120].replace("\n", " ")
        print(f"[p{el.page}] {el.type.value}: {preview}")
