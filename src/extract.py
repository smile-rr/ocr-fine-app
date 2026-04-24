"""PDF / image extraction — multiple peer engines, no auto routing.

Engines exposed (caller picks one explicitly):
    - pdfplumber       : text-PDF tables only (legacy baseline)
    - pymupdf4llm      : text-PDF -> LLM-grade Markdown (fastest full-doc)
    - docling          : layout-aware, auto-OCR fallback, hierarchy + bbox
    - mineru           : best on Chinese / scans / cross-page tables (2026)
    - vision_llm:local : local Qwen2-VL (MLX/HF) on rendered pages
    - vision_llm:openai: OpenAI gpt-4o-vision on rendered pages

Core entry:
    extract_document(pdf_path, engine=...)  -> list[DocElement]
    to_markdown(elements)                   -> str    (LLM-ready)

The DocElement stream still feeds chunking.py for the existing RAG path.
detect_pdf_type() is kept for UI hints only — it never auto-routes.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import pandas as pd

logger = logging.getLogger(__name__)


# Engine IDs exposed in UI / CLI
ENGINES = (
    "pdfplumber",
    "pymupdf4llm",
    "docling",
    "mineru",
    "vision_llm:local",
    "vision_llm:openai",
)


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
# pdfplumber 引擎 —— 文本 + 表格（全页内容）
# ============================================================
def extract_with_pdfplumber(pdf_path: str | Path) -> list[DocElement]:
    """Full pdfplumber extraction: per-page text PARAGRAPH + per-table TABLE.

    pdfplumber has BOTH `page.extract_text()` and `page.extract_tables()`;
    earlier versions of this engine only used the latter. We now emit:
      - one PARAGRAPH per page (full text, including paragraphs the other
        engines see), in reading order
      - one TABLE per detected table (after the page's paragraph)

    Both come from pdfplumber's text layer — no OCR. Returns [] only when
    the PDF has no text layer at all (scanned).
    """
    import pdfplumber
    from .pdf_utils import rows_to_markdown

    pdf_path = Path(pdf_path)
    elements: list[DocElement] = []
    table_idx = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pi, page in enumerate(pdf.pages, 1):
                # 1) Page text as PARAGRAPH (best-effort; pdfplumber returns ""
                #    on text-less pages)
                try:
                    text = (page.extract_text() or "").strip()
                except Exception as e:
                    logger.warning(f"pdfplumber page {pi} text failed: {e}")
                    text = ""
                if text:
                    elements.append(DocElement(
                        type=ElementType.PARAGRAPH,
                        text=text,
                        page=pi,
                        id=f"pp_p_{pi}",
                    ))

                # 2) Tables as TABLE elements
                try:
                    tables = page.extract_tables() or []
                except Exception as e:
                    logger.warning(f"pdfplumber page {pi} tables failed: {e}")
                    tables = []
                for t in tables:
                    if not t or len(t) < 2:
                        continue
                    headers = [(h or "").strip() for h in t[0]]
                    rows = [[(c or "").strip() for c in r] for r in t[1:]]
                    df = None
                    try:
                        if headers and rows:
                            df = pd.DataFrame(rows, columns=headers)
                    except Exception:
                        df = None
                    elements.append(DocElement(
                        type=ElementType.TABLE,
                        text=rows_to_markdown(headers, rows),
                        page=pi,
                        id=f"pp_table_{table_idx}",
                        data=df,
                    ))
                    table_idx += 1
    except Exception as e:
        logger.error(f"pdfplumber open {pdf_path} failed: {e}")

    n_tables = sum(1 for e in elements if e.type == ElementType.TABLE)
    n_paras = sum(1 for e in elements if e.type == ElementType.PARAGRAPH)
    logger.info(f"pdfplumber -> {n_paras} paragraphs + {n_tables} tables")
    return elements


# ============================================================
# pymupdf4llm 引擎 —— 整篇文档 -> markdown -> DocElement[]
# ============================================================
def extract_with_pymupdf4llm(pdf_path: str | Path) -> list[DocElement]:
    """Fast native-text PDF → Markdown. No OCR.

    pymupdf4llm.to_markdown returns one big markdown string covering all
    pages. We split on page-break markers (PyMuPDF inserts `-----` between
    pages by default with `page_chunks=True`) and emit per-page elements.
    Tables and headings are *kept inline* in the page paragraph rather than
    re-parsed — pymupdf4llm already puts markdown tables / `#` headings into
    the text stream, and downstream `to_markdown()` simply concatenates.
    """
    try:
        import pymupdf4llm
    except ImportError as e:
        raise RuntimeError(
            "pymupdf4llm 未安装。运行: uv sync --extra extract"
        ) from e

    pdf_path = Path(pdf_path)
    logger.info(f"pymupdf4llm extracting {pdf_path.name}")
    # page_chunks=True returns list[dict] with per-page text + metadata
    chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    elements: list[DocElement] = []
    for idx, ch in enumerate(chunks):
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        # pymupdf4llm pages are 0-based in metadata
        page = int(ch.get("metadata", {}).get("page", idx)) + 1
        elements.append(DocElement(
            type=ElementType.PARAGRAPH,
            text=text,
            page=page,
            id=f"p4l_page_{page}",
        ))
    logger.info(f"pymupdf4llm -> {len(elements)} page-chunks")
    return elements


# ============================================================
# vision_llm 引擎 —— 渲染每页 -> 多模态 LLM -> markdown
# ============================================================
_VISION_PROMPT = (
    "Extract ALL content from this page as well-structured Markdown. "
    "Preserve headings (# / ##), paragraphs, lists, and especially tables "
    "(use proper Markdown table syntax). Do not invent content; output "
    "only what is visible in the image. If the page is blank, output an "
    "empty string."
)


def _render_pdf_pages(pdf_path: Path, dpi: int = 200) -> list[tuple[int, "PIL.Image.Image"]]:
    """Render each PDF page to a PIL Image. Returns [(page_no_1based, img), ...]."""
    from PIL import Image
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    out: list[tuple[int, Any]] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        out.append((i + 1, img))
    doc.close()
    return out


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def extract_with_vision_llm(
    pdf_path: str | Path,
    provider: str = "local",
    dpi: int = 200,
    max_tokens: int = 2048,
) -> list[DocElement]:
    """Multimodal LLM as an extractor. Treats each page (or the input image)
    as one PARAGRAPH DocElement containing the LLM's markdown.

    provider:
      - "local"  : reuse src/infer.py:load_vlm() (MLX or HF Qwen2-VL)
      - "openai" : OpenAI gpt-4o-class vision via OPENAI_API_KEY
                   (override model with VISION_OPENAI_MODEL, default gpt-4o-mini)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    # Build (page_no, PIL.Image) list for both PDF and direct image input
    if _is_image(pdf_path):
        from PIL import Image
        pages = [(1, Image.open(pdf_path).convert("RGB"))]
    else:
        pages = _render_pdf_pages(pdf_path, dpi=dpi)

    if provider == "local":
        markdowns = _vision_local(pages, max_tokens=max_tokens)
    elif provider == "openai":
        markdowns = _vision_openai(pages, max_tokens=max_tokens)
    else:
        raise ValueError(f"unknown vision_llm provider: {provider}")

    elements: list[DocElement] = []
    for page_no, md in markdowns:
        if not md.strip():
            continue
        elements.append(DocElement(
            type=ElementType.PARAGRAPH,
            text=md.strip(),
            page=page_no,
            id=f"vlm_page_{page_no}",
            metadata={"vision_provider": provider},
        ))
    logger.info(f"vision_llm[{provider}] -> {len(elements)} pages")
    return elements


def _vision_local(pages, max_tokens: int) -> list[tuple[int, str]]:
    """Run local Qwen2-VL on each page via mlx_vlm or transformers."""
    from .infer import load_vlm, _mlx_vlm_config, _unwrap_mlx
    from . import config as C
    backend, model, processor = load_vlm(adapter=None)
    out: list[tuple[int, str]] = []
    if backend == "mlx":
        from mlx_vlm import generate as mlx_gen
        from mlx_vlm.prompt_utils import apply_chat_template
        cfg = _mlx_vlm_config(model, C.STAGE1_VLM_MLX)
        with tempfile.TemporaryDirectory() as td:
            for page_no, img in pages:
                p = Path(td) / f"page_{page_no}.png"
                img.save(p)
                formatted = apply_chat_template(
                    processor, config=cfg, prompt=_VISION_PROMPT, num_images=1,
                )
                result = mlx_gen(model, processor, formatted, image=[str(p)],
                                 max_tokens=max_tokens, verbose=False)
                out.append((page_no, _unwrap_mlx(result)))
    else:
        for page_no, img in pages:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": _VISION_PROMPT},
            ]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img],
                               return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            md = processor.batch_decode(
                gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )[0]
            out.append((page_no, md))
    return out


def _vision_openai(pages, max_tokens: int) -> list[tuple[int, str]]:
    """Send each page image to OpenAI vision (or any OpenAI-compatible
    endpoint that supports image_url with data: URIs)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai 未安装。运行: uv sync --extra extract") from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY 未设置。设置后再用 vision_llm:openai。"
        )
    base_url = os.environ.get("OPENAI_BASE_URL")  # optional, e.g. Azure / DashScope
    model_name = os.environ.get("VISION_OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    out: list[tuple[int, str]] = []
    for page_no, img in pages:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _VISION_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        out.append((page_no, resp.choices[0].message.content or ""))
    return out


# ============================================================
# MinerU 引擎（通过 CLI）
# ============================================================
def extract_with_mineru(pdf_path: str | Path,
                        work_dir: str | Path | None = None) -> list[DocElement]:
    """用 MinerU CLI 抽取，读它的 JSON 输出归一化。

    MinerU 2.x CLI flags we care about (run `mineru --help`):
      -b BACKEND   pipeline | hybrid-auto-engine | vlm-auto-engine | ...
                   `pipeline` is 5-10x faster than the default
                   `hybrid-auto-engine`; default here = pipeline so
                   MacBook users don't wait 70s/page.
      -m METHOD    auto | txt | ocr   (only valid for `pipeline` / `hybrid-*`)
                   `txt` skips OCR entirely — set MINERU_METHOD=txt for
                   pure native-text PDFs.
      -l LANG      ch | en | ...  (improves OCR accuracy)

    Override via env vars:
      MINERU_BACKEND  (default: pipeline)
      MINERU_METHOD   (default: auto)
      MINERU_LANG     (default: ch)
    """
    pdf_path = Path(pdf_path)

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mineru_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    # Defaults tuned for fast native-text PDFs on Apple Silicon. Override
    # for scanned docs by setting MINERU_METHOD=ocr (or auto) and
    # MINERU_LANG=ch when the input is Chinese.
    backend = os.environ.get("MINERU_BACKEND", "pipeline")
    method = os.environ.get("MINERU_METHOD", "txt")
    lang = os.environ.get("MINERU_LANG", "en")

    cmd = [
        "mineru", "-p", str(pdf_path), "-o", str(work_dir),
        "-b", backend, "-m", method, "-l", lang,
    ]
    logger.info(f"mineru extracting {pdf_path.name} -> {work_dir} ({' '.join(cmd[1:])})")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
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
    """两个表格是否构成跨页续接？需要 bbox 才能判断；缺 bbox 时一律不合并。"""
    if curr.page != prev.page + 1:
        return False
    if prev.data is None or curr.data is None:
        return False
    if prev.data.shape[1] != curr.data.shape[1]:
        return False
    if not (prev.bbox and curr.bbox):
        return False
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
# 主入口 —— 显式 engine（不再有 auto）
# ============================================================
def extract_document(
    pdf_path: str | Path,
    engine: str,
    merge_tables: bool = True,
    llm_verify: callable | None = None,
) -> list[DocElement]:
    """Dispatch to the named engine and return the DocElement stream.

    Args:
        pdf_path: PDF or image path.
        engine:   one of ENGINES — must be passed explicitly. No auto-routing.
        merge_tables: only meaningful for engines that produce TABLE elements
                      with bbox/data (docling, mineru). No-op for the others.
        llm_verify:   optional LLM verifier for cross-page table merging.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if engine == "pdfplumber":
        elements = extract_with_pdfplumber(pdf_path)
    elif engine == "pymupdf4llm":
        elements = extract_with_pymupdf4llm(pdf_path)
    elif engine == "docling":
        elements = extract_with_docling(pdf_path)
    elif engine == "mineru":
        elements = extract_with_mineru(pdf_path)
    elif engine.startswith("vision_llm:"):
        provider = engine.split(":", 1)[1]
        elements = extract_with_vision_llm(pdf_path, provider=provider)
    else:
        raise ValueError(
            f"unknown engine: {engine!r}; choose one of {ENGINES}"
        )

    if merge_tables:
        before = sum(1 for e in elements if e.type == ElementType.TABLE)
        elements = merge_cross_page_tables(elements, llm_verify=llm_verify)
        after = sum(1 for e in elements if e.type == ElementType.TABLE)
        if before != after:
            logger.info(f"cross-page merge: {before} -> {after} tables")

    return elements


# ============================================================
# to_markdown —— DocElement[] → 单一 markdown 文档（喂给字段抽取 LLM）
# ============================================================
def to_markdown(elements: list[DocElement]) -> str:
    """Serialize a DocElement stream into one cohesive Markdown document.

    Layout:
      HEADING(level=L)  -> '\\n' + '#'*L + ' ' + text + '\\n'
      PARAGRAPH         -> text + '\\n'
      LIST              -> text + '\\n'
      TABLE             -> '\\n<!-- table id=... page=... [cross_page] -->\\n'
                           + markdown + '\\n'
      CAPTION/FOOTNOTE  -> '> ' + text + '\\n'
      FIGURE            -> '![caption](page-N)' + '\\n'
      FORMULA           -> '$$ ... $$' + '\\n'

    Designed for downstream LLM field extraction — preserves structure cheaply
    and keeps a per-table breadcrumb the LLM can cite.
    """
    parts: list[str] = []
    for el in elements:
        if not el.text and el.type != ElementType.FIGURE:
            continue
        if el.type == ElementType.HEADING:
            level = max(1, min(6, el.level or 2))
            parts.append("\n" + "#" * level + " " + el.text + "\n")
        elif el.type == ElementType.TABLE:
            cp = " cross_page" if el.cross_page else ""
            parts.append(
                f"\n<!-- table id={el.id} page={el.page}{cp} -->\n{el.text}\n"
            )
        elif el.type == ElementType.LIST:
            parts.append(el.text + "\n")
        elif el.type in (ElementType.CAPTION, ElementType.FOOTNOTE):
            parts.append("> " + el.text.replace("\n", "\n> ") + "\n")
        elif el.type == ElementType.FIGURE:
            cap = el.text or "figure"
            parts.append(f"\n![{cap}](page-{el.page})\n")
        elif el.type == ElementType.FORMULA:
            parts.append(f"\n$$\n{el.text}\n$$\n")
        else:  # PARAGRAPH / OTHER
            parts.append(el.text + "\n")
    return "".join(parts).strip() + "\n"


# ============================================================
# 调试: python -m src.extract <pdf> <engine>
# ============================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if len(sys.argv) < 3:
        print(f"usage: python -m src.extract <pdf> <engine>\n"
              f"  engines: {' | '.join(ENGINES)}")
        sys.exit(1)

    elements = extract_document(sys.argv[1], engine=sys.argv[2])
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

    if len(sys.argv) > 3 and sys.argv[3] == "--md":
        print("\n=== to_markdown() ===")
        print(to_markdown(elements)[:2000])
