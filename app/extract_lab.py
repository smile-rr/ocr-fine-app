"""Multi-engine PDF / image extraction lab.

Single-page Streamlit. Upload one document, pick which engines to run,
hit Run — each engine runs in its own background thread and its column
populates the moment that engine finishes. The agreement matrix at the
bottom appears once at least 2 engines have completed.

Launch:
    uv run streamlit run app/extract_lab.py
"""
from __future__ import annotations
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config as C
from src.extract import (
    ENGINES, ElementType, detect_pdf_type, extract_document, to_markdown,
)
from src.field_compare import agreement_matrix, style_agreement
from src.field_extractor import ExtractedDoc, extract_fields, get_llm

st.set_page_config(
    page_title="Extraction Lab",
    page_icon="🧪",
    layout="wide",
)

# Tighten the default Streamlit page chrome — strips the ~6rem of empty
# space at the top and shrinks block padding so the controls sit near the
# top of the viewport.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem !important; padding-bottom: 1rem !important;}
      header[data-testid="stHeader"] {height: 0; visibility: hidden;}
      div[data-testid="stToolbar"] {display: none;}
      /* Make file_uploader's drop zone a single tight row */
      [data-testid="stFileUploaderDropzone"] {min-height: 38px; padding: 4px 10px;}
      [data-testid="stFileUploaderDropzoneInstructions"] > div > span,
      [data-testid="stFileUploaderDropzoneInstructions"] > div > small {font-size: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

UPLOAD_DIR = C.DATA_DIR / "uploads" / "lab"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
TEXT_PDF_ENGINES = {"pdfplumber", "pymupdf4llm", "docling", "mineru"}
VISION_ENGINES = {"vision_llm:local", "vision_llm:openai"}


# ============================================================
# Session state
# ============================================================
def _init_state():
    ss = st.session_state
    ss.setdefault("uploaded_path", None)
    ss.setdefault("uploaded_kind", None)        # 'pdf-text' / 'pdf-scanned' / 'image'
    ss.setdefault("engines_in_run", [])
    ss.setdefault("run_id", 0)
    ss.setdefault("running", False)


_init_state()


# ============================================================
# Background worker
# ============================================================
# Module-level result store: {run_id: {engine: result_dict}}
# Background threads cannot write to st.session_state (no ScriptRunContext),
# so we keep results here and the fragments read from this dict.
_RESULTS_LOCK = Lock()
_RESULTS: dict[int, dict[str, dict]] = {}


def _set_status(run_id: int, engine: str, **patch):
    with _RESULTS_LOCK:
        bucket = _RESULTS.setdefault(run_id, {})
        cur = bucket.setdefault(engine, {"status": "queued", "engine": engine})
        cur.update(patch)


def _get_results(run_id: int) -> dict[str, dict]:
    with _RESULTS_LOCK:
        return dict(_RESULTS.get(run_id, {}))


@st.cache_resource
def _executor() -> ThreadPoolExecutor:
    # Engines load multi-GB models AND drive native subprocesses (mineru CLI,
    # tesseract, MLX). On a MacBook even 2 in parallel segfaulted (exit 139)
    # — native libs share state and step on each other. Sequential is the
    # safe default; bump LAB_MAX_WORKERS on a workstation if desired.
    n = max(1, int(os.environ.get("LAB_MAX_WORKERS", "1")))
    return ThreadPoolExecutor(max_workers=n, thread_name_prefix="extract")


@st.cache_resource(show_spinner=False)
def _prewarm_imports() -> dict:
    """Import every heavy library + load the local LLM ONCE in the main
    thread. Workers then hit fully-imported modules and a cached LLM.

    Without this, two threads doing first-time `import transformers` (one via
    docling, one via mlx_lm) collide and leave the module half-loaded —
    that's the source of both the `tqdm._lock` AttributeError and the
    `StoppingCriteria` ImportError reported on parallel runs.
    """
    out = {"errors": []}
    for name in ("transformers", "huggingface_hub", "tqdm", "pymupdf4llm"):
        try:
            __import__(name)
        except Exception as e:
            out["errors"].append(f"{name}: {e}")
    # docling / mineru imports are heavier and optional — best-effort
    try:
        import docling.document_converter  # noqa: F401
    except Exception as e:
        out["errors"].append(f"docling: {e}")
    try:
        import mineru  # noqa: F401
    except Exception as e:
        # mineru is CLI-driven; we can swallow if the python entry isn't there
        pass
    return out


def _prewarm_llm(provider: str):
    """Load the LLM (esp. local mlx_lm) in the main thread so the cached
    instance is what every worker sees."""
    try:
        llm = get_llm(provider)
        # Trigger a real load by asking for a 1-token completion if local
        if provider == "local":
            from src.infer import load_llm
            load_llm(adapter=None)
    except Exception as e:
        st.warning(f"LLM prewarm ({provider}) failed: {e}")


def _run_one(engine: str, pdf_path: str, llm_provider: str) -> dict:
    """Worker — run one engine end-to-end. Pure function (no Streamlit calls)."""
    out: dict = {
        "status": "running",
        "engine": engine,
        "extract_ms": None,
        "field_ms": None,
        "n_elements": None,
        "n_tables": None,
        "n_paragraphs": None,
        "markdown": "",
        "fields": None,           # ExtractedDoc
        "error": None,
    }
    try:
        t0 = time.time()
        elements = extract_document(pdf_path, engine=engine)
        md = to_markdown(elements)
        out["extract_ms"] = (time.time() - t0) * 1000
        out["n_elements"] = len(elements)
        out["n_tables"] = sum(1 for e in elements if e.type == ElementType.TABLE)
        out["n_paragraphs"] = sum(1 for e in elements if e.type == ElementType.PARAGRAPH)
        out["markdown"] = md

        if not md.strip():
            out["status"] = "empty"
            return out

        # Field extraction
        llm = get_llm(llm_provider)
        ed: ExtractedDoc = extract_fields(md, llm=llm, engine=engine)
        out["field_ms"] = ed.latency_ms
        out["fields"] = ed
        out["status"] = "done"
    except Exception as e:
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    return out


def _start_run(engines: list[str], pdf_path: Path, llm_provider: str):
    st.session_state.run_id += 1
    rid = st.session_state.run_id
    st.session_state.engines_in_run = list(engines)
    st.session_state.running = True

    # Pre-warm imports + LLM in the main thread to avoid concurrent first-time
    # import collisions in workers (transformers/tqdm/docling are not
    # thread-safe under simultaneous first import).
    with st.spinner("Loading models / libraries (one-time per session)…"):
        warm = _prewarm_imports()
        for err in warm.get("errors", []):
            st.warning(err)
        _prewarm_llm(llm_provider)

    # Seed module-level store BEFORE submitting any thread
    with _RESULTS_LOCK:
        _RESULTS[rid] = {eng: {"status": "queued", "engine": eng} for eng in engines}

    pool = _executor()

    def _wrap(eng: str, run_id: int = rid):
        _set_status(run_id, eng, status="running")
        result = _run_one(eng, str(pdf_path), llm_provider)
        with _RESULTS_LOCK:
            _RESULTS[run_id][eng] = result

    for eng in engines:
        pool.submit(_wrap, eng)


# ============================================================
# Top control bar — single row, advanced inside a popover (no extra row)
# ============================================================
# Detect kind FIRST (we need it for the engine multiselect default below).
cur_kind = st.session_state.uploaded_kind

c_up, c_eng, c_adv, c_run = st.columns([4, 5, 1, 1.5])

with c_up:
    up = st.file_uploader(
        "Upload PDF / image",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp"],
        label_visibility="collapsed",
        help="One file at a time. PDF (text or scanned) or image.",
    )
    if up is None:
        # User cleared the uploader (X) — wipe everything tied to that doc.
        if st.session_state.uploaded_path is not None:
            st.session_state.uploaded_path = None
            st.session_state.uploaded_kind = None
            st.session_state.engines_in_run = []
            st.session_state.running = False
            cur_kind = None
    else:
        tgt = UPLOAD_DIR / up.name
        tgt.write_bytes(up.read())
        if st.session_state.uploaded_path != str(tgt):
            # New file (different from current) — reset any stale run state
            st.session_state.uploaded_path = str(tgt)
            st.session_state.engines_in_run = []
            st.session_state.running = False
            if tgt.suffix.lower() in IMAGE_EXTS:
                st.session_state.uploaded_kind = "image"
            else:
                try:
                    kind = detect_pdf_type(tgt)
                except Exception:
                    kind = "text"
                st.session_state.uploaded_kind = (
                    "pdf-text" if kind == "text" else "pdf-scanned"
                )
            cur_kind = st.session_state.uploaded_kind

with c_eng:
    disabled_engines = TEXT_PDF_ENGINES if cur_kind == "image" else set()
    default_engines = (
        ["pdfplumber", "pymupdf4llm", "docling", "mineru"]
        if cur_kind in (None, "pdf-text", "pdf-scanned")
        else ["vision_llm:local"]
    )
    selectable = [e for e in ENGINES if e not in disabled_engines]
    selected_engines = st.multiselect(
        "Engines",
        options=selectable,
        default=[e for e in default_engines if e in selectable],
        label_visibility="collapsed",
        placeholder="Pick engines…",
    )

with c_adv:
    with st.popover("⚙️", use_container_width=True, help="Advanced settings"):
        llm_provider = st.selectbox(
            "Field-extraction LLM",
            options=["local", "ollama", "openai"],
            index=0,
            help=(
                "local = src.infer (Qwen2.5-0.5B MLX/HF — smoke only). "
                "ollama = local Ollama (qwen2.5:1.5b-instruct default). "
                "openai = OPENAI_API_KEY (Azure/DashScope via OPENAI_BASE_URL)."
            ),
        )
        if llm_provider == "local":
            st.caption(
                "ℹ️ Local 0.5B is for plumbing only — use **ollama** or "
                "**openai** for real extraction quality."
            )
        if "mineru" in selected_engines:
            st.divider()
            st.caption("**MinerU options** (defaults tuned for fast text PDFs)")
            mineru_method = st.selectbox(
                "method",
                options=["txt", "auto", "ocr"],
                index=0,
                help=(
                    "txt = no OCR (fastest, text PDFs only). "
                    "auto = per-page OCR detect. "
                    "ocr = force OCR (scans)."
                ),
            )
            mineru_backend = st.selectbox(
                "backend",
                options=["pipeline", "hybrid-auto-engine", "vlm-auto-engine"],
                index=0,
                help=(
                    "pipeline = fast (default). "
                    "hybrid = max accuracy (~5-10× slower). "
                    "vlm = pure VLM (slowest)."
                ),
            )
            mineru_lang = st.selectbox(
                "language",
                options=["en", "ch", "ch_lite", "japan", "korean"],
                index=0,
                help="OCR language hint (only used when method ≠ txt).",
            )
            os.environ["MINERU_METHOD"] = mineru_method
            os.environ["MINERU_BACKEND"] = mineru_backend
            os.environ["MINERU_LANG"] = mineru_lang

# Safety: if running flag is True but no engine is actually pending
# (worker may have segfaulted / been killed), clear it so Run re-enables.
if st.session_state.running:
    rid = st.session_state.run_id
    cur_for_check = _get_results(rid)
    pending = [
        e for e in st.session_state.engines_in_run
        if cur_for_check.get(e, {}).get("status") in ("queued", "running")
    ]
    if not pending:
        st.session_state.running = False

run_disabled = (
    st.session_state.uploaded_path is None or st.session_state.running
)
with c_run:
    run_clicked = st.button(
        "▶ Run",
        type="primary",
        use_container_width=True,
        disabled=run_disabled or not selected_engines,
    )

# Tight 1-line caption only when a doc is loaded (sits inline below the row)
if cur_kind:
    st.caption({
        "image": "🖼️ image input — only `vision_llm` engines apply",
        "pdf-text": "📄 text PDF detected — all engines applicable",
        "pdf-scanned":
            "🧾 scanned PDF — `pdfplumber` / `pymupdf4llm` will likely return "
            "empty; switch MinerU method to `auto`/`ocr` in ⚙️",
    }[cur_kind])

if run_clicked:
    _start_run(selected_engines, Path(st.session_state.uploaded_path), llm_provider)


# ============================================================
# Result columns — one fragment each, polls session_state
# ============================================================
def _badge(status: str) -> str:
    return {
        "queued":  "🟦 queued",
        "running": "⏳ running…",
        "done":    "✅ done",
        "empty":   "⚠️ empty",
        "error":   "❌ error",
    }.get(status, status)


def _render_engine_column(eng: str, run_id: int):
    """Tab body — full width, markdown + extracted fields side-by-side.
       Stats are in the always-visible perf table above; this view focuses
       on the actual content."""
    r = _get_results(run_id).get(eng, {"status": "queued"})
    status = r.get("status", "queued")

    if status in ("queued", "running"):
        st.info("⏳ " + ("Waiting in queue…" if status == "queued" else "Extracting…"))
        return

    if status == "error":
        st.error(r.get("error", "unknown error"))
        with st.expander("traceback"):
            st.code(r.get("traceback", ""), language="text")
        return

    md = r.get("markdown") or ""
    ed: ExtractedDoc | None = r.get("fields")

    # Layout: markdown is the main read (≈2/3), JSON is supporting (≈1/3)
    left, right = st.columns([2, 1])

    with left:
        head_l, head_r = st.columns([3, 2])
        head_l.markdown("**📄 Markdown output**")
        view = head_r.radio(
            "view",
            ["rendered", "raw"],
            horizontal=True,
            index=0,                # default: rendered (human-readable)
            key=f"mdview_{eng}_{run_id}",
            label_visibility="collapsed",
        )
        if not md:
            st.info("(empty markdown)")
        else:
            cap = md[:30_000]
            trunc = "" if len(md) <= 30_000 else "\n\n…[truncated to 30 000 chars]…"
            if view == "rendered":
                st.markdown(cap + trunc)
            else:
                st.code(cap + trunc, language="markdown")

    with right:
        st.markdown("**🧾 LLM JSON output**")
        if ed is not None:
            st.caption(f"LLM: `{ed.llm_model}`")
            st.code(ed.raw_response, language="json")
        else:
            st.info("(no JSON output)")


# Module-level fragments — Streamlit identifies fragments by qualified name,
# so defining them inside a loop collapses N fragments into one fragment ID
# and only one re-polls. Define them once at module scope and call with args.

@st.fragment(run_every=0.5)
def _frag_engine_cell(eng: str, run_id: int):
    _render_engine_column(eng, run_id)


@st.fragment(run_every=0.5)
def _frag_perf_table(engines: tuple, run_id: int):
    """Always-visible performance + key-fields comparison table.
       Polls _RESULTS so it updates live as engines finish."""
    rows = []
    cur = _get_results(run_id)
    for eng in engines:
        r = cur.get(eng, {"status": "queued"})
        status = r.get("status", "queued")
        em = r.get("extract_ms")
        fm = r.get("field_ms")
        ed: ExtractedDoc | None = r.get("fields")
        f = ed.fields if ed else None
        rows.append({
            "engine": eng,
            "status": _badge(status),
            "extract (s)": round(em / 1000, 2) if em is not None else None,
            "fields (s)": round(fm / 1000, 2) if fm is not None else None,
            "total (s)": round(((em or 0) + (fm or 0)) / 1000, 2) if em is not None else None,
            "elements": r.get("n_elements"),
            "tables": r.get("n_tables"),
            "paragraphs": r.get("n_paragraphs"),
            "chars": len(r.get("markdown") or "") if r.get("markdown") else None,
            "doc#": (f.document_number if f else None),
            "currency": (f.currency if f else None),
            "total": (str(f.total_amount) if f and f.total_amount is not None else None),
            "LLM": (ed.llm_model if ed else None),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_tabs(engines: list[str], run_id: int):
    """Full-width tabs, one per engine. Each tab body calls the module-level
    polling fragment so it keeps updating regardless of which tab is active."""
    tabs = st.tabs([f"  {eng}  " for eng in engines])
    for tab, eng in zip(tabs, engines):
        with tab:
            _frag_engine_cell(eng, run_id)


engines_in_run = st.session_state.engines_in_run
current_results = _get_results(st.session_state.run_id) if engines_in_run else {}

if engines_in_run:
    # Always-visible performance + key-fields comparison table
    st.markdown("##### 📊 Performance & key fields")
    _frag_perf_table(tuple(engines_in_run), st.session_state.run_id)

    # Full-width content per engine in tabs
    st.markdown("##### 📄 Per-engine output")
    _render_tabs(engines_in_run, st.session_state.run_id)

    # Stop the polling loop once everything has finished or errored
    statuses = [current_results.get(e, {}).get("status") for e in engines_in_run]
    if st.session_state.running and all(
        s in ("done", "empty", "error") for s in statuses
    ):
        st.session_state.running = False
else:
    st.info("Upload a document and click **Run** to start.")


# ============================================================
# Agreement matrix (appears once ≥ 2 engines have finished)
# ============================================================
done_results: dict[str, ExtractedDoc] = {
    eng: r["fields"]
    for eng, r in current_results.items()
    if r.get("status") == "done" and r.get("fields") is not None
}
if len(done_results) >= 2:
    st.markdown("---")
    st.markdown("### 📊 Agreement matrix")
    st.caption(
        "🟢 all engines agree · 🟡 some silent but answers given agree · "
        "🔴 disagreement · ⚪ all null"
    )
    df = agreement_matrix(done_results)
    st.dataframe(style_agreement(df), use_container_width=True, height=560)

    # Latency comparison
    st.markdown("### ⏱ Latency")
    rows = []
    for eng, r in current_results.items():
        if r.get("status") not in ("done", "empty"):
            continue
        rows.append({
            "engine": eng,
            "extract (s)": (r.get("extract_ms") or 0) / 1000,
            "fields (s)": (r.get("field_ms") or 0) / 1000,
        })
    if rows:
        ldf = pd.DataFrame(rows).set_index("engine")
        st.bar_chart(ldf, stack=True)
