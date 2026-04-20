"""Streamlit UI：文档智能 + RAG 问答。

启动：
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src import config as C
from src.pdf_utils import pdf_to_images, extract_tables
from src.rag import TableVectorStore, parse_markdown_table, build_rag_prompt

st.set_page_config(page_title="Doc Intelligence + RAG", layout="wide")
st.title("📄 文档智能 + 表格 RAG 问答")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ 模型配置")
    use_ft_stage1 = st.checkbox("Stage1 使用微调 LoRA", value=C.STAGE1_ADAPTER.exists())
    use_ft_stage2 = st.checkbox("Stage2 使用微调 LoRA", value=C.STAGE2_ADAPTER.exists())
    st.divider()
    st.subheader("🔬 抽取引擎")
    use_pdfplumber = st.checkbox("✅ pdfplumber (文本层, 老版本)", value=True,
                                  help="只抽表格；仅对文本层 PDF 有效。")
    use_vlm = st.checkbox("✅ VLM (Qwen2-VL, 老版本)", value=False,
                          help="扫描件可用但不准；慢。")
    use_docling = st.checkbox("⭐ V2: Docling / MinerU", value=True,
                               help="2026 推荐。文本+扫描自动路由，跨页合并，多粒度 chunking。")
    v2_engine = st.radio(
        "V2 引擎",
        ["auto", "docling", "mineru"],
        horizontal=True, disabled=not use_docling,
        help="auto=探测文本层; docling=文字 PDF; mineru=扫描件/中文复杂版式",
    )
    v2_contextual = st.checkbox(
        "⭐ Contextual Retrieval (Anthropic)", value=False, disabled=not use_docling,
        help="为每个 chunk 生成 50 字上下文再 embed, 召回 +35%。需要 Ollama 本地服务。",
    )
    v2_ollama_model = st.text_input(
        "Ollama 模型 (context 生成)",
        value="qwen2.5:0.5b-instruct-q4_K_M",
        disabled=not (use_docling and v2_contextual),
    )
    st.divider()
    st.caption(f"Stage1 adapter: {'✅' if C.STAGE1_ADAPTER.exists() else '❌'}")
    st.caption(f"Stage2 adapter: {'✅' if C.STAGE2_ADAPTER.exists() else '❌'}")

s1_adapter = str(C.STAGE1_ADAPTER) if use_ft_stage1 and C.STAGE1_ADAPTER.exists() else None
s2_adapter = str(C.STAGE2_ADAPTER) if use_ft_stage2 and C.STAGE2_ADAPTER.exists() else None


# ---------------- Session state ----------------
if "store" not in st.session_state:
    st.session_state.store = None
if "tables" not in st.session_state:
    st.session_state.tables = []
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
# V2 新增
if "v2_pipe" not in st.session_state:
    st.session_state.v2_pipe = None
if "v2_report" not in st.session_state:
    st.session_state.v2_report = None
if "v2_elements" not in st.session_state:
    st.session_state.v2_elements = []
if "engine_timings" not in st.session_state:
    st.session_state.engine_timings = {}


@st.cache_resource
def get_store() -> TableVectorStore:
    return TableVectorStore()


def _build_v2_pipeline():
    """按当前 sidebar 配置构建 V2 Pipeline (缓存到 session_state)."""
    from src.rag_v2 import Pipeline
    from src.chunking import OllamaContextGen

    ctx_gen = None
    if v2_contextual:
        try:
            ctx_gen = OllamaContextGen(model=v2_ollama_model)
        except Exception as e:
            st.warning(f"构建 Ollama context 生成器失败: {e}; 关闭 contextual")
    return Pipeline(
        engine=v2_engine,
        contextual=v2_contextual,
        context_gen=ctx_gen,
    )


# ---------------- Tabs ----------------
tab_upload, tab_tables, tab_compare_engines, tab_qa, tab_compare = st.tabs(
    ["📤 上传与抽取", "📊 表格查看", "🔬 抽取引擎对比", "💬 RAG 问答", "🆚 微调前后对比"]
)


# ========== Tab 1: Upload ==========
with tab_upload:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        up = st.file_uploader("上传 PDF（或选择内置样例）", type=["pdf"])
    with col_r:
        samples = sorted(C.SAMPLES_DIR.glob("*.pdf"))
        sample_pick = st.selectbox("内置样例",
                                   ["(无)"] + [p.name for p in samples])

    pdf_path: Path | None = None
    if up is not None:
        pdf_path = C.DATA_DIR / "uploads" / up.name
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(up.read())
    elif sample_pick != "(无)":
        pdf_path = C.SAMPLES_DIR / sample_pick

    # 图片 PDF 的提示
    if pdf_path:
        from src.extract import detect_pdf_type
        try:
            pdf_kind = detect_pdf_type(pdf_path)
        except Exception:
            pdf_kind = "unknown"
        if pdf_kind == "scanned":
            st.warning(
                "⚠️ 检测到这是**扫描件 / 图片型 PDF**。\n\n"
                "- pdfplumber: ❌ 无法处理（没有文本层）\n"
                "- VLM: ⚠️ 能处理但易错\n"
                "- **Docling / MinerU (V2)**: ✅ 推荐（MinerU 对扫描件/中文最强）"
            )
        elif pdf_kind == "text":
            st.info(f"📄 检测到**文本型 PDF**；三种引擎都可用。")

    if pdf_path and st.button("🚀 开始抽取", type="primary"):
        with st.status("处理中...", expanded=True) as status:
            doc_id = pdf_path.stem
            st.write(f"📄 {pdf_path.name}")

            timings: dict[str, float] = {}
            all_tables: list[dict] = []

            # ===== pdfplumber =====
            if use_pdfplumber:
                st.write("⏳ pdfplumber 抽取...")
                t0 = time.time()
                try:
                    pp = extract_tables(pdf_path)
                    dt = time.time() - t0
                    timings["pdfplumber"] = dt
                    st.write(f"  ✓ pdfplumber: {len(pp)} 张表 ({dt:.1f}s)")
                    all_tables += [{**t, "source": "pdfplumber"} for t in pp]
                except Exception as e:
                    st.error(f"pdfplumber 失败: {e}")
                    timings["pdfplumber"] = -1

            # ===== VLM =====
            if use_vlm:
                st.write("⏳ VLM 抽取（加载模型需要 10–30s）...")
                t0 = time.time()
                try:
                    from src.infer import extract_table_from_image
                    imgs = pdf_to_images(pdf_path, C.DATA_DIR / "preview" / doc_id)
                    vlm_tables = []
                    for i, img in enumerate(imgs, 1):
                        md = extract_table_from_image(img, adapter=s1_adapter, max_tokens=512)
                        if "无表格" not in md and "|" in md:
                            vlm_tables.append({"page": i, "markdown": md, "source": "vlm"})
                        st.write(f"  ✓ VLM page {i}")
                    dt = time.time() - t0
                    timings["vlm"] = dt
                    st.write(f"  ✓ VLM: {len(vlm_tables)} 张表 ({dt:.1f}s)")
                    all_tables += vlm_tables
                except Exception as e:
                    st.error(f"VLM 失败: {e}")
                    timings["vlm"] = -1

            # ===== Docling / MinerU (V2) =====
            if use_docling:
                st.write(f"⏳ V2 抽取（engine={v2_engine}）...")
                t0 = time.time()
                try:
                    pipe = _build_v2_pipeline()
                    report = pipe.ingest_pdf(pdf_path, doc_id=doc_id)
                    dt = time.time() - t0
                    timings[f"v2-{report.engine_used}"] = dt
                    st.session_state.v2_pipe = pipe
                    st.session_state.v2_report = report

                    # 同时把 V2 抽出的表格加到表格查看
                    from src.extract import extract_document, ElementType
                    elements = extract_document(pdf_path, engine=v2_engine)
                    st.session_state.v2_elements = elements
                    for el in elements:
                        if el.type == ElementType.TABLE and el.data is not None:
                            tag = f"v2-{report.engine_used}"
                            if el.cross_page:
                                tag += " (跨页合并)"
                            all_tables.append({
                                "page": el.page, "markdown": el.text,
                                "source": tag,
                            })

                    st.write(f"  ✓ V2 ({report.engine_used}): "
                             f"{report.n_elements} elements, "
                             f"{report.n_tables} tables "
                             f"({report.n_merged_cross_page} 跨页合并), "
                             f"{report.n_chunks} chunks "
                             f"({dt:.1f}s)")
                except Exception as e:
                    st.error(f"V2 抽取失败: {e}")
                    timings[f"v2-{v2_engine}"] = -1

            # 总表
            st.session_state.tables = all_tables
            st.session_state.current_doc = doc_id
            st.session_state.engine_timings = timings

            # V1 向量库入库（保留给 Tab 4 的 RAG 对比用）
            st.write("⏳ V1 入 ChromaDB (pdfplumber/VLM 的表格)...")
            store = get_store()
            v1_chunks = 0
            for t in all_tables:
                if t["source"].startswith("v2-"):
                    continue
                df = parse_markdown_table(t["markdown"])
                if not df.empty:
                    v1_chunks += store.add(df, doc_id=doc_id, page=t["page"])
            st.session_state.store = store
            st.write(f"  ✓ V1: {v1_chunks} chunks")

            status.update(
                label=f"完成 · {len(all_tables)} 张表 · V1 {v1_chunks} chunks "
                      f"· V2 {(st.session_state.v2_report.n_chunks if st.session_state.v2_report else 0)} chunks",
                state="complete",
            )


# ========== Tab 2: Table viewer ==========
with tab_tables:
    if not st.session_state.tables:
        st.info("先在「上传与抽取」tab 处理一个 PDF。")
    else:
        # 按 source 分组展示，便于对比三个引擎的差异
        sources_seen: dict[str, list[dict]] = {}
        for t in st.session_state.tables:
            sources_seen.setdefault(t.get("source", "?"), []).append(t)

        for src, tables in sources_seen.items():
            st.subheader(f"{src} — {len(tables)} 张表")
            for i, t in enumerate(tables):
                with st.expander(f"Page {t['page']} · table #{i}", expanded=i == 0):
                    st.markdown(t["markdown"])
                    df = parse_markdown_table(t["markdown"])
                    if not df.empty:
                        st.dataframe(df, width="stretch")
            st.divider()


# ========== Tab 3: 抽取引擎对比 ==========
with tab_compare_engines:
    st.write("## 🔬 抽取引擎并排对比")
    st.caption(
        "每个启用的引擎都跑了一遍同一份 PDF。下面按维度对比它们的输出。"
    )

    if not st.session_state.engine_timings:
        st.info("先在「上传与抽取」tab 处理一个 PDF。")
    else:
        timings = st.session_state.engine_timings
        v2_report = st.session_state.v2_report

        # ===== 指标 1: 时间 =====
        st.subheader("⏱️ 时间")
        cols = st.columns(len(timings) or 1)
        for col, (eng, dt) in zip(cols, timings.items()):
            if dt < 0:
                col.metric(eng, "失败")
            else:
                col.metric(eng, f"{dt:.1f}s")

        # ===== 指标 2: 抽到多少东西 =====
        st.subheader("📊 抽取结果统计")
        # 按 source 聚合 tables
        from collections import Counter
        source_table_counts = Counter(
            t.get("source", "?") for t in st.session_state.tables
        )

        stats_rows = []
        if use_pdfplumber and "pdfplumber" in timings:
            stats_rows.append({
                "引擎": "pdfplumber",
                "表格数": source_table_counts.get("pdfplumber", 0),
                "正文段落": "❌ 不抽取",
                "跨页合并": "❌ 不支持",
                "chunk 数": "—",
                "适用类型": "仅文本层 PDF",
            })
        if use_vlm and "vlm" in timings:
            stats_rows.append({
                "引擎": "VLM (Qwen2-VL)",
                "表格数": source_table_counts.get("vlm", 0),
                "正文段落": "⚠️ 一起被当表格抽",
                "跨页合并": "❌ 不支持",
                "chunk 数": "—",
                "适用类型": "扫描件可用但不准",
            })
        if use_docling and v2_report:
            v2_tables_cnt = sum(
                c for k, c in source_table_counts.items() if k.startswith("v2-")
            )
            stats_rows.append({
                "引擎": f"V2 ({v2_report.engine_used})",
                "表格数": v2_tables_cnt,
                "正文段落": f"✅ {v2_report.chunks_by_type.get('paragraph', 0)} 个段落 chunks",
                "跨页合并": f"✅ 合并 {v2_report.n_merged_cross_page} 次",
                "chunk 数": str(v2_report.n_chunks),     # 字符串化，避免混类型
                "适用类型": "文本 + 扫描通杀",
            })
        if stats_rows:
            st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)

        # ===== 指标 3: V2 详细报告 =====
        if v2_report:
            st.subheader("⭐ V2 详细报告")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("元素总数", v2_report.n_elements)
            c2.metric("表格数", v2_report.n_tables)
            c3.metric("跨页合并", v2_report.n_merged_cross_page)
            c4.metric("Chunks", v2_report.n_chunks)

            st.write("**Chunk 类型分布**")
            st.json(v2_report.chunks_by_type)

            st.write("**Contextual Retrieval:** " +
                      ("✅ 已开启" if v2_report.contextual_enriched else "❌ 未开启"))
            if v2_report.contextual_enriched:
                st.caption(f"Context 生成耗时: {v2_report.elapsed_context_ms:.0f}ms")

            st.write("**时间分解 (ms)**")
            st.json({
                "extract": round(v2_report.elapsed_extract_ms, 0),
                "chunk": round(v2_report.elapsed_chunk_ms, 0),
                "context": round(v2_report.elapsed_context_ms, 0),
                "embed": round(v2_report.elapsed_embed_ms, 0),
            })

            st.write("**前 8 个元素预览**")
            st.dataframe(pd.DataFrame(v2_report.elements_preview),
                          width="stretch", hide_index=True)

        # ===== 指标 4: 并排表格样本 =====
        if source_table_counts:
            st.subheader("🧪 同页表格的三路对比")
            # 找一页同时有多个引擎输出的，展示第一个
            pages_with_multi = Counter()
            for t in st.session_state.tables:
                pages_with_multi[t["page"]] += 1
            if pages_with_multi:
                # 挑有最多引擎命中的页
                best_page = max(pages_with_multi, key=pages_with_multi.get)
                st.caption(f"展示第 {best_page} 页")
                tables_on_page = [
                    t for t in st.session_state.tables if t["page"] == best_page
                ]
                cols = st.columns(len(tables_on_page) or 1)
                for col, t in zip(cols, tables_on_page):
                    with col:
                        st.caption(f"**{t['source']}**")
                        st.markdown(t["markdown"])


# ========== Tab 4: RAG QA ==========
with tab_qa:
    if not st.session_state.current_doc:
        st.info("先处理一个 PDF。")
    else:
        st.caption(f"当前文档：{st.session_state.current_doc}")
        # 选向量库
        rag_backend = st.radio(
            "检索后端",
            ["V1 (老 rag.py)", "V2 (rag_v2, 多粒度 + 可选 Contextual)"],
            horizontal=True, key="rag_backend",
            help="V2 包含正文段落、表格汇总、行级 chunks;"
                 "V1 只有表格行。",
        )
        q = st.text_input("🔍 问题", placeholder="如：营收最高的一年是？")
        top_k = st.slider("Top-K", 1, 10, 5)
        if q and st.button("提问"):
            from src.infer import chat

            t0 = time.time()
            if rag_backend.startswith("V2"):
                pipe = st.session_state.v2_pipe
                if pipe is None:
                    pipe = _build_v2_pipeline()
                    st.session_state.v2_pipe = pipe
                hits = pipe.search(q, top_k=top_k,
                                    doc_filter=st.session_state.current_doc)
            else:
                store = get_store()
                hits = store.search(q, top_k=top_k,
                                     doc_filter=st.session_state.current_doc)
            retr_ms = (time.time() - t0) * 1000

            st.subheader("📚 检索结果")
            for h in hits:
                meta = h.get("metadata", {})
                type_tag = f" [{meta.get('type','?')}]" if rag_backend.startswith("V2") else ""
                ctx = meta.get("context", "")
                ctx_tag = f"  🔖 {ctx}" if ctx else ""
                st.write(f"`{h['score']:.2f}`{type_tag} · {h['text'][:200]}{ctx_tag}")

            t1 = time.time()
            msgs = build_rag_prompt(q, hits)
            ans = chat(msgs, adapter=s2_adapter, max_tokens=400)
            gen_ms = (time.time() - t1) * 1000

            st.subheader("✅ 答案")
            st.markdown(ans)
            st.caption(f"检索 {retr_ms:.0f}ms · 生成 {gen_ms:.0f}ms · backend={rag_backend}")


# ========== Tab 4: Before/After ==========
with tab_compare:
    st.write("同一问题下，两个模型的输出并排对比。")
    st.caption("💡 本 Tab 自包含 —— 左侧 sidebar 的「微调 LoRA」勾选对这里无效，"
               "两边模型完全由下方 radio 决定。")
    if not st.session_state.current_doc:
        st.info("先处理一个 PDF。")
    else:
        # 模式选择：默认「v1 vs v2」不需要微调；有 adapter 时可切「base vs LoRA」
        has_adapter = C.STAGE2_ADAPTER.exists()
        modes = {"🔀 v1 vs v2（不同尺寸基座）": "v1_v2"}
        if has_adapter:
            modes["🎯 基座 vs LoRA（微调前后）"] = "base_lora"
        mode_label = st.radio("对比模式", list(modes.keys()), horizontal=True,
                              key="compare_mode")
        mode = modes[mode_label]

        q = st.text_input("问题", key="compare_q",
                          value="表格中数值最大的一项是什么？")

        if q and st.button("对比生成", key="compare_btn"):
            from src.infer import chat
            store = get_store()
            hits = store.search(q, top_k=5, doc_filter=st.session_state.current_doc)
            msgs = build_rag_prompt(q, hits)

            col1, col2 = st.columns(2)
            if mode == "v1_v2":
                v1_name = C.STAGE2_LLM_MLX.split("/")[-1]
                v2_name = C.STAGE2_LLM_MLX_V2.split("/")[-1]
                with col1:
                    st.subheader(f"v1 · {v1_name}")
                    st.caption("0.5B · 快 · 简短")
                    with st.spinner("生成中..."):
                        st.markdown(chat(msgs, adapter=None, max_tokens=400))
                with col2:
                    st.subheader(f"v2 · {v2_name}")
                    st.caption("1.5B · 首次加载约 30–60s · 答案更完整")
                    with st.spinner("生成中..."):
                        try:
                            st.markdown(chat(msgs, adapter=None,
                                             model_id=C.STAGE2_LLM_MLX_V2,
                                             max_tokens=400))
                        except Exception as e:
                            st.error(f"加载 v2 失败：{e}\n\n"
                                     f"确认 MLX 能访问 `{C.STAGE2_LLM_MLX_V2}`，"
                                     "或在 src/config.py 改成你本地已有的 MLX 模型。")
            else:  # base_lora
                with col1:
                    st.subheader("基线 (base)")
                    with st.spinner("生成中..."):
                        st.markdown(chat(msgs, adapter=None, max_tokens=400))
                with col2:
                    st.subheader("微调 (LoRA)")
                    with st.spinner("生成中..."):
                        st.markdown(chat(msgs, adapter=str(C.STAGE2_ADAPTER),
                                         max_tokens=400))
