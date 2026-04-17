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
    use_vlm = st.checkbox("启用 VLM 抽表（慢，但对扫描件更好）", value=False)
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


@st.cache_resource
def get_store() -> TableVectorStore:
    return TableVectorStore()


# ---------------- Tabs ----------------
tab_upload, tab_tables, tab_qa, tab_compare = st.tabs(
    ["📤 上传与抽取", "📊 表格查看", "💬 RAG 问答", "🆚 微调前后对比"]
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

    if pdf_path and st.button("🚀 开始抽取", type="primary"):
        with st.status("处理中...", expanded=True) as status:
            doc_id = pdf_path.stem
            st.write(f"📄 {pdf_path.name}")

            # pdfplumber
            st.write("⏳ pdfplumber 抽取...")
            t0 = time.time()
            pp = extract_tables(pdf_path)
            st.write(f"  ✓ {len(pp)} 张表 ({time.time()-t0:.1f}s)")

            # VLM（可选）
            vlm_tables: list[dict] = []
            if use_vlm:
                st.write("⏳ VLM 抽取（加载模型需要 10–30s）...")
                from src.infer import extract_table_from_image
                imgs = pdf_to_images(pdf_path, C.DATA_DIR / "preview" / doc_id)
                for i, img in enumerate(imgs, 1):
                    md = extract_table_from_image(img, adapter=s1_adapter, max_tokens=512)
                    if "无表格" not in md and "|" in md:
                        vlm_tables.append({"page": i, "markdown": md, "source": "vlm"})
                    st.write(f"  ✓ page {i}")

            all_tables = [{**t, "source": "pdfplumber"} for t in pp] + vlm_tables
            st.session_state.tables = all_tables
            st.session_state.current_doc = doc_id

            # 入库
            st.write("⏳ 向量化写入 ChromaDB...")
            store = get_store()
            total_chunks = 0
            for t in all_tables:
                df = parse_markdown_table(t["markdown"])
                if not df.empty:
                    total_chunks += store.add(df, doc_id=doc_id, page=t["page"])
            st.session_state.store = store
            st.write(f"  ✓ {total_chunks} chunks")
            status.update(label=f"完成 · {len(all_tables)} 表 · {total_chunks} chunks",
                          state="complete")


# ========== Tab 2: Table viewer ==========
with tab_tables:
    if not st.session_state.tables:
        st.info("先在「上传与抽取」tab 处理一个 PDF。")
    else:
        for i, t in enumerate(st.session_state.tables):
            with st.expander(f"Page {t['page']} · {t.get('source','?')} · table #{i}",
                             expanded=i == 0):
                st.markdown(t["markdown"])
                df = parse_markdown_table(t["markdown"])
                if not df.empty:
                    st.dataframe(df, use_container_width=True)


# ========== Tab 3: RAG QA ==========
with tab_qa:
    if not st.session_state.current_doc:
        st.info("先处理一个 PDF。")
    else:
        st.caption(f"当前文档：{st.session_state.current_doc}")
        q = st.text_input("🔍 问题",
                          placeholder="如：营收最高的一年是？")
        top_k = st.slider("Top-K", 1, 10, 5)
        if q and st.button("提问"):
            from src.infer import chat

            store = get_store()
            t0 = time.time()
            hits = store.search(q, top_k=top_k, doc_filter=st.session_state.current_doc)
            retr_ms = (time.time() - t0) * 1000

            st.subheader("📚 检索结果")
            for h in hits:
                st.write(f"`{h['score']:.2f}` · {h['text']}")

            t1 = time.time()
            msgs = build_rag_prompt(q, hits)
            ans = chat(msgs, adapter=s2_adapter, max_tokens=400)
            gen_ms = (time.time() - t1) * 1000

            st.subheader("✅ 答案")
            st.markdown(ans)
            st.caption(f"检索 {retr_ms:.0f}ms · 生成 {gen_ms:.0f}ms")


# ========== Tab 4: Before/After ==========
with tab_compare:
    st.write("同一问题下，基线模型 vs 微调模型的输出对比。")
    if not st.session_state.current_doc:
        st.info("先处理一个 PDF。")
    else:
        q = st.text_input("问题", key="compare_q",
                          value="表格中数值最大的一项是什么？")
        if q and st.button("对比生成", key="compare_btn"):
            from src.infer import chat
            store = get_store()
            hits = store.search(q, top_k=5, doc_filter=st.session_state.current_doc)
            msgs = build_rag_prompt(q, hits)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("基线 (base)")
                with st.spinner("生成中..."):
                    st.markdown(chat(msgs, adapter=None, max_tokens=400))
            with col2:
                st.subheader("微调 (LoRA)")
                if not C.STAGE2_ADAPTER.exists():
                    st.warning("stage2_adapter 不存在，先在 notebook 03 训练。")
                else:
                    with st.spinner("生成中..."):
                        st.markdown(chat(msgs, adapter=str(C.STAGE2_ADAPTER), max_tokens=400))
