# Project Map — 功能清单与源码阅读顺序

> 按「理解系统的最短路径」排序。先读 Tier 1 把运行时跑通、读懂，再往训练/数据上游走。
> 每个功能给出：**入口 → 文件 → 递归调用树**，方便对着读源码。

---

## 功能总览（24 项）

### 🏠 主项目（`src/` + `app/` + `scripts/` + Docker）

| # | 功能 | Tier | 入口 |
|---|---|---|---|
| 1 | RAG 推理 HTTP API（Docker 主角） | 🟢 Tier 1 | `docker compose up` → `uvicorn src.serve.api:app` |
| 2 | 表格向量化 + ChromaDB 检索 | 🟢 Tier 1 | `src.rag.TableVectorStore` |
| 3 | Markdown 表格解析与清洗 | 🟢 Tier 1 | `src.rag.parse_markdown_table` |
| 4 | 模型懒加载 + 热加载（LRU / atomic swap） | 🟢 Tier 1 | `src.infer.load_vlm / load_llm`、`api.py` 的 `_swap_model` |
| 5 | Stage 1：VLM 抽表（图→Markdown） | 🟡 Tier 2 | `src.infer.extract_table_from_image`、`POST /extract` |
| 6 | Stage 2：LLM 表格问答 | 🟡 Tier 2 | `src.infer.chat`、`POST /query` |
| 7 | PDF → 图片 + pdfplumber ground truth（V1，老版本） | 🟡 Tier 2 | `src.pdf_utils` |
| **7b** | **V2 抽取 + 多粒度 Chunking + Contextual Retrieval** ⭐ | 🟡 Tier 2 | `src.extract` + `src.chunking` + `src.rag_v2` |
| 8 | 训练数据集构建（sharegpt / alpaca） | 🟠 Tier 3 | `scripts/prepare_stage1.py`、`scripts/prepare_stage2.py` |
| 9 | HF 数据集 + 样例 PDF 下载 | 🟠 Tier 3 | `scripts/download_data.py` |
| 10 | QLoRA 微调（MLX，Mac 本地） | 🟠 Tier 3 | `notebooks/02_finetune_stage1.ipynb`、`03_finetune_stage2.ipynb` |
| 11 | LoRA 融合导出 HF 格式 | 🟠 Tier 3 | `scripts/fuse_model.py` |
| 12 | Streamlit 交互 UI（5 个 Tab，含 V1/V2 并排引擎对比） | 🔵 Tier 4 | `streamlit run app/streamlit_app.py` |
| 13 | 评估指标（TEDS / EM / F1 / ROUGE） | 🔵 Tier 4 | `src.eval` |
| 14 | Docker 部署 + demo 模型准备 | 🔵 Tier 4 | `Dockerfile`、`docker-compose.yml`、`scripts/setup_demo_models.sh` |

### 🏭 企业级参考（`finetuning/` + `inference/`）—— 独立于主项目，不影响 Streamlit

| # | 功能 | Tier | 入口 |
|---|---|---|---|
| **15** | **Raw HuggingFace 微调（Trainer/PEFT/TRL）** ⭐ | 🟣 Tier 5 | `finetuning/huggingface/train_{lora,qlora,sft_trl}.py` |
| 16 | LLaMA-Factory QLoRA（Stage1 VLM + Stage2 LLM） | 🟣 Tier 5 | `finetuning/llamafactory/scripts/setup.sh` |
| 17 | Axolotl QLoRA（Stage2 LLM，对照学习） | 🟣 Tier 5 | `finetuning/axolotl/scripts/setup.sh` |
| **18** | **Raw HuggingFace 推理（6 个递进示例）** ⭐ | 🟣 Tier 5 | `inference/huggingface/*.py` |
| 19 | Ollama 本地推理（Mac 可 run + OpenAI 兼容 API） | 🟣 Tier 5 | `ollama serve` + `uv run python inference/ollama/rag_server_ollama.py` |
| 20 | vLLM / TGI 生产推理（Linux GPU 参考） | 🟣 Tier 5 | `inference/vllm/docker compose up` |
| 21 | K8s / OpenShift 生产部署（金丝雀 + adapter 安全 ops） | 🟣 Tier 5 | `kubectl apply -k inference/kubernetes/base/` |
| 22 | AWS 部署参考（Bedrock / SageMaker / Fargate / EKS） | 🟣 Tier 5 | `cloud/aws/` |
| 23 | Azure 部署参考（OpenAI / AML / Container Apps / AKS） | 🟣 Tier 5 | `cloud/azure/` |

**推荐阅读顺序**：1 → 2 → 3 → 4 → 5 → 6 → 7 → 12 → 8 → 9 → 10 → 11 → 13 → 14 → **15 → 18 → 19 → 16 → 17 → 20 → 21 → 22 / 23**

为什么 **15 和 18 排最前**：
- 先 **15（Raw HF 微调）**：所有 YAML 框架（LLaMA-Factory/Axolotl）都是它的包装；不懂它=不懂微调
- 再 **18（Raw HF 推理）**：`src/serve/api.py` 用的就是这个栈；读懂后才明白 vLLM 在优化什么
- 然后 **19（Ollama）**：Mac 上真能跑通"业务层 + 推理层分离"
- 之后 **16/17（LLaMA-Factory/Axolotl）**：看 YAML 框架帮你抽象了什么
- 最后 **20/21/22/23**：生产部署细节

理由：先从 Docker 运行时倒推（功能 1 是唯一「实际跑起来」的入口），把 RAG 闭环（2/3/4/5/6/7）吃透，再看 UI 怎么串同一套库（12），最后才是训练/数据上游（8/9/10/11）。评估和部署细节（13/14）放最后。

---

## 🟢 Tier 1 — 运行时核心（读完就能说清楚系统在做什么）

### 功能 1：RAG 推理 HTTP API（Docker 的主角）

**入口**：`docker compose up` → `Dockerfile:89` 的 `CMD ["uvicorn", "src.serve.api:app", ...]`

**文件**：`src/serve/api.py` (422 行，唯一的 web 层)

**调用树**
```
uvicorn 加载 src.serve.api:app
└── FastAPI(..., lifespan=lifespan)           [api.py:213]
    └── lifespan(app)                          [api.py:133]
        ├── _load_stage2()                    [api.py:74]  ← 启动时 eager load
        │   └── transformers.AutoModelForCausalLM.from_pretrained(...)
        ├── _load_embed_and_chroma()          [api.py:121]
        │   ├── SentenceTransformer(EMBED_MODEL)
        │   └── chromadb.PersistentClient(CHROMA_DIR)
        └── _start_watcher()  (可选)          [api.py:158]  ← AUTO_RELOAD=1 时
            └── watchdog.Observer → _reload_if_changed → _swap_model

路由注册：
├── GET  /health           → health()                 [api.py:244]
├── POST /admin/reload     → admin_reload()           [api.py:259]  ← 原子替换模型
│   ├── _require_admin(X-Admin-Key)                   [api.py:236]
│   ├── _load_stage1 / _load_stage2                   [api.py:74/88]
│   └── _swap_model()                                 [api.py:102]
├── POST /admin/unload     → admin_unload()           [api.py:294]  ← 释放显存
├── POST /extract          → extract(file, doc_id)    [api.py:310]  ← 功能 5 入口
│   ├── _load_stage1() 懒加载                         [api.py:88]
│   ├── fitz.open(...)  PDF → PIL                     (内联，未走 pdf_utils)
│   ├── processor / model.generate()   ← VLM 生成 markdown
│   └── _ingest_markdown_table()                      [api.py:361]
│       ├── src.rag.parse_markdown_table              [rag.py:17]
│       ├── embed.encode([...])
│       └── chroma.add(...)
├── POST /query            → query(QueryIn)           [api.py:377]  ← 功能 6 入口
│   ├── embed.encode + chroma.query     ← 检索
│   └── model.generate()                 ← Stage2 生成答案
└── POST /ingest_markdown  → ingest_markdown(...)     [api.py:416]
    └── _ingest_markdown_table()                      [api.py:361]
```

**关键环境变量**（docker-compose.yml 注入）
- `STAGE1_MODEL_PATH=/app/models/stage1_fused`（VLM）
- `STAGE2_MODEL_PATH=/app/models/stage2_fused`（LLM）
- `ENABLE_STAGE1=1|0`（为 0 时 `/extract` 返回 503，省 3GB+ 内存）
- `AUTO_RELOAD=1`（启动 watchdog 监听 `models/` 目录）
- `ADMIN_API_KEY=...`（`/admin/*` 端点校验）

**读完这一节你应该能回答**：容器里的模型是怎么加载的？`/query` 收到请求后经历哪些步骤？热加载怎么做到不中断请求？

---

### 功能 2：表格向量化 + ChromaDB 检索

**入口**：`TableVectorStore(persist_dir=..., embed_model=...)`

**文件**：`src/rag.py:61-112`

**调用树**
```
TableVectorStore.__init__                     [rag.py:62]
├── chromadb.PersistentClient(path=...)
├── client.get_or_create_collection("tables", {"hnsw:space": "cosine"})
└── SentenceTransformer("BAAI/bge-small-zh-v1.5")

.add(df, doc_id, page)                        [rag.py:87]
├── chunk_table(df, doc_id, page)             [rag.py:75]  ← 每行 1 个 chunk
├── encoder.encode(texts, normalize_embeddings=True)
└── collection.add(ids, documents, embeddings, metadatas)

.search(query, top_k, doc_filter)             [rag.py:101]
├── encoder.encode([query])
└── collection.query(..., where={"doc_id": ...})
    → [{"text", "score", "metadata"}]
```

**注意**：FastAPI 里走的是"手写版"（`api.py:387-393`），逻辑一样但没复用 `TableVectorStore`，因为 API 里模型在 `STATE` 字典里。Streamlit 那边（功能 12）用的是完整类。

---

### 功能 3：Markdown 表格解析与清洗

**入口**：`parse_markdown_table(md) → pd.DataFrame`、`normalize_numbers(df)`

**文件**：`src/rag.py:17-56`

**调用树**
```
parse_markdown_table(md_str)                  [rag.py:17]
├── 按 "|" 切行 → header + body
├── 空列名/重复列名自动加后缀（pandas 对 dup 列会返 DataFrame 而非 Series，下游会崩）
└── pd.DataFrame(rows, columns=unique)

normalize_numbers(df)                         [rag.py:48]
└── 全角数字 → 半角、去千分位
```

**被谁用**：`rag.TableVectorStore.add` / `api._ingest_markdown_table` / `eval.teds` / `eval.cell_f1` / `prepare_stage2.template_qa_from_table` / `streamlit_app`。**整个项目的表格入口函数**。

---

### 功能 4：模型懒加载 + 热加载

**入口**（两套）：
- 脚本/UI 用：`src.infer.load_vlm` / `load_llm`（带 `@lru_cache`）
- API 服务用：`api.py` 的 `STATE` 字典 + `_swap_model`（支持热加载）

**文件**：`src/infer.py:46-118`、`src/serve/api.py:47-212`

**调用树（infer.py 版）**
```
load_vlm(adapter=None)   @lru_cache(maxsize=2)   [infer.py:47]
├── if USE_MLX=1:
│   └── mlx_vlm.load(STAGE1_VLM_MLX, adapter_path=adapter)
└── else:
    ├── transformers.Qwen2VLForConditionalGeneration.from_pretrained(...)
    └── if adapter: peft.PeftModel.from_pretrained(...)
    → (backend, model, processor)

load_llm(adapter=None, model_id=None)  @lru_cache(maxsize=4)  [infer.py:96]
(同构，可覆盖 model_id 用于 v1 vs v2 对比)
```

**调用树（api.py 热加载版）**
```
POST /admin/reload → admin_reload()           [api.py:259]
├── _dir_fingerprint(path)                    [api.py:61]  ← sha1(name+size+mtime) 当版本号
├── loader = _load_stage1 或 _load_stage2
├── new_obj = loader(path)                    ← 新模型加载到局部变量，失败不影响旧模型
└── _swap_model(stage, new_obj, path)         [api.py:102]
    ├── with _lock: STATE[key] = new_obj      ← 锁内原子替换引用
    ├── del old; gc.collect()                 ← 锁外释放旧模型
    └── torch.cuda.empty_cache()  (if CUDA)

AUTO_RELOAD=1 时：watchdog 监听 models/，自动触发 _reload_if_changed [api.py:197]
```

**设计要点**：
- `_dir_fingerprint` 只在权重真变时重载，避免文件 touch 也触发
- 新模型先加载到局部变量再 swap —— 加载失败旧模型完好
- 请求端（`/query`/`/extract`）抓引用后就释放锁，并发安全

---

## 🟡 Tier 2 — RAG 两端：VLM 抽表 + LLM 问答

### 功能 5：Stage 1 —— VLM 抽表（图 → Markdown）

**入口**：
- 脚本：`src.infer.extract_table_from_image(image_path, adapter=None)`
- API：`POST /extract` (上传 PDF/图片)

**文件**：`src/infer.py:66-90`、`src/serve/api.py:310-358`

**调用树**
```
extract_table_from_image(image_path, adapter)   [infer.py:66]
├── load_vlm(adapter)                            [infer.py:47]
├── if mlx:
│   ├── mlx_vlm.prompt_utils.apply_chat_template(processor, config, prompt, 1)
│   └── mlx_vlm.generate(model, processor, formatted, image=[path], max_tokens)
└── else:
    ├── processor.apply_chat_template(messages)
    ├── processor(text, images=..., return_tensors="pt")
    └── model.generate(...) → processor.batch_decode(...)
→ "| 列A | 列B |\n| --- |..." markdown
```

**API 版流程**（`POST /extract`）
```
extract(file, doc_id)                           [api.py:310]
├── 读字节 → fitz.open(stream=...) → 每页 PIL
├── for 每页:
│   ├── processor + model.generate() → markdown
│   └── _ingest_markdown_table(md, doc_id, page)  ← 自动入向量库
└── 返回 {doc_id, n_pages, tables:[{page, markdown}]}
```

---

### 功能 6：Stage 2 —— LLM 表格问答

**入口**：
- 脚本：`src.infer.chat(messages, adapter=None, model_id=None)`
- API：`POST /query`

**文件**：`src/infer.py:121-136`、`src/serve/api.py:377-413`、`src/rag.py:117-131`（prompt 模板）

**调用树（API /query 版，完整 RAG 闭环）**
```
POST /query → query(QueryIn{question, top_k, doc_filter})   [api.py:377]
├── 1. 检索
│   ├── embed.encode([question], normalize=True)
│   └── chroma.query(query_embeddings=..., where={"doc_id": ...})
│   → hits [{text, score, metadata}]
├── 2. 组 prompt
│   ├── 用 src.rag.RAG_SYSTEM                    [rag.py:117]
│   └── user content = "检索到的表格数据：\n{ctx}\n\n问题：{q}"
├── 3. 生成
│   ├── tok.apply_chat_template(messages)
│   ├── model.generate(max_new_tokens=MAX_TOKENS, do_sample=False)
│   └── tok.decode(...)
└── 返回 RAGOut{answer, sources, retrieval_ms, generation_ms}
```

**Prompt 模板**（`src/rag.py:117-130`）
```
RAG_SYSTEM: 4 条硬规则（只用表格数据 / 标明来源 / 数据不足说明 / 展示计算）
build_rag_prompt(query, chunks) → [{"role":"system"...}, {"role":"user"...}]
```

---

### 功能 7：PDF → 图片 + pdfplumber ground truth

**入口**：`src.pdf_utils.pdf_to_images` / `extract_tables`

**文件**：`src/pdf_utils.py` (70 行)

**调用树**
```
pdf_to_images(pdf_path, out_dir, dpi=150, max_size=(1344,1344))  [pdf_utils.py:13]
├── fitz.open(pdf_path)                        ← pymupdf 渲染
├── for each page: get_pixmap(Matrix(zoom, zoom))
├── PIL.Image.frombytes + thumbnail(max_size)
└── img.save(out_dir/xxx_p001.png) → [Path, Path, ...]

extract_tables(pdf_path)                       [pdf_utils.py:43]
├── pdfplumber.open(pdf_path)
├── for each page: page.extract_tables()
└── rows_to_markdown(headers, rows)            [pdf_utils.py:33]
→ [{page, index, headers, rows, markdown}, ...]
```

**用处**：
- Streamlit 里做数字表格的"先 pdfplumber，失败才走 VLM"fallback
- 训练数据 ground truth（功能 8）

**注意**：`POST /extract` 里没走这个模块，直接用 `fitz.open(stream=bytes)` 内联了（约定：API 永远过 VLM，脚本才用 pdfplumber）。

---

### 功能 7b：V2 抽取 + 多粒度 Chunking + Contextual Retrieval ⭐

**入口**：`src.rag_v2.Pipeline(engine="auto", contextual=True, context_gen=OllamaContextGen(...))`

**理论文档**：[docs/pdf-extraction-2026.md](./docs/pdf-extraction-2026.md)

**文件结构（3 个新文件，1 份技术文档）**
```
docs/
└── pdf-extraction-2026.md          ← 理论: 2026 最佳实践 + 设计决策 + 迁移路径

src/
├── extract.py                      ← Docling + MinerU 双引擎 + 跨页表格合并
├── chunking.py                     ← 多粒度 chunking + Contextual Retrieval
└── rag_v2.py                       ← Pipeline 类 (extract + chunk + ctx + embed + search)

pyproject.toml                      ← [optional-dependencies].extract: docling + mineru + openai
                                      安装: uv sync --extra extract
```

**调用树（Pipeline.ingest_pdf 完整流程）**
```
Pipeline.ingest_pdf(pdf_path, doc_id)
├── 1. detect_pdf_type(pdf_path)                [extract.py:75]
│   └── fitz 读前 3 页，文本字节 <100 → scanned，否则 text
│
├── 2. extract_document(path, engine)           [extract.py:352]
│   ├── engine="docling":
│   │   └── extract_with_docling()              [extract.py:94]
│   │       ├── DocumentConverter(PdfPipelineOptions(do_ocr, do_table_structure))
│   │       ├── converter.convert(path).document
│   │       └── for item in doc.iterate_items() → DocElement 流
│   │
│   └── engine="mineru":
│       └── extract_with_mineru()               [extract.py:161]
│           ├── subprocess: mineru -p <pdf> -o <out> -m auto
│           └── 读 content_list.json → DocElement 流
│
├── 3. merge_cross_page_tables(elements)        [extract.py:237]
│   ├── _is_continuation(): 列数 + bbox 启发式
│   ├── _strong_match(): 强确信直接合
│   ├── llm_verify (可选): 弱候选走 LLM 二次确认
│   └── _combine_tables(): pd.concat + 去重复表头
│
├── 4. chunk_document(elements, doc_id)         [chunking.py:127]
│   ├── heading → 记为 current_section，注入后续 chunks 的 metadata
│   ├── table   → chunk_table():                [chunking.py:42]
│   │             ├── Level 1: table_summary (列名 + n 行 + 前 3 行预览)
│   │             └── Level 2: table_row × N (每行前缀带表名)
│   ├── paragraph → chunk_paragraph():          [chunking.py:90]
│   │             按句切到 max_tokens, 带 overlap
│   ├── list    → chunk_list()                  [chunking.py:120] 整块保留
│   └── footnote/caption/formula → 一个 chunk 一条
│
├── 5. enrich_with_context(chunks, full_doc)    [chunking.py:210]  (optional)
│   ├── build_full_doc_text()                   [chunking.py:237]
│   └── for ch in chunks:
│       └── context_fn(full_doc, ch.text)       ← Ollama / Azure OpenAI
│           └── 生成 50 字 context
│       → ch.embed_text = "[context] " + ch.text
│
├── 6. _embed_and_add(chunks, doc_id)           [rag_v2.py:131]
│   ├── embedder.encode([c.embed_text for c in chunks])
│   └── collection.add(ids, documents=[c.text], embeddings, metadatas)
│       ← 存的 documents 是原始 ch.text（给 LLM 看）
│       ← embedding 是 ch.embed_text（带 context 前缀）的向量
│
└── 7. 返回 IngestReport
    ├── engine_used, n_elements, n_tables, n_merged_cross_page
    ├── n_chunks, chunks_by_type
    ├── contextual_enriched
    └── 时间分解: extract_ms / chunk_ms / context_ms / embed_ms
```

**关键设计点**

1. **文本 vs 扫描自动路由**（`detect_pdf_type`）
   - 前 3 页文字 <100 字节 → 判为扫描件 → 走 MinerU
   - 否则走 Docling（Docling 自带 OCR fallback，处理混合 PDF 最稳）

2. **Docling vs MinerU 选型**
   - Docling：布局感知最强、表格结构 + heading hierarchy + 阅读顺序最好、LlamaIndex/LangChain 原生集成
   - MinerU：**中文 PDF 之王**、84 语言 OCR、扫描件复杂版式最强
   - 默认 auto：文字型用 Docling，扫描件用 MinerU

3. **跨页表格合并**（Azure Document Intelligence 启发式）
   - 列数相同
   - prev 底部到页底距离 <80pt，curr 顶部到页顶距离 <80pt
   - 自动去除重复表头（后表第一行 == 前表 header 时 drop）
   - 不确定时走 LLM 验证（可选）

4. **多粒度 Chunking**
   - 老做法：每行一个 chunk，脱离上下文 → 召回差
   - 新做法：summary + row，row 前缀带表名 + 列名 → 召回强
   - 粗查："这份文档有哪些财务表" → 命中 summary
   - 细查："2023 年净利润是多少" → 命中 row

5. **Contextual Retrieval**（Anthropic 2024）
   - 为每个 chunk 用小 LLM 生成 50 字上下文
   - `embed_text = "[context] " + text`
   - 召回 @20 提升 35%（Anthropic 数字）
   - 本项目用 Ollama + Qwen2.5-0.5B 免费跑；也支持 Azure OpenAI / Claude

6. **独立 collection**
   - V2 写 `tables_v2` collection，V1 写 `tables`
   - 互不干扰，可随时切换对比

**Streamlit UI 集成**

`streamlit_app.py` 现在有 5 个 Tab（原 4 + 新"🔬 抽取引擎对比"）：

```
Sidebar:
  ☑ pdfplumber  ☑ VLM  ⭐ V2 (Docling/MinerU)
                       V2 engine: [auto | docling | mineru]
                       ⭐ Contextual Retrieval
                       Ollama model: qwen2.5:0.5b-instruct-q4_K_M

Tab 1 "上传与抽取":
  - 自动探测 PDF 类型，扫描件给出"pdfplumber ❌ / VLM ⚠️ / V2 ✅"提示
  - 勾选的每个引擎都跑，时间 + 统计 → 记录 engine_timings + v2_report

Tab 2 "表格查看": 按 source 分组展示（pdfplumber / vlm / v2-docling / v2-mineru）

Tab 3 "🔬 抽取引擎对比"（新）:
  - 时间对比（metrics 卡片）
  - 抽取统计对比表（表格数 / 正文段落 / 跨页合并 / chunk 数 / 适用类型）
  - V2 详细报告（chunk 类型分布 / 时间分解 / 前 8 元素预览）
  - 同页表格三路并排 markdown

Tab 4 "RAG QA":
  - 单选: V1 backend (老 chroma tables) | V2 backend (tables_v2 多粒度)
  - V2 命中会显示 chunk type + context 前缀

Tab 5 "微调前后对比": 保持原样
```

**读完这一节你应该能回答**
- pdfplumber 丢文字是因为什么设计缺陷？Docling 怎么解决？
- 跨页表格的 5 个启发式判定分别是什么？什么时候必须走 LLM？
- 多粒度 chunking vs 每行一 chunk，为什么召回差这么多？
- Contextual Retrieval 的成本怎么控制？prompt caching 有什么用？
- V2 和 V1 为什么用不同的 Chroma collection？

---

## 🔵 Tier 4 提前 —— Streamlit UI（看得见摸得着的入口）

> 强烈建议看完 Tier 1/2 后立刻读 UI，因为它把整个 pipeline 串了一遍，比 notebook 更直观。

### 功能 12：Streamlit UI（4 个 Tab）

**入口**：`streamlit run app/streamlit_app.py`

**文件**：`app/streamlit_app.py` (217 行)

**调用树**
```
Streamlit 启动 → 顶栏 4 tabs
├── Tab 1 "📤 上传与抽取"                      [streamlit_app.py:58-113]
│   └── 点击「开始抽取」:
│       ├── extract_tables(pdf)                ← 功能 7（pdfplumber 主通道）
│       ├── if use_vlm: extract_table_from_image(img, s1_adapter)  ← 功能 5
│       ├── parse_markdown_table(md)           ← 功能 3
│       └── TableVectorStore.add(df, doc_id, page)  ← 功能 2
├── Tab 2 "📊 表格查看"                        [streamlit_app.py:117-127]
│   └── 显示 session_state.tables + parse_markdown_table
├── Tab 3 "💬 RAG 问答"                        [streamlit_app.py:131-158]
│   └── 点「提问」:
│       ├── store.search(q, top_k, doc_filter) ← 功能 2
│       ├── build_rag_prompt(q, hits)          ← 功能 6 prompt
│       └── chat(msgs, adapter=s2_adapter)     ← 功能 6
└── Tab 4 "🆚 微调前后对比"                    [streamlit_app.py:162-217]
    └── 两种模式:
        ├── v1 vs v2: 同架构不同尺寸（0.5B vs 1.5B）
        └── base vs LoRA: 同尺寸 base vs adapter（需 Stage2_ADAPTER 存在）
```

**核心 session_state**：`store` / `tables` / `current_doc`

**与 Docker API 的关系**：**完全独立**。Streamlit 直接 import `src.*` 在本地进程里跑模型（MLX 后端优先）；Docker 的 API 是独立的 HTTP 服务。

---

## 🟠 Tier 3 — 训练管线上游

### 功能 8：训练数据集构建

**入口**：
- `uv run python scripts/prepare_stage1.py`
- `uv run python scripts/prepare_stage2.py`

**文件**：`scripts/prepare_stage1.py`、`scripts/prepare_stage2.py`、`src/data.py`

**Stage 1 调用树**（image + HTML → image + markdown）
```
prepare_stage1.main()                          [prepare_stage1.py:50]
├── 遍历 data/raw/pubtabnet_sample/ + fintabnet_sample/
├── _extract_image_and_md(row)                 [prepare_stage1.py:29]
│   ├── row["image"] / row["png"]
│   └── src.data.html_table_to_markdown(html)  [data.py:26]  ← 无 lxml 依赖的正则解析
├── img.save(stage1_images/xxx.png)
├── src.data.build_stage1_samples(image_paths, markdowns)  [data.py:65]
│   → [{messages:[{role,user,content:[image+text]},{role,assistant,content:md}]}, ...]
│      （sharegpt 格式，LLaMA-Factory / mlx_vlm 都能吃）
├── split_train_val(samples, 0.1)
└── save_jsonl → data/stage1_train/{train,val}.jsonl
```

**Stage 2 调用树**（html → markdown → 模板生成 QA）
```
prepare_stage2.main()                          [prepare_stage2.py:110]
├── from_paired_dataset("fintabnet_sample")    [prepare_stage2.py:83]
│   ├── _row_to_md(row)                         → markdown
│   └── template_qa_from_table(md, n=3)        [prepare_stage2.py:61]
│       ├── src.rag.parse_markdown_table       ← 功能 3
│       └── 模板:「{col} 最大/最小的是哪一行？」+ 自动计算答案
├── src.data.build_stage2_samples(qa_pairs)    [data.py:82]
│   → [{instruction, input, output}, ...] (alpaca 格式)
├── split_train_val
└── save_jsonl → data/stage2_train/{train,val}.jsonl
```

---

### 功能 9：HF 数据集 + 样例 PDF 下载

**入口**：`uv run python scripts/download_data.py`

**文件**：`scripts/download_data.py` (283 行)

**调用树**
```
main()                                         [download_data.py:257]
├── parse_args()                               [download_data.py:229]
│   解析 --pubtabnet / --fintabnet / --comtqa / --clean / --fresh ...
├── if --clean*: _clean() → 退出
├── download_hf_datasets(args)                 [download_data.py:181]
│   ├── streaming 模式（默认）:
│   │   └── _stream_take(repo, split, n, save_to)   [download_data.py:147]
│   │       └── load_dataset(streaming=True).take(n) → save_to_disk
│   └── bulk 模式（--no-stream）:
│       └── _bulk_download()                  [download_data.py:172]
│           └── src.data.download_hf_dataset  [data.py:12]
│               → datasets.load_dataset(repo, split=f"train[:{n}]")
└── download_sample_pdfs()                     [download_data.py:215]
    └── urlretrieve(url, data/samples/xxx.pdf)  ← Apple/NVIDIA/IRS 公开 PDF
```

**数据集**：
- `apoidea/pubtabnet-html` → 学术论文表格
- `ds4sd/FinTabNet_OTSL` → 金融表格
- `ByteDance/ComTQA` → 只给 image_name + QA（默认跳过，无法单独用）

---

### 功能 10：QLoRA 微调（MLX）

**入口**：notebooks（MacBook 本地）或 colab 版（GPU）
- `notebooks/02_finetune_stage1.ipynb` / `02b_stage1_colab.ipynb`
- `notebooks/03_finetune_stage2.ipynb` / `03b_stage2_colab.ipynb`

**文件**：ipynb，由 `scripts/make_notebooks.py` 生成；MLX 后端用 `mlx_lm.lora` / `mlx_vlm.lora`

**调用树**（简化）
```
notebook 02 (Stage 1)
├── 读 data/stage1_train/train.jsonl
├── 构造 mlx_vlm 训练配置（rank=16, alpha=32, 4bit）
├── python -m mlx_vlm.lora --model ... --train
└── 输出 adapter → models/stage1_adapter/

notebook 03 (Stage 2)
├── 读 data/stage2_train/train.jsonl
├── python -m mlx_lm.lora --model Qwen2.5-0.5B-Instruct-4bit --train
└── 输出 adapter → models/stage2_adapter/
```

训练完后，adapter 可被 `src.infer.load_vlm/load_llm(adapter=...)` 直接挂载（功能 4）。

---

### 功能 11：LoRA 融合导出 HF 格式

**入口**：`uv run python scripts/fuse_model.py --stage all`

**文件**：`scripts/fuse_model.py` (78 行)

**调用树**
```
main()
├── fuse_stage1() [stage1_adapter 存在时]
│   └── subprocess: python -m mlx_vlm.fuse --model ... --adapter-path ... --de-quantize
│       → models/stage1_fused/  (HF 格式，transformers 可读)
└── fuse_stage2()
    └── subprocess: python -m mlx_lm.fuse ...
        → models/stage2_fused/
```

**为什么要融合**：
- MLX LoRA 产物是 adapter，只能被 `mlx_lm.load(..., adapter_path=...)` 读
- Docker 容器里用 transformers（不装 MLX），需要 HF 格式 + de-quantize 后的权重
- 融合 = base + adapter 合成一个"新 base"，去量化后 transformers 就能加载

---

## 🔵 Tier 4 — 评估 + 部署

### 功能 13：评估指标

**入口**：`src.eval.teds / cell_f1 / exact_match / token_f1 / rouge_l`

**文件**：`src/eval.py` (78 行)

**调用树**
```
teds(pred_md, gold_md)                         [eval.py:41]
├── serialize(md)  → [f"{ri}|{col}|{_normalize(val)}"]
│   └── src.rag.parse_markdown_table           ← 功能 3
└── editdistance.eval(p, g) → 1 - dist/max(len)

cell_f1(pred_md, gold_md)                      [eval.py:59]
├── cells(md) → set[(ri, col, norm_val)]
└── intersection / size → precision/recall/f1

exact_match / token_f1 / rouge_l               [eval.py:16-38]
  用于 Stage 2 QA 输出评估
```

现在没被任何地方自动调用，预留给 evaluation notebook（`04_end_to_end_rag.ipynb`）或未来的 `run_eval.sh`。

---

### 功能 14：Docker 部署 + demo 模型准备

**入口**：
1. `bash scripts/setup_demo_models.sh [--stage1] [--v2]` → 下载 HF base 到 `models/`
2. `docker compose up --build` → 构建并启动 API

**文件**：`Dockerfile`、`docker-compose.yml`、`scripts/setup_demo_models.sh`、`scripts/test_api.sh`

**调用树**
```
setup_demo_models.sh
└── huggingface_hub.snapshot_download(repo_id, local_dir, ignore=[*.bin,*.gguf,...])
    ├── Qwen/Qwen2.5-0.5B-Instruct     → models/stage2_fused/       (必下)
    ├── Qwen/Qwen2-VL-2B-Instruct       → models/stage1_fused/       (--stage1)
    └── Qwen/Qwen2.5-1.5B-Instruct      → models/stage2_fused_v2/    (--v2)

Dockerfile (multi-stage)
├── builder: python:3.11-slim
│   ├── apt: libglib2.0-0 libgl1 ...
│   ├── COPY --from=ghcr.io/astral-sh/uv:0.9 /uv
│   ├── 内联 Python 脚本：过滤 mlx/streamlit/jupyter 生成 req-docker.txt
│   └── uv pip install -r req-docker.txt (先装 torch CPU 轮子，再装其它)
└── runtime: python:3.11-slim
    ├── COPY --from=builder site-packages + /usr/local/bin
    ├── COPY src/
    ├── COPY pyproject.toml  (README.md 在 .dockerignore 里，不拷)
    ├── USER app
    └── CMD uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

docker-compose.yml
├── volumes:
│   ├── ./models:/app/models:ro           ← 功能 11/14 的产物挂进来
│   ├── chroma_data:/app/chroma_db        ← ChromaDB 持久化
│   └── hf_cache:/app/.cache/huggingface  ← 避免重下 embedder
└── env:
    ├── ENABLE_STAGE1, ADMIN_API_KEY, AUTO_RELOAD, MAX_TOKENS
    └── EMBED_MODEL=BAAI/bge-small-zh-v1.5

scripts/test_api.sh  ← 跑通后用这个 smoke test
└── curl /health → /extract → /ingest_markdown → /query → /admin/reload
```

---

## 读完之后的全局图

```
数据下载                 → 训练集构建          → 微调                 → 融合           → 部署
download_data.py           prepare_stage{1,2}    notebook 02/03         fuse_model.py    docker compose up
                           (功能 8)               (功能 10)               (功能 11)        (功能 14)
     ↓                          ↓                    ↓                      ↓
data/raw/*_sample/        data/stageN_train/    models/stageN_adapter  models/stageN_fused
                                                                             ↓
                                                           挂载进 Docker (volume)
                                                                             ↓
                                  ┌────────── 运行时 (功能 1) ──────────┐
                                  │  uvicorn → FastAPI app              │
                                  │     ├── Stage1 VLM (懒加载)          │
                                  │     ├── Stage2 LLM (eager load)     │
                                  │     ├── BGE embedder                │
                                  │     └── ChromaDB                    │
                                  │                                     │
                                  │  POST /extract → VLM → markdown →   │
                                  │                    ChromaDB 入库    │
                                  │  POST /query   → 检索 → LLM → 答案  │
                                  │  POST /admin/reload → 原子换模型     │
                                  └──────────────────────────────────────┘
                                                  ↑
              两条独立前端：                      │
              - curl / test_api.sh ──────────────┤
              - Streamlit UI (功能 12) ─────────→┘  (其实是旁路：直接 import src.*，不走 Docker API)
```

**一句话总结**：这是一个 "本地 MLX 微调 → HF 融合导出 → CPU Docker 推理" 的端到端 pipeline，核心是 `src/serve/api.py` 这个 RAG HTTP 服务；`src/{rag,infer,pdf_utils}.py` 是底层原子能力，`scripts/` 是离线数据/训练/融合工具，`app/streamlit_app.py` 是本地开发用的可视化。

---

## 🟣 Tier 5 — 企业级参考（独立目录，可选学习）

> 这部分**和主项目解耦**，不影响 `src/` 和 Streamlit。把主项目的 MLX 方案替换成企业栈的实现，用于"了解工业界真实做法"。

### 功能 15：Raw HuggingFace 微调（Trainer + PEFT + TRL）⭐

**入口**：
```bash
cd finetuning/huggingface
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_lora.py          # 纯 LoRA
python train_qlora.py         # QLoRA (4-bit)
python train_sft_trl.py       # TRL SFTTrainer（现代推荐）
```

**文件结构**
```
finetuning/huggingface/
├── README.md                  ← 裸 HF 为什么重要 + LLaMA-Factory YAML ↔ HF API 对应表
├── requirements.txt
├── config.py                  ← 共享配置（模型、LoRA 超参、路径）
├── dataset_prep.py            ⭐ HF datasets 全流程: load JSONL → apply_chat_template → tokenize → mask prompt labels
├── train_lora.py              ← 经典: Trainer + LoraConfig + get_peft_model
├── train_qlora.py             ← QLoRA: + BitsAndBytesConfig + prepare_model_for_kbit_training
└── train_sft_trl.py           ← TRL SFTTrainer (LLaMA-Factory 的内核)
```

**调用树（train_lora.py，典型流程）**
```
main()
├── build_tokenizer()                              [dataset_prep.py:42]
│   └── AutoTokenizer.from_pretrained + pad_token 兜底
├── prepare_dataset(tokenizer)                     [dataset_prep.py:107]
│   ├── load_dataset("json", data_files=...)
│   ├── map(alpaca_to_messages)                    ← alpaca 转 chat messages
│   └── map(mask_prompt_in_labels)                 ← 生成 labels 并把 prompt 部分设 -100
├── AutoModelForCausalLM.from_pretrained(...)      ← bf16 + sdpa + trust_remote_code
├── model.gradient_checkpointing_enable()          ← 必须在 get_peft_model 前
├── model.enable_input_require_grads()             ← gradient_checkpointing+LoRA 组合的坑
├── get_peft_model(model, LoraConfig(...))         ⭐ peft 核心调用
│   └── model.print_trainable_parameters()         ← "trainable: 1M / all: 500M" = LoRA 的魔法
├── TrainingArguments(lr=2e-4, bf16=True, ...)     ← 所有超参
├── DataCollatorForSeq2Seq(label_pad_token_id=-100) ← 动态 padding
├── Trainer(model, args, train/eval_dataset, collator)
├── trainer.train()                                ← HF Trainer 内循环
└── trainer.save_model(output_dir)                 ← 只保存 adapter weights
```

**train_qlora.py 相对 train_lora.py 的唯一差别**（文件结构完全对齐，便于对照）
```
+ BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                     bnb_4bit_use_double_quant=True, ...)
+ model = from_pretrained(..., quantization_config=bnb_config, device_map="auto")
+ model = prepare_model_for_kbit_training(model)       ⭐ 必须，否则梯度不回传
+ optim = "paged_adamw_8bit"                           ← QLoRA 原论文建议
+ max_grad_norm = 0.3
```

**train_sft_trl.py 的简化**（LLaMA-Factory 的内核）
```
trl.SFTTrainer 自动做了:
  - apply_chat_template (读 messages 字段)
  - mask prompt labels (SFT 标准做法)
  - DataCollator (合适的 padding + label mask)
  - packing=True (多短样本拼到 max_length, 吞吐 2-3×)
  - 统一接口给 DPO/KTO/ORPO (换 Trainer 类名即可)
```

**读完这一节你应该能回答（JD 面试关键）**
- LoRA 和 QLoRA 的代码差别只有这 4 行，分别在做什么？
- `get_peft_model()` 内部做了什么？为什么要 `prepare_model_for_kbit_training`？
- SFT 为什么要把 prompt 部分的 labels 设 -100？
- `apply_chat_template` 和 `tokenize()` 的分工是什么？
- 为什么 `padding_side` 训练用 right，推理用 left？
- `gradient_checkpointing_enable()` 为什么要在 `get_peft_model()` 之前调？

---

### 功能 16：LLaMA-Factory 微调（替代 MLX）

**入口**：`bash finetuning/llamafactory/scripts/setup.sh`（Linux + NVIDIA GPU）

**文件结构**
```
finetuning/llamafactory/
├── README.md                        ← 跑通步骤 + 专属坑
├── configs/
│   ├── dataset_info.json            ← 注册本项目 data/stage{1,2}_train/*.jsonl
│   ├── stage1_qwen2vl_qlora.yaml    ← VLM QLoRA（含 freeze_vision_tower、visual_inputs）
│   └── stage2_qwen25_qlora.yaml     ← LLM QLoRA（注释详尽，每个字段都有说明）
├── scripts/
│   ├── setup.sh                     ← clone LLaMA-Factory + 独立 venv + 软链 dataset_info
│   ├── train_stage1.sh              ← llamafactory-cli train ../configs/stage1_...
│   ├── train_stage2.sh
│   ├── merge_stage1.sh              ← llamafactory-cli export → models/stage1_fused/
│   └── merge_stage2.sh              ← 同上，输出到 models/stage2_fused/
└── LLaMA-Factory/                   ← setup.sh clone 出来的，gitignore 掉
```

**调用流（Stage 2 最典型）**
```
prepare_stage2.py (项目原有，生成 data/stage2_train/*.jsonl)
    ↓
setup.sh                             [clone + pip install -e ".[torch,metrics,bitsandbytes]"]
    ↓  软链接 configs/dataset_info.json → LLaMA-Factory/data/dataset_info.json
train_stage2.sh
    └── llamafactory-cli train configs/stage2_qwen25_qlora.yaml
        ├── 读 dataset_info.json → 找到 alpaca 格式的 train.jsonl
        ├── 加载 Qwen2.5-0.5B-Instruct + 4-bit 量化（bitsandbytes NF4）
        ├── 挂 LoRA adapter (r=8, alpha=16, target=all)
        ├── HF Trainer 训练 3 epoch
        └── 输出 outputs/stage2_lora/adapter_model.safetensors
    ↓
merge_stage2.sh
    └── llamafactory-cli export (base + adapter → 完整 HF 权重)
        → models/stage2_fused/       ← Docker 自动挂载这里
    ↓
curl POST /admin/reload               ← 通知 Docker 热加载新模型
```

**读完这一节你应该能回答**：LoRA 为什么只训 ~20M 参数？QLoRA 的 NF4 和 Double Quantization 在干啥？显存不够怎么降？

---

### 功能 17：Axolotl 微调（对照学习）

**入口**：`bash finetuning/axolotl/scripts/setup.sh`（Linux + NVIDIA GPU）

**文件结构**
```
finetuning/axolotl/
├── README.md                        ← 与 LLaMA-Factory 的字段对比表
├── configs/
│   ├── stage2_qwen25_qlora.yml      ← QLoRA 版
│   └── stage2_qwen25_lora_full.yml  ← 对照：纯 LoRA 非量化
├── scripts/
│   ├── setup.sh                     ← clone + accelerate config default
│   ├── train_stage2.sh              ← accelerate launch -m axolotl.cli.train
│   └── merge_stage2.sh              ← python -m axolotl.cli.merge_lora
└── axolotl/                         ← setup.sh clone 出来
```

**关键差异（vs LLaMA-Factory）**
- 数据集 inline 在 yaml（`datasets: [{path, type}]`），不用 `dataset_info.json`
- 用 `accelerate launch` 启动（单卡也这样，多卡直接 `--num_processes=N`）
- `adapter: qlora` + `load_in_4bit: true` 是 QLoRA 标志
- `sample_packing: true` 默认开，吞吐更高
- 内置支持 `loraplus_lr_ratio`（LoRA+，比普通 LoRA 快 2×）

**调用流**（和 LLaMA-Factory 几乎一模一样，学习目的就是发现共性）
```
prepare_stage2.py  →  data/stage2_train/train.jsonl
    ↓
setup.sh
    ↓
train_stage2.sh
    └── accelerate launch -m axolotl.cli.train configs/stage2_qwen25_qlora.yml
        → outputs/stage2_qlora/adapter_model.safetensors
    ↓
merge_stage2.sh
    └── python -m axolotl.cli.merge_lora ... → cp 到 models/stage2_fused/
```

---

### 功能 18：Raw HuggingFace 推理（6 个递进示例）⭐

**入口**：
```bash
cd inference/huggingface
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python basic_generate.py       # 最基础
python with_adapter.py         # 挂 adapter / merge / 多 adapter 路由
python streaming.py            # TextIteratorStreamer + Thread
python batched.py              # 手动 batching（演示 vLLM 优化对比）
python pipeline_api.py         # pipeline() 快捷方式
python serve_fastapi.py        # 最小 FastAPI wrapper (:8004)
```

**文件结构**
```
inference/huggingface/
├── README.md                   ← HF 推理核心概念 + 6 个文件导读 + vs vLLM 性能对比
├── requirements.txt
├── basic_generate.py           ⭐ 面试白板题模板 (AutoModelForCausalLM + generate)
├── with_adapter.py             ← PeftModel.from_pretrained / merge_and_unload / set_adapter
├── streaming.py                ← TextIteratorStreamer + Thread (SSE 基础)
├── batched.py                  ← tokenizer(padding=True) + generate (+ 对比 vLLM 解释)
├── pipeline_api.py             ← transformers.pipeline() 快捷方式
└── serve_fastapi.py            ← FastAPI + run_in_executor + MODEL_LOCK 串行保护
```

**调用树（basic_generate.py，最简流程）**
```
main()
├── AutoTokenizer.from_pretrained(MODEL_PATH)
│   ├── 补 pad_token (Qwen/Llama 默认无)
│   └── padding_side = "left"          ← 推理必须左 padding
├── AutoModelForCausalLM.from_pretrained(
│       MODEL_PATH,
│       torch_dtype=torch.bfloat16,    ← ⭐ 不设会 fp32 爆显存
│       device_map="auto",             ← ⭐ 不设模型在 CPU，慢 1000×
│       attn_implementation="sdpa",    ← flash_attention_2 / sdpa / eager
│   )
├── model.eval()                       ← 关 dropout 等
├── tokenizer.apply_chat_template(messages, add_generation_prompt=True)
│                                      ← ⭐ 忘了 add_generation_prompt 模型不生成
├── inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
├── model.generate(
│       **inputs,
│       max_new_tokens=300,
│       do_sample=False,               ← greedy 最稳定，业务对比用
│       use_cache=True,                ← ⭐ 必开，10× 加速
│   )
└── tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:], ...)
                                       ← ⭐ 必须切掉 input 部分
```

**调用树（streaming.py，流式核心）**
```
TextIteratorStreamer(tokenizer, skip_prompt=True)    ⭐ 流式的核心
    ↓
Thread(target=model.generate, kwargs={..., "streamer": streamer}).start()
    ↓                        ↑
    │                        └── 阻塞调用，所以另起线程
    ▼
for token_chunk in streamer:      ← 主线程从队列消费 token
    print(token_chunk, end="")
```

**读完这一节你应该能回答**
- `from_pretrained` 的 3 个最重要参数是什么？不设会出什么问题？
- `apply_chat_template` 的 `add_generation_prompt=True` 起什么作用？
- generate 返回的 `output_ids` 为什么要切 `[input_len:]`？
- 为什么流式输出必须用 Thread？为什么 streamer 要 `skip_prompt=True`？
- transformers 的 batch 和 vLLM 的 continuous batching 差在哪里？
- FastAPI 里 `model.generate()` 为什么要 `run_in_executor` + Lock？

**关键认知**：这 6 个文件加起来 ~400 行代码，是 `src/serve/api.py` 和整个 `inference/*/` 的基石。读懂它就懂了一切的底层。

---

### 功能 19：Ollama 本地推理（Mac 能 run 的企业级参考）

**入口**：
```bash
brew install ollama && ollama serve &
ollama pull qwen2.5:0.5b-instruct-q4_K_M
uv run python inference/ollama/rag_server_ollama.py   # :8001
```

**文件结构**
```
inference/ollama/
├── README.md                        ← 原理 + 和 vLLM 对比 + 5 分钟跑通
├── Modelfile                        ← 自制模型定义（FROM gguf + TEMPLATE + SYSTEM）
├── scripts/
│   └── hf_to_gguf.sh                ← HF 格式 → GGUF 4-bit 量化（用本项目合并后的权重）
├── client_example.py                ← OpenAI SDK 调用 3 种姿势（同步/流式/RAG）
└── rag_server_ollama.py             ← 🎯 核心：企业级 RAG 业务层示例
```

**核心调用树（rag_server_ollama.py，和主项目 api.py 对比）**
```
uvicorn rag_server_ollama:app --port 8001       ← 和主 API :8000 共存

GET /health:
├── openai.OpenAI(base_url=http://localhost:11434/v1).models.list()   ← 查 Ollama 上都有啥模型
├── chroma.count()                                                    ← 共用项目 chroma_db/
└── 返回 {backend: ollama, vector_count, ...}

POST /query(QueryIn):
├── 1. 检索（本进程做，embedder 留在这里）
│   ├── SentenceTransformer.encode([question])
│   └── chroma.query(...) → hits
├── 2. 组 prompt（复用 src.rag.RAG_SYSTEM）
└── 3. 调 Ollama（关键差异）
    └── openai.OpenAI.chat.completions.create(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:0.5b-instruct-q4_K_M",
            messages=[...]
        )
    ← 不再本地 model.generate()！业务层和推理层解耦
```

**这就是企业级「业务 FastAPI + 独立推理 server」模式**：
| | 主项目 src/serve/api.py | 参考版 rag_server_ollama.py |
|---|---|---|
| 模型加载 | 进程内 transformers.from_pretrained | 无，靠 HTTP 调 Ollama |
| 推理 | model.generate() | openai.chat.completions.create() |
| 扩容 | 业务和模型一起扩（浪费） | 业务/推理独立扩 |
| 换模型 | 重启或 /admin/reload | 改环境变量 OLLAMA_MODEL |
| 多模型路由 | 手写 STATE 字典 | 请求时 `model: xxx` 字段切 |

---

### 功能 20：vLLM / TGI 生产推理（Linux GPU，不 run 只读）

**入口**（需要 NVIDIA GPU，Mac 跑不起来）
```bash
cd inference/vllm  &&  docker compose up -d
# 或
cd inference/tgi   &&  docker compose up -d
```

**文件结构**
```
inference/vllm/
├── README.md                        ← Continuous batching / PagedAttention / S-LoRA 讲解
├── docker-compose.yml               ← vllm/vllm-openai:latest 完整配置
└── client_example.py                ← OpenAI SDK 同步/并发 demo（展示 continuous batching 效果）

inference/tgi/
├── README.md                        ← 和 vLLM 对比（维护方、许可证、集成差异）
└── docker-compose.yml               ← HF 官方 TGI 镜像 + shm_size 配置
```

**为什么只读不 run**：vLLM/TGI 需要 NVIDIA CUDA，Mac 上无法启动。但 compose 配置和客户端代码在 Linux 机器上**原样可用**，重点看：
- vLLM 的 `--enable-chunked-prefill`、`--enable-lora` 等参数
- 客户端用 OpenAI SDK 完全无感知切换（Ollama → vLLM → OpenAI 官方，一行 base_url 改完）
- 多 LoRA adapter 动态加载（S-LoRA）

---

### 功能 21：K8s / OpenShift 生产部署（金丝雀 + adapter 安全 ops）

**入口**：`kubectl apply -k inference/kubernetes/base/`（需要 K8s 集群 + GPU 节点）

**文件结构**
```
inference/kubernetes/
├── README.md                           ← 架构图 + 整体导读 + OpenShift 差异速览
│
├── base/                               ← 最小可部署的完整栈（Kustomize）
│   ├── kustomization.yaml              ← apply -k 入口
│   ├── namespace.yaml
│   ├── pvc-models.yaml                 ← 模型 + adapter 分开的 RWX PVC
│   ├── configmap-adapters.yaml         ⭐ adapter 清单（GitOps 的 Source of Truth）
│   ├── deployment-vllm.yaml            ← vLLM Deployment (GPU request, nonroot, probes)
│   ├── service-vllm.yaml               ⭐ ClusterIP only，不对外暴露
│   ├── networkpolicy.yaml              ⭐ 只放 app-layer + adapter-controller + Prometheus
│   ├── hpa.yaml                        ← GPU 利用率驱动扩缩
│   └── pdb.yaml                        ← 防 drain 全灭
│
├── canary/                             ← 金丝雀发布
│   ├── README.md                       ← 蓝绿 vs 金丝雀 vs A/B + 3 种 K8s 实现对比
│   ├── argo-rollouts-vllm.yaml         ← Rollout + AnalysisTemplate (Prometheus 自动分析)
│   └── istio-virtualservice.yaml       ← Mesh 层流量切分（header 路由 + 权重）
│
├── adapter-ops/                        ⭐ 回答「URL 不安全」的正确姿势
│   ├── README.md                       ← 威胁模型 + 3 种生产方案 + 安全自检清单
│   ├── gitops-argocd-app.yaml          ← 方案 A: ArgoCD 监听 Git
│   ├── gitops-sync-job.yaml            ← 方案 A: PostSync Hook Job 调 vLLM 内部 API
│   ├── mediator-admin-api.py           ← 方案 B: 业务层网关（JWT + RBAC + 审计）
│   ├── mediator-deployment.yaml        ← 方案 B: K8s 部署 + OIDC Ingress
│   └── crd-loraadapter.yaml            ← 方案 C: 自定义 CRD（LoRAAdapter）
│
└── openshift/                          ← OpenShift 专属
    ├── README.md                       ← 必改点（SCC、Route）+ 原生工具速查
    ├── scc.yaml                        ← SecurityContextConstraints 配置
    ├── route.yaml                      ← Route + alternateBackends（原生金丝雀）
    └── kserve-inferenceservice.yaml    ← RHOAI + KServe 一个 CR 搞定的玩法
```

**核心架构（企业级部署）**
```
                     ┌───────────────────────────────┐
                     │   Ingress / Route (TLS + OIDC) │
                     └─────────────┬─────────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │   app-layer (多副本 CPU)   │  ← 业务 FastAPI
                     │   - RAG 编排              │     做认证、RBAC、审计
                     │   - OpenAI SDK 调 vLLM    │
                     └───┬───────────────────┬───┘
                         │ ClusterIP          │
                         ▼                   ▼
                  ┌──────────┐      ┌──────────────┐
                  │ Vector DB│      │  vLLM Pods    │
                  │ Qdrant   │      │  (GPU 节点)    │
                  └──────────┘      └──────┬───────┘
                                           │ NetworkPolicy
                                           │ 只许 app-layer +
                                           │     adapter-ctrl +
                                           │     Prometheus 访问
                                           ▼
                                    ┌──────────────┐
                                    │ adapter-ctrl │ ← 方案 B / C
                                    │ (或 Job 方案 A)│
                                    └──────────────┘
                                           │
                                 ┌─────────┴──────────┐
                                 │                    │
                         ┌───────▼──────┐    ┌────────▼──────┐
                         │ ArgoCD Sync  │    │  PVC(RWX)      │
                         │ from Git     │    │ /models        │
                         │              │    │ /adapters      │
                         └──────────────┘    └───────────────┘
```

**读完这一节你应该能回答**：
- 为什么 vLLM Service 必须是 ClusterIP？
- `/v1/load_lora_adapter` URL 直调的 4 个安全漏洞分别是什么？
- GitOps 方案怎么做到"每次 adapter 变更都可审计 + 可回滚"？
- 金丝雀和蓝绿什么时候各选哪个？Argo Rollouts 怎么基于 Prometheus 指标自动回滚？
- OpenShift 上 vLLM 起不来是因为 SCC 拒了什么？KServe 能帮你省什么？

**阅读顺序**：`README.md` → `adapter-ops/README.md`（理解威胁模型最重要）→ `base/` 所有 YAML → `canary/README.md` → `openshift/README.md`

---

### 功能 22 / 23：AWS / Azure 云部署参考

**入口**：`cloud/aws/` 和 `cloud/azure/`（AWS/Azure 账号可选；IaC 当文档读也能学）

**为什么两朵云并列**：JD 里常要求"AWS/Azure 熟悉"，两边概念一一对应，一次学完一举两得。

**文件结构（对称）**
```
cloud/
├── README.md                                   ← managed vs self-host 决策树 + AWS↔Azure 对照
│
├── aws/
│   ├── README.md                               ← 4 种方案对比 + 何时选哪种 + IRSA
│   ├── bedrock-client/                         ← 方案 1: 不 host 自己模型
│   │   ├── README.md
│   │   ├── client.py                           ← Converse/InvokeModel/Stream/LiteLLM 4 种
│   │   ├── rag_with_bedrock.py                 ← 业务层集成示例（配合 ChromaDB）
│   │   └── terraform/                          ← IAM + Guardrails + PrivateLink 注释
│   ├── sagemaker-endpoint/                     ← 方案 2: 托管自家模型
│   │   ├── README.md                           ← DJL-LMI / 金丝雀 / 自动扩缩 / 成本
│   │   ├── client.py                           ← boto3 sagemaker-runtime + 流式
│   │   ├── deploy.sh                           ← 一键: tar → S3 → terraform apply
│   │   └── terraform/                          ← Model + EndpointConfig + Endpoint + AutoScaling
│   ├── ecs-fargate/                            ← 方案 3: 业务层跑 Fargate（GPU 限制）
│   │   ├── README.md
│   │   └── terraform/                          ← VPC + ALB + ECS Service + Task Role
│   └── eks-vllm/                               ← 方案 4: 自建 K8s 栈
│       ├── README.md                           ← 对接 inference/kubernetes/base/ 的改造
│       └── terraform/
│           ├── main.tf                         ← VPC + EKS + GPU Node Group + NVIDIA Device Plugin
│           ├── efs.tf                          ← EFS 共享存储 (对应 K8s RWX PVC)
│           ├── ecr.tf                          ← 镜像仓库
│           ├── irsa.tf                         ⭐ IAM Roles for Service Accounts（零凭证）
│           ├── variables.tf / outputs.tf
│
└── azure/
    ├── README.md                               ← Azure 4 种 + AWS 对应表 + Managed Identity
    ├── openai-service-client/                  ← 方案 1: Azure OpenAI（对应 Bedrock）
    │   ├── README.md                           ← vs OpenAI 官网 / 内容过滤 / PTU
    │   ├── client.py                           ← API Key + Managed Identity + 流式 + embedding
    │   └── bicep/main.bicep                    ← Cognitive Services + Deployment + RBAC + PrivateEndpoint
    ├── aml-endpoint/                           ← 方案 2: Azure ML Online Endpoint（对应 SageMaker）
    │   ├── README.md                           ← AML 四件套 + 和 SageMaker 详细对比 + 金丝雀
    │   ├── endpoint.yaml + deployment.yaml     ← az ml CLI 用
    │   └── bicep/main.bicep                    ← Workspace + Endpoint + Deployment 一次搞定
    ├── container-apps/                         ← 方案 3: Container Apps（对应 Fargate）
    │   ├── README.md                           ← Revisions / 原生金丝雀 / KEDA / Scale-to-zero
    │   └── bicep/main.bicep                    ← Environment + App + ACR Pull + MI + LogAnalytics
    └── aks-vllm/                               ← 方案 4: AKS（对应 EKS）
        ├── README.md                           ← 和 EKS 详细差异 + KAITO Operator + 成本
        └── bicep/main.bicep                    ← AKS + GPU Pool + Azure Files + Workload Identity + ACR
```

**读完你应该能回答（JD 面试关键）**
- Bedrock / Azure OpenAI 什么时候比自 host 便宜？什么时候更贵？
- SageMaker Endpoint 和 AML Online Endpoint 的"模型抽象"有什么不同？
- AWS IRSA 和 Azure Workload Identity 原理是不是一样？怎么配？
- 同一套 `inference/kubernetes/base/` YAML，在 EKS 和 AKS 上各要改哪几个字段？
- 金丝雀发布在 5 个位置（Argo Rollouts / Istio / SageMaker Variants / AML traffic split / Container Apps Revisions）分别怎么做？
- 什么场景该选 Container Apps 的 scale-to-zero？Fargate 为什么做不到？

**实操 vs 学术**：
- 有 AWS / Azure 账号 → 按 `deploy.sh` / `az deployment group create` 跑（记得 `destroy` 别烧钱）
- 没账号 → 读 `main.tf` / `main.bicep` 就能学完 **90% 的云上 LLM 部署知识**，面试足够
- 两边对比着读 —— 概念一一对应，一次学会跨云迁移思维

---

## 🗂️ 全新目录树

```
ocr-fine-app/
├── src/                    ← 主项目核心库（功能 2-7）
├── app/                    ← Streamlit UI（功能 12）
├── scripts/                ← 数据/训练/合并脚本（功能 8-11, 14）
├── notebooks/              ← MLX 微调 notebook（功能 10）
├── config/, data/, models/, chroma_db/, logs/    ← 数据与产物
├── Dockerfile + docker-compose.yml               ← 功能 14
├── PROJECT_MAP.md          ← 本文件
│
├── finetuning/             ← 🟣 企业级微调（功能 15, 16）
│   ├── README.md           ← LoRA/QLoRA 理论 + 框架对比 + 通用坑
│   ├── llamafactory/       ← 功能 15
│   └── axolotl/            ← 功能 16
│
└── inference/              ← 🟣 企业级推理（功能 17, 18）
    ├── README.md           ← 4 种引擎对比 + 业务分离架构讲解
    ├── ollama/             ← 功能 17（Mac 能 run）
    ├── vllm/               ← 功能 18
    └── tgi/                ← 功能 18
```

---

## 🧭 全新阅读路线（学完能上生产那种）

**第 1 周 —— 吃透主项目运行时**
- Day 1-2：功能 1 + 4（Docker 怎么跑起来 / 热加载怎么做原子 swap）
- Day 3：功能 2 + 3（向量检索 + markdown 解析）
- Day 4：功能 5 + 6（VLM 抽表 + LLM 问答的完整链路）
- Day 5：功能 12（Streamlit 对着代码把四个 Tab 跑一遍）

**第 2 周 —— 数据与训练上游**
- Day 6-7：功能 7 + 8 + 9（PDF 处理 + 数据集构建 + 下载）
- Day 8-9：**功能 15** 跟着 LLaMA-Factory README 跑 Stage 2 QLoRA（Colab T4 就够）
- Day 10：功能 11（合并 adapter）+ 用 `/admin/reload` 把新模型热换进 Docker

**第 3 周 —— 企业级栈**
- Day 11：**功能 17** Mac 上跑 Ollama + rag_server_ollama.py，对比主项目 api.py
- Day 12：**功能 16** Axolotl 跑同一任务，对比配置
- Day 13-14：**功能 18** 读 vLLM README + docker-compose，理解 continuous batching / S-LoRA

**第 4 周 —— 改造主项目（可选）**
- 把 `src/serve/api.py` 改成调 Ollama（开发）/ vLLM（生产）HTTP，删掉 `transformers` 依赖
- Docker 镜像从 6GB 减到 <1GB（只剩 FastAPI + embedder + chroma）
- 加 Prometheus metrics + Langfuse trace（你 docker ps 里已经跑着 Langfuse，现成）
