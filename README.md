# OCR-Fine-App · 文档智能 + RAG 问答（MacBook 小内存友好版）

一个端到端实验：用 **QLoRA** 微调小模型，从 PDF 图片抽取表格，再做表格感知问答。
全部设计为 **MacBook 8–16GB** 能跑，Notebook 一步步学，Streamlit 可交互测试。

---

## 🎯 模型与资源占用

| 阶段 | 模型 | 大小 | 用途 | 约需内存 |
|---|---|---|---|---|
| Stage 1 | `Qwen2-VL-2B-Instruct` (4-bit) | ~1.3GB | 图片 → Markdown 表格 | 6–8 GB |
| Stage 2 | `Qwen2.5-0.5B-Instruct` (4-bit) | ~0.4GB | 表格 QA | 2–3 GB |
| Embed   | `bge-small-zh-v1.5` | ~95MB | 向量检索 | < 1GB |

**主框架**：[MLX](https://github.com/ml-explore/mlx) + [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) + [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) — Apple Silicon 原生，统一内存，4-bit QLoRA。

**Colab 备选**：若本地慢，训练可放 Colab（`requirements-colab.txt` + Unsloth，或 `uv sync --extra colab` 装 bitsandbytes/trl）。

---

## 🌍 按地区/行业的模型选型参考

本仓库默认用 Qwen 系列（中国供应商），**这只适合 APAC / 国内团队**。金融等强监管行业在不同区域有不同首选——模型来源、许可、数据主权都会卡 Model Risk Committee。下表供你把本项目 fork 后按本地合规替换 Stage 1/2/Embed：

### APAC（中国大陆 / 港澳 / 东南亚中资）

| 层 | 首选 | 备选 |
|---|---|---|
| LLM (Stage 2) | **Qwen2.5 / Qwen3**、**DeepSeek-V3 / R1**、**GLM-4** | Yi (零一万物)、Baichuan、MiniMax abab |
| VLM (Stage 1) | **Qwen2-VL / Qwen2.5-VL**、**InternVL 2.5**、**MiniCPM-V** | GLM-4V、DeepSeek-VL |
| Embedding | **BGE-M3 / BGE-large-zh**（BAAI）、**Conan-embedding**、**GTE-Qwen2** | M3E、text2vec |
| 部署 | 阿里云 PAI / 百炼、华为 ModelArts、腾讯 TI-ONE、火山引擎 | 本地 vLLM + Ollama |

日本/韩国团队另有 **Stockmark (JP)**、**Rinna Youri (JP)**、**KoAlpaca / SOLAR (KR, Upstage)**；东南亚本地化可看 **SEA-LION (AI Singapore)** 和 **Sahabat-AI (ID)**。

### 美国（US 银行 / 金融机构）

严格遵循 **SR 11-7**、OCC、SEC 的 Model Risk Management；**不允许中国来源模型**，优先选 hyperscaler 「数据留租户」承诺的闭源模型。

| 层 | 首选（闭源，云托管）| 备选（开源权重，私有化）|
|---|---|---|
| LLM (Stage 2) | **Azure OpenAI GPT-4o / 4.1 / o-series**、**AWS Bedrock Claude (Anthropic)**、**Vertex AI Gemini 2.x** | **Llama 3.3 / 3.1**（Meta）、**Phi-4**（微软）、**Gemma 2/3**（Google）、**Granite**（IBM, Apache 2.0，银行最爱）|
| VLM (Stage 1) | **GPT-4o vision**、**Claude 3.5 Sonnet vision**、**Gemini 2.0 Flash** | **Llama 3.2-Vision 11B/90B**、**Pixtral-12B**（Mistral, 走美区合规路径）、**Granite-Vision** |
| Embedding | **OpenAI text-embedding-3-large**、**Cohere Embed v3**、**Voyage-3**（金融专用） | **Snowflake arctic-embed**、**NV-Embed**、**nomic-embed-text-v2** |
| 部署 | Azure OpenAI on Your Data、Bedrock Knowledge Bases、Vertex RAG Engine | SageMaker JumpStart、**vLLM / TGI** on-prem、Databricks Model Serving |

> 真实案例：JPMorgan 自研 **IndexGPT / LLM Suite**（GPT-4 on Azure 私有化）；Morgan Stanley 用 GPT-4；Goldman Sachs 内部 **GS AI Platform** 用 Gemini + 自研。

### 欧洲（UK / EU 银行）

受 **EU AI Act**、**GDPR**、**DORA**、UK **PRA SS1/23** 约束；偏好欧盟原产或可完全本地化的模型。中国来源和部分美国来源都会被数据主权委员会筛查。

| 层 | 首选 | 备选 |
|---|---|---|
| LLM (Stage 2) | **Mistral Large 2 / Small 3**（法国，La Plateforme 或 Azure EU）、**Mixtral 8x22B**、**Llama 3.3** 自托管 | **Aleph Alpha Pharia**（德国，主权 AI）、**Teuken-7B**（德国，24 种欧盟语言）、**EuroLLM**、**BLOOM** |
| VLM (Stage 1) | **Pixtral Large**（Mistral）、**Claude Sonnet** via AWS Bedrock Frankfurt/Stockholm | **Llama 3.2-Vision**、**Qwen2-VL** 仅在完全隔离私有化时偶尔使用 |
| Embedding | **Mistral-embed**、**Jina Embeddings v3**（德国柏林）、**Cohere Embed multilingual**（加拿大，EU region） | **BGE-M3**（可本地跑，MIT 许可；但 BAAI 背景在某些银行仍需审批） |
| 部署 | **Scaleway**（法）、**OVHcloud**（法）、**IONOS AI Model Hub**（德）、Azure/AWS EU sovereign regions | **vLLM / TGI** on-prem (Kubernetes) |

> 真实案例：BNP Paribas 与 Mistral 签战略合作；ING / ABN AMRO 用 Azure OpenAI EU region；Deutsche Bank 用 Google Cloud + Gemini；瑞士银行（UBS / CS 遗产）因银行保密法严格，多用 Azure 瑞士区 + 自托管 Llama。

### 选型速查（按合规硬约束）

| 约束 | 直接排除 | 推荐路径 |
|---|---|---|
| **美国实体 + 涉华制裁敏感** | Qwen、DeepSeek、GLM、Yi、InternVL | Llama + Azure OpenAI / Bedrock |
| **EU AI Act 高风险系统** | 训练数据不透明的美系闭源 | Mistral + Aleph Alpha，或闭源 + 完整审计合同 |
| **GDPR 数据不出境** | 任何非 EU region 的 API | EU sovereign cloud + 开源自托管 |
| **银行保密（瑞士/列支）** | 任何多租户 API | 完全私有化 vLLM + 开源权重 |
| **APAC 内资机构** | 无硬性排除 | Qwen / DeepSeek + 本土云 |
| **跨国集团（多区部署）** | — | 抽象层（LiteLLM / LangChain）+ 按 region 路由到合规模型 |

> 本项目的 `src/serve/api.py` 只要替换 `STAGE1_MODEL_PATH` / `STAGE2_MODEL_PATH` / `EMBED_MODEL` 三个环境变量即可切换模型，微调脚本换 base 即可。RAG 逻辑、热加载、向量库都与具体模型解耦。

### 🔒 Fine-tuning + 私有化部署（重点）

**核心约束**：闭源 API 模型（GPT-4o / Claude / Gemini）**无法真正私有化微调** —— 即便 OpenAI / Bedrock / Vertex 提供 fine-tuning 服务，权重仍托管在供应商云上，训练数据也要上传。只要你的 MRM / DPO 要求「权重+训练数据留在机构内网」，就必须走**开源权重 + 本地 QLoRA/LoRA**。

#### 硬性筛选：哪些模型真能本地微调？

| 类别 | 可私有化微调 | ❌ 不行（闭源） |
|---|---|---|
| LLM | Llama 3.x、Mistral、Mixtral、Qwen2.5/3、DeepSeek、Gemma、Phi、Granite、Pharia、Teuken、SOLAR | GPT-4o、Claude、Gemini |
| VLM | Llama 3.2-Vision、Pixtral、Qwen2-VL、InternVL、MiniCPM-V、Granite-Vision | GPT-4o vision、Claude vision |
| Embedding | BGE、E5、Arctic、NV-Embed、Jina、Nomic、GTE | OpenAI text-embedding-3、Cohere、Voyage |

> 许可证陷阱：**Llama 3.x 的 community license** 对月活 >7 亿的公司要求额外授权（大行一般已通过 Meta 法务）；**Gemma** 禁止某些用途；**Qwen2.5-72B** 商用要申请。银行法务要逐条审。

#### 按地区的私有化微调 stack

| 地区 | Base 模型（微调起点）| 微调框架 | 量化 / 方法 | 部署运行时 | 硬件 |
|---|---|---|---|---|---|
| **APAC** | Qwen2.5-7B/14B、DeepSeek-V2-Lite、GLM-4-9B、InternVL2-8B | **LLaMA-Factory**、MS-Swift、Unsloth、本仓库 MLX | QLoRA 4-bit (NF4)、LoRA、DoRA | vLLM、SGLang、LMDeploy、TGI、Ollama（开发） | A100 80G、H100、国产替代：昇腾 910B、寒武纪 MLU370 |
| **US** | **Llama 3.3-70B / 3.1-8B**、Granite 3.x（IBM, Apache 2.0，银行尤爱）、Phi-4、Gemma 2 | **Axolotl**（行业标准）、TRL、Unsloth、NVIDIA **NeMo Customizer** | QLoRA、**FSDP2 + LoRA**（多卡）、PEFT | vLLM、**NVIDIA NIM**（打包成 container 进 OpenShift）、TGI、TensorRT-LLM | H100、H200、B200；AWS `p5`/`p5e`、Azure ND H100 v5（私有 VPC） |
| **Europe** | **Mistral 7B / Small 3 / Mixtral**、Llama 3.x（走 Meta EU DPA）、**Aleph Alpha Pharia-1-7B**（主权 AI，欧盟原产）、**Teuken-7B**（24 语种 EU 基座） | Axolotl、TRL、**Mistral fine-tuning API**（权重可导出）、Pharia Studio（Aleph Alpha 自家）| QLoRA、LoRA | vLLM、TGI、**IONOS AI Model Hub**、Scaleway Dedibox、OVH AI Endpoints | 本地 H100 集群、OVH `HGR-AI`、Scaleway H100 PCIe；**要求：数据中心在 EU**（Frankfurt / Paris / Gravelines / Zurich）|

#### 跟本仓库的对接方式

本项目已按「**Base + LoRA adapter + fuse → HF 格式**」的解耦流水线设计，换地区只改 3 处：

```python
# src/config.py
STAGE1_VLM_HF  = "Qwen/Qwen2-VL-2B-Instruct"   # APAC
# STAGE1_VLM_HF = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # US
# STAGE1_VLM_HF = "mistralai/Pixtral-12B-2409"               # Europe

STAGE2_LLM_HF  = "Qwen/Qwen2.5-0.5B-Instruct"  # APAC
# STAGE2_LLM_HF = "ibm-granite/granite-3.1-2b-instruct"       # US (Apache 2.0)
# STAGE2_LLM_HF = "mistralai/Mistral-7B-Instruct-v0.3"        # Europe

EMBED_MODEL    = "BAAI/bge-small-zh-v1.5"      # APAC
# EMBED_MODEL  = "Snowflake/snowflake-arctic-embed-l-v2.0"    # US
# EMBED_MODEL  = "jinaai/jina-embeddings-v3"                  # Europe
```

训练管线（LLaMA-Factory / Axolotl）产出 `adapter/` → `scripts/fuse_model.py` 合并回 base → Docker volume 挂 `models/stageX_fused/` → API 以 transformers 加载。整个链路**不依赖任何在线 API**，满足「权重+训练数据完全不出内网」。

#### 审计 / 合规检查清单（银行最常问）

- [ ] Base 模型许可证是否允许商用 + 金融场景？（Llama ✅、Gemma 有限制、Qwen 商用需申请）
- [ ] 训练数据是否全部来自内部 + 签署 DPA 的公开数据？
- [ ] 推理时有无出站流量？（`EMBED_MODEL` 首次下载是唯一的外联点，下载后可离线）
- [ ] 模型权重是否落在加密存储 + 最小权限访问？（本项目 `models/` 挂 `ro` 已符合）
- [ ] 版本可追溯？（`/health.versions` 提供文件指纹，配 Git LFS / DVC 做 adapter 版本化）
- [ ] PII / MNPI 是否在训练前被清洗？（建议在 `scripts/prepare_stage*.py` 加 presidio / 自研 PII 检测）
- [ ] 红队 / 偏见评估报告？（EU AI Act 高风险系统强制，本项目 `src/eval.py` 可扩展加 HELM / TruthfulQA）

---

## 📂 目录结构

```
ocr-fine-app/
├── README.md                    ← 本文件（操作指南）
├── pyproject.toml               ← 依赖定义（uv 管理）
├── requirements-colab.txt       ← Colab 训练依赖（pip）
├── src/                         ← 核心 Python 模块
│   ├── config.py                  全局路径 & 模型名
│   ├── pdf_utils.py               PDF ↔ 图片 & pdfplumber 表格
│   ├── data.py                    数据集构建（sharegpt/alpaca）
│   ├── rag.py                     表格清洗 + ChromaDB + prompt
│   ├── infer.py                   MLX 推理封装（Stage1 / Stage2，本地用）
│   ├── eval.py                    TEDS / EM / F1 / ROUGE-L
│   └── serve/api.py               FastAPI 推理服务（Docker 用）
├── notebooks/
│   ├── 01_explore_data.ipynb      数据探索
│   ├── 02_finetune_stage1.ipynb   VLM 微调
│   ├── 03_finetune_stage2.ipynb   LLM 微调
│   └── 04_end_to_end_rag.ipynb    端到端 demo
├── scripts/
│   ├── download_data.py           一键下载 HF 数据集 + 样例 PDF
│   ├── prepare_stage1.py          构建 Stage1 训练集
│   ├── prepare_stage2.py          构建 Stage2 训练集
│   ├── fuse_model.py              合并 LoRA → HF 格式（部署前跑）
│   ├── test_api.sh                curl 测 Docker API
│   └── make_notebooks.py          生成 notebooks
├── app/
│   └── streamlit_app.py           4-tab UI（dev 用）
├── Dockerfile                    ← 多阶段构建（CPU 推理）
├── docker-compose.yml            ← 容器编排
├── .dockerignore
├── data/                        ← gitignore
│   ├── raw/                       HF 数据集子集
│   ├── samples/                   手工测试 PDF
│   ├── stage1_train/              Stage1 jsonl
│   └── stage2_train/              Stage2 jsonl
└── models/                      ← gitignore
    ├── stage1_adapter/            LoRA 权重
    └── stage2_adapter/
```

---

## 🚀 操作步骤

### Step 0 · 环境初始化（5 分钟）

本项目用 [**uv**](https://github.com/astral-sh/uv) 管理依赖（不用 pip / venv）。

```bash
# 先装 uv（若没装）
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /Users/pc-rn/ws/ai-lab/ocr-fine-app

# 一键同步：读 pyproject.toml，自动建 .venv 并装全部依赖
uv sync

# 验证 MLX
uv run python -c "import mlx.core as mx; print('MLX device:', mx.default_device())"
```

之后所有命令都用 `uv run <cmd>`（自动激活 venv），或手动 `source .venv/bin/activate`。

> ⚠️ **非 Apple Silicon**（Intel Mac / Linux / Windows）：先编辑 `pyproject.toml` 删掉 `mlx*` 三行，再 `uv sync`，训练走 Colab。

### Step 1 · 下载数据（默认 streaming，1–5 分钟）

`scripts/download_data.py` 默认走 **HF streaming 模式**：下载量严格按 `--N` 条数走，不会拉整个 shard。对 PubTables-1M 这种 WebDataset 分片的大数据集尤其有效（非 streaming 哪怕切片 500 条也会拉 2–6 GB 的整 shard）。

```bash
# 默认：streaming · pubtables 500 + fintabnet 300 + comtqa 1000 + 3 个 PDF
uv run python scripts/download_data.py

# 国内镜像 + hf_transfer 并发加速（最快）
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_ENABLE_HF_TRANSFER=1 \
  uv run python scripts/download_data.py

# 只要 PDF 测试数据：
uv run python scripts/download_data.py --skip-hf

# 自定义采样量（0 = 跳过该集；--pubtables 是 --pubtabnet 的兼容别名）
uv run python scripts/download_data.py --pubtabnet 100 --fintabnet 0 --comtqa 500

# 禁用 streaming（整 shard 下载，慢但可 resume）
uv run python scripts/download_data.py --no-stream

# 🧹 清理历史下载（任何 --clean* 都是「只清不下」，完事就退出）
uv run python scripts/download_data.py --clean             # 清 data/raw/*_sample/
uv run python scripts/download_data.py --clean-cache       # 清 data/hf_cache/（整 shard 缓存，可能上 GB）
uv run python scripts/download_data.py --clean-all         # 样本 + cache + PDF 全删
uv run python scripts/download_data.py --clean --dry-run   # 只预览要删什么

# 清理 + 重下：用 --fresh（等价于 clean-all 后再跑一次）
uv run python scripts/download_data.py --fresh
uv run python scripts/download_data.py --fresh --pubtables 100 --fintabnet 0
```

**数据源**（都是已配对好的 parquet，`image` + `html` 直接可用，streaming 按条数拉不翻 shard）：

| 角色 | repo | 默认 N | 字段 |
|---|---|---|---|
| Stage 1 学术表 | `apoidea/pubtabnet-html` | 500 | `image`(PIL) + `html` |
| Stage 1 金融表 | `ds4sd/FinTabNet_OTSL` | 300 | `image`(PIL) + `html` + `otsl` |
| Stage 2 QA 语料 | `ByteDance/ComTQA` | 1000 | `image_name` + `question` + `answer` |

> ⚠️ 老版 `bsmock/pubtables-1m` 已弃用——那是多 shard WebDataset，streaming 会只拉到 XML annotation shard（没图没结构，对 VLM 训练无用）。迁移后 `data/raw/pubtables_sample/` 目录失效，可用 `--clean-all` 一起清掉。

**实际下载量对比**（默认 n=500/300/1000）：

| 模式 | PubTabNet | FinTabNet_OTSL | ComTQA | 总计 |
|---|---|---|---|---|
| streaming（默认）| ~180 MB | ~80 MB | ~5 MB | **~270 MB** |
| `--no-stream` | 2–4 GB | 0.5–1 GB | ~10 MB | **2.5–5 GB** |

**产物**：
- `data/raw/pubtabnet_sample/` — 论文表格 (image + html)
- `data/raw/fintabnet_sample/` — 金融表格 (image + html + OTSL)
- `data/raw/comtqa_sample/` — 表格 QA（注：只有 `image_name`，单独训不了 Stage 2，需和前两个按文件名 join）
- `data/samples/*.pdf` — Apple Q4 2023 / NVIDIA Q2 2024 / IRS Form 1040

> 已存在的 `data/raw/*_sample/` 目录会被跳过；想重下用 `--clean`（或 `--clean-all` 连 HF cache + 旧 pubtables_sample 一起清）。

### Step 2 · 探索数据（notebook 01，15 分钟）

```bash
uv run jupyter lab notebooks/
# 打开 01_explore_data.ipynb，逐个 cell 执行
```

看完你会清楚：
- 每个数据集的 schema
- 如何把 HTML 表格转 Markdown
- pdfplumber 能从哪些页抽出表

### Step 3 · 构建训练集（⚠️ 必须跑完再开 notebook 02/03）

```bash
uv run python scripts/prepare_stage1.py   # → data/stage1_train/{train,val}.jsonl + data/stage1_images/*.png
uv run python scripts/prepare_stage2.py   # → data/stage2_train/{train,val}.jsonl
```

> notebook 02/03 的第一个 cell 有自检逻辑：若 jsonl 不存在会自动帮你补跑 `prepare_stage*.py`；但前提是 `data/raw/*_sample/` 已经下好。

### Step 4 · Stage 1 VLM 微调（notebook 02，20–40 分钟）

打开 `02_finetune_stage1.ipynb`，第 2 个 code cell 会产出 `data/stage1_mlx/`。
然后在终端跑：

```bash
uv run python -m mlx_vlm.lora \
    --model-path mlx-community/Qwen2-VL-2B-Instruct-4bit \
    --dataset data/stage1_mlx \
    --iters 300 \
    --batch-size 1 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --learning-rate 1e-4 \
    --output-path models/stage1_adapter
```

> 新版 `mlx_vlm` (`>=0.3`) CLI 改过：`--model-path` / `--dataset` / `--lora-rank`；无需再加 `--train`（给了 `--dataset` 就默认训练）。
>
> ⚠️ `--output-path` vs `--adapter-path`：前者是**保存**路径（首次训练用这个），后者是**恢复**路径（从已有 LoRA 接着训）。首次就上 `--adapter-path` 会被当作 resume，目录不存在就报 `FileNotFoundError`。接着训则两个都填（都指同一目录）。
>
> 想连 vision encoder 一起训：加 `--train-vision`（更吃内存）；只对 assistant 段算 loss 更稳：加 `--train-on-completions`。
>
> 显存不够 → `--batch-size 1 --lora-rank 4`；继续 OOM 换更小模型（如 `mlx-community/SmolVLM-Instruct-4bit`）。

**回到 notebook 跑最后一个 cell 做微调前/后对比。**

### Step 5 · Stage 2 LLM 微调（notebook 03，10–20 分钟）

```bash
uv run python -m mlx_lm.lora \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --train \
    --data data/stage2_mlx \
    --iters 600 \
    --batch-size 2 \
    --num-layers 8 \
    --learning-rate 2e-4 \
    --adapter-path models/stage2_adapter
```

### Step 6 · 端到端 Demo（notebook 04）

跑完 `04_end_to_end_rag.ipynb`，验证：PDF → 抽表 → 入库 → 检索 → QA 整链。

### Step 7 · 启动 Streamlit UI

```bash
uv run streamlit run app/streamlit_app.py
```

默认 http://localhost:8501。4 个 tab：

| Tab | 功能 |
|---|---|
| 上传与抽取 | 上传 PDF 或选内置样例，pdfplumber + VLM 抽表，向量库入库 |
| 表格查看 | 查看本次抽到的所有表（Markdown + DataFrame） |
| RAG 问答 | 输入问题，显示 Top-K 检索 + LLM 答案 + 延迟 |
| 微调前后对比 | 同一问题下 base vs LoRA 两栏输出 |

---

## 🐛 常见问题

### MLX 安装失败
- Apple Silicon (M1/M2/M3/M4) 才支持。Intel Mac / Linux 请用 `requirements-colab.txt` 走 Colab。
- Python 必须 3.9+（推荐 3.10/3.11）。

### OOM（内存不足）
- Stage 1：`--lora-layers 4`，换 `SmolVLM-500M-Instruct`。
- Stage 2：`--batch-size 1`。
- 关闭其他 app，macOS 活动监视器看「内存压力」。

### HuggingFace 下载慢
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### pdfplumber 抽不到表
扫描 PDF → 开 Streamlit 侧栏「启用 VLM 抽表」，由 Stage1 模型从图片里认。

### Streamlit 推理第一次很慢
模型冷加载要 10–30s。`src/infer.py` 用了 `lru_cache`，后续请求会快。

### Colab 训练（两个现成 notebook）

本仓库提供了 ready-to-run 的 Colab notebook，用 **Unsloth** 跑 4-bit QLoRA，支持 HF token 登录。点击徽章直接打开：

| Notebook | 训练任务 | 预计时长（T4）| 一键打开 |
|---|---|---|---|
| `notebooks/02b_stage1_colab.ipynb` | Qwen2-VL-2B VLM 微调 | 5–15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smile-rr/ocr-fine-app/blob/main/notebooks/02b_stage1_colab.ipynb) |
| `notebooks/03b_stage2_colab.ipynb` | Qwen2.5-0.5B LLM 微调 | 3–8 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smile-rr/ocr-fine-app/blob/main/notebooks/03b_stage2_colab.ipynb) |

> 徽章默认指向 `smile-rr/ocr-fine-app` 的 `main` 分支。repo 名不一样就改 URL 里的 `smile-rr/ocr-fine-app`（两个 notebook 顶部 + 这个表格都有）。私有 repo 需要用 `https://colab.research.google.com/github.com/...`（域名带 `.com`）并先在 Colab 里授权 GitHub。

**完整流程**：

```bash
# 1) 本地：打包数据（需要先跑过 prepare_stage*.py）
bash scripts/pack_for_colab.sh           # 产出 stage1_colab.zip + stage2_colab.zip
# 或单独：bash scripts/pack_for_colab.sh stage1 / stage2
```

```
# 2) Colab：
#   a. File → Upload notebook → 选 notebooks/02b_stage1_colab.ipynb
#   b. Runtime → Change runtime type → T4 GPU
#   c. （可选）左侧 🔑 图标 → 加 Colab Secret：Name=HF_TOKEN Value=hf_xxxxx
#   d. Run all；第 4 cell 会弹文件选择，上传 stage1_colab.zip
#   e. 训练完成后最后一 cell 触发 files.download('stage1_adapter.zip')
```

```bash
# 3) 本地：解压 adapter + 合并 + 热加载
unzip stage1_adapter.zip -d models/      # → models/stage1_adapter/
# 合并到 base（改用 transformers + peft 路径，因为 Colab 产出 HF 格式）
# 参考 02b notebook 末尾的 "🔁 回到本地部署" 段
```

**HF 登录**的两种方式（notebook 里都支持，自动选一个）：
- **Colab Secret（推荐）**：左侧 🔑 图标 → Add new secret → Name `HF_TOKEN` / Value 你的 `hf_xxxxx` → 勾「Notebook access」。之后每次打开这个 notebook 自动用，不用重输。
- **手动输入**：没设 Secret 就 fallback 到 `getpass` 提示，运行时粘贴 token 即可。

**Unsloth 优势**：
- VLM 和 LLM 都官方支持 Qwen2 系
- 2× 训练速度 + 30% VRAM 节省
- 直接存 HF 格式的 adapter，和 Docker 服务端的 transformers 无缝衔接

---

## 🐳 Docker 部署（infra 要求）

### 设计说明

| 环节 | 选型 | 原因 |
|---|---|---|
| 训练 | MLX（Mac 本地）| Apple Silicon 最快、4-bit QLoRA |
| 推理后端 | **transformers + FastAPI** | 跨平台，Docker 能跑 |
| LoRA 处理 | 训练后 **fuse 回 base**，导出 HF 格式 | 容器里不依赖 MLX |
| 模型分发 | **volume 挂载**（不打进 image）| 镜像小，模型可替换 |
| API | REST：`/health` `/extract` `/query` `/ingest_markdown` |  |

### Step A · 合并 LoRA 到 base（在 Mac 本地跑）

```bash
# 把 models/stageX_adapter 和 base 合并，产出 HF 格式到 models/stageX_fused/
uv run python scripts/fuse_model.py --stage all
# 或单独跑：--stage 1 / --stage 2
```

产物：
```
models/
├── stage1_fused/         # Qwen2-VL-2B + LoRA 合并后（~4GB fp16）
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json ...
└── stage2_fused/         # Qwen2.5-0.5B + LoRA 合并后（~1GB fp16）
```

### Step B · 构建镜像

```bash
# 构建（Mac 上可构跨架构，生产常见 linux/amd64）
docker build -t ocr-fine-app:latest .

# 或者 buildx 多平台：
# docker buildx build --platform linux/amd64 -t ocr-fine-app:latest --load .
```

镜像大小约 **2.5–3GB**（不含模型）。

### Step C · 启动服务

**推荐用 compose**（模型 volume 挂载 + ChromaDB 持久化）：

```bash
docker compose up -d
docker compose logs -f api   # 看启动日志
docker compose ps
```

只想手工跑：
```bash
docker run -d --name ocr-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v ocr_chroma:/app/chroma_db \
  -e DEVICE=cpu \
  -e ENABLE_STAGE1=1 \
  ocr-fine-app:latest
```

首次启动会：
1. 加载 `stage2_fused` LLM（~10s，CPU）
2. 加载 `bge-small-zh-v1.5`（首次下载 ~95MB）
3. Stage1 VLM 懒加载，首次 `/extract` 再加载

### Step D · 测试

```bash
bash scripts/test_api.sh
# 或：
curl http://localhost:8000/health
```

`test_api.sh` 会依次打 `/health` → `/extract`（若有 sample PDF）→ `/ingest_markdown` → `/query` → `/admin/reload`，端到端冒烟。

**端点速查**：

| Method | Path | 用途 |
|---|---|---|
| GET | `/health` | 状态/模型加载情况/向量数/版本指纹/loaded_at/device/auto_reload |
| POST | `/extract` | 上传 PDF/图片，VLM 抽表 + 自动入库 |
| POST | `/ingest_markdown` | 直接入一段 Markdown（跳过 VLM） |
| POST | `/query` | RAG 问答，返回答案 + 来源 + 延迟 |
| POST | `/admin/reload` | 热加载 Stage1/2（需 `X-Admin-Key`） |
| POST | `/admin/unload` | 卸载某个 stage 释放内存（需 `X-Admin-Key`） |

**示例**（端到端）：

```bash
# 1) 上传财报
curl -X POST http://localhost:8000/extract \
  -F "file=@data/samples/apple_2023_q4.pdf" \
  -F "doc_id=apple_q4"

# 2) 问答
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"iPhone 营收是多少？","doc_filter":"apple_q4"}'
```

### 🔁 模型热加载（Hot Reload）

重新训练完模型后，**不停机**替换运行中的 adapter。服务端实现了 3 条路径。

> `ADMIN_API_KEY` 在 `docker-compose.yml` 默认为 `change-me-in-prod`，生产请改；若为空则 admin 端点放行（dev 友好）。

#### 1. 手动触发（API 端点）

```bash
# 默认路径 + 仅在 fingerprint 变化时加载
curl -X POST http://localhost:8000/admin/reload \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"stage":2}'

# 强制重载（即使 fingerprint 没变）
curl -X POST http://localhost:8000/admin/reload \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"stage":2,"force":true}'

# 指定别的路径（A/B 切换）
curl -X POST http://localhost:8000/admin/reload \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -d '{"stage":2,"path":"/app/models/stage2_fused_v2"}'
```

返回示例：
```json
{
  "status": "reloaded",
  "stage": 2,
  "old_version": "a1b2c3d4e5f6",
  "new_version": "f6e5d4c3b2a1",
  "elapsed_s": 3.4
}
```

#### 2. 自动监听（watchdog）

docker-compose 里 `AUTO_RELOAD=1` 会监听 `/app/models` 目录，检测到文件变化（防抖 2s）自动 reload。工作流：

```bash
# 1) 在 Mac 上重新训练 + 合并
uv run python scripts/fuse_model.py --stage 2

# 2) 把新模型 rsync 到容器挂载目录（就地覆盖）
#    compose 已把 ./models 挂成 /app/models:ro
#    → 写文件到 ./models/stage2_fused/ 即可被容器看见
#    容器内 watchdog 检测到 mtime 变化 → 3s 后自动 reload

# 3) 看日志确认
docker compose logs -f api | grep hot-reload
```

#### 3. 卸载释放内存

```bash
curl -X POST "http://localhost:8000/admin/unload?stage=1" \
  -H "X-Admin-Key: $ADMIN_API_KEY"
```

#### 原子性保证

- 所有 STATE 读写用 `threading.RLock` 保护
- 新模型**先完整加载成功**，再在锁内原子替换引用（old → new）
- 正在处理的请求持有旧模型引用，不受影响
- 旧模型引用计数归零后由 GC 自动释放（`torch.cuda.empty_cache()` 兜底）
- **内存峰值 ≈ 2× 模型大小**（0.5B：约 2GB；2B VLM：约 8GB，生产需注意）

#### 版本指纹

`/health` 返回当前每个 stage 的 `versions`（基于文件 mtime+size 的 SHA1），CD pipeline 可用来验证新版本已生效：

```bash
curl -s http://localhost:8000/health | jq '.versions'
# {"stage1": "a1b2c3...", "stage2": "f6e5d4..."}
```

#### 生产强化方案（Blue-Green）

单进程原子替换解决了 99% 问题，但模型 > 5GB 时 2× 内存吃不消。此时用 **双容器 + 反向代理**：

```
Traefik/Nginx
    ├──→ ocr-api-blue   (v1, 承载所有流量)
    └──→ ocr-api-green  (v2, 预热中 → 就绪后切过来 → blue 停)
```

- docker-compose 扩展两份 service，健康检查通过后切路由
- K8s 场景直接用 Deployment + rolling update，`readinessProbe` 探 `/health`

---

### 生产建议（不在 MVP 范围，但要知道）

- **GPU**：把 `Dockerfile` 第一行换 `FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04`，装 `torch` CUDA 版，compose 加 `deploy.resources.reservations.devices`。
- **vLLM**：0.5B 用 vLLM OpenAI-compatible server 替代 transformers，吞吐 10×。
- **Kubernetes**：镜像无状态，可直接 Deployment + PVC（挂 `models/` 和 `chroma_db`）。
- **模型热替换**：把 `models/` 做成独立 image 或 init container 拉下来，API 用 SIGHUP reload。
- **认证**：当前无 auth，生产加 API key / OAuth2 middleware。

### 常见坑

- `ENABLE_STAGE1=1` 时内存峰值 ~6GB；若只做 QA 可关掉它（设 `0`）。
- ARM Mac 本机跑 `docker run` 是 arm64 容器，**没问题**；但推 prod (amd64) 需 `buildx`。
- 首次 compose up 要 `docker compose up --build`（否则不会重建）。
- 模型没合并前 API 启动会 warning 而非 crash；可先用 `/ingest_markdown` + `/query` 做纯 RAG 测试。

---

## 📊 评估

`src/eval.py` 提供：

- `teds(pred_md, gold_md)` — 表格结构相似度 [0,1]
- `cell_f1(pred_md, gold_md)` — 单元格级 P/R/F1
- `exact_match / token_f1 / rouge_l(pred, gold)` — QA 指标

notebook 03 最后一个 cell 演示了对 val 集批量评估。

---

## 🧪 手工测试数据

`data/samples/` 下自动下载 3 份公开 PDF：

| 文件 | 领域 | 表格难度 |
|---|---|---|
| apple_2023_q4.pdf | 财报 | ★★★（多级表头） |
| nvidia_2024_q2.pdf | 财报 | ★★ |
| irs_form_1040.pdf | 政府表单 | ★（规则网格） |

你可以把自己的 PDF 扔进 `data/samples/`，Streamlit 侧栏会自动识别。

---

## 🔗 参考

- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) — Apple Silicon VLM 微调
- [MLX-LM LoRA](https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md)
- [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) / [MLX 4bit](https://huggingface.co/mlx-community/Qwen2-VL-2B-Instruct-4bit)
- [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m) / [FinTabNet.c](https://huggingface.co/datasets/bsmock/FinTabNet.c) / [ComTQA](https://huggingface.co/datasets/ByteDance/ComTQA)
- [Unsloth Colab](https://github.com/unslothai/unsloth) — Colab 训练备选
