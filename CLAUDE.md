# CLAUDE.md — QLoRA Document Intelligence + RAG Q&A 实验

## 项目概述

构建一个端到端的文档智能系统，分两个阶段：

- **Stage 1**：QLoRA 微调多模态视觉模型（Qwen2-VL-7B），从 PDF/图片中提取结构化表格
- **Stage 2**：QLoRA 微调小型 LLM（Qwen2.5-7B），结合 RAG 做表格感知问答
- **部署**：全部本地运行，Ollama / vLLM 提供推理服务

---

## 目录结构（目标）

```
project/
├── CLAUDE.md                    # 本文件
├── README.md                    # 自动生成
├── requirements.txt
├── config/
│   ├── stage1_lora.yaml         # LLaMA-Factory Stage 1 配置
│   ├── stage2_lora.yaml         # LLaMA-Factory Stage 2 配置
│   └── rag_config.yaml          # RAG 参数配置
├── data/
│   ├── raw/                     # 原始 PDF / 图片
│   ├── stage1_train/            # Stage 1 训练集 (image + conversation JSON)
│   ├── stage1_eval/             # Stage 1 评估集
│   ├── stage2_train/            # Stage 2 QA 训练集
│   └── stage2_eval/             # Stage 2 评估集
├── src/
│   ├── data_pipeline/
│   │   ├── pdf_extractor.py     # PDF → 图片切片
│   │   ├── table_parser.py      # pdfplumber 提取真实表格作为 ground truth
│   │   ├── dataset_builder.py   # 构建 Stage 1 / Stage 2 数据集
│   │   └── qa_generator.py      # 用 LLM API 自动生成 QA 对
│   ├── structured/
│   │   ├── table_cleaner.py     # Pandas 表格清洗 / 校验
│   │   ├── embedder.py          # BGE-M3 向量化
│   │   └── vector_store.py      # ChromaDB 增删查
│   ├── rag/
│   │   ├── retriever.py         # Top-K 相似度召回
│   │   ├── prompt_builder.py    # 组装 RAG prompt
│   │   └── pipeline.py          # 端到端 RAG 推理
│   ├── eval/
│   │   ├── teds_score.py        # TEDS 表格提取评估
│   │   ├── qa_metrics.py        # EM / F1 / ROUGE 问答评估
│   │   └── rag_eval.py          # RAGAs Recall@K
│   └── serve/
│       ├── api_server.py        # FastAPI 推理接口
│       └── Modelfile            # Ollama 模型配置
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_stage1_training.ipynb
│   ├── 03_stage2_training.ipynb
│   └── 04_end_to_end_demo.ipynb
├── scripts/
│   ├── download_data.sh         # 下载公开数据集
│   ├── train_stage1.sh          # 启动 Stage 1 训练
│   ├── train_stage2.sh          # 启动 Stage 2 训练
│   ├── merge_lora.sh            # 合并 LoRA 权重
│   └── run_eval.sh              # 跑全套评估
└── tests/
    ├── test_pdf_extractor.py
    ├── test_table_cleaner.py
    ├── test_retriever.py
    └── test_api_server.py
```

---

## 依赖与环境

### Python 依赖（requirements.txt）

```
# Core ML
torch>=2.1.0
transformers>=4.45.0
peft>=0.13.0
bitsandbytes>=0.43.0
accelerate>=0.34.0
trl>=0.11.0

# Vision & Document Processing
pymupdf>=1.24.0         # fitz，PDF 转图片
pdfplumber>=0.11.0      # 表格结构提取
Pillow>=10.0.0
pytesseract>=0.3.13     # OCR 备用

# Data & Structured Processing
pandas>=2.1.0
duckdb>=1.0.0
jsonlines>=4.0.0

# Embeddings & Vector Store
sentence-transformers>=3.0.0    # BGE-M3
chromadb>=0.5.0

# RAG & Evaluation
ragas>=0.1.0
rouge-score>=0.1.2
editdistance>=0.8.0     # TEDS 计算

# Serving
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0

# Training Framework (外部安装)
# LLaMA-Factory: git clone https://github.com/hiyouga/LLaMA-Factory
```

### 系统要求

- GPU: 至少 1x RTX 3090 (24GB) 或 A100 40GB（推荐）
- Stage 1 (7B VLM): 需 ~20GB VRAM（4-bit QLoRA）
- Stage 2 (7B LLM): 需 ~16GB VRAM（4-bit QLoRA）
- 硬盘: 至少 200GB（模型权重 + 数据集）
- Python: 3.10+，CUDA 12.1+

---

## 实现任务清单

> Claude Code 请按照下面的 Task 顺序逐步实现，每完成一个模块运行对应测试后再继续。

---

### Task 0：环境初始化

**文件**：`scripts/setup_env.sh`

实现内容：
1. 创建 Python venv 并安装 `requirements.txt`
2. Clone LLaMA-Factory 到 `./LLaMA-Factory/` 并执行 `pip install -e ".[torch,metrics]"`
3. 检测 GPU 可用性，打印 CUDA 版本和 VRAM 信息
4. 下载 BGE-M3 模型到 `./models/bge-m3/`

```bash
#!/bin/bash
set -e

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory ./LLaMA-Factory
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')"

python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-m3', cache_folder='./models/bge-m3')
print('BGE-M3 downloaded.')
"
```

---

### Task 1：数据下载脚本

**文件**：`scripts/download_data.sh`

实现以下数据集下载：

1. **PubTables-1M（子集）**：从 HuggingFace 下载 5000 条训练样本
   ```python
   from datasets import load_dataset
   ds = load_dataset("bsmock/pubtables-1m", split="train[:5000]")
   ds.save_to_disk("./data/raw/pubtables_5k")
   ```

2. **FinTabNet（可选）**：如果 HuggingFace 有镜像则下载，否则打印手动下载指引

3. **示例 PDF**：下载 5 份公开金融报告 PDF 用于自定义数据集构建
   - 来源：中国上市公司年报（SEC EDGAR 或 CNINFO 公开文件）
   - 保存到 `./data/raw/sample_pdfs/`

---

### Task 2：PDF 预处理模块

**文件**：`src/data_pipeline/pdf_extractor.py`

```python
class PDFExtractor:
    """
    将 PDF 文件转换为：
    1. 高分辨率页面图片（用于 VLM 输入）
    2. pdfplumber 提取的结构化表格（用于 ground truth）
    """

    def extract_pages_as_images(
        self,
        pdf_path: str,
        output_dir: str,
        dpi: int = 150,          # 150 DPI 对 7B VLM 足够，更高增加计算量
        max_size: tuple = (1344, 1344)  # Qwen2-VL 最优输入尺寸
    ) -> list[str]:
        """
        用 pymupdf (fitz) 将 PDF 每页渲染为 PNG。
        返回图片路径列表。
        注意：对超过 max_size 的图片做等比例缩放。
        """
        ...

    def extract_tables_ground_truth(
        self,
        pdf_path: str
    ) -> list[dict]:
        """
        用 pdfplumber 提取结构化表格作为 ground truth。
        返回格式：
        [
          {
            "page": 1,
            "table_index": 0,
            "headers": ["列1", "列2", ...],
            "rows": [["v1", "v2"], ...],
            "markdown": "| 列1 | 列2 |\n|---|---|\n| v1 | v2 |"
          },
          ...
        ]
        对提取失败的页面记录 warning 日志而不是抛出异常。
        """
        ...

    def align_images_with_tables(
        self,
        images: list[str],
        tables: list[dict]
    ) -> list[dict]:
        """
        将图片与对应页面的表格 ground truth 对齐。
        一张图片可能对应多个表格（多个表格合并为一个 markdown 块）。
        """
        ...
```

**测试**：`tests/test_pdf_extractor.py`
- 用 `tests/fixtures/sample.pdf`（一份 3 页含表格的测试 PDF）跑单元测试
- 验证：提取图片数量 == PDF 页数，表格 markdown 格式正确

---

### Task 3：Stage 1 训练数据集构建

**文件**：`src/data_pipeline/dataset_builder.py`

实现 `DatasetBuilder.build_stage1_dataset()`：

```python
def build_stage1_dataset(
    self,
    sources: list[str],       # PDF 路径列表 或 PubTables 数据集路径
    output_path: str,
    max_samples: int = 5000,
    val_ratio: float = 0.1
) -> dict:
    """
    构建 Stage 1 训练集，输出 LLaMA-Factory 兼容的 JSON 格式。

    每条样本格式（sharegpt 格式）：
    {
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image", "image": "<base64 or path>"},
            {"type": "text",  "text": "请提取图中所有表格，以标准 Markdown 格式输出。如无表格则输出 '无表格'。"}
          ]
        },
        {
          "role": "assistant",
          "content": "| 年份 | 营收(亿元) | 净利润(亿元) |\n|---|---|---|\n| 2023 | 120.5 | 18.3 |"
        }
      ]
    }

    注意：
    - 图片路径使用相对路径（相对于 data/ 目录）
    - 对 PubTables 数据集，image 字段直接使用 HuggingFace 的 PIL Image 序列化
    - 过滤掉 ground truth 表格为空的样本
    - 按 8:1:1 分割 train / val / test
    - 保存为 data/stage1_train/train.json, val.json, test.json
    """
    ...
```

同时实现 `DatasetBuilder.build_stage2_dataset()`：

```python
def build_stage2_dataset(
    self,
    tables: list[dict],       # Stage 1 提取的表格结果 或 ground truth 表格
    output_path: str,
    qa_per_table: int = 5,    # 每张表格生成 5 个 QA 对
    use_llm_api: bool = True  # 是否调用 API 生成，否则使用模板
) -> None:
    """
    构建 Stage 2 问答训练集。

    如果 use_llm_api=True，调用 qa_generator.py 中的 generate_qa_pairs()。
    如果 use_llm_api=False，使用规则模板生成（适合无 API 的离线环境）：
      - 模板问题类型：
        * "表格中[列名]最大/最小的是哪一行？"
        * "[行名]对应的[列名]是多少？"
        * "哪一年的[指标]同比增长最大？"

    输出格式（alpaca 格式，适合 LLaMA-Factory）：
    {
      "instruction": "基于以下表格数据回答问题，引用具体数值。",
      "input": "表格：\\n| 年份 | 营收 |\\n|---|---|\\n| 2022 | 100 |\\n| 2023 | 120 |\\n\\n问题：2023年营收是多少？",
      "output": "根据表格数据，2023年营收为120（单位与原表格一致）。数据来源：第2行。"
    }
    """
    ...
```

---

### Task 4：QA 自动生成器

**文件**：`src/data_pipeline/qa_generator.py`

```python
class QAGenerator:
    """
    调用 LLM API 为每张表格生成多样化 QA 对。
    支持：OpenAI API、DashScope（通义千问）、本地 Ollama
    """

    SYSTEM_PROMPT = """你是一个数据分析专家，负责基于表格生成高质量问答对。
每个问题必须：
1. 答案明确存在于表格中（不要推测表格外的信息）
2. 覆盖不同问题类型：数值查询、比较分析、趋势判断、条件筛选
3. 答案包含具体数值和来源行列信息

输出严格的 JSON 数组，不含 markdown 代码块：
[{"question": "...", "answer": "..."}, ...]"""

    def generate_qa_pairs(
        self,
        table_markdown: str,
        n: int = 5,
        provider: str = "openai"   # "openai" | "dashscope" | "ollama"
    ) -> list[dict]:
        """
        调用 LLM API 生成 n 个 QA 对。
        捕获 API 错误，失败时 fallback 到模板生成。
        返回 [{"question": str, "answer": str}, ...]
        """
        ...
```

---

### Task 5：LLaMA-Factory 训练配置

**文件**：`config/stage1_lora.yaml`

```yaml
### Stage 1: QLoRA VLM 表格提取训练配置
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct   # 需提前 huggingface-cli 下载
trust_remote_code: true

### 数据集
dataset: stage1_table_extraction                  # 在 LLaMA-Factory/data/dataset_info.json 中注册
dataset_dir: ../data/stage1_train
template: qwen2_vl
cutoff_len: 4096

### QLoRA 参数
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj

### 量化（4-bit NF4，节省显存）
quantization_bit: 4
quantization_type: nf4
double_quantization: true

### 训练超参
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8     # 等效 batch_size = 8
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
max_grad_norm: 1.0

### 精度与优化
bf16: true                          # A100/H100 用 bf16，3090 改为 fp16
optim: adamw_torch
flash_attn: fa2                     # 需 pip install flash-attn

### 输出
output_dir: ./outputs/stage1_lora
logging_steps: 10
eval_steps: 200
save_steps: 500
save_total_limit: 3

### 评估
val_size: 0.05
evaluation_strategy: steps
```

**文件**：`config/stage2_lora.yaml`

```yaml
### Stage 2: QLoRA LLM 表格问答训练配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct

dataset: stage2_table_qa
dataset_dir: ../data/stage2_train
template: qwen
cutoff_len: 2048

finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: q_proj,v_proj

quantization_bit: 4
quantization_type: nf4
double_quantization: true

num_train_epochs: 2
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
optim: adamw_torch
flash_attn: fa2

output_dir: ./outputs/stage2_lora
logging_steps: 10
eval_steps: 100
save_steps: 300
save_total_limit: 3
val_size: 0.05
evaluation_strategy: steps
```

---

### Task 6：结构化数据处理模块

**文件**：`src/structured/table_cleaner.py`

```python
class TableCleaner:
    """
    清洗和规范化从 Stage 1 提取的 Markdown 表格。
    """

    def parse_markdown_table(self, markdown: str) -> pd.DataFrame:
        """
        将 Markdown 表格字符串解析为 DataFrame。
        处理边缘情况：
        - 列数不一致的行（丢弃或填充 NaN）
        - 包含换行符的单元格
        - 中文全角字符的数字（"１２３" → 123）
        返回清洗后的 DataFrame。
        """
        ...

    def normalize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测并转换数值列：
        - 移除千位分隔符（1,234 → 1234）
        - 处理百分比（12.3% → 0.123）
        - 处理单位标注（120亿 → 12000000000）
        - 无法转换的列保持原始字符串类型
        """
        ...

    def validate_and_repair(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基础校验：
        - 过滤全空行
        - 去除重复标题行（有些 VLM 会重复输出表头）
        - 对缺失值填充 "N/A"
        """
        ...
```

**文件**：`src/structured/vector_store.py`

```python
class TableVectorStore:
    """
    将清洗后的表格行 chunk 化并存入 ChromaDB。
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.encoder = SentenceTransformer("BAAI/bge-m3", cache_folder="./models/bge-m3")
        self.collection = self.client.get_or_create_collection(
            name="table_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def chunk_table(self, df: pd.DataFrame, doc_id: str, page: int) -> list[dict]:
        """
        将 DataFrame 转换为语义 chunks，每行一个 chunk。
        格式："[doc_id | page X | row Y] 列名1: 值1 | 列名2: 值2 | ..."
        返回包含 text、metadata 的字典列表。
        """
        ...

    def add_table(self, df: pd.DataFrame, doc_id: str, page: int) -> int:
        """
        向量化并存入 ChromaDB。返回新增 chunk 数量。
        使用 BGE-M3 的 query_instruction_for_retrieval 前缀以提升召回率。
        """
        ...

    def search(self, query: str, top_k: int = 5, doc_filter: str = None) -> list[dict]:
        """
        语义检索，返回 top_k 个相关 chunks。
        支持按 doc_id 过滤（doc_filter 参数）。
        返回 [{"text": ..., "score": ..., "metadata": {...}}, ...]
        """
        ...
```

---

### Task 7：RAG 推理 Pipeline

**文件**：`src/rag/pipeline.py`

```python
class TableRAGPipeline:
    """
    完整的 RAG 推理流水线：
    Query → 检索 → Prompt 组装 → LLM 推理 → 结构化答案
    """

    SYSTEM_PROMPT = """你是一个精确的数据分析助手。
规则：
1. 只基于给定的表格数据回答，不要引入外部知识
2. 答案必须包含具体数值，并注明数据来源（哪个文档、第几页、哪一行）
3. 如果数据不足以回答问题，直接说明"数据不足"，不要猜测
4. 涉及计算时展示计算过程"""

    def __init__(
        self,
        vector_store: TableVectorStore,
        llm_base_url: str = "http://localhost:11434/v1",   # Ollama OpenAI 兼容接口
        model_name: str = "table-qa",
        top_k: int = 5
    ):
        ...

    def run(self, query: str, doc_filter: str = None) -> dict:
        """
        完整推理，返回：
        {
          "query": str,
          "answer": str,
          "sources": [{"doc_id": str, "page": int, "chunk": str, "score": float}],
          "latency_ms": float
        }
        """
        ...

    def build_prompt(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        构建 chat messages，格式：
        - system: SYSTEM_PROMPT
        - user: "基于以下数据回答问题：\n\n[检索内容]\n\n问题：{query}"
        对 chunks 按 score 降序排列，截断到 max_context_tokens（默认 2048）
        """
        ...
```

---

### Task 8：评估模块

**文件**：`src/eval/teds_score.py`

```python
def compute_teds(pred_markdown: str, gold_markdown: str) -> float:
    """
    计算 TEDS (Tree-Edit Distance based Similarity)。
    将 Markdown 表格解析为树结构后计算编辑距离。

    步骤：
    1. parse_markdown_table → 解析为 (headers, rows) 结构
    2. 构建树：根节点 → 行节点 → 单元格节点
    3. 用 editdistance 计算树编辑距离
    4. TEDS = 1 - (edit_distance / max(|T_pred|, |T_gold|))

    返回 [0, 1] 之间的浮点数。
    """
    ...

def compute_cell_f1(pred_markdown: str, gold_markdown: str) -> dict:
    """
    计算单元格级别的 Precision / Recall / F1。
    以 (行索引, 列索引, 归一化值) 为匹配键。
    返回 {"precision": float, "recall": float, "f1": float}
    """
    ...
```

**文件**：`src/eval/qa_metrics.py`

```python
def compute_exact_match(pred: str, gold: str) -> float:
    """归一化后精确匹配，返回 0 或 1。"""
    ...

def compute_f1(pred: str, gold: str) -> float:
    """词级别 F1（中英文分词处理）。"""
    ...

def evaluate_stage2(
    predictions_path: str,    # JSONL，每行 {"query": ..., "pred": ..., "gold": ...}
    output_path: str
) -> dict:
    """
    批量评估 Stage 2，返回并保存：
    {"exact_match": float, "f1": float, "rouge_l": float, "n_samples": int}
    """
    ...
```

---

### Task 9：FastAPI 推理服务

**文件**：`src/serve/api_server.py`

```python
"""
FastAPI 服务，暴露以下接口：

POST /extract
  - 输入：上传 PDF 文件 或 图片 base64
  - 功能：调用 Stage 1 模型提取表格，存入向量库
  - 返回：{"doc_id": str, "n_tables": int, "tables": [{"page": int, "markdown": str}]}

POST /query
  - 输入：{"question": str, "doc_id": str (可选)}
  - 功能：RAG 检索 + Stage 2 模型推理
  - 返回：{"answer": str, "sources": [...], "latency_ms": float}

GET /health
  - 返回：{"status": "ok", "stage1_loaded": bool, "stage2_loaded": bool, "vector_store_count": int}
"""

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
...

app = FastAPI(title="Document Intelligence API", version="0.1.0")
```

实现要求：
- Stage 1 / Stage 2 模型在启动时懒加载（首次请求时加载，之后复用）
- `/extract` 接口支持异步处理（大 PDF 不阻塞）
- 所有接口加入 latency logging
- 启动命令：`uvicorn src.serve.api_server:app --host 0.0.0.0 --port 8000`

---

### Task 10：训练启动脚本

**文件**：`scripts/train_stage1.sh`

```bash
#!/bin/bash
set -e

cd LLaMA-Factory

# 注册数据集（在 data/dataset_info.json 中添加条目）
python scripts/register_dataset.py \
  --name stage1_table_extraction \
  --data_path ../data/stage1_train/train.json \
  --format sharegpt

# 启动训练
llamafactory-cli train \
  --config ../config/stage1_lora.yaml \
  2>&1 | tee ../logs/stage1_training.log

echo "Stage 1 training complete. Outputs at: ../outputs/stage1_lora"
```

**文件**：`scripts/merge_lora.sh`

```bash
#!/bin/bash

STAGE=$1   # "1" 或 "2"

if [ "$STAGE" = "1" ]; then
  BASE_MODEL="Qwen/Qwen2-VL-7B-Instruct"
  LORA_PATH="./outputs/stage1_lora"
  OUT_PATH="./merged_models/stage1_merged"
elif [ "$STAGE" = "2" ]; then
  BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
  LORA_PATH="./outputs/stage2_lora"
  OUT_PATH="./merged_models/stage2_merged"
fi

cd LLaMA-Factory
llamafactory-cli export \
  --model_name_or_path $BASE_MODEL \
  --adapter_name_or_path $LORA_PATH \
  --export_dir $OUT_PATH \
  --export_size 4 \
  --export_legacy_format false

echo "Model merged to $OUT_PATH"
```

---

### Task 11：端到端测试脚本

**文件**：`scripts/run_eval.sh`

```bash
#!/bin/bash

echo "=== Stage 1 Evaluation: Table Extraction ==="
python -m src.eval.teds_score \
  --pred_dir ./outputs/stage1_predictions \
  --gold_dir ./data/stage1_eval \
  --output ./results/stage1_eval.json

echo "=== Stage 2 Evaluation: Q&A ==="
python -m src.eval.qa_metrics \
  --predictions ./outputs/stage2_predictions.jsonl \
  --output ./results/stage2_eval.json

echo "=== RAG Evaluation ==="
python -m src.eval.rag_eval \
  --test_set ./data/stage2_eval/test.json \
  --output ./results/rag_eval.json

echo "=== Summary ==="
python scripts/print_results.py ./results/
```

---

### Task 12：Ollama 部署配置

**文件**：`src/serve/Modelfile`

```
FROM ./merged_models/stage2_merged

SYSTEM """你是一个专业的数据分析助手，擅长基于表格数据回答问题。
你的答案必须：
1. 只引用给定表格中的数据
2. 包含具体数值并注明来源
3. 对计算类问题展示计算过程"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

**文件**：`scripts/deploy_ollama.sh`

```bash
#!/bin/bash

# 创建 Ollama 模型
ollama create table-qa -f ./src/serve/Modelfile

# 测试
curl -s http://localhost:11434/api/chat -d '{
  "model": "table-qa",
  "messages": [{"role": "user", "content": "2023年营收是多少？\n表格：\n| 年份 | 营收 |\n|---|---|\n| 2023 | 120亿 |"}]
}' | python -m json.tool
```

---

## 执行顺序

Claude Code 请严格按以下顺序执行，每步完成后打印 ✅ 确认：

```
Step 1  → Task 0  环境初始化
Step 2  → Task 1  数据下载
Step 3  → Task 2  PDF 预处理模块 + 单元测试
Step 4  → Task 3  数据集构建（Stage 1 + Stage 2）
Step 5  → Task 4  QA 生成器（use_llm_api=False 优先验证）
Step 6  → Task 5  写入 LLaMA-Factory 训练配置 + 注册数据集
Step 7  → Task 6  结构化处理模块（TableCleaner + VectorStore）+ 单元测试
Step 8  → Task 7  RAG Pipeline + 集成测试（用 mock LLM 验证）
Step 9  → Task 8  评估模块
Step 10 → Task 9  FastAPI 服务
Step 11 → Task 10 训练脚本（先做 dry-run 验证配置正确）
Step 12 → Task 11 端到端评估
Step 13 → Task 12 Ollama 部署
```

---

## 关键约定

1. **所有模块必须有 docstring 和类型注解**（Python 3.10+ typing）
2. **错误处理**：IO 操作和模型推理必须 try/except，异常写入 `./logs/` 而不是崩溃
3. **日志**：使用 `logging` 模块，级别 INFO，格式 `%(asctime)s | %(levelname)s | %(name)s | %(message)s`
4. **配置优先**：数值超参（batch size、top_k、DPI 等）不要 hardcode，从 `config/rag_config.yaml` 读取
5. **路径**：所有路径用 `pathlib.Path`，不用字符串拼接
6. **单元测试**：每个模块对应 `tests/` 中的测试文件，用 `pytest`，测试用 fixture 不依赖真实模型（用 mock）

---

## 常见问题预处理

- **OOM**：Stage 1 训练时若 OOM，将 `per_device_train_batch_size` 改为 1，`gradient_accumulation_steps` 改为 16
- **Flash Attention 安装失败**：在 `config/` 中将 `flash_attn: fa2` 改为 `flash_attn: sdpa`（PyTorch 内置）
- **Qwen2-VL 下载慢**：使用 `HF_ENDPOINT=https://hf-mirror.com` 环境变量
- **pdfplumber 无法提取表格**：表格可能是扫描件，fallback 到 pytesseract OCR 流程（`src/data_pipeline/pdf_extractor.py` 中的 OCR 分支）
