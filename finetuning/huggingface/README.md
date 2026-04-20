# Raw Hugging Face 微调 —— 理解下面一切的基础

> LLaMA-Factory / Axolotl 都是建在 **HF Trainer + `peft` + `trl` + `datasets`** 之上的包装。本目录不抽象、不 YAML —— **纯 Python，手写训练流程**，让你知道：
>
> - LoRA / QLoRA 的 `peft` API 长什么样
> - `Trainer` 是怎么循环的
> - `datasets` 怎么加载 JSONL + 做 tokenization
> - chat template 怎么正确应用到训练数据
> - TRL 的 `SFTTrainer` 比 `Trainer` 帮你多做了什么

## 为什么要学裸 HF

面试里"熟悉 Transformers / Hugging Face datasets"的真实含义：**能独立 debug 训练过程**。以下场景必须用 raw HF：

- 训练卡住 / loss 异常 / adapter 合并后效果不对 —— 没办法靠 YAML 定位
- 自定义 loss / 数据采样策略 / 奇怪的多模态输入
- 读别人的 repo（MIT/Stanford/DeepMind 的 repo 基本都是裸 HF，很少用 YAML 框架）
- 面试手写"用 HF 微调 7B 模型"的白板题

## 本目录的 4 个递进示例

| 文件 | 用的 API | 场景 | 依赖 |
|---|---|---|---|
| `train_lora.py` | `transformers.Trainer` + `peft.LoraConfig` | 纯 LoRA，fp16/bf16 | transformers + peft + datasets |
| `train_qlora.py` | 上面 + `BitsAndBytesConfig` + `prepare_model_for_kbit_training` | QLoRA，4-bit + LoRA | + bitsandbytes |
| `train_sft_trl.py` | `trl.SFTTrainer` | 现代推荐，自动处理 packing / chat template | + trl |
| `train_full_ft.py` | `Trainer` 不挂 adapter | 全参数微调（对照） | — |

`dataset_prep.py` 是公共的数据加载模块，四个脚本都 import。

## HF 训练栈概览（训练前必须看懂这张图）

```
┌────────────────────── HF 训练栈 ──────────────────────────┐
│                                                           │
│    datasets          tokenizer           model            │
│    ─────────         ─────────           ─────            │
│    load JSONL        apply_chat_template AutoModelForCausalLM
│    map() tokenize    tokenizer()         .from_pretrained │
│        │                 │                    │           │
│        ▼                 ▼                    ▼           │
│    tokenized_ds ──► DataCollator ──► Trainer ──► Model    │
│                      (padding/label)    │           (权重) │
│                                         │                 │
│                                         ▼                 │
│                                  TrainingArguments        │
│                                  (LR, epoch, bs, bf16...) │
│                                                           │
│    peft:                                                  │
│    LoraConfig ──► get_peft_model(model, config)           │
│      (r, alpha, target_modules, dropout)                  │
│                                                           │
│    bitsandbytes (QLoRA):                                  │
│    BitsAndBytesConfig ──► 传给 from_pretrained            │
│      (load_in_4bit, bnb_4bit_quant_type="nf4", ...)       │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## 关键对应：LLaMA-Factory YAML ↔ HF Python API

和 `finetuning/llamafactory/configs/stage2_qwen25_qlora.yaml` 一一对应着看：

| YAML 字段 | HF Python | 在哪个文件里 |
|---|---|---|
| `model_name_or_path: Qwen/...` | `AutoModelForCausalLM.from_pretrained("Qwen/...")` | 每个 train_*.py |
| `finetuning_type: lora` | `peft.get_peft_model(model, LoraConfig(...))` | `train_lora.py` |
| `lora_rank: 8` | `LoraConfig(r=8, ...)` | `train_lora.py` |
| `lora_target: all` | `target_modules="all-linear"` 或显式列表 | `train_lora.py` |
| `quantization_bit: 4` | `BitsAndBytesConfig(load_in_4bit=True, ...)` | `train_qlora.py` |
| `quantization_type: nf4` | `bnb_4bit_quant_type="nf4"` | `train_qlora.py` |
| `double_quantization: true` | `bnb_4bit_use_double_quant=True` | `train_qlora.py` |
| `dataset: stage2_table_qa` | `load_dataset("json", data_files=...)` | `dataset_prep.py` |
| `template: qwen` | `tokenizer.apply_chat_template(...)` | `dataset_prep.py` |
| `cutoff_len: 2048` | `tokenizer(..., max_length=2048, truncation=True)` | `dataset_prep.py` |
| `per_device_train_batch_size: 2` | `TrainingArguments(per_device_train_batch_size=2)` | 每个 train_*.py |
| `learning_rate: 2.0e-4` | `TrainingArguments(learning_rate=2e-4)` | 每个 train_*.py |
| `bf16: true` | `TrainingArguments(bf16=True)` | 每个 train_*.py |
| `flash_attn: fa2` | `from_pretrained(attn_implementation="flash_attention_2")` | 每个 train_*.py |
| `packing: true` | `SFTTrainer(packing=True, ...)` | `train_sft_trl.py` |

## 安装

```bash
cd finetuning/huggingface
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 跑

```bash
# 先生成数据（如果还没跑过）
cd /path/to/ocr-fine-app
uv run python scripts/prepare_stage2.py

# 回到本目录跑训练
cd finetuning/huggingface
source .venv/bin/activate

# 从简单到复杂选一个:
python train_lora.py        # 纯 LoRA (需要 ~12GB VRAM 训 0.5B)
python train_qlora.py       # QLoRA (需要 ~4GB VRAM 训 0.5B, 或 ~8GB 训 7B)
python train_sft_trl.py     # TRL SFTTrainer (推荐日常用这个)

# 输出: ./outputs/stage2_lora_hf/
```

## 训完怎么用

```python
# 直接在 Python 里 load（也是 inference/huggingface/with_adapter.py 做的事）
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "./outputs/stage2_lora_hf/final")

# 合并（如果要给 vLLM / Docker 用）
merged = model.merge_and_unload()
merged.save_pretrained("/path/to/ocr-fine-app/models/stage2_fused")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").save_pretrained("/path/to/ocr-fine-app/models/stage2_fused")
```

## 和 LangChain 是什么关系

JD 里说的 "LangChain, Transformers, or Hugging Face datasets" —— 注意是 **or**（会其一即可）：

- **Transformers** + **HF datasets** = 本目录的栈。**微调和推理的基础**，必会。
- **LangChain** = RAG / Agent 编排层，基于上面的栈往上一层。**业务层的黏合剂**，非核心。

你本项目的 `src/rag.py` + `src/serve/api.py` 就是用 Transformers + 手写 RAG，**没用 LangChain**（故意的，它太重）。面试时可以这么讲：
> "我们选 raw transformers + 手写 RAG，是因为 LangChain 的抽象层次不匹配我们的需求——RAG 的检索策略、prompt 组装、路由都是业务定制的，LangChain 的 Chain 抽象反而拖慢迭代。用 OpenAI SDK + 显式 chroma 调用，代码清晰 debug 快。"

## 常见坑（和 LLaMA-Factory 不一样的地方）

- **`pad_token` 没设** —— 很多模型（Qwen/Llama3）原生 tokenizer 没 pad_token，训练会报错。见 `dataset_prep.py` 里的处理
- **`attention_mask` 不对齐** —— tokenize 后要 `DataCollatorForLanguageModeling` 或 `DataCollatorForSeq2Seq` 统一处理 padding
- **`labels` 没 mask prompt** —— SFT 时通常只算 response 部分的 loss；prompt 部分的 labels 要设 -100（`train_sft_trl.py` 里 `SFTTrainer` 自动做了；`train_lora.py` 手写演示）
- **`prepare_model_for_kbit_training` 忘调** —— QLoRA 一定要调这个，否则梯度回传有 bug
- **`gradient_checkpointing_enable()` 位置** —— 要在 `get_peft_model()` 之前调，否则 checkpoint 不生效
