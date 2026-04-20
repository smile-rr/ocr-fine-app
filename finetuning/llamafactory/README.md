# LLaMA-Factory 微调（Stage 1 VLM + Stage 2 LLM）

> **先读** [../README.md](../README.md) 了解 LoRA/QLoRA 原理。本文件只讲**怎么用 LLaMA-Factory 跑通本项目的两个 stage**。

## 0. 适用环境

| 条件 | 支持 |
|---|---|
| Linux + NVIDIA CUDA | ✅ 推荐（3090/4090/A100/H100） |
| WSL2 + CUDA | ✅ |
| macOS (M 系列) | ❌ bitsandbytes 无 Mac 包；用本项目原 MLX 方案 |
| CPU-only | ❌ 速度不可接受 |
| Google Colab T4 / V100 | ✅ QLoRA 7B 以下都行 |

**最低显存建议**
- Stage 2 (Qwen2.5-0.5B QLoRA) → 4GB（Colab T4 都够）
- Stage 2 (Qwen2.5-7B QLoRA) → 10GB（3060/3090）
- Stage 1 (Qwen2-VL-2B QLoRA) → 8GB
- Stage 1 (Qwen2-VL-7B QLoRA) → 20GB（3090 满卡）

---

## 1. 安装

```bash
# 从本目录跑
cd finetuning/llamafactory
bash scripts/setup.sh
```

`setup.sh` 做了：
1. Clone `LLaMA-Factory` 到 `finetuning/llamafactory/LLaMA-Factory/`（独立目录，不污染主项目 `.venv`）
2. 在里面建独立 venv 并 `pip install -e ".[torch,metrics,bitsandbytes]"`
3. 下载 base 模型到 `models/` (HuggingFace cache)

---

## 2. 注册数据集

**关键概念**：LLaMA-Factory 不直接读 JSONL，它用一个 `dataset_info.json` 索引所有数据集。跑训练前必须先把我们的数据注册进去。

本目录已经给好了 `configs/dataset_info.json`，里面注册了两条：

```json
{
  "stage1_table_extraction": {
    "file_name": "../../../data/stage1_train/train.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
  },
  "stage2_table_qa": {
    "file_name": "../../../data/stage2_train/train.jsonl",
    "formatting": "alpaca",
    "columns": {"prompt": "instruction", "query": "input", "response": "output"}
  }
}
```

`scripts/setup.sh` 会把这个 `dataset_info.json` **软链接**到 `LLaMA-Factory/data/` 下，保持单一来源。

---

## 3. Stage 2 训练（先从这个开始，30 分钟跑完）

```bash
# 先生成训练数据（如果还没跑过）
cd /path/to/ocr-fine-app
uv run python scripts/prepare_stage2.py

# 启动训练
cd finetuning/llamafactory
bash scripts/train_stage2.sh
```

**脚本里实际做的事**：
```bash
cd LLaMA-Factory
source .venv/bin/activate
llamafactory-cli train ../configs/stage2_qwen25_qlora.yaml
```

**输出**：`finetuning/llamafactory/outputs/stage2_lora/`
- `adapter_model.safetensors` ← LoRA 权重（~20MB）
- `adapter_config.json`
- `tokenizer_*`
- `training_args.bin`
- `trainer_log.jsonl` ← 每步 loss/lr，后面画图用

**TensorBoard 看训练曲线**：
```bash
tensorboard --logdir outputs/stage2_lora
```

---

## 4. Stage 1 训练（多模态 VLM）

Stage 1 比 Stage 2 麻烦的地方：
- 模型大（2B VLM ≈ 5GB fp16）
- 数据包含图片路径，必须能访问
- Qwen2-VL processor 需要 `trust_remote_code: true`

```bash
cd /path/to/ocr-fine-app
uv run python scripts/prepare_stage1.py   # 生成 train.jsonl + stage1_images/
cd finetuning/llamafactory
bash scripts/train_stage1.sh
```

**配置差异点**（见 `configs/stage1_qwen2vl_qlora.yaml`）：
- `template: qwen2_vl`
- `visual_inputs: true`
- `freeze_vision_tower: true` ← **默认冻结 ViT**，只微调 LLM 部分，省显存且更稳
- `cutoff_len: 4096` ← 图片 token 会占 800+，文本预算要大些

---

## 5. 合并 adapter → HF 格式

训完后 adapter 不能直接被本项目 Docker 里的 `transformers.from_pretrained` 读（Docker 里 base 是 fp16，训练时是 4-bit 量化的，权重布局不兼容）。必须合并。

```bash
bash scripts/merge_stage2.sh   # 或 merge_stage1.sh
```

脚本实际做的：
```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_name_or_path outputs/stage2_lora \
  --template qwen \
  --finetuning_type lora \
  --export_dir ../../models/stage2_fused \
  --export_size 4 \
  --export_legacy_format false
```

输出到 `models/stage2_fused/`，然后 `docker compose up` 就自动用这个新 base + 合并后权重了。

---

## 6. 跑通后怎么验证

```bash
# 方式 1：LLaMA-Factory 自带 CLI chat
llamafactory-cli chat ../configs/stage2_qwen25_qlora.yaml

# 方式 2：用项目的 Streamlit（默认就会找 models/stage2_adapter/；合并后改指向 stage2_fused/）
cd /path/to/ocr-fine-app
uv run streamlit run app/streamlit_app.py
# 勾选 "Stage2 使用微调 LoRA" 进 Tab 4 对比

# 方式 3：项目 Docker API
docker compose up
curl -X POST http://localhost:8000/admin/reload -H "X-Admin-Key: change-me-in-prod" \
  -H "Content-Type: application/json" -d '{"stage":2,"force":true}'
# 然后 POST /query 看新答案
```

---

## 7. 常见 LLaMA-Factory 专属坑

- **`FileNotFoundError: dataset_info.json`**
  → `dataset_info.json` 必须在 `LLaMA-Factory/data/` 下；`setup.sh` 会做软链接，手动重建：
  ```bash
  ln -sf "$(pwd)/configs/dataset_info.json" LLaMA-Factory/data/dataset_info.json
  ```

- **报 `ValueError: Please specify a template`**
  → YAML 里 `template:` 字段没填或拼错。Qwen2.5 用 `qwen`，Qwen2-VL 用 `qwen2_vl`。查模型支持列表：
  ```bash
  llamafactory-cli webui   # 打开 WebUI 左边有完整列表
  ```

- **`trust_remote_code` 报错**
  → YAML 里加 `trust_remote_code: true`；或者环境变量 `HF_ALLOW_REMOTE_CODE=1`

- **训练速度巨慢**
  → 检查：
  1. 是不是用了 `flash_attn: fa2`（`pip install flash-attn --no-build-isolation`）
  2. 是不是开了 `packing: true`（短样本拼接，提升 2-3 倍）
  3. `num_workers` 默认 0，改成 4–8

- **验证集一直是 0.0**
  → `val_size` 太小，`compute_metrics` 没配；加 `val_size: 0.05` + `evaluation_strategy: steps`

---

## 8. 更多 LLaMA-Factory 能力（本项目没用但企业级常用）

- **DPO / KTO / ORPO** 偏好对齐：在 YAML 里把 `stage: sft` 改成 `stage: dpo` 并加偏好数据集
- **RLHF + Reward Model**：`stage: rm` → `stage: ppo`
- **长文本训练**：`rope_scaling: linear` + `cutoff_len: 16384`，配合 Qwen2 长上下文版
- **多 GPU**：直接 `CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train`，DDP 自动启用
- **DeepSpeed ZeRO-3**（训 70B）：加 `deepspeed: examples/deepspeed/ds_z3_config.json`
- **WebUI 零代码训练**：`llamafactory-cli webui`
