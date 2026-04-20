# Axolotl 微调（Stage 2 纯文本 QLoRA）

> **先读** [../README.md](../README.md) 了解 LoRA/QLoRA 原理。Axolotl 和 LLaMA-Factory 是同类工具，本目录用来**对照学习**——同一个任务两个框架各跑一遍，就能对企业微调栈有直观理解。

## Axolotl vs LLaMA-Factory 一眼对比

| 维度 | Axolotl | LLaMA-Factory |
|---|---|---|
| 维护方 | OpenAccess AI Collective + HuggingFace | 北航 hiyouga 团队 |
| 配置文件 | `config.yml` (扁平) | `config.yaml` (扁平) |
| 启动命令 | `accelerate launch -m axolotl.cli.train` | `llamafactory-cli train` |
| 数据集注册 | 配置里 inline 写 path | 独立 `dataset_info.json` 注册 |
| WebUI | ❌ | ✅ |
| VLM 原生支持 | ⚠️ 有限（社区 fork） | ✅ 一流 |
| DeepSpeed/FSDP | ✅ 成熟 | ✅ 齐全 |
| HF 集成 | 深度（AutoTrain 底层）| 深度 |
| 工业界使用度 | 更高（Replicate/Together 用它微调） | 高（科研多） |
| 学习曲线 | 中（要懂 HF Trainer） | 低（零代码 WebUI） |

**本项目只给 Axolotl 配 Stage 2**（纯文本），Stage 1 多模态走 LLaMA-Factory 更省心。

---

## 0. 环境要求

和 LLaMA-Factory 一样：Linux + NVIDIA CUDA，或 WSL2。Mac 不支持（bitsandbytes 限制）。

---

## 1. 安装

```bash
cd finetuning/axolotl
bash scripts/setup.sh
```

`setup.sh` 做的事：
1. Clone Axolotl → `finetuning/axolotl/axolotl/`
2. 建独立 venv
3. `pip install -e '.[flash-attn,deepspeed]'`（Axolotl 依赖比 LF 重些，预计 3-5 分钟）

---

## 2. 训练配置讲解

`configs/stage2_qwen25_qlora.yml` 是核心，几个关键字段和 LLaMA-Factory 的对应关系：

| Axolotl | LLaMA-Factory | 说明 |
|---|---|---|
| `base_model` | `model_name_or_path` | base 模型 |
| `adapter: qlora` | `finetuning_type: lora` + `quantization_bit: 4` | Axolotl 把 LoRA/QLoRA 作为一个 adapter 类型 |
| `load_in_4bit: true` | `quantization_bit: 4` | QLoRA 标志 |
| `datasets:` (list) | `dataset:` + `dataset_info.json` | Axolotl 直接 inline 写数据路径 |
| `chat_template: chatml` | `template: qwen` | Qwen 用 chatml 模板 |
| `sequence_len` | `cutoff_len` | 最大 token 长度 |
| `micro_batch_size` | `per_device_train_batch_size` | 单卡单步 batch |
| `gradient_accumulation_steps` | 同名 | 梯度累积 |
| `lora_r` / `lora_alpha` | `lora_rank` / `lora_alpha` | 命名略有差别 |
| `lora_target_modules: [...]` | `lora_target: all` | Axolotl 要显式列出 |
| `lora_target_linear: true` | `lora_target: all` | 相当于"所有线性层" |

---

## 3. 启动训练

```bash
cd finetuning/axolotl
bash scripts/train_stage2.sh
```

**脚本里实际做的**：
```bash
cd axolotl
source .venv/bin/activate
accelerate launch -m axolotl.cli.train ../configs/stage2_qwen25_qlora.yml
```

**为什么要 `accelerate launch` 而不是直接 python**：
- HuggingFace Accelerate 自动处理多卡 DDP / FSDP
- 单卡时它自动退化成普通 python，**不会有副作用**，建议始终这样启动
- 多卡时：`CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 ...`

**输出**：`finetuning/axolotl/outputs/stage2_qlora/`
- `adapter_model.safetensors`
- `adapter_config.json`
- `trainer_state.json`（记录 loss、lr 等）

---

## 4. 合并 adapter

```bash
bash scripts/merge_stage2.sh
```

实际命令：
```bash
python -m axolotl.cli.merge_lora configs/stage2_qwen25_qlora.yml \
    --lora_model_dir outputs/stage2_qlora
```

默认合并到配置里的 `output_dir/merged/`，我们脚本里额外复制到项目 `models/stage2_fused/` 方便 Docker 直接挂载。

---

## 5. Axolotl 专属特性（本配置没用到但值得知道）

### 5.1 数据集 **自动加载 + 多数据集混合**

```yaml
datasets:
  - path: Open-Orca/SlimOrca
    type: sharegpt
    split: train
    train_on_split: train
  - path: HuggingFaceH4/no_robots
    type: sharegpt
    split: train
    shards: 10          # 只拿 1/10
  - path: ./local/data.jsonl
    type: alpaca
```

不需要提前下载 —— Axolotl 自动从 HF 拉数据，还支持按比例 shard。

### 5.2 Sample Packing（比 LLaMA-Factory 更强）

```yaml
sample_packing: true           # 多个短样本拼到 sequence_len，吞吐 2-3×
eval_sample_packing: false     # eval 时关掉，不影响指标
pad_to_sequence_len: true
```

Axolotl 的 packing 实现更成熟，对 loss 影响更小。

### 5.3 RLHF / DPO 一键切换

```yaml
rl: dpo                        # 或 ipo, kto, orpo
datasets:
  - path: HuggingFaceH4/ultrafeedback_binarized
    type: chatml.ultra
```

### 5.4 多机多卡（FSDP）

`configs/deepspeed_configs/` 里有预设，训 70B+ 模型时用：
```yaml
deepspeed: deepspeed_configs/zero3_bf16.json
# 或 FSDP:
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_offload_params: true
```

### 5.5 LoRA+（比传统 LoRA 快 2×）

```yaml
loraplus_lr_ratio: 16          # B 矩阵学习率设为 A 的 16 倍
```

LoRA 的 A 和 B 梯度不对称，不同学习率能显著加速收敛。这是 2024 年的新技巧，LLaMA-Factory 刚加进来，Axolotl 更早支持。

---

## 6. 常见 Axolotl 专属坑

- **报 `accelerate config not found`** 
  → 第一次跑需要 `accelerate config --default` 生成默认配置；`setup.sh` 已经做了

- **`pad_token` 相关报错** 
  → Qwen 没有 pad_token，在 yaml 里加：
  ```yaml
  special_tokens:
    pad_token: "<|endoftext|>"
  ```

- **loss 一直 nan**
  → 几乎都是 `bf16: false` + `fp16: true` 在老卡上的问题；bf16 能开就开

- **报 `unsupported chat_template`**
  → Axolotl 内置模板比 LF 少；用 `chatml`（Qwen/Yi）、`llama3`、`vicuna`；其他的在 yaml 写 `chat_template_jinja: |` 贴 jinja2 源码

- **`FileNotFoundError` 读不到 jsonl**
  → Axolotl 的 `path: ./foo.jsonl` 是相对于**当前工作目录**的，不是配置文件所在目录。脚本里我们会 `cd` 到合适位置。

---

## 7. 训练完验证

```bash
# 方式 1：Axolotl 自带 inference
python -m axolotl.cli.inference configs/stage2_qwen25_qlora.yml \
    --lora_model_dir outputs/stage2_qlora \
    --prompt "基于以下表格回答..."

# 方式 2：合并后走项目 Docker
cd ../../
docker compose up
# 合并脚本已把文件放到 models/stage2_fused/
# 容器启动自动用新模型
```
