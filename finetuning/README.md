# Finetuning — 企业级 LoRA / QLoRA 参考

用于替代 `notebooks/02_finetune_*.ipynb` 里的 MLX 原生方式。**先读本节的"主流微调框架对比"和"LoRA/QLoRA 原理"**，再选一个框架跟它自己的 README 跑。

---

## 主流微调框架全景

整个生态可以分成 **三层**：

```
┌─────────────────────────────────────────────────────────────┐
│ 应用层（YAML 配置 + CLI，本目录实现的就在这层）               │
│  LLaMA-Factory · Axolotl · Unsloth · SWIFT · torchtune      │
│  LitGPT · NeMo · DeepSpeed-Chat                             │
├─────────────────────────────────────────────────────────────┤
│ 库层（Python API，上面的框架都调用它们）                      │
│  TRL · PEFT · Accelerate · DeepSpeed · FSDP                 │
├─────────────────────────────────────────────────────────────┤
│ 底层（PyTorch / MLX / JAX + 量化库）                          │
│  PyTorch · MLX · bitsandbytes · GPTQ · AWQ                  │
└─────────────────────────────────────────────────────────────┘
```

### 应用层框架对比（选型主要看这张表）

| 框架 | 平台 | 速度 | 模型广度 | 多模态 | RLHF | 多节点 | 易用度 | 适合 |
|---|---|---|---|---|---|---|---|---|
| **LLaMA-Factory** ✅ | Linux+CUDA/ROCm | ★★★ | 100+，最新 | ✅ 一流 | SFT/DPO/KTO/PPO/ORPO/SimPO | DeepSpeed+FSDP | ★★★★★ CLI+WebUI | **综合首选**，个人到中厂 |
| **Axolotl** ✅ | Linux+CUDA | ★★★ | 主流都有 | ⚠️ 社区 fork | SFT/DPO/KTO/ORPO | FSDP 成熟 | ★★★ YAML | 工业界生产管线（Together/Replicate 在用） |
| **Unsloth** | Linux+CUDA（**单卡**） | ★★★★★ **2-5×** | Llama/Qwen/Mistral/Gemma | ❌ | SFT/DPO | ❌ 不支持多卡 | ★★★★ Notebook | **速度极致**，Colab/单 4090 个人用 |
| **SWIFT**（ms-swift） | Linux+CUDA/Ascend | ★★★ | 200+，含国产 | ✅ | SFT/DPO/KTO/PPO | DeepSpeed+FSDP | ★★★★ CLI+WebUI | **国产模型 + 昇腾卡**，阿里出品 |
| **torchtune** | Linux+CUDA/CPU | ★★★ | 主流开源 | ⚠️ 初步 | SFT/DPO/PPO | FSDP native | ★★ 纯 Python | **PyTorch 官方正统**，可读性好 |
| **TRL** (库) | Linux+CUDA | ★★★ | 任意 HF 模型 | 需自拼 | SFT/DPO/PPO/GRPO | Accelerate | ★ 写代码 | **库**，上面所有框架都在用它 |
| **LitGPT** | Linux+CUDA/TPU | ★★★ | Llama/Phi/Mistral 等 | ❌ | SFT/DPO | Fabric | ★★★ YAML | Lightning AI 生态，教育/研究 |
| **NeMo**（NVIDIA） | Linux+CUDA | ★★★★ | NVIDIA 优化过的 | ✅ | SFT/DPO/RLHF | Megatron 超大规模 | ★★ 复杂 | **超大模型 + 多节点**，企业级 |
| **DeepSpeed-Chat** | Linux+CUDA | ★★★ | 主流 | ❌ | SFT/RLHF 全流程 | ZeRO 1/2/3 | ★★ 脚本 | 纯 RLHF 流水线参考实现 |
| **MLX-LM/VLM** | **macOS (M 系列)** | ★★★ | Llama/Qwen/Phi 等 | ✅ | SFT/DPO | ❌ 单机 | ★★★★ CLI | **Mac 上唯一选择**（本项目原方案） |

✅ 标记 = 本目录已给出完整配置。下面 5 种给出**在什么场景该选**的快速判断：

### 一句话选型指南

| 场景 | 选这个 | 理由 |
|---|---|---|
| 个人 / 科研 / 快速原型 / **多模态** | **LLaMA-Factory** | 模型最全，WebUI 零代码，多模态一流 |
| **生产管线 / CI 训练** | **Axolotl** | 配置即代码，HF 官方加持，分布式稳 |
| **Colab / 单卡极致速度** | **Unsloth** | 自研 Triton kernel，2-5× 快，40% 省显存 |
| **Mac M 系列本地训练** | **MLX-LM** | 唯一选择，本项目原方案保留 |
| **国产模型 / 昇腾卡** | **SWIFT**（ms-swift） | 阿里原厂，Qwen/DeepSeek/百川都一流 |
| **超大模型（100B+）多节点** | **NeMo** 或 **Megatron-LM** | NVIDIA 自家优化 |
| **想懂底层，PyTorch 原生** | **torchtune** | 代码最干净，适合改 |
| **想研究 RLHF 全流程** | **TRL** 直接用 | 上面框架的源头，PPO/DPO/GRPO 原版实现 |

### 本目录选了哪两个、为什么

- **[llamafactory/](./llamafactory/)** — 最主流、多模态一流。我们 Stage 1（VLM）+ Stage 2（LLM）都给配置
- **[axolotl/](./axolotl/)** — 工业界生产栈，**对照学习**，同样 Stage 2 配一份看差异
- **MLX-LM** — 本项目 `notebooks/02_*.ipynb` 和 `scripts/fuse_model.py` 已经是 MLX 路线，Mac 上要用就跟那套

**没选 Unsloth 的原因**：它速度是真快，但不支持多卡，且对模型架构有侵入性（重写了部分 forward），学到东西偏"Unsloth 特定优化"而非"通用微调"。如果你就是一台 4090 想极速，单独跑一遍 Unsloth 的 notebook 会有惊喜。

### 库层 —— 无论用哪个框架，底下都离不开这些

读代码时会反复看见，先认个脸：

- **TRL** (HuggingFace Transformer Reinforcement Learning) — SFT/DPO/PPO/GRPO Trainer 都在这。LLaMA-Factory / Axolotl 的训练循环内核就是它。
- **PEFT** (HuggingFace Parameter-Efficient Fine-Tuning) — LoRA / QLoRA / IA³ / Prefix Tuning 的官方实现。**所有框架的 adapter 逻辑都来自它**。
- **Accelerate** (HuggingFace) — 抽象"单机单卡 / 多卡 DDP / 多机 FSDP / DeepSpeed"的启动差异。`accelerate launch` 就是它。
- **bitsandbytes** — 4-bit / 8-bit 量化 + 8-bit Adam。QLoRA 的基石。Mac 没有，所以 Mac 只能走 MLX。
- **DeepSpeed** (Microsoft) — ZeRO 1/2/3 显存优化，训超大模型必备。
- **FSDP** (PyTorch) — DeepSpeed 同类，PyTorch 官方实现。新项目倾向用 FSDP。

---

## 1. LoRA 是什么 —— 一分钟原理

**问题**：直接全参数微调 7B 模型要 140GB+ 显存（fp16 梯度 + 优化器状态），个人/小团队跑不动。

**LoRA 的核心思想**：冻结 base 权重，只训练一个**低秩分解矩阵**加到原权重上。

```
原 forward:   h = W @ x                    (W: d×d, 比如 4096×4096)
LoRA forward: h = W @ x + (B @ A) @ x * α/r
               ↓         ↓
            (冻结)   (可训练, A: r×d, B: d×r, r=8~64)
```

- `r` 叫 **rank**（低秩维度），通常取 8 / 16 / 32 / 64
- `alpha` 是缩放因子，惯例 `alpha = 2 * r`，作用类似于学习率的放大器
- 训练参数量：从 `d×d ≈ 16M` 减到 `2×r×d ≈ 128K`（r=16 时），**省 ~100×**
- 推理时两种用法：
  - **合并**：`W' = W + BA`，与 base 同速，但每个 adapter 都要一个 merged 模型
  - **挂载**：保留两组权重，推理时动态加 BA，多个 adapter 可共享 base（企业级做法）

**关键超参**：
| 参数 | 典型值 | 作用 |
|---|---|---|
| `r` (lora_rank) | 8 / 16 / 32 | 越大容量越强，也越容易过拟合 |
| `alpha` (lora_alpha) | `2r` 或 `r` | 缩放，大了会 dominate base |
| `dropout` | 0.05 ~ 0.1 | 正则化，小数据集建议开 |
| `target_modules` | `q,k,v,o,gate,up,down` | **覆盖越全效果越好**，不要只挂 QV |

---

## 2. QLoRA 是什么 —— 让 24GB 卡能训 7B

**QLoRA = 4-bit 量化 base + LoRA adapter**

```
              ┌──────── 冻结 + 4-bit NF4 量化 ────────┐
forward:   h = W_nf4 @ x (反量化到 bf16 算) + (B @ A) @ x * α/r
                                                  ↓
                                           bf16, 可训练
```

**三大关键技术**（QLoRA 论文）：
1. **NF4**（Normal Float 4-bit）— 比 INT4 更贴合神经网络权重的正态分布，精度损失小
2. **Double Quantization** — 对 quantization constants 再量化，再省 0.4 bits/param
3. **Paged Optimizers** — GPU 显存不足时自动换页到 CPU 内存（NVIDIA unified memory）

**显存对比**（训 7B 模型，序列长度 2048）：
| 方法 | 显存 | 训练速度 | 精度损失 |
|---|---|---|---|
| 全参数 FP16 | ~140GB | 最快 | 0% |
| LoRA FP16 | ~20GB | 90% | <1% |
| **QLoRA NF4** | **~8GB** | 70% | <2% |

所以 24GB 的 3090/4090 就能训 13B，40GB A100 能训 70B。

**什么时候用 LoRA vs QLoRA**：
- 显存够 → LoRA（训得更快，精度略好）
- 显存紧 → QLoRA（本项目默认选项）
- 极致部署精度 → 全参数微调（需要多卡 DeepSpeed ZeRO-3）

---

## 3. LLaMA-Factory vs Axolotl 深入对比（本目录两框架的取舍细节）

| 维度 | LLaMA-Factory | Axolotl |
|---|---|---|
| **易用性** | CLI + YAML + WebUI（零代码） | YAML 为主，略硬核 |
| **模型支持** | 100+ 主流模型，更新快 | 主流都支持，新模型稍滞后 |
| **多模态** | ✅ 原生 Qwen2-VL / LLaVA / InternVL | ⚠️ 需自己拼 processor |
| **数据格式** | alpaca / sharegpt / 自定义，自动注册 | alpaca / sharegpt / jsonl，显式配置 |
| **分布式** | DeepSpeed ZeRO-0/2/3 + FSDP | FSDP 更成熟，DeepSpeed 齐全 |
| **量化** | bitsandbytes / AQLM / HQQ | bitsandbytes / AWQ |
| **RLHF** | DPO / KTO / PPO / ORPO 全 | DPO / KTO |
| **推理** | `llamafactory-cli chat/api` | 无，需外接 vLLM/TGI |
| **活跃度** | GitHub 43k+ stars，每周更新 | 10k+，月度更新 |
| **适合谁** | 个人、科研、快速原型 | 工业界生产管线 |

**本项目里三份对照**：
- [**`huggingface/`** ⭐ 从这里开始学](./huggingface/) —— 裸 HF Trainer + PEFT + TRL，**理解下面一切的基础**
- [`llamafactory/`](./llamafactory/) —— LLaMA-Factory 配置（Stage 1 VLM + Stage 2 LLM）
- [`axolotl/`](./axolotl/) —— Axolotl 配置（Stage 2，对照 LLaMA-Factory 学差异）

**推荐阅读顺序**：先 `huggingface/` 懂原理 → 再 `llamafactory/` 和 `axolotl/` 对比看它们帮你抽象了什么

---

## 4. 典型微调流水线（两框架通用）

```
1. 数据准备
   data/stage2_train/{train,val}.jsonl    ← src/data.py 已生成 alpaca 格式
                                            (和这两个框架直接兼容)

2. 框架安装
   llamafactory/scripts/setup.sh  或  axolotl/scripts/setup.sh
   → 建独立 venv（不污染项目主环境）

3. 训练配置
   配置 yaml：模型、数据、LoRA 超参、训练超参

4. 启动训练
   llamafactory-cli train config.yaml
   或
   accelerate launch -m axolotl.cli.train config.yml
   → 输出 outputs/stageN_lora/  (adapter_model.safetensors + adapter_config.json)

5. 合并权重（可选）
   llamafactory-cli export  /  axolotl.cli.merge_lora
   → models/stageN_fused/    (完整 HF 格式，transformers/vLLM 可直接加载)

6. 推理
   见 ../inference/README.md
```

---

## 5. 常见问题 —— Cheat Sheet

### 🔴 显存相关

**问题**：`CUDA out of memory`

| 症状 | 原因 | 解决 |
|---|---|---|
| forward 就 OOM | batch size 太大 / 序列太长 | `per_device_train_batch_size=1`，`cutoff_len` 减半 |
| backward 才 OOM | 梯度 + 激活值 | 开 `gradient_checkpointing: true`（慢 30% 但省 50% 显存） |
| 优化器初始化 OOM | Adam 状态是权重 2× | 换 `optim: adamw_bnb_8bit`（8-bit Adam）或 `adafactor` |
| 加载模型就爆 | fp16 base 放不下 | 改 `quantization_bit: 4`（QLoRA） |
| 多卡 OOM | 单卡放不下全模型 | 用 DeepSpeed ZeRO-2/3 或 FSDP |

**标准降显存流水线**：
```
OOM → bs=1 → grad_accum 提到 8+ → grad_checkpoint → QLoRA → 8-bit adam → ZeRO-3
```

### 🟠 训练不收敛 / Loss 异常

**症状 → 诊断**

- **Loss 一直高不下降** 
  - LR 太小 → 拉到 `2e-4`（LoRA 标准值，比全参数大 10×）
  - `target_modules` 只挂了 Q/V → 改成 all-linear（Qwen: `q,k,v,o,gate,up,down`）
  - Prompt 模板错了 → 确认 `template: qwen / qwen2_vl / llama3` 匹配 base

- **Loss NaN** 
  - fp16 下梯度爆炸 → 换 `bf16: true`（A100/H100/3090 以上都支持）
  - LR 过大 → 降到 `1e-4`
  - 数据里有异常 token（很长的零宽字符、emoji） → 预处理清洗

- **Loss 下降但验证集变差**（过拟合）
  - `num_train_epochs` 太多 → 降到 2–3 epoch（LoRA 很快就饱和）
  - 数据量太少（<500）→ 加数据增强 / 降低 `lora_rank` / 增大 `lora_dropout`

- **Loss 震荡很大** → `warmup_ratio: 0.03~0.05`；scheduler 用 `cosine`

### 🟡 模板与数据

- **chat template 不对**（输出乱码 / 不停机）
  - 每个模型都有官方 template：Qwen → `qwen`，Qwen2-VL → `qwen2_vl`，Llama3 → `llama3`
  - LLaMA-Factory 会自动匹配；Axolotl 需要显式指定 `chat_template: chatml / llama3`
  - 调试：`tokenizer.apply_chat_template(msgs, tokenize=False)` 打出来看

- **数据格式**（两个主流 + 一个多模态）
  ```json
  // alpaca
  {"instruction": "...", "input": "...", "output": "..."}

  // sharegpt
  {"conversations": [{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}

  // 多模态 sharegpt (LLaMA-Factory)
  {"messages":[
    {"role":"user","content":[{"type":"image","image":"path.png"},{"type":"text","text":"..."}]},
    {"role":"assistant","content":"..."}
  ]}
  ```

### 🔵 量化与精度

- **QLoRA 训完合并后精度掉**
  - `nf4` base + 直接 merge 会把 adapter 量化掉 → 训练时 `--de-quantize` 或推理时保持 adapter 挂载
  - 本项目 `scripts/fuse_model.py` 就是做这个的

- **bitsandbytes 装不上**（CPU / Mac / AMD）
  - Mac/CPU 不支持 bnb；只能用 `mlx_lm` 或 Linux + CUDA
  - AMD ROCm 要装 `bitsandbytes-rocm`
  - 可以先在 Colab 训练，下载 adapter 回本地合并

- **Flash Attention 装不上**
  - `flash_attn` 需要 CUDA + Ampere 以上（3090/4090/A100）
  - Fallback：`attn_implementation: sdpa`（PyTorch 内置，慢 20% 但兼容）
  - Axolotl: `flash_attention: false` + `sdp_attention: true`

### 🟢 Adapter 管理

- **多个 adapter 合并选哪个**
  - 不同任务的 LoRA **不要互相合并**（会互相干扰），合并只用于"base + 这一个 adapter → 新 base"
  - 多任务路由：用 PEFT 的 `MultiAdapterModel` 或 S-LoRA（vLLM 内置支持）

- **adapter_config.json 丢了模型装不回去**
  - 训练配置必须保存；建议每次训练连同 yaml 一起 tar 到 `outputs/` 里

- **合并后大小翻倍** 
  - 正常：LoRA 只有 ~100MB，merge 后是 base 的完整大小（7B fp16 = 14GB）
  - 生产环境建议**保持 adapter 挂载**，vLLM/TGI 都支持动态 swap

### ⚪ 性能

- **训练太慢**
  - 检查是否在用 flash-attn（`flash_attn: fa2`）
  - `packing: true`（多个短样本拼一条，利用率高 2–3×）
  - 升级到 `transformers >= 4.45`（Liger Kernel 支持，省 20% 显存 + 快 20%）

- **推理太慢**（合并后）
  - 不要用 `transformers.generate()` 跑线上，换 vLLM / TGI（吞吐 10×+）
  - 见 [../inference/README.md](../inference/README.md)

---

## 6. 下一步

1. 先去 [llamafactory/README.md](./llamafactory/README.md) 跟着跑 Stage 2（最简单，纯文本 QLoRA）
2. 看完对照 [axolotl/README.md](./axolotl/README.md) 理解两个框架的差异
3. 训完的 adapter 合并到 `models/stage2_fused/`，然后 `docker compose up` 就能用新模型
4. 想再深入，去 [../inference/README.md](../inference/README.md) 用 vLLM/TGI 替代 transformers 推理
