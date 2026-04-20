# Raw Hugging Face 推理 —— 理解 vLLM / TGI / Ollama 在优化什么

> 本项目主服务 `src/serve/api.py` 就是用 HF `transformers.generate()`。本目录把这套 API **拆到最基础**，一个文件一个主题，方便读懂和面试复用。

## 为什么要学裸 HF 推理

- 面试题："用 `transformers` 跑一个 7B 模型生成" —— 必须会白板手写
- 调试 vLLM 出问题时 fallback 到 transformers 验证是**模型问题**还是**服务器问题**
- 小流量 / 单请求 / 本地 demo —— transformers 够用，不值得上 vLLM
- 理解 vLLM 的优化到底在省什么：看完这里对比 `inference/vllm/` 就懂了

## 本目录 6 个递进示例

| 文件 | 主题 | 关键 API | 对应常见问题 |
|---|---|---|---|
| `basic_generate.py` | 最基础的 CausalLM 推理 | `AutoModelForCausalLM.from_pretrained` + `generate` | "怎么用 HF 跑一个模型" |
| `with_adapter.py` | 挂 LoRA adapter / 合并 | `PeftModel.from_pretrained` + `merge_and_unload` | "训完 adapter 怎么用" |
| `streaming.py` | 流式输出（token-by-token） | `TextIteratorStreamer` + `Thread` | "怎么像 ChatGPT 那样一个字一个字出" |
| `batched.py` | 手动 batching（多请求一起推） | `tokenizer(..., padding=True)` + `generate` | "为什么我 100 个请求要 100 次 forward" |
| `pipeline_api.py` | `pipeline()` 快捷方式 | `transformers.pipeline` | "最快 5 行代码跑推理" |
| `serve_fastapi.py` | 最小 FastAPI wrapper | FastAPI + 全局 model | "怎么把推理变成 HTTP 服务" |

读完这 6 个，你就完全理解：
- vLLM 为什么快（continuous batching + PagedAttention）
- TGI 在实现啥（Rust router + 同样的 continuous batching）
- Ollama 在帮你省什么（GGUF 量化 + llama.cpp）

## 安装

```bash
cd inference/huggingface
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 跑

```bash
# 需要本项目 models/stage2_fused/ 存在（跑过 setup_demo_models.sh 或微调合并过）
python basic_generate.py             # 最基础
python streaming.py                   # 流式
python batched.py                     # batch
python with_adapter.py                # 挂 adapter (adapter 路径要改)
python pipeline_api.py                # pipeline
python serve_fastapi.py               # HTTP 服务，另起端子 curl localhost:8003/query
```

## HF 推理核心概念（读代码前先看）

### 1. 模型加载 3 个关键参数

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,        # 精度：bf16 省一半显存 + 速度快
    device_map="auto",                 # 自动把模型切到 GPU（单卡=cuda:0，多卡切分）
    attn_implementation="sdpa",        # flash_attention_2 / sdpa / eager
)
```

- **torch_dtype**：不设会用 fp32，显存爆炸
- **device_map**：不设模型在 CPU，推理会慢 1000×
- **attn_implementation**：`sdpa` 是 PyTorch 内置，最通用；flash-attn 更快但要 Ampere+ 卡

### 2. `generate()` 的几个重要参数

```python
outputs = model.generate(
    input_ids,                 # [batch, seq_len] tensor
    max_new_tokens=500,        # 生成多少 token（不是总长度）
    do_sample=True,            # True = 采样, False = greedy
    temperature=0.7,           # 采样温度
    top_p=0.9,                 # nucleus sampling
    top_k=50,                  # top-k sampling
    repetition_penalty=1.1,    # 重复惩罚
    eos_token_id=...,          # 遇到这些 token 停
    pad_token_id=...,          # padding token
    use_cache=True,            # KV cache，推理必开
)
```

**关键**：`do_sample=False`（greedy）推理最稳定，对比业务逻辑时用。`do_sample=True` 才有温度/top_p 意义。

### 3. Chat Template

Qwen / Llama / Mistral 等 chat 模型都有自己的对话模板（特殊 token 包围）：

```python
messages = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "你好"},
]
# 模板会产生：<|im_start|>system\n你是助手<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,        # 加上最后的 assistant 引导，让模型开始生成
)
```

**坑**：忘了 `add_generation_prompt=True` 模型不会生成（因为 prompt 以 `<|im_end|>` 结束，像个完成的对话）。

### 4. 解码

```python
# generate 返回的是 input + generated 的完整 ids
# 需要切掉 input 部分才是真正的回复
response_ids = outputs[0][input_ids.shape[1]:]
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
```

## HF 推理 vs vLLM —— 真实性能对比

同一模型（Qwen2.5-7B-Instruct, A100 40GB）:

| 指标 | HF transformers | vLLM | 差距 |
|---|---|---|---|
| 单请求 TTFT | 250ms | 100ms | 2.5× |
| 单请求完成（500 token） | 8s | 3s | 2.7× |
| **并发 16 吞吐** | **80 tok/s** | **2500 tok/s** | **30×** |
| GPU 利用率 | 20-40% | 85-95% | — |
| KV Cache 浪费 | 50-70% | <4% | — |

**什么场景 HF 够用**：
- 单请求、低 QPS（< 1 req/s）
- 开发 / 调试 / 本地
- 内部工具 / 批处理（大 batch 离线跑）

**什么场景必须上 vLLM**：
- 多用户并发
- 需要 p99 延迟 < 2s
- QPS > 5 req/s

## 还可以进阶的地方（本目录没展示但要知道）

| 技术 | API | 目的 |
|---|---|---|
| **KV Cache** | `past_key_values` | 多轮对话时复用上一轮计算，自己管理可省 30%+ |
| **Speculative Decoding** | `assistant_model` 参数 | 小模型起草 + 大模型校验，2× 加速 |
| **8-bit / 4-bit 推理** | `load_in_8bit=True` | 推理时降精度，省显存（和 QLoRA 量化不同） |
| **torch.compile** | `model = torch.compile(model)` | PyTorch 2.0+ JIT，10-30% 加速 |
| **Batch generation** | 手动 stack 多请求 | 见 `batched.py` |
| **Beam Search** | `num_beams=4` | 探索多条生成路径，质量好速度慢 |
| **Constrained Decoding** | `LogitsProcessor` | 强制 JSON / 特定格式输出 |
