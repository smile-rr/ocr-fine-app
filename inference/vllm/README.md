# vLLM —— LLM 推理生产标杆

> **不能在 Mac 上跑**（需要 NVIDIA CUDA）。本目录给一个完整的 Docker Compose 配置作为**参考**，等你上 Linux GPU 机器时可以直接用。

## 为什么 vLLM 是事实标准

- **PagedAttention**：KV Cache 分页管理，显存浪费 <4%（传统 60%+）
- **Continuous Batching**：token-level 动态批处理，吞吐 10-20×
- **OpenAI 兼容 API**：`/v1/chat/completions`、`/v1/embeddings`，直接复用业务代码
- **量化**：AWQ / GPTQ / FP8 / INT8 / bitsandbytes 都原生支持
- **S-LoRA**：单 base 模型同时挂载多个 LoRA adapter，按请求路由

GitHub 40k+ stars，被 Anyscale / Mistral / IBM / Cloudflare 等在生产用。

---

## 1. 快速跑 Docker Compose（需要 NVIDIA GPU + nvidia-container-toolkit）

```bash
cd inference/vllm
docker compose up -d

# 跟日志
docker compose logs -f vllm
# 看到 "Uvicorn running on http://0.0.0.0:8000" 就 OK
```

验证：
```bash
curl http://localhost:8000/v1/models
```

---

## 2. 配置讲解（docker-compose.yml）

关键参数：
- `--model Qwen/Qwen2.5-7B-Instruct` — 会从 HF 自动下载；也可以指向本地路径 `/models/stage2_fused`
- `--tensor-parallel-size 2` — 跨 2 张 GPU 切分权重（单卡设 1）
- `--gpu-memory-utilization 0.9` — GPU 显存使用上限（0.9 = 90%）
- `--max-model-len 4096` — 支持的最大上下文长度
- `--quantization awq` — 如果模型是 AWQ 量化版，加这个；原始 fp16 不要加

## 3. 和本项目对接

把 `docker-compose.yml` 里的 model 指向合并后的模型：
```yaml
environment:
  - MODEL_PATH=/models/stage2_fused
volumes:
  - ../../models:/models:ro
command: >
  --model /models/stage2_fused
  --served-model-name stage2
  ...
```

然后业务层（`src/serve/api.py` 或自己写的）通过 OpenAI SDK 调：
```python
from openai import OpenAI
llm = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = llm.chat.completions.create(model="stage2", messages=[...])
```

完整的业务层示例见 `inference/ollama/rag_server_ollama.py` —— 把 `base_url` 从 Ollama 改成 vLLM 就完事。

---

## 4. 多 LoRA Adapter（S-LoRA，vLLM 独有强项）

生产场景：一个 base 模型服务多租户，每个租户有自己的 LoRA 定制。

```bash
# 启动时声明允许挂 LoRA
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 32 \
    --lora-modules tenant-a=/models/adapter-a tenant-b=/models/adapter-b

# 请求时通过 model 字段路由
curl /v1/chat/completions -d '{"model":"tenant-a","messages":[...]}'
```

动态热加载（运行时加新 adapter）：
```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -d '{"lora_name":"tenant-c","lora_path":"/models/adapter-c"}'
```

---

## 5. 性能基准（7B 模型，1 × A100 40GB）

| 引擎 | 单请求 latency (TTFT/P99) | 并发 16 吞吐 |
|---|---|---|
| transformers.generate | 300ms / 5s | 80 tok/s |
| TGI | 120ms / 2s | 1200 tok/s |
| **vLLM** | **100ms / 1.5s** | **2500 tok/s** |
| SGLang | 90ms / 1.4s | 3000 tok/s |

vLLM ≈ 30× 于本项目现在的 `transformers.generate`。

---

## 6. 常见问题

- **CUDA out of memory at startup** → 降 `--gpu-memory-utilization` 到 0.85，或减 `--max-model-len`
- **`This model does not support LoRA`** → 确认 base 架构 vLLM 支持 LoRA（大部分 Qwen/Llama/Mistral 都支持）
- **生成慢 / 吞吐低** → 没开 continuous batching？检查 `--max-num-seqs`（默认 256 够了）
- **prefill 巨慢** → 开 `--enable-chunked-prefill`（长 prompt 场景大幅提速）
- **OpenAI SDK 报 404** → URL 要是 `/v1`，不是 `/` 根路径
