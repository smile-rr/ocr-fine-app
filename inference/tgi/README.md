# TGI (Text Generation Inference) —— HuggingFace 官方推理服务

> 和 vLLM 同类的竞品，HuggingFace 官方维护。企业里两者都有人用，主要差别在生态。**不能在 Mac 上跑**（需要 CUDA/ROCm）。

## TGI vs vLLM

| 维度 | TGI | vLLM |
|---|---|---|
| 维护方 | HuggingFace | Anyscale/UC Berkeley |
| 吞吐 | 高 | 略高 (~20%) |
| API | 原生 `/generate` + OpenAI 兼容 | OpenAI 兼容 |
| LoRA 多 adapter | ✅ | ✅ S-LoRA |
| 许可证 | HFOIL (商用有限制，v2.0 后改 Apache) | Apache 2.0 |
| 集成 | HF Inference Endpoints / SageMaker 原生 | 独立 |
| 架构 | Rust router + Python model worker | 纯 Python |
| 量化 | AWQ/GPTQ/EETQ/bitsandbytes | AWQ/GPTQ/FP8/INT8 |
| 适合 | 深度用 HF 生态（SageMaker/Inference Endpoints） | 所有场景 |

**选型建议**：没特殊原因就用 vLLM。TGI 的优势是 AWS SageMaker / HF Endpoints 一键部署，业务已经在 HF 生态里的话省心。

---

## 1. 启动（需要 NVIDIA GPU）

```bash
cd inference/tgi
docker compose up -d
docker compose logs -f tgi
```

验证：
```bash
curl http://localhost:8080/health
curl http://localhost:8080/info   # 看当前 loaded model
```

---

## 2. 调用（两种方式）

### 原生 `/generate`
```bash
curl http://localhost:8080/generate \
    -H 'Content-Type: application/json' \
    -d '{
      "inputs": "请介绍 RAG",
      "parameters": {"max_new_tokens": 200, "temperature": 0.1}
    }'
```

### OpenAI 兼容 `/v1/chat/completions`
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="tgi",  # TGI 的 model 字段无意义
    messages=[{"role": "user", "content": "你好"}],
)
```

---

## 3. LoRA 热加载

TGI 2.0+ 支持 LoRA，启动时预声明：
```yaml
command: >
  --model-id Qwen/Qwen2.5-7B-Instruct
  --lora-adapters adapter-a=/data/adapter-a,adapter-b=/data/adapter-b
```

请求时用 `adapter_id` 参数路由：
```bash
curl /generate -d '{"inputs":"...","parameters":{"adapter_id":"adapter-a"}}'
```

---

## 4. 常见问题

- **镜像超大（>20GB）** → TGI 官方镜像就是这么大；首次拉需要 30 分钟+
- **报 `flash-attn` 不兼容** → GPU 架构太老；TGI 用 `--attention sdpa` fallback
- **`Repository Not Found`** → 私有模型要设 `HUGGING_FACE_HUB_TOKEN` 环境变量
- **内存不够加载 7B** → `--quantize bitsandbytes-nf4` 走 4-bit
- **多卡** → `--num-shard 2` 会 tensor parallel 切 2 张 GPU
