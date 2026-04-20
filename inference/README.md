# Inference — 企业级推理框架参考

> 本项目主 Docker API (`src/serve/api.py`) 用 `transformers.generate()` 跑模型——简单但**不适合生产**：单请求串行、无 continuous batching、GPU 利用率 ~20%。
>
> 本目录提供 3 种企业级推理方案的参考实现，和当前代码**完全独立**，不会影响 Streamlit / Docker 主服务。

---

## 四大主流推理引擎对比

| 引擎 | 平台 | 吞吐 | 批处理 | 接口 | 量化 | 多 adapter | 适合 |
|---|---|---|---|---|---|---|---|
| **vLLM** | Linux + CUDA | ★★★★★ | Continuous + PagedAttention | OpenAI 兼容 | AWQ/GPTQ/FP8/INT8 | ✅ S-LoRA | **LLM 生产首选** |
| **TGI** | Linux + CUDA/ROCm | ★★★★ | Continuous | 自研 + OpenAI | 同 vLLM | ✅ | HuggingFace 生态深耦合 |
| **SGLang** | Linux + CUDA | ★★★★★ | RadixAttention | OpenAI 兼容 | 多样 | ✅ | 需结构化输出/复杂 prompt |
| **Ollama** | **Mac/Linux/Win** | ★★ | ❌ 串行 | OpenAI 兼容 | GGUF (Q4/Q5/Q8) | 手动切换 | **本地开发/Mac 首选** |
| Triton | 全平台 | ★★★★ | 多样 | gRPC/HTTP | 多样 | ✅ | 多框架多模型混布 |

**本目录实现**：
| 目录 | 能在你机器上 run 吗 | 用途 |
|---|---|---|
| [**huggingface/**](./huggingface/) ⭐ | ✅ Mac/Linux 都能 | **裸 HF 推理栈**，理解下面一切的基础（6 个递进示例） |
| [vllm/](./vllm/) | ❌ 需 NVIDIA GPU | 生产级，看代码理解架构 |
| [ollama/](./ollama/) | ✅ Mac/Linux 都能 | 本地开发 + 真能跑通 |
| [tgi/](./tgi/) | ❌ 需 NVIDIA GPU | 参考 HF 官方方案 |
| [kubernetes/](./kubernetes/) | ⚠️ 需 K8s/OpenShift 集群 | **生产部署 + adapter 安全 ops + 金丝雀** |

---

## 核心概念速记

### 1. Continuous Batching（vLLM / TGI / SGLang 核心）

传统 batching：**等齐一批请求才开始推理** → 延迟高，如果一条请求 token 长就拖慢全部。

Continuous batching：**每生成一个 token 就重新决定 batch 组成**
- 新请求随时加入（不用等）
- 生成完成的请求立即退出
- 吞吐量提升 **10-20 倍**

### 2. PagedAttention（vLLM 首创）

KV Cache 是推理的主要显存消耗（比权重还大），传统分配是连续的，浪费严重。

PagedAttention：把 KV Cache 切成固定大小的 block（类似 OS 虚拟内存分页）
- 按需分配，浪费 <4%（传统 60%+）
- 多请求共享相同 prefix（system prompt、few-shot examples）
- 直接把同容量的 GPU 吞吐量提升 2-4×

### 3. OpenAI 兼容 API（行业事实标准）

vLLM / TGI / Ollama / SGLang 都提供 `/v1/chat/completions` 接口，和 OpenAI 官方 SDK 完全兼容：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(model="qwen2.5-0.5b", messages=[...])
```

这意味着：
- 业务代码只写一份，可以随时在 OpenAI 官方 / 自托管 / 多家云供应商之间切换
- 客户端不变，底层推理引擎随意换
- 你现在 `src/serve/api.py` 里自定义的 `/extract` `/query` 是业务编排层，**不应该自己跑模型**，应该调 `localhost:8000/v1/chat/completions` 让专门的推理引擎跑

### 4. 量化方案对比

| 方案 | 位宽 | 精度损失 | 推理速度 | 训练 | 适合 |
|---|---|---|---|---|---|
| BF16 (原始) | 16 | 0% | 基线 | ✅ | - |
| AWQ | 4 | <1% | 2× | ❌ | 生产推理首选 |
| GPTQ | 4 | 1-2% | 2× | ❌ | 历史老方案 |
| FP8 | 8 | <0.5% | 1.5× | ❌ | H100 独占 |
| NF4 (QLoRA) | 4 | 2-3% | 慢 0.7× | ✅ | **只为训练**，不要拿去推理 |
| GGUF Q4_K_M | 4 | <2% | 2× | ❌ | llama.cpp/Ollama |

**关键**：训练用的 QLoRA (NF4) 权重**不适合推理**，要重新用 AWQ 量化一次。这是很多人搞错的坑。

---

## 推荐部署拓扑

```
生产环境（云端 GPU）                        开发环境（你的 Mac）
─────────────────────────                    ─────────────────────────
客户端 ──► FastAPI 业务层                    Streamlit/CLI ──► FastAPI
             │ (RAG 编排)                               │ (RAG 编排)
             ▼                                         ▼
          vLLM Server                              Ollama Server
          (4× H100 continuous batch)               (本地 GGUF 量化)
             │                                         │
             ▼                                         ▼
          Qdrant/Milvus                            ChromaDB (本地)
```

业务层（FastAPI）**只做编排**，不 host 模型；推理层（vLLM/Ollama）独立部署独立扩缩。

---

## 怎么和本项目对接

**现状**：`src/serve/api.py` 把三件事捆在一起
1. HTTP 业务接口（/extract, /query）
2. RAG 编排（检索 → 组 prompt）
3. 模型推理（`model.generate()`）

**改造目标**：把 #3 剥出去，业务层通过 HTTP 调推理 server。

**最小改动示例**（伪代码，**本次不改**）：
```python
# src/serve/api.py 的 query 端点改成
from openai import OpenAI
llm = OpenAI(base_url="http://vllm-server:8000/v1", api_key="dummy")

@app.post("/query")
def query(req: QueryIn):
    hits = chroma.query(...)
    messages = build_rag_prompt(req.question, hits)
    resp = llm.chat.completions.create(
        model="qwen2.5-0.5b",
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return {"answer": resp.choices[0].message.content, "sources": hits}
```

改完后：
- `transformers` 依赖可以从 Docker 镜像里删掉（大幅减小镜像）
- 多个业务容器可以共享一个 vLLM
- vLLM 可以独立扩缩、换模型、热加载 adapter

---

## 下一步

- **先读裸 HF** → [huggingface/](./huggingface/) ⭐（6 个单文件示例，面试白板题必会）
- 有 NVIDIA GPU → 去 [vllm/](./vllm/) 跑一遍，体会 continuous batching 的吞吐
- Mac 或没 GPU → 去 [ollama/](./ollama/) 跑本地版（用你现在 `models/stage2_fused/` 的权重都行）
- 想理解 HuggingFace 官方生态 → 看 [tgi/](./tgi/) 的 compose 文件
- **准备上生产 / K8s / OpenShift** → **[kubernetes/](./kubernetes/)** 里有完整的：
  - `base/` 可 kubectl apply 的部署
  - `canary/` 金丝雀发布（Argo Rollouts + Istio）
  - `adapter-ops/` ⭐ **LoRA 热加载的 3 种安全做法**（回答"URL 直调不安全"的正确姿势）
  - `openshift/` OpenShift 专属差异（SCC / Route / KServe）
