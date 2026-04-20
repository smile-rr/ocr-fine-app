# Ollama —— Mac 上真能跑起来的推理 server

Ollama 是把 llama.cpp 包装成一个 OpenAI 兼容的本地 HTTP server。虽然不是"高性能"引擎（没 continuous batching），但它是 Mac 上**唯一能真正跑生产风格 API 的方案**。

## 为什么选 Ollama 做 Mac 上的参考

| 方案 | Mac 能跑 | OpenAI API | 性能 |
|---|---|---|---|
| vLLM | ❌ (无 CUDA) | ✅ | 极高 |
| TGI | ❌ | ✅ | 高 |
| SGLang | ❌ | ✅ | 极高 |
| **Ollama** | ✅ **Apple Silicon GPU (Metal)** | ✅ | 中（够 demo） |
| llama.cpp 裸跑 | ✅ | ❌ | 中 |

Ollama = llama.cpp + HTTP server + 模型仓库。Mac M1/M2/M3 上自动用 Metal 加速，吞吐不如 vLLM 但比 `transformers.generate()` 快 2-5×。

---

## 1. 安装（5 分钟）

```bash
# Mac
brew install ollama

# 启动 server（后台常驻）
ollama serve
# 或 `brew services start ollama` 开机自启
```

验证：`curl http://localhost:11434/api/version`

---

## 2. 跑官方预量化的 Qwen2.5

最快路径——直接用 Ollama 仓库的预量化模型：

```bash
# 0.5B 版本，300MB 左右
ollama pull qwen2.5:0.5b-instruct-q4_K_M

# 1.5B 版本
ollama pull qwen2.5:1.5b-instruct-q4_K_M

# 测试
ollama run qwen2.5:0.5b-instruct-q4_K_M "介绍一下你自己"
```

默认模型文件在 `~/.ollama/models/`。

---

## 3. 加载本项目合并后的模型（可选，进阶）

如果你想用 `models/stage2_fused/`（项目自己微调合并出来的模型），要先转成 GGUF 格式。

### 3.1 用 llama.cpp 转 GGUF

```bash
# 从 inference/ollama/ 目录跑
bash scripts/hf_to_gguf.sh
```

脚本做的事：
1. Clone `llama.cpp` 到临时目录
2. `python convert_hf_to_gguf.py models/stage2_fused/` → `stage2.gguf` (fp16)
3. `./llama-quantize stage2.gguf stage2-q4_k_m.gguf Q4_K_M`（4-bit 量化）

### 3.2 创建 Ollama Modelfile

见 `Modelfile` —— 声明 FROM stage2.gguf + 系统 prompt + 参数。

```bash
ollama create ocr-stage2 -f Modelfile
ollama run ocr-stage2
```

---

## 4. 用 OpenAI SDK 调用（企业级做法）

Ollama 在 `:11434` 暴露 `/v1/chat/completions`，SDK 直接无缝切换：

```bash
pip install openai
uv run python client_example.py
```

见 [client_example.py](./client_example.py)。核心就一句：
```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

---

## 5. 和 RAG Pipeline 集成

这里给一个**参考版 RAG endpoint**，它和当前项目的 `src/serve/api.py` 并存但不冲突（独立端口，调用 Ollama 而不是本地 transformers）：

```bash
uv run python rag_server_ollama.py
# 启动在 :8001（主 API 在 :8000），也调你项目的 chroma_db/
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"哪年营收最高？","top_k":3}'
```

见 [rag_server_ollama.py](./rag_server_ollama.py)。**这个文件就是企业级改造示例**：RAG 业务层只做编排，推理通过 HTTP 调 Ollama。

---

## 6. 常见问题

- **`connection refused :11434`** → `ollama serve` 没启动；或用 `brew services start ollama`

- **模型跑起来但吐乱码** → Modelfile 里模板不对。Qwen 用 `chatml`，Llama3 用 `llama3`。`ollama show --modelfile qwen2.5:0.5b-instruct-q4_K_M` 看官方是怎么写的。

- **想多个模型并发** → Ollama 默认每个模型单独加载（会占显存）。设 `OLLAMA_NUM_PARALLEL=4` 提高单模型并发；`OLLAMA_MAX_LOADED_MODELS=2` 控制同时加载几个。

- **内存不够** → `ollama pull ... --insecure` 不会帮你，只能选更小的量化（`q4_K_S` > `q3_K_M`）或更小的模型。

- **M 系列 GPU 没满载** → `OLLAMA_NUM_GPU=99` 把所有层放 GPU（默认是自动）。

- **生产环境能用吗**：勉强，但没 continuous batching，吞吐差。Mac 部署 / 个人 demo / 离线场景 OK。服务器用 vLLM。
