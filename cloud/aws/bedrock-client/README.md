# AWS Bedrock —— 不 host 自己模型的玩法

Bedrock 是 AWS 的"托管 foundation model API"：调用 Claude、Llama、Mistral、Titan 等一堆模型，按 token 计费，**不用管 GPU / 不用部署 / 不用微调基础设施**。

和你前面所有 host-your-own 的路径（vLLM / SageMaker / EKS）是互补关系 —— 业务如果根本不需要自己微调，Bedrock 是最省心的选择。

## 什么时候选 Bedrock

✅ **适合**：
- 业务用通用大模型就够（总结、翻译、分类、普通 QA）
- 不想管 GPU 成本和运维
- 要合规 / 数据不出 AWS（Bedrock 保证数据不用于训练）
- Anthropic Claude 是刚需（AWS 是 Claude 的主力云）

❌ **不适合**：
- 要微调自有数据（Bedrock 支持有限的 continued pre-training，但不如自己微调灵活）
- 要 LoRA 热加载
- 要成本最优且 QPS 稳定（长期算反而比自 host 贵）
- 要 OpenAI 兼容 API（Bedrock 是自己的协议，需要 adapter）

## Bedrock vs OpenAI API vs 自 host vLLM

| 维度 | Bedrock | OpenAI API | 自 host vLLM |
|---|---|---|---|
| 部署 | 0 代码 | 0 代码 | 你自己 |
| 模型 | Claude/Llama/Mistral/Titan 等 | GPT 系列 | 任意开源 |
| 计费 | per-token | per-token | per-hour 实例 |
| 数据隔离 | ✅ | ⚠️ opt-out 训练 | ✅ 自己控制 |
| 延迟 | ~1s TTFT | ~1s | 取决于部署 |
| 细调 | 有限（CPT） | 有（GPT-4 fine-tune） | 完全控制 |
| 定制 system prompt | ✅ | ✅ | ✅ |
| 合规/VPC | ✅ PrivateLink | ⚠️ 只有企业版 | ✅ |

## Bedrock 模型速览（2026-04）

Bedrock 上能调的主要模型（region 有差异）：

| 模型家族 | 代表 | 上下文 | 擅长 |
|---|---|---|---|
| **Anthropic Claude** | claude-3-5-sonnet, claude-3-opus, claude-haiku | 200K | 综合能力顶级，长文推理 |
| **Meta Llama** | llama-3.1-70b-instruct, llama-3.2-90b-vision | 128K | 开源旗舰，多模态 |
| **Mistral** | mistral-large-2, mixtral-8x22b | 128K | 欧洲合规，代码好 |
| **Cohere** | command-r-plus | 128K | RAG 优化 |
| **Amazon Titan** | titan-text-express, titan-embeddings | 32K | AWS 自家，便宜 |
| **AI21** | jamba-1-5-large | 256K | 长上下文 |

## 三种调用方式

Bedrock 有两代 API，新业务用 **Converse API**（统一接口，兼容所有模型）。

### 1. Converse API（推荐，2024 新）

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

resp = client.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[
        {"role": "user", "content": [{"text": "什么是 RAG？"}]}
    ],
    system=[{"text": "你是数据分析助手。"}],
    inferenceConfig={"maxTokens": 500, "temperature": 0.1},
)
print(resp["output"]["message"]["content"][0]["text"])
```

### 2. InvokeModel API（老接口，每个模型参数不同）

```python
import json, boto3
client = boto3.client("bedrock-runtime")

resp = client.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": "你好"}],
    }),
)
print(json.loads(resp["body"].read())["content"][0]["text"])
```

### 3. 用 OpenAI SDK 兼容层（LiteLLM 或 bedrock-access-gateway）

如果业务代码已经写了 OpenAI SDK，不想改：
```bash
pip install litellm
```
```python
from litellm import completion
resp = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "你好"}],
)
```

LiteLLM 在本地做协议转换，客户端代码和调 OpenAI 完全一样。

## IaC：最小资源（主要是 IAM）

Bedrock 本身不需要 provision 任何资源 —— 模型是 AWS 自己的。你要准备的只有 **IAM policy** 和可选的 **guardrails**（内容安全策略）。见 `terraform/main.tf`。

## 本目录文件

```
bedrock-client/
├── README.md                     ← 本文件
├── client.py                     ← Converse + LiteLLM 两种示例
├── rag_with_bedrock.py           ← 把 Bedrock 接到你现有的 RAG pipeline
└── terraform/
    ├── main.tf                   ← IAM role + Guardrails + PrivateLink VPC endpoint
    └── variables.tf
```
