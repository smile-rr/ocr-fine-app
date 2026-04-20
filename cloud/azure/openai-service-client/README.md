# Azure OpenAI Service —— 企业级 OpenAI API

Azure 版本的 OpenAI API。**等同于** OpenAI 官网 API，但：
- 数据留在 Azure region 内（不经 OpenAI 公网）
- 用 Azure AD 认证（Managed Identity，不用 API key）
- 可配私有 endpoint + VNet integration

## Azure OpenAI vs OpenAI 官网 API

| | Azure OpenAI | OpenAI 官网 |
|---|---|---|
| 模型 | GPT-4o / GPT-4 / GPT-3.5 / Embeddings / DALL-E / Whisper | 同 + o1 系列（Azure 新发布滞后 2-4 周） |
| 认证 | Entra ID (Managed Identity) 或 API Key | API Key |
| 数据 | 不离开你选的 region | OpenAI 自家服务器（承诺不训练，但商务合同弱） |
| 私网 | ✅ Private Endpoint + VNet | ❌ |
| 合规 | SOC 2, HIPAA, FedRAMP High（美国政府） | SOC 2 |
| 计费 | 按 Azure 订阅 | OpenAI 单独账单 |
| Rate Limit | 按部署（Deployment）配 TPM/RPM | 按账户 tier |

**企业里几乎都走 Azure OpenAI**，原因是数据合规 + 统一账单 + Azure AD 集成。

## Azure OpenAI 的关键概念

1. **Resource** —— Cognitive Services account，region 相关
2. **Deployment** —— 在 Resource 下部署一个具体模型（`gpt-4o-2024-08-06`），分配 TPM 配额
3. **Endpoint URL** —— `https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-08-01-preview`

和 OpenAI 官网 URL 结构差很多，所以 SDK 调用姿势略不同。

## 两种 SDK 调用姿势

### 方式 1: OpenAI Python SDK（推荐）

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://my-resource.openai.azure.com",
    api_version="2024-08-01-preview",
    # 二选一：
    api_key="...",                                           # API Key 方式
    # 或者 azure_ad_token_provider=token_provider,          # Entra ID 方式
)

resp = client.chat.completions.create(
    model="gpt-4o",                     # 这里填 DEPLOYMENT 名（不是模型名）
    messages=[{"role": "user", "content": "你好"}],
)
```

### 方式 2: Managed Identity（生产推荐，零凭证）

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint="https://my-resource.openai.azure.com",
    api_version="2024-08-01-preview",
    azure_ad_token_provider=token_provider,
)
```

`DefaultAzureCredential` 按优先级查凭证：
- 本地 dev: `az login` 的 token
- VM / Container Apps / Functions: **System-assigned Managed Identity**
- AKS Pod: **Workload Identity** (Pod SA → AKS OIDC → Entra ID)
- CI/CD: GitHub Actions OIDC

## Bicep 部署

见 `bicep/main.bicep`。做的事：
1. 创建 Cognitive Services account（`kind: OpenAI`）
2. 部署 `gpt-4o` 和 `text-embedding-3-small` 两个模型
3. 配 Private Endpoint 到 VNet（生产必备）
4. 给指定 Managed Identity `Cognitive Services OpenAI User` 角色

```bash
cd bicep
az deployment group create \
    --resource-group rg-ocr-fine-app \
    --template-file main.bicep \
    --parameters prefix=ocr prefix=eastus
```

## Quota 和 PTU

**TPM (Tokens Per Minute) 配额**：
- 新资源默认 60K-120K TPM（很少，生产要申请）
- 单独申请上限：https://aka.ms/oai/quotaincrease

**PTU (Provisioned Throughput Units)**：
- 按月/按年买保证吞吐（贵，但可预测）
- 适合 QPS 稳定的大客户
- 最小 50 PTU 起，大致 $50K/月起步

**选择**：
- MVP / 原型 → PAYG（按 token 计费）
- 稳定生产 → 观察实际 QPS → 算清楚后决定是否 PTU

## Responsible AI & Content Filter

Azure OpenAI **强制**启用内容过滤（Content Filter），所有请求走一个 Severity Score 分类器。

可以自定义过滤策略：
```bash
az cognitiveservices account deployment create \
    --raiPolicy "Microsoft.Default"    # 或 custom policy
```

不能完全关，违规场景必须关时需要申请 Abuse Monitoring 豁免。

## 本目录文件

```
openai-service-client/
├── README.md                     ← 本文件
├── client.py                     ← 同步 / 流式 / Managed Identity 三种
├── rag_with_azure_openai.py     ← RAG Pipeline 参考
└── bicep/
    ├── main.bicep                ← Cognitive Services + Deployment + Private Endpoint + RBAC
    └── parameters.json
```
