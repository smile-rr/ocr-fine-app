# Azure 部署参考

> Azure 上 host LLM 的 4 种姿势。目录结构和 AWS 镜像 —— 对照读效果最好。

## 4 种方案速览

| 子目录 | Azure 服务 | AWS 对应 | 何时选 |
|---|---|---|---|
| [openai-service-client/](./openai-service-client/) | **Azure OpenAI Service** | Bedrock | 只调 API，不 host 自己模型 |
| [aml-endpoint/](./aml-endpoint/) | **Azure ML Online Endpoint** | SageMaker Endpoint | host 自家微调模型 |
| [container-apps/](./container-apps/) | **Container Apps** | ECS Fargate | 容器业务，不上 K8s |
| [aks-vllm/](./aks-vllm/) | **AKS** | EKS | 完全控制，多模型 / 多租户 |

## 选型 —— 和 AWS 的关键差异

### 1. Azure OpenAI vs Bedrock

Azure OpenAI 几乎只卖 **OpenAI 家的模型**（GPT-4o、GPT-4, GPT-3.5, Embeddings, DALL-E）。没有 Claude、Llama、Mistral。

优势：
- **数据不出 Azure**（OpenAI API 官网也对企业承诺，但 Azure 合同更严）
- **区域 + 私有 Endpoint** 让流量不走公网
- **Entra ID（原 AAD）集成** 免 API Key（用 Managed Identity）

劣势：
- 模型仅限 OpenAI 自家
- Region 覆盖有限（部分模型仅在 East US、West Europe）
- Quota 要申请（尤其 GPT-4o）

### 2. Azure ML vs SageMaker

| | Azure ML Online Endpoint | SageMaker Endpoint |
|---|---|---|
| 模型注册 | **MLflow / custom** | SageMaker Model Registry |
| 部署单位 | Deployment（一个 Endpoint 下多个 Deployment） | Variant（一个 Endpoint 下多个 Variant） |
| 金丝雀 | 流量切分 = `az ml online-endpoint update --traffic "blue=90 green=10"` | Variant 权重 |
| 推理容器 | 自定义 image + `MLflowModel` / `MLModel` | DJL-LMI / TGI / PyTorch DLC |
| 计费 | 实例/小时 | 实例/小时 |
| 抽象层 | 更贴近"数据科学家"体验（AML Studio UI 好用） | 更贴近"工程师"体验（SDK 清爽） |
| 学习曲线 | 概念多（workspace/compute/environment/endpoint） | 概念少（model/endpoint/variant） |

**Azure ML 额外特性**：
- **Workspace** 统一管理数据集、模型、实验、endpoint（类似 mini-MLOps 平台）
- **Managed Compute** 训练集群，跑微调任务（对应 SageMaker Training Jobs）
- **Prompt Flow**：可视化 RAG/LLM pipeline 编排

### 3. Container Apps vs Fargate

Container Apps 是 Azure 的"无 K8s 容器"托管服务，**比 Fargate 多给你**：
- 内置 **Dapr**（服务网格 lite）
- 内置 **KEDA** 事件驱动扩缩（按 Kafka / Queue 深度缩放）
- **Scale-to-zero**（Fargate 做不到真零实例）
- **Built-in ingress + TLS**（不用单独 ALB）

不过 Container Apps 的 **GPU 支持同样受限**：
- Consumption plan: ❌ 无 GPU
- Dedicated plan: ⚠️ 预览期（2025-）

GPU 推理还是走 AML Endpoint 或 AKS。

### 4. AKS vs EKS

高度平行，主要差异：
- **节点镜像**：AKS 用 Ubuntu/AzureLinux，EKS 用 Amazon Linux 2/Bottlerocket
- **身份**：AKS → **Workload Identity**（对应 EKS IRSA）
- **存储**：AKS → **Azure Files/Disk/NetApp Files**（对应 EFS/EBS/FSx）
- **自动扩容**：AKS 用 **Cluster Autoscaler** 或 **Karpenter-AKS**（预览）
- **推理优化**：Azure 有 **KAITO**（K8s AI Toolchain Operator）—— 微调 + 推理 CRD，对应 AWS 没有的层级

## 共同前置

```bash
# 1. Azure CLI
brew install azure-cli
az login

# 2. 订阅和 resource group
export SUB="your-subscription-id"
az account set --subscription $SUB
az group create --name rg-ocr-fine-app --location eastus

# 3. Bicep CLI（Azure CLI 自带）
az bicep version

# 4. 确认 GPU quota
az vm list-usage --location eastus --query "[?contains(name.value, 'NC') || contains(name.value, 'ND')]" -o table
```

**成本 Tip**：
- 用 `eastus` / `southcentralus` 模型最全
- GPU VM 系列对比：NC (V100) > NCas_T4 > NCads_A100 > ND
- Low priority VM = Spot，便宜 80%，但可被抢占
- Azure OpenAI 是 PAYG，便宜；但 PTU (Provisioned Throughput Units) 贵

## Azure 身份最佳实践

| 场景 | 推荐 |
|---|---|
| 本地 dev | `az login` / Azure CLI credential |
| VM / Container Apps | **System-assigned Managed Identity** |
| AKS Pod | **Workload Identity** (OIDC, 等同 IRSA) |
| CI/CD | GitHub Actions + OIDC → Service Principal |
| 凭证 | **Key Vault** + 引用，不在代码里 |

**绝不要用 Service Principal + client secret 硬编码**。Managed Identity 优先。
