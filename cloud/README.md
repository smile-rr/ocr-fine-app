# Cloud Deployment — AWS / Azure 参考

> 把前面的所有方案（Docker 主服务、vLLM、K8s/OpenShift）落到两朵主流云上。回答的核心问题：
>
> **同一个 LLM 服务，在云上我该选哪种部署方式？**

## 决策树（一图看懂）

```
         ┌─ 不想 host 模型 / 只调 API ─► Bedrock / Azure OpenAI          ← 最省事，但锁定云且贵
         │
         ├─ 完全托管，不想管容器 ─────► SageMaker / Azure ML Endpoint     ← 次省事，每小时按实例计费
你的模型 ─┤
         ├─ 要容器但不上 K8s ─────────► ECS Fargate / Container Apps     ← 中等，会 Docker 就行
         │
         └─ 要完全控制 / 多模型 / 自研 ─► EKS / AKS + vLLM               ← 最自由，运维成本最高
```

## AWS ↔ Azure 对照速查

| 用途 | AWS | Azure | 本目录位置 |
|---|---|---|---|
| **托管 API（不 host 自己的模型）** | Bedrock | Azure OpenAI Service | `aws/bedrock-client/`、`azure/openai-service-client/` |
| **托管推理端点（host 自己微调的）** | SageMaker Endpoint | Azure ML Online Endpoint | `aws/sagemaker-endpoint/`、`azure/aml-endpoint/` |
| **托管容器（不上 K8s）** | ECS Fargate / App Runner | Container Apps / ACI | `aws/ecs-fargate/`、`azure/container-apps/` |
| **托管 K8s + 自建栈** | EKS | AKS | `aws/eks-vllm/`、`azure/aks-vllm/` |
| 对象存储（模型权重） | S3 | Blob Storage | — |
| 镜像仓库 | ECR | ACR | — |
| 身份 | IAM + IRSA | Managed Identity + Workload Identity | — |
| 监控 | CloudWatch + X-Ray | Monitor + App Insights | — |
| 配置/密钥 | SSM Parameter Store + Secrets Manager | Key Vault + App Config | — |
| CI/CD | CodePipeline / GitHub Actions + OIDC | Azure DevOps / GitHub Actions + OIDC | — |
| IaC 主流 | **Terraform** / CDK / CloudFormation | **Bicep** / Terraform / ARM | 本目录用 Terraform / Bicep |

## 成本与锁定速查

| 方案 | 门槛 | 每小时 (~7B 模型, A10G/T4) | 锁定度 | 冷启动 |
|---|---|---|---|---|
| Bedrock / Azure OpenAI | 零 | 按 token 计费（$15-75 per M tokens，和自 host 差不多） | 🔴 高（模型选择有限） | ~1s |
| SageMaker / Azure ML Endpoint | 低 | ~$0.75-$3/h | 🟡 中（IaC 可迁移） | 2-5 min 首次 |
| ECS Fargate / Container Apps | 中 | ~$0.5-$2/h | 🟢 低（Docker 镜像可迁） | 1-3 min |
| EKS / AKS + vLLM | 高 | ~$0.5-$2/h + 集群费 $73/月 | 🟢 低（K8s manifests 跨云通用） | 2-5 min |

**实操建议**：
- POC / Demo → **托管 API**（Bedrock / Azure OpenAI），快
- 小团队生产（<10 QPS）→ **SageMaker / Azure ML Endpoint**，省运维
- 中大规模 / 已有 K8s 栈 → **EKS / AKS**，复用 `inference/kubernetes/` 整套
- 纯粹想省钱 → **ECS Fargate / Container Apps**，不用养 K8s

## IaC 工具选择

本目录默认：
- **AWS → Terraform**（跨云可复用，AWS provider 最成熟；CDK/CloudFormation 仅在 README 里提）
- **Azure → Bicep**（微软官方现代选择，YAML 友好；也可以用 Terraform）

两边都给：**IaC + 一个 deploy.sh**，读者用自己的账号跑。没账号的话把 IaC 当文档读也能学到。

## 如何把之前建好的东西塞进云

| 你之前的产物 | AWS 对应 | Azure 对应 |
|---|---|---|
| `models/stage2_fused/`（本地目录） | S3 `s3://my-bucket/stage2/` | Blob `container/stage2/` |
| `Dockerfile`（主服务镜像） | ECR `123.dkr.ecr.../ocr-api:v1` | ACR `myregistry.azurecr.io/ocr-api:v1` |
| `inference/kubernetes/base/` | EKS 集群 `kubectl apply -k` | AKS 集群 `kubectl apply -k` |
| `inference/kubernetes/adapter-ops/` | EKS + IRSA + EFS RWX | AKS + Workload Identity + Azure Files |
| `inference/vllm/docker-compose.yml` | Fargate TaskDefinition | Container App |
| OpenAI-SDK 客户端 | Bedrock Converse API / SageMaker Runtime | Azure OpenAI（原样 SDK + 改 base_url） |

---

下一步：进 `aws/README.md` 或 `azure/README.md` 选一朵云开始读。两边目录结构平行，对比学习效果最好。
