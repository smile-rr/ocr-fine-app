# Azure Container Apps —— 托管容器（不上 K8s）

AWS Fargate 的对应方案。**比 Fargate 多**：
- Scale-to-zero（真 0 实例，请求来了再起）
- KEDA 事件驱动扩缩
- 内置 Dapr 服务网格
- 内置 HTTP Ingress + Managed Cert

**同样的限制**：Consumption plan 没 GPU。LLM 推理层走 Azure OpenAI 或 AML Endpoint，Container Apps 跑**业务层**（FastAPI）。

## Container Apps vs AKS vs Fargate

| | Container Apps | AKS | Fargate |
|---|---|---|---|
| 抽象 | App + Revision | Pod | Task |
| K8s API 访问 | ❌ | ✅ | ❌ |
| Scale-to-zero | ✅ | 需 KEDA | ❌ |
| 冷启动 | 1-3s | 无 | 30s-1min |
| GPU | ⚠️ Dedicated only (预览) | ✅ | ❌ |
| 多容器 per App | ✅ sidecar | ✅ | ✅ |
| Ingress + TLS | 内置 | 要装 Ingress Controller | 要 ALB |
| Dapr | 内置 | 要装 | ❌ |
| 适合 | 微服务、business logic、Web API | 完整控制 / 大规模 | AWS 侧的 CA 对偶 |

## 架构

```
Internet
   │
   ▼
Container Apps Environment (托管的 K8s-lite)
   │
   ├── ocr-api (2 replicas, 0.5 vCPU, 1 GB)
   │      │
   │      ├─► Azure OpenAI  (推理层)
   │      └─► Azure AI Search / Cosmos DB  (RAG / 元数据)
   │
   └── (可选) adapter-controller (1 replica)
```

## 部署步骤

### 1. 推镜像到 ACR

```bash
# 如果还没有 ACR
az acr create --resource-group rg-ocr-fine-app --name ocracr --sku Basic

# 本地构建 + 推
az acr login --name ocracr
docker build -t ocracr.azurecr.io/ocr-api:v1 .
docker push ocracr.azurecr.io/ocr-api:v1
```

### 2. Bicep apply

```bash
cd bicep
az deployment group create \
    --resource-group rg-ocr-fine-app \
    --template-file main.bicep \
    --parameters \
        prefix=ocr \
        acrName=ocracr \
        imageTag=v1 \
        azureOpenAIEndpoint="https://ocr-openai.openai.azure.com"
```

### 3. 测试

```bash
# 拿公网 FQDN
FQDN=$(az containerapp show -n ocr-api -g rg-ocr-fine-app --query properties.configuration.ingress.fqdn -o tsv)
curl https://$FQDN/health
```

## Revisions —— Container Apps 的版本抽象

每次改 App 配置或 image tag，Container Apps 生成一个新 **Revision**。流量可以在多个 Revision 之间切分：

```bash
# 查看 revisions
az containerapp revision list --name ocr-api -g rg-ocr-fine-app -o table

# 金丝雀: 当前 revision 90%, 新 revision 10%
az containerapp ingress traffic set \
    --name ocr-api -g rg-ocr-fine-app \
    --revision-weight ocr-api--rev1=90 ocr-api--rev2=10

# 全切
az containerapp ingress traffic set \
    --name ocr-api -g rg-ocr-fine-app \
    --revision-weight ocr-api--rev2=100
```

这是比 Fargate（要自己接 CodeDeploy）更优雅的原生金丝雀方案。

## 事件驱动扩缩（KEDA）

Container Apps 内建 KEDA，按 HTTP 并发 / Queue 长度 / Cron 等指标扩缩：

```bicep
scale: {
  minReplicas: 0        // scale-to-zero，闲时不花钱
  maxReplicas: 10
  rules: [
    {
      name: 'http-concurrency'
      http: {
        metadata: {
          concurrentRequests: '50'   // 每实例目标 50 并发
        }
      }
    }
    {
      name: 'servicebus-queue'
      custom: {
        type: 'azure-servicebus'
        metadata: {
          queueName: 'extract-tasks'
          messageCount: '5'
        }
      }
    }
  ]
}
```

`minReplicas: 0` 是 Container Apps **大杀器** —— 对于 QPS 很低的业务（定时任务 / 内部工具），月成本可以做到几美元。

## 常见坑

- **FQDN 没出来** → Ingress 要显式启用 `configuration.ingress.external: true`
- **Scale-to-zero 冷启动用户有感** → 加个 warm-up ping（定时任务每 5 min GET /health）或把 minReplicas 设 1
- **Managed Identity 调 ACR 失败** → Environment 级别要绑 MI 到 ACR `AcrPull` role；Bicep 里已经配好
- **日志不出** → Environment 要绑 Log Analytics workspace；Bicep 里已配

## 本目录文件

```
container-apps/
├── README.md                   ← 本文件
└── bicep/
    ├── main.bicep              ← Environment + App + MI + Log Analytics
    └── parameters.json
```
