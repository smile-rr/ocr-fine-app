# Azure ML Online Endpoint —— 托管自家微调模型

和 AWS SageMaker Endpoint 对偶。你在 `finetuning/` 训完的模型合并成 HF 格式后，扔 Azure ML 托管。

## 核心概念（AML 四件套）

Azure ML 比 SageMaker 概念多一层 **Workspace**：

```
Azure Subscription
└── Resource Group
    └── Azure ML Workspace   ← 所有 ML 资源的顶层容器
        ├── Compute          ← 训练/推理用的计算资源（cluster / instance）
        ├── Datastore        ← 数据源（Blob / ADLS / File Share）
        ├── Environment      ← Docker image 定义（conda.yaml / Dockerfile）
        ├── Model            ← 模型 artifact + metadata
        └── Endpoint         ← 真正对外的推理服务
            └── Deployment   ← Endpoint 下的具体部署（金丝雀靠多个 Deployment）
```

SageMaker 里 `Model → EndpointConfig → Endpoint`，AML 里 `Model + Environment → Deployment → Endpoint`。类似但不一样。

## 和 SageMaker 的具体对应

| SageMaker | Azure ML | 说明 |
|---|---|---|
| SageMaker Model | Model + Environment | AML 把"权重"和"运行环境"分开注册 |
| EndpointConfig | Deployment | AML 每个 Deployment 自带 instance 配置 |
| Endpoint | Online Endpoint | 一样，对外 HTTPS |
| Production Variant | Deployment (多个) + `traffic` 字段 | `az ml online-endpoint update --traffic "blue=90 green=10"` |
| DJL-LMI container | 自定义 Dockerfile / AML curated image | AML 没有预设 LMI 镜像，得自己拼或用 HF Inference Server |
| Auto Scaling | Autoscale Settings | 都是 CPU/request-count 驱动 |

## 部署步骤

### 1. 上传模型到 AML Model Registry

```bash
# 项目根
az ml model create \
    --name stage2-fused \
    --version 1 \
    --path ./models/stage2_fused \
    --type custom_model \
    --resource-group rg-ocr-fine-app \
    --workspace-name ocr-aml-ws
```

或者用 Python SDK:
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

ml = MLClient(DefaultAzureCredential(), sub_id, rg, ws_name)
ml.models.create_or_update(Model(
    name="stage2-fused",
    version="1",
    path="./models/stage2_fused",
    type="custom_model",
))
```

### 2. 定义 Environment（推理容器）

最简单用 HuggingFace Inference Server（AML curated）:

```yaml
# environment.yaml
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: vllm-serve-env
version: 1
image: mcr.microsoft.com/azureml/curated/foundation-model-inference:33
```

或自己写 Dockerfile 跑 vLLM（见 `bicep/main.bicep` 注释）。

### 3. 创建 Endpoint + Deployment

```bash
# 一次性 Bicep
cd bicep
az deployment group create \
    --resource-group rg-ocr-fine-app \
    --template-file main.bicep \
    --parameters workspaceName=ocr-aml-ws

# 或者用 az ml CLI (更贴近日常运维)
az ml online-endpoint create -f endpoint.yaml
az ml online-deployment create -f deployment.yaml --all-traffic
```

### 4. 调用

```python
import urllib.request, json
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
token = credential.get_token("https://ml.azure.com/.default").token

payload = {
    "messages": [{"role": "user", "content": "你好"}],
    "max_new_tokens": 100,
}
req = urllib.request.Request(
    "https://ocr-stage2.eastus.inference.ml.azure.com/score",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
print(urllib.request.urlopen(req).read())
```

或者开启 **OpenAI 兼容路径**（需要 HF Inference Server image）— 客户端代码和调 OpenAI 一模一样。

## 金丝雀（AML 原生）

```bash
# 创建 blue deployment (v1)
az ml online-deployment create \
    --endpoint-name ocr-stage2 \
    --name blue \
    --model stage2-fused:1 \
    --instance-type Standard_NC6s_v3 \
    --instance-count 2

# 创建 green deployment (v2), 初始 0% 流量
az ml online-deployment create \
    --endpoint-name ocr-stage2 \
    --name green \
    --model stage2-fused:2 \
    --instance-count 1

# 金丝雀: 给 green 10%
az ml online-endpoint update \
    --name ocr-stage2 \
    --traffic "blue=90 green=10"

# 观察 metric，OK 就全切
az ml online-endpoint update \
    --name ocr-stage2 \
    --traffic "blue=0 green=100"

# 最后删 blue
az ml online-deployment delete -e ocr-stage2 -n blue
```

流量切分是**请求级**（每个请求独立 roll dice），不是"把某个实例整个切过去"。

## 成本参考

| 实例 | GPU | $/h | 适合 |
|---|---|---|---|
| Standard_NC4as_T4_v3 | 1× T4 16GB | ~$0.53 | 7B 量化 / 1B 非量化 |
| Standard_NC6s_v3 | 1× V100 16GB | ~$3.06 | 7B bf16 |
| Standard_NC24ads_A100_v4 | 1× A100 80GB | ~$3.67 | 13B-70B |
| Standard_ND96asr_v4 | 8× A100 40GB | ~$27 | 大型训练 |

AML Endpoint 没 Serverless（SageMaker 有）。QPS 极低场景建议用 Azure OpenAI 或 Container Apps + initContainer。

## 常见坑

- **Deployment 卡在 `Creating` 20 分钟+** → 查 `az ml online-deployment get-logs -e <ep> -n <dep>`，多半是镜像拉不动或 entry script 报错
- **GPU quota 不够** → `az vm list-usage` 看配额；申请提配 https://aka.ms/AMLGPUQuotaIncrease
- **超时** → AML Endpoint 默认 request timeout 90s；`request_settings.request_timeout_ms` 调大（上限 5 分钟）
- **Managed Identity 调失败** → Deployment 必须 `identity: type: SystemAssigned`，然后给这个 MI 授权数据源（Storage / Key Vault）

## 本目录文件

```
aml-endpoint/
├── README.md                   ← 本文件
├── endpoint.yaml               ← az ml CLI 用
├── deployment.yaml             ← az ml CLI 用
├── client.py                   ← Python 客户端示例
└── bicep/
    ├── main.bicep              ← Workspace + Endpoint + Deployment 一次搞定
    └── parameters.json
```
