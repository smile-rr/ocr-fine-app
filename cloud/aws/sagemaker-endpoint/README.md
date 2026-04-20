# SageMaker Endpoint —— 托管推理（自家微调模型上生产）

你在 `finetuning/` 训完的模型合并成 HF 格式后，如果不想碰容器/K8s，**SageMaker Endpoint 是最顺的上线路径**。

## 核心概念（三件套）

SageMaker 把部署拆成三个资源，Terraform 里会依次创建：

```
S3 (model artifact)                            ECR (inference image)
  │    ↓                                                ↓
  │  s3://bucket/stage2_fused.tar.gz         763104351884.dkr.ecr.../djl-lmi:latest
  │                      │                              │
  │                      ▼                              ▼
  │             ┌───────────────────────┐    ┌──────────────────────┐
  │             │  SageMaker Model       │◄───│  Inference Image      │
  │             │  (把 artifact + image │    │  (AWS DLC / 自构建)    │
  │             │   绑一起)              │    │                       │
  │             └───────────┬───────────┘    └──────────────────────┘
  │                         │
  │                         ▼
  │             ┌───────────────────────┐
  │             │ EndpointConfig         │  ← 定义 instance type + 扩缩容
  │             │  - InitialInstanceCount│
  │             │  - InstanceType:        │
  │             │      ml.g5.xlarge       │
  │             │  - ProductionVariants   │  ← 金丝雀在这里配
  │             └───────────┬───────────┘
  │                         │
  │                         ▼
  │             ┌───────────────────────┐
  └────────────►│ Endpoint              │  ← 对外的 HTTPS URL
                │  https://runtime.sagemaker.│
                │    region.amazonaws.com/   │
                │    endpoints/stage2/invocations
                └───────────────────────┘
```

## 4 种推理容器选项

AWS 提供几种**预构建的 Deep Learning Container**（DLC），直接挂模型就能跑：

| Container | 底层 | 适合 | 镜像名 |
|---|---|---|---|
| **DJL-LMI** | DeepJavaLibrary + vLLM / TensorRT-LLM / LMI-Dist | **LLM 推理首选**，一键 continuous batching | `763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124` |
| TGI (Hugging Face) | TGI | 和 HF 生态深度集成 | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.0-tgi2.4.0-gpu-py311-cu124-ubuntu22.04` |
| PyTorch DLC | 裸 PyTorch | 自己写 `inference.py`，非 LLM 模型 | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker` |
| 自建镜像 | vLLM 官方等 | 有特殊依赖 | 你自己 push 到 ECR |

**本目录默认用 DJL-LMI**（LLM 上 SageMaker 事实标准）。想切到 TGI 改 `image_uri` 即可。

## 部署步骤

### 1. 打包模型到 S3

```bash
# 从 finetuning/ 合并出的 HF 格式模型
cd models/stage2_fused
tar -czf /tmp/stage2_fused.tar.gz .

# 上传
aws s3 cp /tmp/stage2_fused.tar.gz s3://YOUR-BUCKET/sagemaker/stage2_fused.tar.gz
```

DJL-LMI 还支持**直接从 HF Hub 拉**（容器内下），不用打 tar：
```hcl
environment = {
  "HF_MODEL_ID" = "Qwen/Qwen2.5-7B-Instruct"
  "OPTION_ROLLING_BATCH" = "vllm"
  "OPTION_MAX_ROLLING_BATCH_SIZE" = "32"
}
```

### 2. Terraform apply

```bash
cd terraform
terraform init
terraform apply \
    -var="s3_model_path=s3://YOUR-BUCKET/sagemaker/stage2_fused.tar.gz" \
    -var="instance_type=ml.g5.xlarge"
```

### 3. 调用

```bash
# 通过 SageMaker Runtime API (AWS SigV4 签名)
uv run python client.py
```

## 关键参数（DJL-LMI 环境变量）

```hcl
environment = {
  "HF_MODEL_ID"                     = "Qwen/Qwen2.5-7B-Instruct"
  "OPTION_ROLLING_BATCH"            = "vllm"          # vllm / lmi-dist / scheduler
  "OPTION_DTYPE"                    = "bf16"
  "OPTION_MAX_MODEL_LEN"            = "8192"
  "OPTION_MAX_ROLLING_BATCH_SIZE"   = "32"
  "OPTION_TENSOR_PARALLEL_DEGREE"   = "1"             # 多 GPU 才 >1
  "OPTION_GPU_MEMORY_UTILIZATION"   = "0.9"
  "OPTION_ENABLE_LORA"              = "true"
  "OPTION_MAX_LORAS"                = "4"
  "OPTION_QUANTIZE"                 = "awq"           # 或 gptq / squeezellm
}
```

这些参数等价于你 vLLM docker-compose.yml 里的 command line args。

## 金丝雀 / 蓝绿（SageMaker 原生支持）

**Production Variants** 一个 Endpoint 可挂多个 Variant，按权重切流量：

```hcl
production_variants {
  variant_name           = "v1"
  model_name             = aws_sagemaker_model.stage2_v1.name
  initial_instance_count = 2
  instance_type          = "ml.g5.xlarge"
  initial_variant_weight = 0.9   # 90% 流量
}
production_variants {
  variant_name           = "v2-canary"
  model_name             = aws_sagemaker_model.stage2_v2.name
  initial_instance_count = 1
  instance_type          = "ml.g5.xlarge"
  initial_variant_weight = 0.1   # 10% 流量
}
```

**蓝绿升级** 可以用 `update_endpoint_weights_and_capacities`，零停机切流量。

## Auto Scaling

SageMaker 支持按 `SageMakerVariantInvocationsPerInstance` 自动扩缩：
```hcl
resource "aws_appautoscaling_target" "sagemaker" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "endpoint/${aws_sagemaker_endpoint.stage2.name}/variant/v1"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "sagemaker" {
  name               = "invocations-per-instance"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker.resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 100   # 每实例目标 100 QPS
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
  }
}
```

## Serverless Inference（成本敏感选项）

如果 QPS 很低（< 5 req/s），可以用 **Serverless Inference** 按请求计费：

```hcl
resource "aws_sagemaker_endpoint_configuration" "stage2_serverless" {
  production_variants {
    variant_name         = "v1"
    model_name           = aws_sagemaker_model.stage2.name
    serverless_config {
      max_concurrency   = 10
      memory_size_in_mb = 6144
    }
  }
}
```

**限制**：不支持 GPU（只能小模型 CPU 推理）、冷启动 1-2 分钟。7B 模型跑不动，1B 以下勉强。

## 常见坑

- **`ModelError: [Errno 28] No space left on device`** → 模型太大，把 `VolumeSizeInGB` 调大或用 `/tmp` 之外的路径
- **Endpoint 卡在 `Creating` 10 分钟+** → 多半是镜像拉不动或模型下载失败，看 CloudWatch Logs `/aws/sagemaker/Endpoints/<name>`
- **`InvocationsPerInstance` 指标是 0** → 客户端没调到真实 endpoint，检查 `invoke_endpoint` 的 `EndpointName`
- **成本爆炸** → SageMaker Endpoint **只要 endpoint 还在就计费**（不管有没有流量）。用完 `terraform destroy` 或切 Serverless
- **Multi-Model Endpoint (MME) 延迟波动大** → MME 是懒加载，冷模型首次调用延迟 3-5s；推理敏感场景用 SingleModel

## 本目录文件

```
sagemaker-endpoint/
├── README.md                       ← 本文件
├── client.py                       ← boto3 调 endpoint 示例
├── deploy.sh                       ← 一键：打包模型 → 上 S3 → terraform apply
└── terraform/
    ├── main.tf                     ← Model + EndpointConfig + Endpoint + AutoScaling
    ├── variables.tf
    └── outputs.tf
```
