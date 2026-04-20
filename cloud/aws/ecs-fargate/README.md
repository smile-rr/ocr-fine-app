# ECS + Fargate —— 托管容器（不上 K8s）

> **警告**：Fargate **不支持 GPU**。本目录的 ECS Fargate 示例适合部署 **业务层**（本项目 `src/serve/api.py` 那种 CPU 服务），推理层用 Bedrock 或 SageMaker。
>
> 如果要跑 vLLM（需要 GPU），得用 **ECS on EC2**（GPU 实例 + ECS 调度），下面也给了配置。

## 两种 ECS launch type

| Launch Type | GPU | 何时用 |
|---|---|---|
| **Fargate** ✅ | ❌ | CPU 业务、无状态 API、成本敏感（秒级计费，无闲置实例） |
| **EC2** ✅ | ✅（g 系列实例） | 需要 GPU、需要 privileged mode、需要自定义网络 |

**本目录示例**：
- `terraform/main.tf` 跑**业务层**（FastAPI + CPU）on Fargate
- `terraform/ec2-gpu.tf.example` 跑 **vLLM** on ECS/EC2 GPU（参考）

## ECS vs EKS vs Fargate-Serverless

| | ECS + Fargate | ECS + EC2 | EKS | Lambda |
|---|---|---|---|---|
| 调度抽象 | Service + Task | Service + Task | Pod | Function |
| 管理集群 | ❌ AWS 托管 | ✅ 自己管节点 | ✅ 自己管节点（EKS 管 control plane） | ❌ 完全托管 |
| GPU | ❌ | ✅ | ✅ | ❌ |
| 容器大小 | 0.25-16 vCPU, 0.5-120GB | 任意 EC2 size | 任意 | ≤10GB |
| 冷启动 | 30s-1min | 无（节点常驻） | 无 | 100ms-5s |
| 调度粒度 | Task | Task | Pod | Request |
| 学习成本 | ★★ | ★★★ | ★★★★★ | ★★ |

**选型**：
- 业务层（FastAPI，CPU） → **Fargate**（不用养节点）
- vLLM（GPU） → **ECS EC2** 或 **EKS**（Fargate 没 GPU）
- 事件驱动短逻辑 → Lambda
- 已有 K8s → 所有都塞 EKS

## 架构（本目录部署的）

```
Internet
   │
   ▼
ALB (Application Load Balancer)
   │
   ▼
ECS Service (Fargate, 2+ Tasks)
   │
   ├── Task: ocr-api Container (FastAPI :8000)
   │    │
   │    ├─► 调用 SageMaker Endpoint  或  Bedrock（推理层）
   │    └─► 调用 OpenSearch / RDS / DynamoDB（向量/元数据）
   │
   └── 日志 → CloudWatch Logs
       指标 → CloudWatch Metrics / Container Insights
```

## 部署步骤

```bash
# 1. 构建并推送业务镜像到 ECR
cd /path/to/ocr-fine-app
docker build -t ocr-api:latest .   # 项目根 Dockerfile
aws ecr create-repository --repository-name ocr-api --region us-east-1
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag ocr-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ocr-api:v1
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ocr-api:v1

# 2. terraform apply
cd cloud/aws/ecs-fargate/terraform
terraform apply \
    -var="image_uri=<account>.dkr.ecr.us-east-1.amazonaws.com/ocr-api:v1" \
    -var="sagemaker_endpoint_name=ocr-fine-app-stage2"   # 如果推理在 SageMaker

# 3. 验证
curl http://$(terraform output -raw alb_dns)/health
```

## Fargate 成本计算

Fargate 按 **vCPU-hour + GB-hour** 计费：
- 0.5 vCPU + 1 GB, 24h × 30day = **~$13/月**
- 2 vCPU + 4 GB = **~$50/月**

对比：
- Lambda: 1M 次调用 + 400K GB-s = ~$6.67
- EC2 t3.small 同规格: ~$15/月（要自己管 patch）

**什么时候 Fargate 比 Lambda 划算**：
- 请求 > 百万次/月（Lambda 按调用次数计费）
- 单次请求 > 15分钟（Lambda 有超时）
- 需要持久连接（WebSocket / SSE）

## 金丝雀发布

ECS 原生支持 **CodeDeploy Blue/Green**：

```hcl
# Service 加这段
deployment_controller {
  type = "CODE_DEPLOY"
}

# CodeDeploy 会：
# 1. 起一组新 Tasks（绿）
# 2. 切 ALB 流量到绿（可配 10% / 50% / 100% 逐步）
# 3. 观察指标（CloudWatch Alarm），坏了自动回滚
# 4. 好了停掉蓝
```

或简单版 **Rolling Update**（默认策略）：
```hcl
deployment_controller { type = "ECS" }
deployment_minimum_healthy_percent = 100
deployment_maximum_percent         = 200   # 允许起双倍 Task 做替换
```

## 常见坑

- **Task 起不来 `CannotPullContainerError`** → VPC 里没 NAT 或 VPC Endpoint 拉不到 ECR；要么公网 subnet，要么加 ECR VPC Endpoint
- **Fargate 网络慢** → 默认 `awsvpc` 模式每 Task 一张 ENI，冷启动拉长；`vpc-lattice` 更快
- **Task CPU 100%** → Fargate CPU 是共享单元，定义的是 vCPU **上限**；监控 `CPUUtilization` 看是否到限
- **日志不出** → Task 里必须配 `awslogs` driver，并且 execution role 有 `logs:*` 权限
- **换 image 不生效** → ECS Service 默认只对 Task Definition 变化敏感；改 tag 要 `aws ecs update-service --force-new-deployment`

## 本目录文件

```
ecs-fargate/
├── README.md                       ← 本文件
└── terraform/
    ├── main.tf                     ← Fargate 业务层 (ALB + ECS Service + CloudWatch)
    ├── variables.tf
    ├── outputs.tf
    └── ec2-gpu.tf.example          ← 参考：vLLM on ECS/EC2 GPU
```
