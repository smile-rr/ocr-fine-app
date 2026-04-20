# EKS + vLLM —— 把 `inference/kubernetes/` 跑在 AWS 上

> 本目录的 Terraform 只负责**起集群 + GPU 节点组 + 共享存储 + IAM**。
> 真正的应用部署还是用前面写好的 **`inference/kubernetes/base/`**（YAML 原样 kubectl apply）。

## 架构

```
AWS                                    Your Laptop
┌────────────────────────────┐         ┌──────────────┐
│  VPC                        │         │  kubectl     │
│  ├─ EKS Control Plane       │◄────────┤  + aws CLI  │
│  ├─ Node Group: general     │         └──────────────┘
│  │    (t3.medium, 2 nodes)  │                │
│  │    跑 app-layer / Ingress│                │
│  └─ Node Group: gpu         │                │
│       (g5.xlarge A10G)       │ kubectl apply -k
│       跑 vLLM Pods           │◄──── inference/kubernetes/base/
│                              │
│  EFS (RWX 存储 models/)      │
│  ECR (镜像仓库)              │
│  IAM OIDC Provider (IRSA)    │
└────────────────────────────┘
```

## Terraform 做了什么

**本目录 `terraform/main.tf` 配置的资源**：
1. **VPC**（2 AZ，public + private subnets，NAT Gateway）
2. **EKS Cluster**（`v1.30`）+ OIDC provider（IRSA 前置）
3. **General Node Group**（t3.medium × 2，跑 Ingress / 业务层）
4. **GPU Node Group**（g5.xlarge × 1-4，taint 只允许 GPU Pod 调度）
5. **NVIDIA Device Plugin**（让 K8s 识别 GPU 资源）
6. **EFS CSI Driver** + **EFS filesystem**（RWX PVC，多 Pod 共享模型权重）
7. **ECR Repository** （业务镜像 + adapter-controller 镜像）
8. **IAM Roles for Service Accounts (IRSA)**：
   - `vllm-runner` — 让 vLLM Pod 能从 S3 拉 adapter 权重
   - `adapter-controller` — 让它能访问 EFS 和 SSM Parameter Store

## 部署（分三步，每步验证）

### 1. 建集群 + 基础设施

```bash
cd cloud/aws/eks-vllm/terraform
terraform init
terraform apply -var="cluster_name=ocr-prod"

# 15-20 分钟后拿到 kubeconfig
aws eks update-kubeconfig --name ocr-prod --region us-east-1

# 验证 GPU 节点 ready
kubectl get nodes -L node.kubernetes.io/instance-type,karpenter.sh/capacity-type
kubectl describe node <gpu-node> | grep nvidia.com/gpu
# 期望: Allocatable: nvidia.com/gpu: 1
```

### 2. 上传模型到 EFS（或 S3）

**方案 A：EFS 作模型仓库**（简单，对应 K8s 里的 RWX PVC）
```bash
# 起一个临时 Pod 挂 EFS PVC，copy 模型进去
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: model-uploader
  namespace: ocr-inference
spec:
  containers:
    - name: uploader
      image: amazon/aws-cli:latest
      command: ["sleep", "3600"]
      volumeMounts:
        - name: models
          mountPath: /models
  volumes:
    - name: models
      persistentVolumeClaim:
        claimName: models
EOF

# rsync 本地模型到 Pod
kubectl -n ocr-inference cp models/stage2_fused model-uploader:/models/

# 清理
kubectl -n ocr-inference delete pod model-uploader
```

**方案 B：S3 作模型仓库 + initContainer 拉**（更云原生）
- Deployment 里加 initContainer，启动时 `aws s3 sync s3://bucket/models/ /models/`
- vLLM 容器 `depends_on` initContainer 完成
- 优点：模型更新走 S3 object version + CloudTrail，合规好；缺点：首次启动慢

本目录默认 A；B 的 patch 见 `eks-s3-initcontainer.yaml.example`。

### 3. Apply `inference/kubernetes/base/`

```bash
# 项目根
kubectl apply -k inference/kubernetes/base/

# 等 vLLM Pod 变 Running（2-5 min 加载模型）
kubectl -n ocr-inference get pods -w
```

## 关键决策点

### IRSA 为什么重要

**别在 Pod 里塞 AWS access key**。IRSA 用 K8s ServiceAccount + OIDC 换 AWS STS token，Pod 到 AWS 的调用走 STS 签名。

本 Terraform 里已经配好：
```hcl
module "eks" {
  enable_irsa = true
}

resource "aws_iam_role" "vllm_runner" {
  assume_role_policy = ...   # trust policy 认 OIDC provider
}

# K8s ServiceAccount annotation 绑 role ARN
```

Pod 里的应用拿到 `AWS_ROLE_ARN` + `AWS_WEB_IDENTITY_TOKEN_FILE` 环境变量，boto3 自动 assume role，零凭证泄漏风险。

### Karpenter vs Cluster Autoscaler

**Cluster Autoscaler**（传统）：按 ASG 扩缩，节点类型固定
**Karpenter**（推荐）：按 Pod 需求动态选实例类型（容器要 16GB 自动起 g5.xlarge，要 80GB 自动起 g5.4xlarge）

本 Terraform 用 **Karpenter**（EKS 推荐）—— 成本更优、扩缩更快。配置在 `karpenter.tf`。

### Spot 实例（省钱 70%）

GPU node group 可以开 Spot 混部：
```hcl
capacity_type = "SPOT"   # 或 ["SPOT", "ON_DEMAND"] 混部
```

Spot 风险：中断时 Pod 被强制调度。对 LLM 推理的影响：
- 有副本时 PDB 保证 `minAvailable: 1` 不中断
- Graceful shutdown 让 in-flight 请求完成
- 生产建议：**关键 Service 至少 1 个 OnDemand + N 个 Spot**

### 存储选型

| 方案 | 用途 | 性能 | 成本 |
|---|---|---|---|
| **EFS** ⭐ | 模型权重共享（多 Pod 读） | 中（IO 共享） | $0.30/GB-月 |
| **EBS gp3** | 单 Pod 高 IO | 高 | $0.08/GB-月 |
| **FSx for Lustre** | HPC / 多 Pod 高 IO | 极高 | $0.125/GB-月 + |
| **S3 (+ initContainer)** | 模型分发 | 启动慢，之后不读 | $0.023/GB-月 |

7B 模型 15GB，EFS 月成本 $4.5，完全 OK。

## 成本估算（生产最小配置）

| 项 | 配置 | $/月 |
|---|---|---|
| EKS Control Plane | | $73 |
| 2× t3.medium (general) | OnDemand | ~$60 |
| 1× g5.xlarge (GPU) | OnDemand | ~$730 |
| EFS 20GB | | $6 |
| ALB | | $25 |
| NAT Gateway | | $32 + 流量 |
| ECR 存储 | 10GB | $1 |
| **合计** | | **~$930/月** |

GPU 节点是大头。省钱方案：
- Spot: $730 → $220
- Serverless GPU（AWS 2024 新）: $0 idle + per-request（测试阶段）
- 多模型共享 GPU（vLLM S-LoRA）

## 本目录文件

```
eks-vllm/
├── README.md                      ← 本文件
├── deploy.sh                      ← 一键编排
└── terraform/
    ├── main.tf                    ← EKS + VPC + 2 个 Node Group
    ├── karpenter.tf               ← Karpenter + GPU NodePool
    ├── efs.tf                     ← EFS 文件系统 + CSI driver + StorageClass
    ├── ecr.tf                     ← 镜像仓库
    ├── irsa.tf                    ← IAM Roles for Service Accounts
    ├── variables.tf
    └── outputs.tf
```
