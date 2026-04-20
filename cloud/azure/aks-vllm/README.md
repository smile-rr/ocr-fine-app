# AKS + vLLM —— 把 `inference/kubernetes/` 跑在 Azure 上

AWS EKS 的对偶。Bicep 只负责**起集群 + GPU 节点池 + 共享存储 + Workload Identity**，应用部署还是 `kubectl apply -k inference/kubernetes/base/`。

## 架构

```
Azure                                   Your Laptop
┌─────────────────────────────┐         ┌──────────────┐
│  Virtual Network             │         │  kubectl     │
│  ├─ AKS Control Plane        │◄────────┤  + az CLI   │
│  ├─ System Node Pool         │         └──────────────┘
│  │    (Standard_D2s_v3)      │                │
│  └─ GPU Node Pool            │                │
│       (Standard_NC6s_v3 V100)│ kubectl apply -k
│                              │◄──── inference/kubernetes/base/
│  Azure Files (RWX) models/   │
│  ACR (镜像仓库)               │
│  Entra ID OIDC (Workload ID) │
└─────────────────────────────┘
```

## AKS vs EKS 具体差异

| | AKS | EKS |
|---|---|---|
| 节点 OS | Ubuntu 22.04 / Azure Linux | Amazon Linux 2 / Bottlerocket |
| 身份 | **Workload Identity** (OIDC 换 token) | **IRSA** (OIDC 换 token) |
| 存储 | Azure Files / Azure Disk / Azure NetApp Files | EFS / EBS / FSx |
| 自动扩容 | Cluster Autoscaler / **Karpenter-AKS** (preview) | Karpenter / CA |
| GPU Operator | 自装 NVIDIA Device Plugin 或用 **KAITO** | 自装 / Karpenter |
| 网络 | Azure CNI / CNI Overlay / BYOCNI | VPC CNI / Cilium |
| 对外 | Azure Load Balancer / App Gateway / Front Door | ALB / NLB |
| 控制面费用 | Free tier / Standard ($0.10/h) | $0.10/h |
| 托管 Ingress | Web App Routing Addon (nginx) | AWS Load Balancer Controller |

## Bicep 部署

```bash
cd bicep
az deployment group create \
    --resource-group rg-ocr-fine-app \
    --template-file main.bicep \
    --parameters prefix=ocr location=eastus sshPublicKey="$(cat ~/.ssh/id_rsa.pub)"

# 拉 kubeconfig
az aks get-credentials --resource-group rg-ocr-fine-app --name ocr-aks

# 验证 GPU
kubectl get nodes -L agentpool
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

## Workload Identity (对应 EKS IRSA)

原理一样：K8s ServiceAccount → OIDC → Entra ID Federated Credential → Managed Identity。

Pod 里调用 Azure SDK 自动拿到 Managed Identity 的 token，零凭证。

**设置步骤**（Bicep 已做）：
1. AKS 启用 `oidcIssuerProfile.enabled: true` + `securityProfile.workloadIdentity.enabled: true`
2. 创建 User-assigned Managed Identity
3. 给 MI 加 Federated Credential（trust AKS OIDC issuer 的特定 SA）
4. ServiceAccount 加 annotation `azure.workload.identity/client-id: <mi-client-id>`
5. Pod 加 label `azure.workload.identity/use: "true"`

Python 里 `DefaultAzureCredential` 自动用这条链路。

## 存储：Azure Files

对应 EFS 的 RWX 共享文件系统。Bicep 配了：
- Storage Account (Premium FileStorage)
- File Share `models`
- CSI driver 已经是 AKS 内置 addon
- StorageClass `azurefile-csi-premium` 自动 bind PVC

用法：`inference/kubernetes/base/pvc-models.yaml` 把 `storageClassName` 改成 `azurefile-csi-premium`。

## GPU 节点池

```bicep
{
  name: 'gpu'
  vmSize: 'Standard_NC6s_v3'      // 1× V100 16GB (~$3/h)
  // 更便宜: Standard_NC4as_T4_v3  (1× T4 16GB, ~$0.53/h)
  // 更强:   Standard_NC24ads_A100_v4  (1× A100 80GB, ~$3.67/h)
  count: 1
  enableAutoScaling: true
  minCount: 1
  maxCount: 4
  mode: 'User'
  nodeTaints: [
    'nvidia.com/gpu=true:NoSchedule'    // 只让 GPU Pod 调度
  ]
  nodeLabels: {
    accelerator: 'nvidia-gpu'
  }
  spotMaxPrice: -1                      // -1 = 不用 Spot；0.5 = 最多 $0.5/h (Spot)
}
```

## NVIDIA GPU Operator 或 Device Plugin

AKS GPU 节点需要 nvidia device plugin 让 K8s 识别 `nvidia.com/gpu`。两种装法：

**方式 1: AKS GPU addon**（最简单）
```bash
az aks nodepool add --cluster-name ocr-aks --resource-group rg-ocr-fine-app \
    --name gpu --node-vm-size Standard_NC6s_v3 \
    --node-count 1 --enable-gpu-driver-install   # 自动装驱动 + plugin
```

**方式 2: Helm 装 NVIDIA GPU Operator**（功能全，支持 MIG / GPU sharing）
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace
```

Bicep 例子用方式 1。

## KAITO —— Azure 独有的 K8s AI Toolchain Operator

Azure 2024 推的 Operator，把"微调 + 推理"做成 CRD：

```yaml
apiVersion: kaito.sh/v1alpha1
kind: Workspace
metadata:
  name: workspace-stage2
resource:
  instanceType: "Standard_NC6s_v3"
  count: 1
inference:
  preset:
    name: "qwen-2-5-7b-instruct"
```

KAITO 自动：
- 拉 GPU VM
- 装 vLLM + 模型
- 配 Service
- 管理 LoRA adapters

等于把 AWS SageMaker 的体验搬到 K8s 上。**对本项目**：Bicep 出 AKS 集群，然后可以用 KAITO 作为 `inference/kubernetes/` 的升级版。

## 成本估算

| 项 | 配置 | $/月 |
|---|---|---|
| AKS Standard tier | | ~$73 |
| 2× Standard_D2s_v3 (system) | | ~$140 |
| 1× Standard_NC6s_v3 (GPU, V100) | | ~$2,200 |
| Azure Files Premium 100GB | | $24 |
| Load Balancer Standard | | $20 |
| ACR Standard | | $20 |
| **合计** | | **~$2,500/月** |

Azure GPU 比 AWS 贵。省钱：
- 换 T4 (`Standard_NC4as_T4_v3`): $2,200 → $380
- Spot (`spotMaxPrice: 0.5`): 省 70-90%
- Scale-to-zero GPU（Karpenter-AKS 预览）
- 用 AML Endpoint + Serverless 而不是养集群

## 本目录文件

```
aks-vllm/
├── README.md                   ← 本文件
└── bicep/
    ├── main.bicep              ← AKS + GPU pool + Azure Files + Workload Identity + ACR
    └── parameters.json
```
