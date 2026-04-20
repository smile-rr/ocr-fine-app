# AWS 部署参考

> AWS 上 host LLM 有 4 种主流姿势，本目录四个子目录各示范一种。**先读本页的对比表挑一个子目录深入**，别一次全看。

## 4 种方案速览

| 子目录 | AWS 服务 | 何时选 | 谁管什么 |
|---|---|---|---|
| [bedrock-client/](./bedrock-client/) | **Bedrock** | 业务要 LLM，但不想自己 host 和微调 | AWS 管一切；你只写 client 代码 |
| [sagemaker-endpoint/](./sagemaker-endpoint/) | **SageMaker Endpoint** | host 自己微调的模型，但不想碰容器/K8s | AWS 管容器、GPU、Auto Scaling；你管模型 + 代码 |
| [ecs-fargate/](./ecs-fargate/) | **ECS + Fargate** | 自己打镜像（vLLM / 自家业务），但嫌 K8s 重 | AWS 管调度；你管 Task Definition + 镜像 |
| [eks-vllm/](./eks-vllm/) | **EKS**（复用本项目 `inference/kubernetes/`） | 完全控制，多模型 / 多租户 / 自研 Operator | 你管 K8s 栈，AWS 只管节点 |

## 详细对比

| 维度 | Bedrock | SageMaker Endpoint | ECS Fargate | EKS |
|---|---|---|---|---|
| **抽象层** | API (黑盒) | 模型 + 推理容器 | 容器 | Pod |
| **输入** | 只要 prompt | 模型 artifact (S3) + image | Docker image + Task Def | Deployment/Service YAML |
| **GPU 支持** | ✅ 托管 | ✅（选 `ml.g5.xlarge` 等） | ⚠️ 仅 EC2 launch type（Fargate 无 GPU） | ✅（自己加 GPU node group） |
| **Auto Scaling** | AWS 托管 | ✅ built-in | ✅ Service scaling | ✅ HPA + Karpenter |
| **多模型 / LoRA 热加载** | ❌ | ✅ Multi-Model Endpoint | ⚠️ 自己搞 | ✅（S-LoRA） |
| **金丝雀** | ❌ 无版本概念 | ✅ Production Variants | ⚠️ CodeDeploy 做 | ✅ Argo Rollouts |
| **冷启动** | ~1s | 2-5 min（首次） | 1-3 min | 2-5 min |
| **单请求延迟** | 依赖模型 | 取决于实例 | 取决于镜像 | 取决于 Pod |
| **计费粒度** | per-token | per-hour 实例 | per-hour vCPU/mem + GPU | 节点 per-hour + EKS $0.10/h |
| **IaC 成熟度** | ★★★ | ★★★★★ | ★★★★★ | ★★★★★ |
| **学习曲线** | ★ | ★★ | ★★★ | ★★★★★ |
| **锁定度** | 🔴 高（模型有限，API 不兼容 OpenAI） | 🟡 中（模型可迁，IaC 不通用） | 🟢 低（Docker 可迁） | 🟢 低（YAML 通用） |

## 选型建议（按场景）

- **你公司 AWS 中心化很深 + 预算灵活 + 模型 Bedrock 有** → Bedrock
- **微调后的 7B 模型部署，团队只懂 Python 不懂 K8s** → SageMaker Endpoint
- **成本敏感 + 偶尔用 / QPS 低** → SageMaker Serverless Inference 或 Fargate
- **已经有 EKS 跑其他业务** → EKS + 本项目 `inference/kubernetes/`
- **正在从 Azure/本地迁过来** → 从 EKS 开始，manifest 最不变

## 共同前置（所有子目录都需要）

```bash
# 1. AWS CLI + 认证
brew install awscli
aws configure           # 或 aws sso login

# 2. Terraform
brew install terraform

# 3. 确认 region 支持 GPU 实例
aws ec2 describe-instance-type-offerings --region us-east-1 \
    --filters Name=instance-type,Values='g5.*' --output table

# 4. 确认 Bedrock 模型可用（region 限制）
aws bedrock list-foundation-models --region us-east-1
```

**省钱 Tip**：
- 开发/测试用 `us-east-1`（模型/实例种类最全 + 便宜）
- 用完立刻 `terraform destroy`，SageMaker Endpoint 不用也计费
- Bedrock on-demand 没有最低费，适合偶尔测试

## 常见 AWS 身份认证模式

| 场景 | 推荐 |
|---|---|
| 本地开发跑 SDK | `aws configure` / SSO |
| EC2 实例跑服务 | Instance Profile |
| ECS Task 跑容器 | Task Role |
| EKS Pod 跑容器 | **IRSA** (IAM Roles for Service Accounts) ⭐ |
| Lambda | Execution Role |
| GitHub Actions CI | OIDC（零长期凭证） |

**永远不要在代码/镜像里硬编码 access key**。四个子目录都演示了正确姿势。
