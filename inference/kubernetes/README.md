# Kubernetes / OpenShift 部署

> 从 Docker Compose 升级到 K8s 的完整参考。回答三个典型企业级问题：
>
> 1. 如何在 K8s 上跑 vLLM？（本目录 `base/`）
> 2. 金丝雀 / 蓝绿发布怎么做？（本目录 `canary/`）
> 3. **LoRA 热加载的 URL 方案为什么不安全，生产上怎么做？**（本目录 `adapter-ops/`）

## 目录

```
inference/kubernetes/
├── base/             ← 可直接 kubectl apply 的最小部署（Deployment+Service+PVC+...）
├── canary/           ← 金丝雀发布（Argo Rollouts / Istio / Flagger 三种方案）
├── adapter-ops/      ← 🔐 LoRA 热加载的 3 种安全做法（GitOps / Mediator / Operator）
└── openshift/        ← OpenShift 专属差异（Route、SCC、OpenShift AI / KServe）
```

**读法**：先 `base/` 看整个部署长什么样 → 再 `adapter-ops/` 理解 ops 安全模型 → 最后 `canary/` 看发布流程 → OpenShift 有就看，没有跳过。

---

## 整体架构

```
                    ┌──────────────────────────────────┐
                    │         Ingress / Gateway         │   ← 认证、限流、TLS 终止
                    │   (Istio IngressGateway 或        │
                    │    OpenShift Route)              │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────┴─────────────────┐
                    │       app-layer (Service)        │   ← 业务 FastAPI
                    │   (可多副本，CPU-only, 轻量)     │      做 RAG 编排
                    └────┬────────────────────────┬────┘
                         │                        │
                    ┌────▼────────┐          ┌───▼───────┐
                    │  Vector DB  │          │   vLLM    │   ← GPU Pod
                    │  (Qdrant/   │          │  Service  │      ClusterIP only
                    │   Milvus)   │          │ (ClusterIP)│      不对外暴露
                    └─────────────┘          └───┬───────┘
                                                 │
                                        ┌────────▼──────────┐
                                        │   vLLM Pod(s)     │
                                        │   - GPU request   │
                                        │   - PVC 挂模型     │
                                        │   - ConfigMap:     │
                                        │     adapter 清单   │
                                        └───────────────────┘
```

**关键设计决策**：
- **vLLM Service 是 ClusterIP，不是 LoadBalancer/Ingress** —— 外部不可直接访问
- **业务层（app-layer）充当 vLLM 的前置网关**，认证、日志、审计都在这做
- **admin 接口（/load_lora_adapter 之类）只能从特定 namespace 访问**（NetworkPolicy）
- **adapter 加载走 GitOps**，而不是运维手 curl

---

## OpenShift 差异速览

| K8s 标准 | OpenShift 对应 | 说明 |
|---|---|---|
| Ingress | **Route** | 更强的路由，支持 re-encrypt / passthrough |
| RBAC | 同 | 但 OpenShift 默认更严格 |
| PodSecurityPolicy (弃用) | **SecurityContextConstraints (SCC)** | 容器运行时权限控制 |
| 任意 UID 跑容器 | 默认**拒绝 root**，需要 SCC 放行 | 很多官方镜像跑不起来要改 |
| 自建 image registry | **ImageStream** + internal registry | 可选，非必需 |
| 自己装 Argo / Istio | **OpenShift GitOps** (Argo CD) / **Service Mesh** (Istio) Operator | OLM 一键装 |
| KServe 自装 | **OpenShift AI** (RHOAI) 内置 KServe | 推荐生产 LLM 服务 |

**OpenShift 上部署 vLLM 的特别注意**：
- vLLM 官方镜像默认 root 用户 → 要么改用 `restricted-v2` SCC + 写个 `runAsUser` 非零，要么给 SA 加 `anyuid` SCC
- Route 默认 HAProxy，单请求默认超时 30s，长 prompt 会被切断 → 加 `haproxy.router.openshift.io/timeout: 300s` 注解
- GPU 节点需要 NVIDIA GPU Operator + Node Feature Discovery，装过的集群一般都有

详情看 `openshift/README.md`。

---

## 5 分钟快速走查

```bash
# 1. 前置：集群有 GPU 节点 + NVIDIA device plugin
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# 2. 应用 base
kubectl apply -k inference/kubernetes/base/

# 3. 看 Pod 起来
kubectl -n ocr-inference get pods -w
# 等到 vllm-xxx 的 READY 变 1/1 （要 2-5 分钟加载模型）

# 4. 内部测试（不对外）
kubectl -n ocr-inference port-forward svc/vllm 8000:8000
curl http://localhost:8000/v1/models

# 5. 看完 adapter-ops/README.md 理解怎么安全加载 LoRA
```

---

## 相关文件的定位

| 你关心 | 看这里 |
|---|---|
| 我要 kubectl apply 跑起来 | `base/` |
| v1 → v2 模型怎么发布，怎么回滚 | `canary/README.md` |
| 微调新 adapter，怎么上到生产 | `adapter-ops/README.md` ⭐ |
| 我在 OpenShift 上 | `openshift/README.md` |
| 我想做更高层抽象（KServe CRD） | `openshift/README.md` → KServe 章节 |
