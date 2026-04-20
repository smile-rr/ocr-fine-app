# OpenShift 专属差异与最佳实践

> `base/` 和 `canary/` 在 OpenShift 上基本通用。本文档只讲 **必须改的点** 和 OpenShift 原生能力能帮你省事的地方。

## 必改点

### 1. Pod 不能跑 root —— SCC 配置

OpenShift 默认用 `restricted-v2` SCC，拒绝：
- `runAsUser: 0`
- 特权容器
- hostPath、hostNetwork
- 未声明 SELinux context

vLLM 官方镜像默认 root 用户跑，直接 apply 会被拒。三种解法：

**推荐：用 nonroot SCC + 改镜像里的 UID**

```bash
# 创建 ServiceAccount
oc create sa vllm-runner -n ocr-inference

# 允许它用 nonroot-v2 SCC（跑非 root UID 但限制较严）
oc adm policy add-scc-to-user nonroot-v2 -z vllm-runner -n ocr-inference
```

Deployment 里加：
```yaml
spec:
  template:
    spec:
      serviceAccountName: vllm-runner
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000            # OpenShift 会尊重这个；如果冲突会自动分配 namespace 范围内的 UID
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile: { type: RuntimeDefault }
```

**不推荐但简单：加 anyuid SCC**
```bash
oc adm policy add-scc-to-user anyuid -z vllm-runner -n ocr-inference
```
让容器能以任意 UID（包括 root）跑，但违反最小权限原则，**生产不建议**。

### 2. Ingress → Route

OpenShift 用 Route 而不是 Ingress（功能更强但 YAML 不同）：

```yaml
# 见 openshift/route.yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: adapter-controller
  namespace: ocr-inference
  annotations:
    # vLLM 生成长 prompt 会慢；默认 HAProxy 超时 30s 会切断
    haproxy.router.openshift.io/timeout: 300s
spec:
  host: vllm-admin.apps.example.com
  to:
    kind: Service
    name: adapter-controller
    weight: 100
  port:
    targetPort: 8080
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
```

Route 优势：**内置金丝雀**。直接加 `alternateBackends` 字段就能做流量切分：
```yaml
spec:
  to:
    kind: Service
    name: vllm-stable
    weight: 90
  alternateBackends:
    - kind: Service
      name: vllm-canary
      weight: 10
```
改 weight + oc apply = 金丝雀完成，**不用 Istio/Argo Rollouts 也能做**（但没自动分析）。

### 3. GPU Operator

NVIDIA GPU 在 OpenShift 上需要安装 **NVIDIA GPU Operator**（从 OperatorHub）：

```bash
# Web UI: Administrator → OperatorHub → NVIDIA GPU Operator → Install
# 或 CLI:
oc apply -f - <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: gpu-operator-certified
  namespace: nvidia-gpu-operator
spec:
  channel: stable
  name: gpu-operator-certified
  source: certified-operators
  sourceNamespace: openshift-marketplace
EOF
```

装完 `oc get nodes` 会看到 GPU 节点有 `nvidia.com/gpu: 1` 的 allocatable 资源。

---

## OpenShift 原生能力（用了能省事）

### OpenShift AI (RHOAI) + KServe

RHOAI = Red Hat OpenShift AI（前身 OpenShift Data Science），开箱即用：
- **KServe** —— K8s 原生的模型 serving CRD（比我们自己写 Deployment 更抽象）
- **Data Science Pipelines** —— Kubeflow Pipelines
- **Model Registry** —— 模型版本管理
- **Jupyter / VSCode Workbench**

装上之后，部署 vLLM 只要一个 CR：

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: stage2
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      storage:
        key: localMinIO
        path: stage2_fused/
      resources:
        limits:
          nvidia.com/gpu: 1
    # 金丝雀一行字段搞定
    canaryTrafficPercent: 10
```

KServe 自动帮你：
- 起 vLLM Pod
- 配 Service + Route
- 金丝雀流量切分
- Scale-to-zero（闲时不占 GPU）

**评价**：如果你 OpenShift 集群已经装了 RHOAI，**别自己写 Deployment 了，直接用 InferenceService**。本仓 `base/` 里那些 YAML 就是 KServe 之前的手工版。

### OpenShift GitOps (ArgoCD)

OpenShift 自带 ArgoCD，不用自己装。直接：
```bash
oc apply -f https://operatorhub.io/install/openshift-gitops-operator.yaml
```
装完 `adapter-ops/gitops-argocd-app.yaml` 可以直接用。

### OpenShift Service Mesh (Istio)

OpenShift 自带 Istio Operator。装完我们 `canary/istio-virtualservice.yaml` 直接能用。

### OpenShift Pipelines (Tekton)

CI/CD 跑训练、融合、上传 adapter 的 pipeline，比 Jenkins 轻量，K8s 原生。

---

## 速查表

| 需求 | 原生 K8s | OpenShift 等价/更好 |
|---|---|---|
| 外部访问 | Ingress + Nginx | **Route**（自带 HAProxy） |
| TLS 证书 | cert-manager | **Service Serving Certificates**（集群自签，内部用） |
| Pod 权限 | PSS / PodSecurity labels | **SCC** |
| GitOps | 装 ArgoCD | **OpenShift GitOps Operator** |
| 金丝雀 | Argo Rollouts | Route alternateBackends **or** Argo Rollouts |
| Mesh | 装 Istio | **OpenShift Service Mesh** |
| 模型服务 | 自写 Deployment | **KServe (via RHOAI)** |
| CI/CD | Jenkins/GitLab | **OpenShift Pipelines (Tekton)** |
| 日志 | EFK | **OpenShift Logging**（Loki 栈） |
| 监控 | Prometheus Operator | **内置 Monitoring Stack**（默认装好了） |

**底线**：原生 K8s 能做的 OpenShift 都能做；OpenShift 在"运维 / 合规 / 多租户"上默认配置更严、现成工具更多。公司有 OpenShift 订阅就尽量用 RHOAI + KServe 这条路线，自己写 Deployment 只作为"出问题后 debug 下层"的后备。
