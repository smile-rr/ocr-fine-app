# 金丝雀 / 蓝绿发布

> 回答「v1 → v2 模型怎么平滑发布、出问题秒级回滚」。

## 金丝雀 vs 蓝绿

| 模式 | 流量切法 | 特点 |
|---|---|---|
| **蓝绿** | 0% → 100% 一次切 | 切换快，但切的瞬间全量切换；需要两倍容量 |
| **金丝雀** | 0% → 5% → 25% → 100% | 逐步放量，能在小比例时捕获问题 |
| **A/B** | 按 header / cookie 定向 | 同一用户始终路由到同一版本；适合产品实验 |

生产上**金丝雀是默认选项**。蓝绿只用于：
- 数据库 schema 变更（渐进式放量会错乱）
- 必须原子切的场景

---

## K8s 上三种实现方案

| 工具 | 原理 | 复杂度 | 自动回滚 |
|---|---|---|---|
| **Argo Rollouts** | 替代原生 Deployment，内置 Canary/BlueGreen 策略 | 低 | ✅ 基于 AnalysisTemplate 自动判断 |
| **Istio VirtualService** | 在 Mesh 层做流量切分 | 中（需要 Istio） | ❌ 手动切或脚本联动 |
| **Flagger** | 监听 Deployment，自动渐进切流量 | 中 | ✅ Prometheus 指标驱动 |

**本目录只给 Argo Rollouts 的完整示例**（成熟度最高、学习曲线最平）。Istio 给 VirtualService 骨架作参考。

---

## 方案 1：Argo Rollouts（推荐）

### 前置

```bash
# 集群装 Argo Rollouts
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

# CLI 插件
brew install argoproj/tap/kubectl-argo-rollouts
```

### 核心思路

用 `Rollout` CR 替代 `Deployment`：
- 保持和 Deployment 几乎一样的 spec
- 多一个 `strategy.canary` 定义放量步骤
- 每一步（比如 10%、25%）之间可以跑 **AnalysisTemplate** 自动检查指标
- 指标坏了自动回滚，不坏了继续放量

见 `argo-rollouts-vllm.yaml`（把 base/deployment-vllm.yaml 转成 Rollout）。

### 发布流程

```bash
# 1. 新版本通过改 image 或 args 触发
kubectl argo rollouts set image vllm vllm=vllm/vllm-openai:v0.6.0

# 2. 看阶段（每一步都有 Paused）
kubectl argo rollouts get rollout vllm

# 3. 人工 approve 每阶段（或让 AnalysisTemplate 自动判断）
kubectl argo rollouts promote vllm

# 4. 有问题回滚
kubectl argo rollouts abort vllm
kubectl argo rollouts undo vllm
```

### AnalysisTemplate —— 金丝雀期间的"健康检查"

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  metrics:
    - name: success-rate
      interval: 30s
      successCondition: result[0] >= 0.95
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            sum(rate(vllm_requests_total{status=~"2..",service="vllm-canary"}[1m]))
            /
            sum(rate(vllm_requests_total{service="vllm-canary"}[1m]))
```

金丝雀阶段跑这个分析：**成功率 >= 95% 才继续放量**，否则自动回滚。

---

## 方案 2：Istio VirtualService（用 Service Mesh 时）

如果集群已经有 Istio，可以用更灵活的流量规则：

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vllm
spec:
  hosts: [vllm]
  http:
    # A/B 测试：beta 用户去 canary
    - match:
        - headers:
            x-beta-tester:
              exact: "true"
      route:
        - destination: { host: vllm, subset: canary }
    # 普通流量：90 / 10 切分
    - route:
        - destination: { host: vllm, subset: stable }
          weight: 90
        - destination: { host: vllm, subset: canary }
          weight: 10
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: vllm
spec:
  host: vllm
  subsets:
    - name: stable
      labels: { version: v1 }
    - name: canary
      labels: { version: v2 }
```

切流量靠改 `weight` + `kubectl apply`。

**优点**：头部路由、Fault Injection、Retry 策略、超时全齐。
**缺点**：需要先引入 Istio（重），单纯为了金丝雀引入不划算。

---

## 方案 3：LoRA 金丝雀（特别情况）

有时发布的"新版本"其实只是一个新 adapter，base 没变。这时**不需要**部署新 Pod，只要：

```
单一 vLLM Deployment
├── base: Qwen-7B（跑 3 个副本）
└── adapters:
    - stage2-prod   (90% 流量 → model="stage2-prod")
    - stage2-canary (10% 流量 → model="stage2-canary")
```

流量切分在**业务层**做：
```python
# app-layer 里
import random

def choose_model(user_id: str) -> str:
    if is_beta_user(user_id):
        return "stage2-canary"
    return "stage2-canary" if random.random() < 0.10 else "stage2-prod"

resp = llm.chat.completions.create(model=choose_model(uid), messages=[...])
```

**优点**：秒级切流量，不用部署 Pod，资源省到极致
**缺点**：base 必须相同；adapter 出问题不能影响 base

**这就是 adapter-ops 方案 C（CRD）里 `spec.traffic.percent` 的场景** —— controller 给 app-layer 下发流量配比。
