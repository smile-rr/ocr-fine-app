# LoRA Adapter 热加载 —— 安全的生产做法

> **你的顾虑是对的**：直接暴露 vLLM 的 `POST /v1/load_lora_adapter` 给外部访问是严重的安全问题。本节把"正确做法"拆开讲清楚。

## 为什么 URL 直调不安全（威胁模型）

vLLM 原生 `/v1/load_lora_adapter` 接口有 **4 个致命问题**：

| 风险 | 说明 |
|---|---|
| **无认证** | 默认开到 0.0.0.0:8000，没 API key 没 mTLS。任何能访问到 vLLM Service 的人都能动 |
| **路径穿越** | 接口接收任意 `lora_path` 字符串，可以是 `/etc/passwd` 或 `/root/.ssh/`（虽然加载会失败，但信息泄漏是可能的） |
| **权重投毒** | 挂载的 adapter 目录如果可写（misconfig），攻击者传恶意权重上去就等于控制模型行为 |
| **无审计** | 谁加载了啥、什么时候、为什么，运维侧完全没记录；故障无法复盘 |

生产环境的答案是：**永远不要把 vLLM 的 admin 接口直接暴露**。

---

## 三种生产级方案

按"工程成本 / 安全度"排序：

| 方案 | 工程成本 | 安全度 | 审计 | 回滚速度 | 适合 |
|---|---|---|---|---|---|
| **A. GitOps (ArgoCD + ConfigMap + Job)** | 低 | ★★★★★ | Git 即审计 | 改 YAML + revert commit | **首选**，中小团队 |
| **B. Mediator API（业务层网关）** | 中 | ★★★★ | 业务层日志 | 调自己的 admin API | 需要"运营人员点按钮加 adapter"时 |
| **C. 自研 Operator + CRD** | 高 | ★★★★★ | 事件流 + CR status | `kubectl delete adapter xxx` | 大平台，多租户 SaaS |

**三种方案的共同要点**（一开始就应该做）：
1. vLLM Service 设为 ClusterIP，只能集群内访问 → `base/service-vllm.yaml` ✅ 已做
2. NetworkPolicy 只允许 app-layer + adapter-controller + Prometheus → `base/networkpolicy.yaml` ✅ 已做
3. 加载触发器从**容器化**的 actor 发起（不是运维手 curl），留下可审计的日志
4. adapter 权重文件来源可信（只从 PVC 的某个由 CI 写入的路径加载，不接受任意 URL）

---

## 方案 A：GitOps（推荐，本目录示例方案）

**核心思路**：adapter 的"应该加载的状态"写在 Git 的 ConfigMap 里。CI 上传权重，Git PR 合并触发 ArgoCD sync，K8s Job 执行真正的加载动作。

```
   开发者                                               运维/平台
      │                                                    │
      ▼                                                    │
   ① CI: 训练 + 融合 + 上传权重到 PVC           ⑥ 看 ArgoCD Dashboard 确认
      │  (独立 Job，跑在 GPU 节点)                          │
      ▼                                                    ▲
   PVC: /adapters/stage2-v3/                              │
      │                                                    │
   ② PR: 改 configmap-adapters.yaml:                     ⑤ ArgoCD Sync
      adapters += { name: stage2-v3,                       │
                    path: /adapters/stage2-v3 }            │
      │                                                    │
      ▼                                                    │
   ③ 代码评审通过，合并到 main                             │
      │                                                    │
      ▼                                                    │
   ④ ArgoCD 监听 Git → 检测到 ConfigMap 变化 ──────────────┤
                                                           │
                                                           ▼
                                                  ⑦ adapter-sync-job:
                                                     - 读 ConfigMap
                                                     - 调 vLLM 内部 API:
                                                       POST /v1/load_lora_adapter
                                                       POST /v1/unload_lora_adapter
                                                     - 调用完自删
                                                           │
                                                           ▼
                                                     vLLM Pod 现在挂上 v3
```

**关键安全性质**：
- **任何 adapter 变更都是 Git commit** —— 谁、什么时候、为什么，永久记录
- **4 眼原则**（4-eyes）：改 ConfigMap 要 PR review
- **回滚 = `git revert`** —— 秒级可追溯
- **vLLM 的 admin 接口只被 Job 调用**，Job 在同 namespace 受 NetworkPolicy 保护
- **整个流程没有任何人手 curl**

### 本目录给的示例文件

| 文件 | 作用 |
|---|---|
| `gitops-argocd-app.yaml` | ArgoCD Application 定义（监听 Git 目录） |
| `gitops-sync-job.yaml` | 每次 ConfigMap 变化时跑的 Job |
| `gitops-upload-job.yaml` | CI 触发的 adapter 权重上传 Job（用 kaniko 或 oras） |

---

## 方案 B：Mediator API（业务层当管控网关）

**核心思路**：业务层 FastAPI 自己实现一个 `/admin/adapters` 端点，做认证、鉴权、审计，然后**它**去调 vLLM 的内部 `/v1/load_lora_adapter`。

```
运维人员
   │ JWT token (OIDC)
   ▼
Ingress ──► app-layer /admin/adapters (POST)
                │
                ├─ 1. 验 JWT (OIDC 集成公司 SSO)
                ├─ 2. 查 RBAC (谁有 adapter.write 权限)
                ├─ 3. 审计日志 (推 Kafka / Loki)
                ├─ 4. 校验 adapter_name 是预期格式（拒绝 ../ 路径）
                ├─ 5. 校验权重在预期 PVC 路径下
                └─ 6. 调 vLLM 内部:
                     POST http://vllm.ocr-inference.svc/v1/load_lora_adapter
                                │
                                ▼
                         vLLM Pod 真正加载
```

**优点**：
- 运营人员可以在 admin UI 上点按钮加 adapter，不用改 Git
- 业务层本来就有认证栈，顺手复用

**缺点**：
- 自己写代码，代码就是攻击面
- 审计需要自己做（Git 天然自带）
- 回滚需要再调一次 API

**适用场景**：需要"非开发人员自助加 adapter"的场景，比如多租户 SaaS 里每个客户上传自己的微调。

### 本目录给的示例文件

| 文件 | 作用 |
|---|---|
| `mediator-admin-api.py` | FastAPI 实现（认证 + 审计 + 代理到 vLLM） |
| `mediator-deployment.yaml` | K8s 部署 admin-api |
| `mediator-ingress.yaml` | Ingress + OIDC / Keycloak 集成 |

---

## 方案 C：自研 Operator + CRD（大平台玩法）

**核心思路**：定义一个 `LoRAAdapter` CRD，让 adapter 成为 K8s 一等公民：

```yaml
apiVersion: ml.example.com/v1
kind: LoRAAdapter
metadata:
  name: stage2-v3
  namespace: ocr-inference
spec:
  baseModel: Qwen/Qwen2.5-7B-Instruct
  weightsURI: s3://ml-artifacts/stage2-v3/
  targetService: vllm
status:
  phase: Loaded
  loadedAt: "2026-04-19T10:30:00Z"
  servingReplicas: 2
```

Operator watch CR → 把权重从 S3 拉到 PVC → 调 vLLM 加载 → 更新 status。

**优点**：
- 完全 K8s 原生体验，`kubectl get adapters`
- 适合多租户平台、需要复杂状态机（加载中/失败/回滚）
- 可以做更高级的调度（跨多个 vLLM 集群同步 adapter）

**缺点**：
- 写 Operator 要懂 controller-runtime / Kubebuilder，3-5 人周的工作量
- 维护成本高

**适用**：公司做内部 ML Platform，要服务几十个团队。参考 KServe / Kubeflow / OpenShift AI 的做法。

### 本目录给的示例文件

| 文件 | 作用 |
|---|---|
| `crd-loraadapter.yaml` | CRD 定义（schema + validation） |
| `operator-deployment.yaml.example` | Operator 部署框架（代码不在本仓） |

---

## 推荐路径

刚开始：**方案 A（GitOps）**。它把"谁改了什么"这件事彻底交给 Git，是安全性/成本比最高的方案。

业务 / 租户复杂了，再在 A 的基础上加方案 B 或升级到 C。**永远不要回到"URL 直接暴露"的起点**。

---

## Bonus：如何判断你的部署是否安全

一分钟自检：

```bash
# 1. vLLM Service 对外暴露了吗？应该只有 ClusterIP
kubectl -n ocr-inference get svc vllm -o jsonpath='{.spec.type}'
# 期望: ClusterIP    ✅

# 2. 从集群外能访问吗？应该 timeout
kubectl get svc vllm -n ocr-inference
# 如果没有 EXTERNAL-IP，就是内部服务 ✅

# 3. NetworkPolicy 存在吗？
kubectl -n ocr-inference get networkpolicy
# 期望: vllm-deny-default    ✅

# 4. 试试从无权 namespace 调 load_lora_adapter
kubectl run attacker --image=curlimages/curl -n default -- sleep 999
kubectl exec -n default attacker -- curl -X POST \
  http://vllm.ocr-inference.svc.cluster.local:8000/v1/load_lora_adapter \
  -d '{"lora_name":"pwn","lora_path":"/tmp/pwn"}'
# 期望: 连接超时（NetworkPolicy 拦了）   ✅

# 5. 审计日志有吗？
argocd app history adapter-config
# 期望: 看到每次 adapter 变更的 Git commit   ✅
```

任何一条 ❌，回去修。
