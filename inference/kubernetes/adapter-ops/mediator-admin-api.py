"""Mediator API —— 方案 B：业务层做 vLLM admin 的安全网关。

职责（按处理顺序）：
    1. 接收外部 HTTP 请求（Ingress 后）
    2. 验证 JWT / OAuth token（公司 SSO）
    3. 检查 RBAC（调用者有 adapter.write 权限吗？）
    4. 审计日志写 Loki/Kafka
    5. 校验 adapter 路径在白名单内
    6. 代理到 vLLM 内部 /v1/load_lora_adapter / unload

部署方式见: mediator-deployment.yaml
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="vLLM Admin Mediator", version="1.0.0")
log = logging.getLogger("mediator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------- 配置（生产从 Secret 读，别硬编码） ----------
VLLM_INTERNAL_URL = os.environ["VLLM_INTERNAL_URL"]         # http://vllm.ocr-inference.svc:8000
OIDC_ISSUER = os.environ["OIDC_ISSUER"]                     # https://keycloak.company.com/realms/ml
OIDC_AUDIENCE = os.environ["OIDC_AUDIENCE"]                 # vllm-admin
ADAPTER_PVC_PATH = os.environ.get("ADAPTER_PVC_PATH", "/adapters")

# JWKS 缓存（生产用 python-jose-cryptodome 或 authlib 更稳）
_jwks = None


def verify_token(token: str) -> dict:
    """验证 JWT 并返回 claims。"""
    global _jwks
    if _jwks is None:
        _jwks = httpx.get(f"{OIDC_ISSUER}/protocol/openid-connect/certs", timeout=5).json()
    try:
        return jwt.decode(
            token,
            _jwks,
            algorithms=["RS256"],
            audience=OIDC_AUDIENCE,
            issuer=OIDC_ISSUER,
        )
    except JWTError as e:
        raise HTTPException(401, f"invalid token: {e}")


def check_rbac(claims: dict, required_scope: str):
    """RBAC 示例：从 JWT claims 的 roles 字段看权限。生产对接公司 IAM。"""
    scopes = claims.get("realm_access", {}).get("roles", [])
    if required_scope not in scopes:
        raise HTTPException(403, f"missing scope: {required_scope}")


def audit_log(action: str, user: str, details: dict, outcome: str):
    """审计日志：生产推到 Loki/Kafka/Splunk。这里只打本地 log。"""
    log.info(
        "AUDIT action=%s user=%s outcome=%s details=%s",
        action, user, outcome, details,
    )


# ---------- Schemas ----------
ADAPTER_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}$")      # DNS-safe


class LoadAdapterIn(BaseModel):
    name: str = Field(..., description="adapter 名字，只接受 DNS-safe 字符")
    path: str = Field(..., description="必须在 ADAPTER_PVC_PATH 下的子目录")

    @field_validator("name")
    @classmethod
    def name_ok(cls, v):
        if not ADAPTER_NAME_RE.match(v):
            raise ValueError("invalid adapter name")
        return v

    @field_validator("path")
    @classmethod
    def path_ok(cls, v):
        # 拒绝路径穿越
        if ".." in v or "//" in v:
            raise ValueError("path contains invalid components")
        if not v.startswith(ADAPTER_PVC_PATH + "/"):
            raise ValueError(f"path must start with {ADAPTER_PVC_PATH}/")
        # 还可以再校验 adapter_config.json 是否存在、SHA256 是否在已知清单里
        return v


# ---------- Endpoints ----------
@app.post("/admin/adapters/load")
async def load_adapter(
    req: LoadAdapterIn,
    request: Request,
    authorization: str = Header(...),
):
    # 1. 认证
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "bearer token required")
    token = authorization[7:]
    claims = verify_token(token)
    user = claims.get("preferred_username", "unknown")

    # 2. 鉴权
    check_rbac(claims, "adapter.write")

    details = {
        "adapter_name": req.name,
        "adapter_path": req.path,
        "client_ip": request.client.host if request.client else "",
        "request_id": hashlib.sha1(str(time.time()).encode()).hexdigest()[:12],
    }

    # 3. 调 vLLM 内部
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{VLLM_INTERNAL_URL}/v1/load_lora_adapter",
                json={"lora_name": req.name, "lora_path": req.path},
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        audit_log("adapter.load", user, details, outcome="failed")
        raise HTTPException(502, f"vLLM upstream failed: {e}")

    # 4. 审计
    audit_log("adapter.load", user, details, outcome="success")
    return {"status": "loaded", "request_id": details["request_id"]}


@app.post("/admin/adapters/unload")
async def unload_adapter(
    name: str,
    request: Request,
    authorization: str = Header(...),
):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "bearer token required")
    claims = verify_token(authorization[7:])
    check_rbac(claims, "adapter.write")
    user = claims.get("preferred_username", "unknown")

    if not ADAPTER_NAME_RE.match(name):
        raise HTTPException(400, "invalid adapter name")

    details = {"adapter_name": name}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{VLLM_INTERNAL_URL}/v1/unload_lora_adapter",
                json={"lora_name": name},
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        audit_log("adapter.unload", user, details, outcome="failed")
        raise HTTPException(502, str(e))

    audit_log("adapter.unload", user, details, outcome="success")
    return {"status": "unloaded"}


@app.get("/admin/adapters")
async def list_adapters(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401)
    claims = verify_token(authorization[7:])
    check_rbac(claims, "adapter.read")

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{VLLM_INTERNAL_URL}/v1/models")
        return resp.json()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
