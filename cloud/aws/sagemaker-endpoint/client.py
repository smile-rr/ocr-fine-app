"""SageMaker Endpoint 客户端示例（boto3 + OpenAI 兼容层两种姿势）。

前置:
    terraform apply 已经把 endpoint 拉起来了
    export ENDPOINT_NAME=ocr-fine-app-stage2

跑:
    uv run python cloud/aws/sagemaker-endpoint/client.py
"""
from __future__ import annotations

import json
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ.get("ENDPOINT_NAME", "ocr-fine-app-stage2")


# ============================================================
# 方式 1: 原生 boto3 sagemaker-runtime
# ============================================================
def demo_native():
    client = boto3.client("sagemaker-runtime", region_name=REGION)

    # DJL-LMI 接受的 payload 格式（等价 vLLM /v1/chat/completions）
    payload = {
        "messages": [
            {"role": "system", "content": "你是数据分析助手。"},
            {"role": "user", "content": "用一句话解释 PagedAttention。"},
        ],
        "max_new_tokens": 200,
        "temperature": 0.1,
    }

    resp = client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(resp["Body"].read())
    print("=== boto3 native ===")
    print(result)


# ============================================================
# 方式 2: DJL-LMI 暴露 OpenAI 兼容接口（从 v11 起）
# ============================================================
# DJL-LMI 容器内部就是 vLLM，支持 /v1/chat/completions 路径
# SageMaker 上要用 "Custom Inference Path": 先配容器 env 开启
# 或者用 OpenAI SDK + sagemaker bridge
def demo_openai_style():
    from openai import OpenAI
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    # SageMaker 不直接兼容 OpenAI HTTP，但可以用 boto3 sign 后转发
    # 生产上更常见做法：前置一个 Lambda/ALB 做协议转换
    print("\n(见代码注释: DJL-LMI 支持 OpenAI 格式 payload，但需要客户端手动签名 SigV4)")


# ============================================================
# 方式 3: 流式输出（InvokeEndpointWithResponseStream）
# ============================================================
def demo_streaming():
    client = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = client.invoke_endpoint_with_response_stream(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({
            "messages": [{"role": "user", "content": "写 50 字关于云原生的文案。"}],
            "max_new_tokens": 200,
            "stream": True,
        }),
    )
    print("\n=== Streaming ===")
    for event in resp["Body"]:
        if "PayloadPart" in event:
            chunk = event["PayloadPart"]["Bytes"].decode()
            # DJL-LMI 流式输出是 JSONL, 每行一个 token 块
            for line in chunk.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {}).get("content", "")
                        print(delta, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
    print()


# ============================================================
# 方式 4: 多 Variant 流量切分（金丝雀触发）
# ============================================================
def demo_target_variant():
    client = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        TargetVariant="v2-canary",          # 强制路由到 canary variant
        Body=json.dumps({
            "messages": [{"role": "user", "content": "test canary"}],
            "max_new_tokens": 50,
        }),
    )
    print("\n=== Forced canary variant ===")
    print(json.loads(resp["Body"].read()))


if __name__ == "__main__":
    demo_native()
    demo_streaming()
    # demo_target_variant()   # 只在配了 canary variant 时跑
