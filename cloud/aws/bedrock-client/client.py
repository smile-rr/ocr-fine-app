"""Bedrock 客户端 —— 3 种调用方式示例。

前置:
    aws configure
    # 首次使用要在 console 申请 model access:
    # https://console.aws.amazon.com/bedrock/home#/modelaccess

跑:
    uv run python cloud/aws/bedrock-client/client.py
"""
from __future__ import annotations

import json
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
)


# ========== 方式 1: Converse API（推荐，统一接口） ==========
def demo_converse():
    client = boto3.client("bedrock-runtime", region_name=REGION)
    resp = client.converse(
        modelId=MODEL_ID,
        messages=[
            {"role": "user", "content": [{"text": "用一句话介绍 RAG 的核心思想。"}]}
        ],
        system=[{"text": "你是简洁的技术编辑。"}],
        inferenceConfig={"maxTokens": 200, "temperature": 0.1, "topP": 0.9},
    )
    print("=== Converse API ===")
    print(resp["output"]["message"]["content"][0]["text"])
    print(f"\nUsage: input={resp['usage']['inputTokens']}, "
          f"output={resp['usage']['outputTokens']}")


# ========== 方式 2: InvokeModel（老 API，每个模型 body 格式不同） ==========
def demo_invoke_model():
    client = boto3.client("bedrock-runtime", region_name=REGION)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0.1,
        "system": "你是简洁的技术编辑。",
        "messages": [{"role": "user", "content": "用一句话说 PagedAttention 在解决什么。"}],
    }
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
    )
    result = json.loads(resp["body"].read())
    print("\n=== InvokeModel ===")
    print(result["content"][0]["text"])


# ========== 方式 3: 流式输出 ==========
def demo_streaming():
    client = boto3.client("bedrock-runtime", region_name=REGION)
    resp = client.converse_stream(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": "写一段关于云原生微调栈的 100 字文案"}]}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.3},
    )
    print("\n=== Streaming ===")
    for event in resp["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                print(delta["text"], end="", flush=True)
        if "messageStop" in event:
            print()  # newline


# ========== 方式 4: LiteLLM 兼容层（保留 OpenAI SDK 习惯） ==========
def demo_litellm():
    try:
        from litellm import completion
    except ImportError:
        print("\n(跳过 LiteLLM 示例，先 pip install litellm)")
        return

    # LiteLLM 把 Bedrock 协议翻译成 OpenAI 格式，业务代码和调 OpenAI 一模一样
    resp = completion(
        model=f"bedrock/{MODEL_ID}",
        messages=[{"role": "user", "content": "用一句话解释 continuous batching。"}],
        max_tokens=150,
    )
    print("\n=== LiteLLM (OpenAI-style) ===")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    demo_converse()
    demo_invoke_model()
    demo_streaming()
    demo_litellm()
