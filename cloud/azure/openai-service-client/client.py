"""Azure OpenAI 客户端 —— API Key + Managed Identity 两种姿势。

前置:
    az login                       # 开发机
    # 生产上用 Managed Identity，不用 az login

    export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o
    export AZURE_OPENAI_API_KEY=...     # 仅 demo，生产用 Managed Identity

跑:
    uv run python cloud/azure/openai-service-client/client.py
"""
from __future__ import annotations

import os

from openai import AzureOpenAI

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")


# ============================================================
# 方式 1: API Key （快速 demo）
# ============================================================
def demo_with_api_key():
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=DEPLOYMENT,       # ⚠️ 这里填 deployment 名，不是模型名
        messages=[
            {"role": "system", "content": "你是数据分析助手。"},
            {"role": "user", "content": "用一句话解释 Managed Identity 和 Service Principal 的区别"},
        ],
        max_tokens=200,
        temperature=0.1,
    )
    print("=== API Key ===")
    print(resp.choices[0].message.content)


# ============================================================
# 方式 2: Managed Identity（推荐生产）
# ============================================================
def demo_with_managed_identity():
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    # DefaultAzureCredential 按优先级查:
    #   EnvironmentCredential → ManagedIdentityCredential → AzureCliCredential → ...
    # 本地 dev 用 `az login`，生产在 VM/AKS 用 Managed Identity
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION,
        azure_ad_token_provider=token_provider,
    )
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": "列出 Azure 里 LLM 推理的 3 种部署方式"}],
        max_tokens=300,
    )
    print("\n=== Managed Identity ===")
    print(resp.choices[0].message.content)


# ============================================================
# 方式 3: 流式输出
# ============================================================
def demo_streaming():
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    stream = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": "写 50 字 Azure OpenAI 优势的描述"}],
        stream=True,
        max_tokens=200,
    )
    print("\n=== Streaming ===")
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# ============================================================
# 方式 4: Embedding
# ============================================================
def demo_embedding():
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    resp = client.embeddings.create(
        model=os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small"),
        input=["这是一段要 embedding 的文本"],
    )
    print("\n=== Embedding ===")
    print(f"dim={len(resp.data[0].embedding)}, first 5={resp.data[0].embedding[:5]}")


if __name__ == "__main__":
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        demo_with_api_key()
        demo_streaming()
        demo_embedding()
    demo_with_managed_identity()
