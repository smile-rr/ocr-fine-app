"""示例：用 OpenAI SDK 调用本地 Ollama server。

前置：
    brew install ollama && ollama serve &
    ollama pull qwen2.5:0.5b-instruct-q4_K_M

跑:
    uv run python inference/ollama/client_example.py
"""
from __future__ import annotations

from openai import OpenAI

# 关键：Ollama 的 OpenAI 兼容接口在 :11434/v1
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama 不校验，随便填
)

MODEL = "qwen2.5:0.5b-instruct-q4_K_M"
# 如果你跑了 hf_to_gguf.sh + ollama create ocr-stage2，换成:
# MODEL = "ocr-stage2"


def simple_chat():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个数据分析助手。"},
            {"role": "user", "content": "2023 年营收 120 亿，2024 年 135 亿，同比增长多少？"},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    print("=== simple chat ===")
    print(resp.choices[0].message.content)
    print(f"usage: {resp.usage}")


def streaming_chat():
    """流式生成，延迟感知。"""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话说清楚什么是 RAG"}],
        stream=True,
        max_tokens=150,
    )
    print("\n=== streaming ===")
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


def table_rag_demo():
    """模拟项目 /query 端点的调用方式。"""
    table_md = """| 年份 | 营收(亿) | 净利润(亿) |
|---|---|---|
| 2022 | 100 | 15 |
| 2023 | 120 | 18 |
| 2024 | 135 | 22 |"""

    system = """你是一个精确的数据分析助手。规则：
1. 只基于给定的表格数据回答
2. 答案必须包含具体数值和来源（哪年）
3. 涉及计算时展示计算过程"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"表格：\n{table_md}\n\n问题：净利润增长最快是哪一年？"},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    print("\n=== table RAG demo ===")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    simple_chat()
    streaming_chat()
    table_rag_demo()
