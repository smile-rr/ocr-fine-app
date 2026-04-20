"""vLLM 客户端示例（用 OpenAI SDK）。

跑:
    # 需要先启动 vLLM server:
    # cd inference/vllm && docker compose up -d
    uv run python inference/vllm/client_example.py
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

MODEL = "qwen2.5-7b"   # 与 docker-compose.yml 里的 --served-model-name 对齐


def demo_chat():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是数据分析助手。"},
            {"role": "user", "content": "介绍一下连续批处理（continuous batching）。"},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    print(resp.choices[0].message.content)
    print(f"\nusage: {resp.usage}")


def demo_concurrent(n=16):
    """并发请求，观察 continuous batching 带来的吞吐提升。"""
    import asyncio
    from openai import AsyncOpenAI
    import time

    aclient = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    async def one_call(i):
        return await aclient.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": f"用一句话回答：{i} 的平方是？"}],
            max_tokens=50,
        )

    async def run():
        t = time.time()
        results = await asyncio.gather(*(one_call(i) for i in range(n)))
        print(f"{n} 并发请求，耗时 {time.time()-t:.1f}s")
        print(f"总生成 tokens: {sum(r.usage.completion_tokens for r in results)}")

    asyncio.run(run())


if __name__ == "__main__":
    demo_chat()
    print("\n--- 并发测试 ---")
    demo_concurrent(16)
