#!/usr/bin/env bash
# 测试 Docker API 端点
# 用法：./scripts/test_api.sh [host:port]
set -e
HOST="${1:-http://localhost:8000}"

echo "=== /health ==="
curl -sS "$HOST/health" | python -m json.tool

SAMPLE_PDF="data/samples/apple_2023_q4.pdf"
if [ -f "$SAMPLE_PDF" ]; then
    echo -e "\n=== /extract (上传 PDF) ==="
    curl -sS -X POST "$HOST/extract" \
        -F "file=@$SAMPLE_PDF" \
        -F "doc_id=apple_2023_q4" | python -m json.tool | head -80
fi

echo -e "\n=== /ingest_markdown (直接入库) ==="
curl -sS -X POST "$HOST/ingest_markdown" \
    -F "doc_id=demo" \
    -F "page=1" \
    -F 'markdown=| 年份 | 营收 | 净利润 |
|---|---|---|
| 2022 | 100 | 15 |
| 2023 | 120 | 18 |
| 2024 | 135 | 22 |' | python -m json.tool

echo -e "\n=== /query (RAG 问答) ==="
curl -sS -X POST "$HOST/query" \
    -H 'Content-Type: application/json' \
    -d '{"question":"哪一年净利润最高？","top_k":5,"doc_filter":"demo"}' \
    | python -m json.tool

echo -e "\n=== /admin/reload (热加载 Stage 2) ==="
ADMIN_KEY="${ADMIN_API_KEY:-change-me-in-prod}"
curl -sS -X POST "$HOST/admin/reload" \
    -H "X-Admin-Key: $ADMIN_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"stage":2,"force":false}' \
    | python -m json.tool
