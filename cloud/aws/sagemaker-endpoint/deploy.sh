#!/usr/bin/env bash
# 一键部署: 打包本地模型 → 上 S3 → terraform apply
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"

# --- 参数 ---
BUCKET="${MODEL_BUCKET:?set MODEL_BUCKET env var (S3 bucket name)}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/stage2_fused}"
REGION="${AWS_REGION:-us-east-1}"
PREFIX="${PREFIX:-ocr-fine-app}"

echo "=== 1. 检查模型目录 ==="
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ $MODEL_DIR 不存在；先跑 scripts/setup_demo_models.sh 或训练流程"
    exit 1
fi

echo
echo "=== 2. 打包 $MODEL_DIR → stage2_fused.tar.gz ==="
TAR_FILE="/tmp/stage2_fused.tar.gz"
tar -C "$MODEL_DIR" -czf "$TAR_FILE" .
echo "  包大小: $(du -h "$TAR_FILE" | cut -f1)"

echo
echo "=== 3. 上传到 s3://$BUCKET/sagemaker/ ==="
S3_PATH="s3://$BUCKET/sagemaker/stage2_fused.tar.gz"
aws s3 cp "$TAR_FILE" "$S3_PATH" --region "$REGION"

echo
echo "=== 4. Terraform apply ==="
cd "$HERE/terraform"
terraform init
terraform apply \
    -var="region=$REGION" \
    -var="prefix=$PREFIX" \
    -var="model_bucket=$BUCKET" \
    -var="s3_model_path=$S3_PATH"

echo
echo "=== ✅ 部署完成 ==="
terraform output

echo
echo "等 endpoint 变 InService (2-5 min):"
echo "  aws sagemaker describe-endpoint --endpoint-name $PREFIX-stage2 \\"
echo "    --query EndpointStatus --region $REGION"
echo
echo "测试:"
echo "  ENDPOINT_NAME=$PREFIX-stage2 uv run python $HERE/client.py"
echo
echo "⚠️  Endpoint 一直计费（~\$1/h for ml.g5.xlarge），用完记得:"
echo "  cd $HERE/terraform && terraform destroy"
