terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

# ============================================================
# 1. S3 bucket for model artifacts (复用已有就不用创建)
# ============================================================
# 如果你已经有模型上传的 bucket，直接传参；下面用已有 bucket 的示例：
# variable "s3_model_path" = "s3://YOUR-BUCKET/sagemaker/stage2_fused.tar.gz"

# ============================================================
# 2. IAM Role for SageMaker Endpoint
# ============================================================
resource "aws_iam_role" "sagemaker_execution" {
  name = "${var.prefix}-sagemaker-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# 如果 bucket 在别的账户，给个显式 S3 policy
resource "aws_iam_role_policy" "s3_read" {
  name = "${var.prefix}-s3-read"
  role = aws_iam_role.sagemaker_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = ["arn:aws:s3:::${var.model_bucket}", "arn:aws:s3:::${var.model_bucket}/*"]
    }]
  })
}

# ============================================================
# 3. SageMaker Model (artifact + inference image)
# ============================================================
resource "aws_sagemaker_model" "stage2" {
  name               = "${var.prefix}-stage2"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    # DJL-LMI (Deep Java Library Large Model Inference) 容器
    # AWS 每个 region 的 DLC URI 见: https://github.com/aws/deep-learning-containers
    image = var.lmi_image_uri

    # 模型权重来源（三选一）：
    #   A) S3 tar.gz（打包好的 HF 模型）
    model_data_url = var.s3_model_path
    #   B) HF Hub 直接拉（见 environment 里的 HF_MODEL_ID）
    #   C) S3 目录（uncompressed）—— 需要 model_data_source block

    environment = {
      # DJL-LMI 的配置都走环境变量（等价 vLLM 的 CLI args）
      "HF_MODEL_ID"                   = var.hf_model_id                  # fallback 如果 S3 没传
      "OPTION_ROLLING_BATCH"          = "vllm"
      "OPTION_DTYPE"                  = "bf16"
      "OPTION_MAX_MODEL_LEN"          = "8192"
      "OPTION_MAX_ROLLING_BATCH_SIZE" = "32"
      "OPTION_TENSOR_PARALLEL_DEGREE" = "1"
      "OPTION_GPU_MEMORY_UTILIZATION" = "0.9"
      "OPTION_ENABLE_LORA"            = "true"
      "OPTION_MAX_LORAS"              = "4"
      # AWS 预下载模型到 /opt/ml/model；HuggingFace hub 要 token 时加:
      # "HF_TOKEN" = var.hf_token
    }
  }
}

# ============================================================
# 4. EndpointConfig - 定义实例规格 + 扩缩容 + 金丝雀
# ============================================================
resource "aws_sagemaker_endpoint_configuration" "stage2" {
  name = "${var.prefix}-stage2-config"

  production_variants {
    variant_name           = "v1"
    model_name             = aws_sagemaker_model.stage2.name
    initial_instance_count = var.initial_instance_count
    instance_type          = var.instance_type    # ml.g5.xlarge (A10G 24GB)
    initial_variant_weight = 1

    # 慢启动：给容器 10 分钟加载（大模型需要）
    model_data_download_timeout_in_seconds = 600
    container_startup_health_check_timeout_in_seconds = 600
  }

  # 金丝雀示例（取消注释启用）：
  # production_variants {
  #   variant_name           = "v2-canary"
  #   model_name             = aws_sagemaker_model.stage2_v2.name
  #   initial_instance_count = 1
  #   instance_type          = "ml.g5.xlarge"
  #   initial_variant_weight = 0.1
  # }

  # 异步推理（长请求 + 批量）或数据捕获
  # data_capture_config {
  #   capture_content_type_header {}
  #   capture_options   { capture_mode = "Input" }
  #   capture_options   { capture_mode = "Output" }
  #   destination_s3_uri = "s3://${var.model_bucket}/captures/"
  #   initial_sampling_percentage = 100
  # }
}

# ============================================================
# 5. Endpoint - 真正对外的 URL
# ============================================================
resource "aws_sagemaker_endpoint" "stage2" {
  name                 = "${var.prefix}-stage2"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.stage2.name

  tags = {
    ManagedBy = "terraform"
    Project   = "ocr-fine-app"
  }
}

# ============================================================
# 6. Auto Scaling
# ============================================================
resource "aws_appautoscaling_target" "sagemaker" {
  max_capacity       = 10
  min_capacity       = var.initial_instance_count
  resource_id        = "endpoint/${aws_sagemaker_endpoint.stage2.name}/variant/v1"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "sagemaker" {
  name               = "${var.prefix}-invocations-per-instance"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker.resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 100   # 每实例 100 QPS 触发扩容
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
  }
}
