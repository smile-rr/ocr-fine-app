# Bedrock 本身不需要 provision 模型资源，但企业环境通常要：
# 1. IAM policy：限制哪个 role 能调哪些模型
# 2. Guardrails：内容安全过滤
# 3. VPC Endpoint（PrivateLink）：流量不出 VPC

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ============================================================
# 1. IAM: 给业务层（Lambda/ECS/EKS Pod）调 Bedrock 的权限
# ============================================================
resource "aws_iam_policy" "bedrock_invoke" {
  name        = "${var.prefix}-bedrock-invoke"
  description = "Invoke specific Bedrock foundation models"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:Converse",
          "bedrock:ConverseStream",
        ]
        # 只允许指定模型，防止误调贵的（比如 Opus）
        Resource = [
          "arn:aws:bedrock:${var.region}::foundation-model/anthropic.claude-3-5-sonnet-*",
          "arn:aws:bedrock:${var.region}::foundation-model/anthropic.claude-3-haiku-*",
          "arn:aws:bedrock:${var.region}::foundation-model/meta.llama3-*",
        ]
      },
      # 如果用 Guardrails，追加这条
      {
        Effect   = "Allow"
        Action   = ["bedrock:ApplyGuardrail"]
        Resource = aws_bedrock_guardrail.content_safety.guardrail_arn
      },
    ]
  })
}

# ============================================================
# 2. Guardrails: 内容安全（拒绝涉政/涉黄/PII 泄漏）
# ============================================================
resource "aws_bedrock_guardrail" "content_safety" {
  name                      = "${var.prefix}-content-safety"
  description               = "Basic content safety for prod LLM calls"
  blocked_input_messaging   = "I cannot process this request."
  blocked_outputs_messaging = "I cannot provide this response."

  # 有害内容过滤
  content_policy_config {
    filters_config {
      input_strength  = "HIGH"
      output_strength = "HIGH"
      type            = "VIOLENCE"
    }
    filters_config {
      input_strength  = "HIGH"
      output_strength = "HIGH"
      type            = "HATE"
    }
    filters_config {
      input_strength  = "HIGH"
      output_strength = "HIGH"
      type            = "SEXUAL"
    }
    filters_config {
      input_strength  = "HIGH"
      output_strength = "HIGH"
      type            = "MISCONDUCT"
    }
    filters_config {
      input_strength  = "NONE"   # Prompt attack 建议应用侧处理
      output_strength = "NONE"
      type            = "PROMPT_ATTACK"
    }
  }

  # PII 过滤（身份证、卡号等）
  sensitive_information_policy_config {
    pii_entities_config {
      action = "BLOCK"
      type   = "EMAIL"
    }
    pii_entities_config {
      action = "BLOCK"
      type   = "PHONE"
    }
    pii_entities_config {
      action = "ANONYMIZE"
      type   = "CREDIT_DEBIT_CARD_NUMBER"
    }
  }
}

# ============================================================
# 3. (可选) PrivateLink VPC Endpoint: 流量不出 VPC
# ============================================================
# 启用条件：你业务跑在 VPC 里（ECS/EKS/Lambda in VPC），且合规要求不走 public internet
#
# 成本提醒：每个 VPC Endpoint 约 $7/月 + 流量费
#
# resource "aws_vpc_endpoint" "bedrock_runtime" {
#   vpc_id              = var.vpc_id
#   service_name        = "com.amazonaws.${var.region}.bedrock-runtime"
#   vpc_endpoint_type   = "Interface"
#   subnet_ids          = var.private_subnet_ids
#   security_group_ids  = [var.endpoint_sg_id]
#   private_dns_enabled = true
#   tags = { Name = "${var.prefix}-bedrock-runtime" }
# }

# ============================================================
# 4. 样例 IAM Role —— 给 EKS Pod (IRSA) 使用
# ============================================================
# 完整 IRSA 配置见 eks-vllm/，这里只示意 trust policy
data "aws_caller_identity" "current" {}

resource "aws_iam_role" "bedrock_client" {
  name = "${var.prefix}-bedrock-client"
  # 这个 trust policy 要根据实际场景改（EKS IRSA / ECS Task / Lambda）
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"   # 改成对应服务
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "bedrock_client" {
  role       = aws_iam_role.bedrock_client.name
  policy_arn = aws_iam_policy.bedrock_invoke.arn
}

output "guardrail_id" {
  value = aws_bedrock_guardrail.content_safety.guardrail_id
}

output "guardrail_version" {
  value = aws_bedrock_guardrail.content_safety.version
}

output "client_role_arn" {
  value = aws_iam_role.bedrock_client.arn
}
