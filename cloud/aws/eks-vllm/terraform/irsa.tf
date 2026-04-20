# IAM Roles for Service Accounts (IRSA)
# 让 K8s 里的 ServiceAccount 能 assume AWS role，Pod 调 AWS API 时零凭证

# ============================================================
# vllm-runner SA: 让 vLLM Pod 从 S3 拉 adapter 权重（可选）
# ============================================================
module "vllm_runner_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.cluster_name}-vllm-runner"

  role_policy_arns = {
    policy = aws_iam_policy.vllm_runner.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["ocr-inference:vllm-runner"]
    }
  }
}

resource "aws_iam_policy" "vllm_runner" {
  name = "${var.cluster_name}-vllm-runner"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.model_bucket}",
          "arn:aws:s3:::${var.model_bucket}/*",
        ]
      }
    ]
  })
}

# ============================================================
# adapter-controller SA: 让 mediator API 调 Bedrock / SSM
# ============================================================
module "adapter_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.cluster_name}-adapter-controller"

  role_policy_arns = {
    policy = aws_iam_policy.adapter_controller.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["ocr-inference:adapter-controller"]
    }
  }
}

resource "aws_iam_policy" "adapter_controller" {
  name = "${var.cluster_name}-adapter-controller"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath",
        ]
        # 限制在 /ocr/ 前缀下
        Resource = "arn:aws:ssm:${var.region}:*:parameter/ocr/*"
      },
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = "*"
      }
    ]
  })
}

# ============================================================
# ⭐ 在 K8s 里创建 ServiceAccount 并打上 role-arn annotation
# 这样 Pod 引用这个 SA 就自动拿到 IRSA
# ============================================================
resource "kubernetes_service_account" "vllm_runner" {
  metadata {
    name      = "vllm-runner"
    namespace = "ocr-inference"
    annotations = {
      "eks.amazonaws.com/role-arn" = module.vllm_runner_irsa.iam_role_arn
    }
  }
  depends_on = [module.eks]
}

resource "kubernetes_service_account" "adapter_controller" {
  metadata {
    name      = "adapter-controller"
    namespace = "ocr-inference"
    annotations = {
      "eks.amazonaws.com/role-arn" = module.adapter_controller_irsa.iam_role_arn
    }
  }
  depends_on = [module.eks]
}
