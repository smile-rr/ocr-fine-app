# ECR 镜像仓库 — 业务层镜像 + adapter-controller 镜像
resource "aws_ecr_repository" "api" {
  name                 = "${var.cluster_name}/ocr-api"
  image_tag_mutability = "IMMUTABLE"                # 推荐：tag 不可覆写

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 30 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 30
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecr_repository" "adapter_controller" {
  name                 = "${var.cluster_name}/adapter-controller"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}
