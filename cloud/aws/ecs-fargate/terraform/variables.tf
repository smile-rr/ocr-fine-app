variable "region" {
  type    = string
  default = "us-east-1"
}

variable "prefix" {
  type    = string
  default = "ocr-fine-app"
}

variable "image_uri" {
  type        = string
  description = "ECR 镜像完整 URI, e.g. 123.dkr.ecr.us-east-1.amazonaws.com/ocr-api:v1"
}

variable "sagemaker_endpoint_name" {
  type        = string
  default     = ""
  description = "如果业务层要调 SageMaker Endpoint，填这里；不填就跳过（业务用 Bedrock）"
}
