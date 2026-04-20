variable "region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "ocr-prod"
}

variable "allowed_cidr_blocks" {
  type        = list(string)
  default     = ["0.0.0.0/0"]
  description = "谁能访问 EKS API server。生产上限制到 VPN/办公网段"
}

variable "model_bucket" {
  type        = string
  default     = "ocr-fine-app-models"
  description = "S3 bucket for model artifacts (vLLM runner 需要读权限)"
}
