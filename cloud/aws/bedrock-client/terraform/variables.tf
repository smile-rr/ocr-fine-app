variable "region" {
  type    = string
  default = "us-east-1"
}

variable "prefix" {
  type        = string
  default     = "ocr-fine-app"
  description = "资源前缀，避免多项目同账号冲突"
}
