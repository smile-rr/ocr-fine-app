variable "region" {
  type    = string
  default = "us-east-1"
}

variable "prefix" {
  type    = string
  default = "ocr-fine-app"
}

variable "model_bucket" {
  type        = string
  description = "S3 bucket name（不含 s3://）"
}

variable "s3_model_path" {
  type        = string
  description = "S3 完整路径到 model.tar.gz, e.g. s3://bucket/key.tar.gz"
}

variable "hf_model_id" {
  type        = string
  default     = "Qwen/Qwen2.5-7B-Instruct"
  description = "如果 s3_model_path 为空，DJL-LMI 会从 HF Hub 拉这个 model id"
}

# DJL-LMI 容器镜像（不同 region 不同 account id，这里是 us-east-1 的）
# 完整列表：https://github.com/aws/deep-learning-containers/blob/master/available_images.md
variable "lmi_image_uri" {
  type    = string
  default = "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124"
}

variable "instance_type" {
  type    = string
  default = "ml.g5.xlarge"
  # g5.xlarge  = 1× A10G 24GB (~$1.0/h)   — 7B QLoRA/AWQ 足够
  # g5.2xlarge = 1× A10G 24GB (~$1.2/h)
  # g6.xlarge  = 1× L4 24GB   (~$0.8/h)   — 更新一代，便宜
  # p4d.24xlarge = 8× A100 40GB (~$32/h)  — 70B+ 才用
}

variable "initial_instance_count" {
  type    = number
  default = 1
}
