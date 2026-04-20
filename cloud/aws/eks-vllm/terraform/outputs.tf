output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "ecr_api_url" {
  value = aws_ecr_repository.api.repository_url
}

output "efs_id" {
  value = aws_efs_file_system.models.id
}

output "update_kubeconfig" {
  value = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.region}"
}

output "next_steps" {
  value = <<-EOT

    === 下一步 ===

    1) 拉 kubeconfig:
       aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.region}

    2) 验证 GPU:
       kubectl get nodes -L node.kubernetes.io/instance-type
       kubectl describe node <gpu-node> | grep nvidia.com/gpu

    3) 部署应用 (从项目根):
       kubectl apply -k inference/kubernetes/base/

    4) 推镜像到 ECR:
       aws ecr get-login-password --region ${var.region} | \
         docker login --username AWS --password-stdin ${aws_ecr_repository.api.repository_url}
       docker tag ocr-api:latest ${aws_ecr_repository.api.repository_url}:v1
       docker push ${aws_ecr_repository.api.repository_url}:v1
  EOT
}
