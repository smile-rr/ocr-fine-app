output "alb_dns" {
  value       = aws_lb.main.dns_name
  description = "业务 API 的公网 URL"
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "service_name" {
  value = aws_ecs_service.api.name
}

output "curl_example" {
  value = "curl http://${aws_lb.main.dns_name}/health"
}
