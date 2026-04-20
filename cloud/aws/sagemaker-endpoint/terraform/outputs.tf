output "endpoint_name" {
  value = aws_sagemaker_endpoint.stage2.name
}

output "endpoint_arn" {
  value = aws_sagemaker_endpoint.stage2.arn
}

output "execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}

# 调用方式示意
output "invoke_example" {
  value = <<-EOT
    aws sagemaker-runtime invoke-endpoint \
      --endpoint-name ${aws_sagemaker_endpoint.stage2.name} \
      --content-type application/json \
      --body fileb://payload.json \
      output.json && cat output.json
  EOT
}
