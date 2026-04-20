# EFS filesystem + StorageClass — 给 models/ 和 adapters/ PVC 用
# EFS CSI Driver 已在 cluster_addons 里装了

resource "aws_efs_file_system" "models" {
  creation_token = "${var.cluster_name}-models"
  encrypted      = true
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Name    = "${var.cluster_name}-models"
    Purpose = "LLM model weights + LoRA adapters"
  }
}

# 每个 private subnet 一个 mount target
resource "aws_efs_mount_target" "models" {
  for_each        = toset(module.vpc.private_subnets)
  file_system_id  = aws_efs_file_system.models.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

resource "aws_security_group" "efs" {
  name   = "${var.cluster_name}-efs"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port       = 2049      # NFS
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

# StorageClass + PV —— 让 inference/kubernetes/base/pvc-models.yaml 能 bind
resource "kubernetes_storage_class" "efs" {
  metadata {
    name = "efs-sc"
  }
  storage_provisioner = "efs.csi.aws.com"
  parameters = {
    provisioningMode = "efs-ap"
    fileSystemId     = aws_efs_file_system.models.id
    directoryPerms   = "700"
  }
  reclaim_policy = "Retain"

  depends_on = [module.eks]
}

# 提醒: inference/kubernetes/base/pvc-models.yaml 的 storageClassName 改成 "efs-sc"
# 本目录 eks-overlay/ 有做好的 Kustomize overlay（见文件结构）
