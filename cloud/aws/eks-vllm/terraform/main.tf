# EKS Cluster with GPU node group, ready to run inference/kubernetes/base/
# 简化版：生产上建议用 terraform-aws-modules/eks/aws 并补多账户 / 多 AZ / 私网 endpoint
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws        = { source = "hashicorp/aws",        version = "~> 5.0" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.25" }
    helm       = { source = "hashicorp/helm",       version = "~> 2.11" }
  }
}

provider "aws" {
  region = var.region
}

data "aws_availability_zones" "available" {}

# ============================================================
# VPC
# ============================================================
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name                 = "${var.cluster_name}-vpc"
  cidr                 = "10.0.0.0/16"
  azs                  = slice(data.aws_availability_zones.available.names, 0, 2)
  public_subnets       = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets      = ["10.0.11.0/24", "10.0.12.0/24"]
  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  # EKS 需要的 tag
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    # Karpenter 靠这个 tag 发现 subnet
    "karpenter.sh/discovery" = var.cluster_name
  }
}

# ============================================================
# EKS Cluster
# ============================================================
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.30"

  cluster_endpoint_public_access       = true
  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # IRSA = IAM Roles for Service Accounts
  enable_irsa = true

  # Addons
  cluster_addons = {
    coredns                = {}
    kube-proxy             = {}
    vpc-cni                = {}
    aws-ebs-csi-driver     = {}
    aws-efs-csi-driver     = {}
    eks-pod-identity-agent = {}
  }

  # General-purpose 节点组
  eks_managed_node_groups = {
    general = {
      instance_types = ["t3.medium"]
      min_size       = 2
      max_size       = 4
      desired_size   = 2
      capacity_type  = "ON_DEMAND"
    }

    # GPU 节点组（Karpenter 不启用时用这个 static 节点组）
    gpu = {
      instance_types = ["g5.xlarge"]   # 1× A10G 24GB
      min_size       = 1
      max_size       = 4
      desired_size   = 1
      capacity_type  = "ON_DEMAND"     # 或 "SPOT" 省 70%

      # GPU 专用 AMI (含 NVIDIA drivers)
      ami_type = "AL2_x86_64_GPU"

      labels = {
        accelerator = "nvidia-gpu"     # 对应 base/deployment-vllm.yaml 的 nodeSelector
      }
      taints = [
        # 只允许声明 GPU 的 Pod 调度到这里
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  # Karpenter 需要的 tag（本文件暂不开 Karpenter，见 karpenter.tf.example）
  node_security_group_tags = {
    "karpenter.sh/discovery" = var.cluster_name
  }
}

# ============================================================
# Kubernetes / Helm provider config (从上面 EKS 拿 credential)
# ============================================================
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# ============================================================
# NVIDIA Device Plugin (让 K8s 识别 nvidia.com/gpu 资源)
# ============================================================
resource "helm_release" "nvidia_device_plugin" {
  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  version    = "0.16.1"
  namespace  = "kube-system"

  set {
    name  = "tolerations[0].key"
    value = "nvidia.com/gpu"
  }
  set {
    name  = "tolerations[0].operator"
    value = "Exists"
  }

  depends_on = [module.eks]
}
