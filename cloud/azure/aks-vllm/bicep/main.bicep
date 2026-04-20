// AKS + GPU Node Pool + Azure Files + Workload Identity + ACR
// 部署:
//   az deployment group create \
//       --resource-group rg-ocr-fine-app \
//       --template-file main.bicep \
//       --parameters prefix=ocr location=eastus sshPublicKey="$(cat ~/.ssh/id_rsa.pub)"

param prefix string = 'ocr'
param location string = 'eastus'
param sshPublicKey string                    // cat ~/.ssh/id_rsa.pub
param kubernetesVersion string = '1.30.4'

// ============================================================
// Managed Identity (cluster identity)
// ============================================================
resource clusterIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-aks-mi'
  location: location
}

// ============================================================
// ACR
// ============================================================
resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: '${prefix}acr${uniqueString(resourceGroup().id)}'
  location: location
  sku: { name: 'Standard' }
  properties: {
    adminUserEnabled: false
  }
}

// 让 cluster 能 pull ACR
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
resource acrPullAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, clusterIdentity.id, acrPullRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: clusterIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ============================================================
// Log Analytics (AKS monitoring)
// ============================================================
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-aks-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ============================================================
// AKS Cluster
// ============================================================
resource aks 'Microsoft.ContainerService/managedClusters@2024-05-01' = {
  name: '${prefix}-aks'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${clusterIdentity.id}': {}
    }
  }
  properties: {
    dnsPrefix: '${prefix}-aks'
    kubernetesVersion: kubernetesVersion
    enableRBAC: true

    // ⭐ 启用 OIDC + Workload Identity（对应 EKS IRSA）
    oidcIssuerProfile: { enabled: true }
    securityProfile: {
      workloadIdentity: { enabled: true }
    }

    // AAD 集成（用 Entra ID 管 RBAC）
    aadProfile: {
      managed: true
      enableAzureRBAC: true
    }

    // Addons
    addonProfiles: {
      azurepolicy: { enabled: true }
      omsagent: {
        enabled: true
        config: {
          logAnalyticsWorkspaceResourceID: logAnalytics.id
        }
      }
      azureKeyvaultSecretsProvider: { enabled: true }  // 用 Key Vault 管 Secret
    }

    // System Node Pool (跑 kube-system, coredns 等)
    agentPoolProfiles: [
      {
        name: 'system'
        mode: 'System'
        count: 2
        vmSize: 'Standard_D2s_v3'
        osDiskSizeGB: 128
        osType: 'Linux'
        osSKU: 'Ubuntu'
        type: 'VirtualMachineScaleSets'
        enableAutoScaling: true
        minCount: 2
        maxCount: 4
        availabilityZones: ['1', '2']
      }
    ]

    linuxProfile: {
      adminUsername: 'azureuser'
      ssh: {
        publicKeys: [{ keyData: sshPublicKey }]
      }
    }

    networkProfile: {
      networkPlugin: 'azure'
      networkPolicy: 'azure'     // NetworkPolicy 支持
      loadBalancerSku: 'standard'
    }

    apiServerAccessProfile: {
      // 生产关 public access，改用 private cluster
      // enablePrivateCluster: true
    }
  }
  dependsOn: [acrPullAssignment]
}

// ============================================================
// GPU Node Pool (用户节点池)
// ============================================================
resource gpuNodePool 'Microsoft.ContainerService/managedClusters/agentPools@2024-05-01' = {
  parent: aks
  name: 'gpu'
  properties: {
    mode: 'User'
    count: 1
    vmSize: 'Standard_NC6s_v3'                // 1× V100 16GB
    // Standard_NC4as_T4_v3 = T4 16GB, 便宜
    // Standard_NC24ads_A100_v4 = A100 80GB
    osType: 'Linux'
    osSKU: 'Ubuntu'
    osDiskSizeGB: 256
    type: 'VirtualMachineScaleSets'
    enableAutoScaling: true
    minCount: 1
    maxCount: 4
    nodeTaints: [
      'nvidia.com/gpu=true:NoSchedule'
    ]
    nodeLabels: {
      accelerator: 'nvidia-gpu'
      workload: 'vllm'
    }
    // scaleSetPriority: 'Spot'         // Spot 省钱但可能被抢
    // spotMaxPrice: -1
    availabilityZones: ['1']
    // 自动装 NVIDIA driver + device plugin
    gpuInstanceProfile: 'MIG1g'           // 或 null 用整卡
  }
}

// ============================================================
// Azure Files (RWX 存储 for models/ PVC)
// ============================================================
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: '${prefix}st${uniqueString(resourceGroup().id)}'
  location: location
  sku: { name: 'Premium_LRS' }               // File Share Premium 要 Premium
  kind: 'FileStorage'
  properties: {
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource modelShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: 'models'
  properties: {
    shareQuota: 200      // GB
    accessTier: 'Premium'
  }
}

// ============================================================
// Workload Identity for vllm-runner
// ============================================================
resource vllmRunnerIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-vllm-runner-mi'
  location: location
}

// Federated Credential: 让 K8s SA "ocr-inference/vllm-runner" 换 token
resource fedCredVllmRunner 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: vllmRunnerIdentity
  name: 'vllm-runner-fc'
  properties: {
    issuer: aks.properties.oidcIssuerProfile.issuerURL
    subject: 'system:serviceaccount:ocr-inference:vllm-runner'
    audiences: ['api://AzureADTokenExchange']
  }
}

// ============================================================
// Outputs
// ============================================================
output aksName string = aks.name
output acrLoginServer string = acr.properties.loginServer
output oidcIssuerURL string = aks.properties.oidcIssuerProfile.issuerURL
output storageAccountName string = storage.name
output fileShareName string = modelShare.name
output vllmRunnerClientId string = vllmRunnerIdentity.properties.clientId
output vllmRunnerPrincipalId string = vllmRunnerIdentity.properties.principalId

output nextSteps string = '''
  === 下一步 ===

  1) 拉 kubeconfig:
     az aks get-credentials --resource-group ${resourceGroup().name} --name ${aks.name}

  2) 验证 GPU:
     kubectl get nodes -L accelerator
     kubectl describe node <gpu-node> | grep nvidia.com/gpu

  3) 给 ServiceAccount 打 Workload Identity annotation:
     kubectl annotate sa -n ocr-inference vllm-runner azure.workload.identity/client-id=<上面 output 里的 clientId>

  4) 部署应用 (从项目根, 注意 storageClassName 改 azurefile-csi-premium):
     kubectl apply -k inference/kubernetes/base/
'''
