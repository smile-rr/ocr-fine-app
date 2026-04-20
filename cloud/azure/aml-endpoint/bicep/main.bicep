// Azure ML Workspace + Online Endpoint + Deployment
//
// 部署:
//   az deployment group create \
//       --resource-group rg-ocr-fine-app \
//       --template-file main.bicep \
//       --parameters prefix=ocr location=eastus

param prefix string = 'ocr'
param location string = 'eastus'

// ============================================================
// 1. 前置资源：Storage + KeyVault + App Insights + ACR (AML 必需)
// ============================================================
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: '${prefix}st${uniqueString(resourceGroup().id)}'
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${prefix}-kv-${uniqueString(resourceGroup().id)}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-ai'
  location: location
  kind: 'web'
  properties: { Application_Type: 'web' }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: '${prefix}acr${uniqueString(resourceGroup().id)}'
  location: location
  sku: { name: 'Standard' }
  properties: { adminUserEnabled: false }
}

// ============================================================
// 2. AML Workspace
// ============================================================
resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01' = {
  name: '${prefix}-aml-ws'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    friendlyName: 'OCR Fine-App ML Workspace'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: acr.id
    publicNetworkAccess: 'Enabled'       // 生产关掉 + 用 Private Link
  }
  sku: { name: 'Basic', tier: 'Basic' }
}

// ============================================================
// 3. Online Endpoint
// ============================================================
resource onlineEndpoint 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2024-10-01' = {
  parent: workspace
  name: '${prefix}-stage2'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    authMode: 'AMLToken'     // AMLToken / Key / AADToken
    // AMLToken 是 AML 自家 token；AADToken 让你走 Entra ID（推荐生产）
    description: 'Stage 2 LLM RAG QA endpoint'
    traffic: {
      blue: 100              // 当前流量：blue 100%
      // green: 0           // 加新版本时这里改 "blue: 90, green: 10"
    }
  }
}

// ============================================================
// 4. Deployment (具体部署)
// ============================================================
resource blueDeployment 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints/deployments@2024-10-01' = {
  parent: onlineEndpoint
  name: 'blue'
  location: location
  sku: {
    name: 'Standard_NC6s_v3'    // 1× V100 16GB
    capacity: 2                 // 实例数
  }
  properties: {
    description: 'Stage 2 v1 deployment'

    // 模型：需要先用 az ml model create 注册
    model: resourceId(
      'Microsoft.MachineLearningServices/workspaces/models/versions',
      workspace.name, 'stage2-fused', '1'
    )

    // Environment: 推理容器
    // 用 curated foundation-model-inference，或者换成 acr 里自建 image
    environmentId: resourceId(
      'Microsoft.MachineLearningServices/workspaces/environments/versions',
      workspace.name, 'vllm-serve-env', '1'
    )

    // 请求配置
    requestSettings: {
      maxConcurrentRequestsPerInstance: 4
      requestTimeout: 'PT5M'      // ISO 8601 duration，5 分钟
      maxQueueWait: 'PT30S'
    }

    // Auto-scale: 按 CPU / request 数
    scaleSettings: {
      scaleType: 'Default'        // 或 'TargetUtilization' 按指标
    }

    // Liveness / Readiness
    livenessProbe: {
      failureThreshold: 30
      successThreshold: 1
      timeout: 'PT2S'
      period: 'PT10S'
      initialDelay: 'PT10M'       // VLM 加载慢，给 10 分钟
    }
    readinessProbe: {
      failureThreshold: 30
      successThreshold: 1
      timeout: 'PT2S'
      period: 'PT10S'
      initialDelay: 'PT10M'
    }

    // Data collection (optional): 抓请求/响应进 Blob
    // dataCollector: {
    //   collections: {
    //     requests: { enabled: 'true', samplingRate: 1.0 }
    //     responses: { enabled: 'true', samplingRate: 1.0 }
    //   }
    // }
  }
}

// ============================================================
// Outputs
// ============================================================
output workspaceName string = workspace.name
output endpointName string = onlineEndpoint.name
output scoringUri string = onlineEndpoint.properties.scoringUri
