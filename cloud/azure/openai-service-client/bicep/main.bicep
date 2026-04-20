// Azure OpenAI Service 部署 - Cognitive Services + 两个模型 deployment + Private Endpoint + RBAC
//
// 部署:
//   az deployment group create \
//       --resource-group rg-ocr-fine-app \
//       --template-file main.bicep \
//       --parameters prefix=ocr location=eastus

@description('资源名前缀')
param prefix string = 'ocr'

@description('Region - 注意 GPT-4o 并非所有 region 都有')
param location string = 'eastus'

@description('给这个 Managed Identity object ID 授权调用 (空则跳过 RBAC)')
param clientPrincipalId string = ''

@description('是否启用 Private Endpoint (需要先有 VNet)')
param enablePrivateEndpoint bool = false

@description('Private Endpoint 要挂的 subnet ID (启用时填)')
param privateEndpointSubnetId string = ''

// ============================================================
// 1. Cognitive Services Account (kind=OpenAI)
// ============================================================
resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${prefix}-openai'
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'           // Standard tier, PAYG
  }
  properties: {
    customSubDomainName: '${prefix}-openai'   // 必需：让 endpoint URL 可用
    publicNetworkAccess: enablePrivateEndpoint ? 'Disabled' : 'Enabled'
    networkAcls: {
      defaultAction: enablePrivateEndpoint ? 'Deny' : 'Allow'
    }
    // 禁用 Local Auth = 强制用 Entra ID / Managed Identity
    // 生产推荐 true；demo 时先 false 能用 API Key
    disableLocalAuth: false
  }
  identity: {
    type: 'SystemAssigned'
  }
}

// ============================================================
// 2. Model Deployments
// ============================================================
// Chat model
resource gpt4o 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: 'gpt-4o'
  sku: {
    name: 'Standard'     // 或 'ProvisionedManaged' 走 PTU
    capacity: 10         // TPM * 1000（10 = 10K TPM）
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-08-06'
    }
    versionUpgradeOption: 'OnceCurrentVersionExpired'
    raiPolicyName: 'Microsoft.DefaultV2'      // 默认内容过滤
  }
}

// Embedding model
resource embed3 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: 'text-embedding-3-small'
  sku: {
    name: 'Standard'
    capacity: 30
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-small'
      version: '1'
    }
  }
  dependsOn: [gpt4o]                         // Azure 要求串行部署
}

// ============================================================
// 3. RBAC - 给指定 principal 权限调用
// ============================================================
// Cognitive Services OpenAI User 角色
var cognitiveServicesOpenAIUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' =
    if (!empty(clientPrincipalId)) {
  name: guid(openai.id, clientPrincipalId, cognitiveServicesOpenAIUserRoleId)
  scope: openai
  properties: {
    roleDefinitionId: subscriptionResourceId(
        'Microsoft.Authorization/roleDefinitions',
        cognitiveServicesOpenAIUserRoleId)
    principalId: clientPrincipalId
    principalType: 'ServicePrincipal'   // 或 'User'
  }
}

// ============================================================
// 4. Private Endpoint (可选)
// ============================================================
resource privateEndpoint 'Microsoft.Network/privateEndpoints@2024-01-01' =
    if (enablePrivateEndpoint) {
  name: '${prefix}-openai-pe'
  location: location
  properties: {
    subnet: {
      id: privateEndpointSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'openai-connection'
        properties: {
          privateLinkServiceId: openai.id
          groupIds: ['account']
        }
      }
    ]
  }
}

// ============================================================
// Outputs
// ============================================================
output openaiEndpoint string = openai.properties.endpoint
output openaiResourceId string = openai.id
output chatDeploymentName string = gpt4o.name
output embedDeploymentName string = embed3.name
