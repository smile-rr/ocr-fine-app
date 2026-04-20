// Container Apps Environment + ocr-api App + ACR Pull + Log Analytics
//
// 部署:
//   az deployment group create \
//       --resource-group rg-ocr-fine-app \
//       --template-file main.bicep \
//       --parameters prefix=ocr acrName=ocracr imageTag=v1

param prefix string = 'ocr'
param location string = 'eastus'
param acrName string                       // 已存在的 ACR 名字
param imageTag string = 'v1'
param azureOpenAIEndpoint string = ''      // 如果业务层要调 Azure OpenAI

// ============================================================
// Log Analytics Workspace
// ============================================================
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ============================================================
// Managed Identity - 用户分配的，便于跨资源复用
// ============================================================
resource userAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-app-mi'
  location: location
}

// 授权 MI 访问 ACR (pull 镜像)
resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' existing = {
  name: acrName
}

var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'   // AcrPull
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, userAssignedIdentity.id, acrPullRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: userAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ============================================================
// Container Apps Environment
// ============================================================
resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    zoneRedundant: false   // 生产开 true + 需要 VNet
  }
}

// ============================================================
// Container App (业务层)
// ============================================================
resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-api'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true                      // 公网访问
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
        traffic: [
          { weight: 100, latestRevision: true }
        ]
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST']
        }
      }
      registries: [
        {
          server: '${acrName}.azurecr.io'
          identity: userAssignedIdentity.id
        }
      ]
      secrets: []     // 需要时填，可引用 Key Vault
    }
    template: {
      containers: [
        {
          name: 'ocr-api'
          image: '${acrName}.azurecr.io/ocr-api:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenAIEndpoint }
            { name: 'ENABLE_STAGE1', value: '0' }
            { name: 'MAX_TOKENS', value: '512' }
            // MI client id，SDK 自动用
            { name: 'AZURE_CLIENT_ID', value: userAssignedIdentity.properties.clientId }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 10
            }
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 60
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0                 // ⚠️ scale-to-zero, 省钱但有冷启动
        // minReplicas: 1              // 改这个消除冷启动
        maxReplicas: 10
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

// ============================================================
// Outputs
// ============================================================
output fqdn string = app.properties.configuration.ingress.fqdn
output appUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output miClientId string = userAssignedIdentity.properties.clientId
output miPrincipalId string = userAssignedIdentity.properties.principalId
