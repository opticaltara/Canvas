// API client for data connections
import axios from "axios"
import { BACKEND_URL } from "@/config/api-config"

export interface Connection {
  id: string
  name: string
  type: string
  config: Record<string, any>
  is_default?: boolean
  created_at?: string
  updated_at?: string
  mcp_status?: "running" | "stopped" | "error"
}

export interface ConnectionType {
  id: string
  name: string
}

const API_BASE_URL = `${BACKEND_URL}/api`

// Export as connectionApi (singular) to match the expected export
export const connectionApi = {
  async listConnections(): Promise<Connection[]> {
    const response = await axios.get(`${API_BASE_URL}/connections`)
    return response.data
  },

  async getConnectionTypes(): Promise<string[]> {
    const response = await axios.get(`${API_BASE_URL}/connections/types`)
    return response.data
  },

  async getConnection(connectionId: string): Promise<Connection> {
    const response = await axios.get(`${API_BASE_URL}/connections/${connectionId}`)
    return response.data
  },

  async createConnection(connectionData: Partial<Connection>): Promise<Connection> {
    // Use type-specific endpoints based on connection type
    if (connectionData.type === "github") {
      const response = await axios.post(`${API_BASE_URL}/connections/github`, {
        name: connectionData.name,
        github_personal_access_token: connectionData.config?.github_personal_access_token || "",
      })
      return response.data
    } else if (connectionData.type === "grafana") {
      const response = await axios.post(`${API_BASE_URL}/connections/grafana`, {
        name: connectionData.name,
        url: connectionData.config?.url || "",
        api_key: connectionData.config?.api_key || "",
      })
      return response.data
    } else if (connectionData.type === "python") {
      const response = await axios.post(`${API_BASE_URL}/connections/python`, {
        name: connectionData.name,
        ...connectionData.config,
      })
      return response.data
    } else {
      // Fallback to generic endpoint
      const response = await axios.post(`${API_BASE_URL}/connections`, {
        name: connectionData.name,
        type: connectionData.type,
        config: connectionData.config,
      })
      return response.data
    }
  },

  async updateConnection(connectionId: string, connectionData: Partial<Connection>): Promise<Connection> {
    // For GitHub connections, we need to handle PAT updates
    if (connectionData.type === "github" && connectionData.config?.github_personal_access_token) {
      const response = await axios.put(`${API_BASE_URL}/connections/${connectionId}`, {
        name: connectionData.name,
        github_personal_access_token: connectionData.config.github_personal_access_token,
      })
      return response.data
    } else {
      // For other connection types or just name updates
      const response = await axios.put(`${API_BASE_URL}/connections/${connectionId}`, {
        name: connectionData.name,
        config: connectionData.config,
      })
      return response.data
    }
  },

  async deleteConnection(connectionId: string): Promise<boolean> {
    await axios.delete(`${API_BASE_URL}/connections/${connectionId}`)
    return true
  },

  async testConnection(connectionData: Partial<Connection>): Promise<{ valid: boolean; message: string }> {
    // For GitHub connections
    if (connectionData.type === "github") {
      const response = await axios.post(`${API_BASE_URL}/connections/test`, {
        type: "github",
        config: {},
        github_personal_access_token: connectionData.config?.github_personal_access_token || "",
      })
      return response.data
    } else if (connectionData.type === "grafana") {
      const response = await axios.post(`${API_BASE_URL}/connections/test`, {
        type: "grafana",
        config: {
          url: connectionData.config?.url || "",
          api_key: connectionData.config?.api_key || "",
        },
      })
      return response.data
    } else if (connectionData.type === "python") {
      const response = await axios.post(`${API_BASE_URL}/connections/test`, {
        type: "python",
        config: connectionData.config || {},
      })
      return response.data
    } else {
      // Generic test endpoint
      const response = await axios.post(`${API_BASE_URL}/connections/test`, {
        type: connectionData.type,
        config: connectionData.config || {},
      })
      return response.data
    }
  },

  async getMcpStatus(connectionId: string): Promise<{ status: "running" | "stopped" | "error"; message?: string }> {
    const response = await axios.get(`${API_BASE_URL}/connections/${connectionId}/mcp/status`)
    return response.data
  },

  async startMcp(connectionId: string): Promise<{ success: boolean; message?: string }> {
    const response = await axios.post(`${API_BASE_URL}/connections/${connectionId}/mcp/start`)
    return response.data
  },

  async stopMcp(connectionId: string): Promise<{ success: boolean; message?: string }> {
    const response = await axios.post(`${API_BASE_URL}/connections/${connectionId}/mcp/stop`)
    return response.data
  },

  async getAllMcpStatus(): Promise<Record<string, { status: "running" | "stopped" | "error"; message?: string }>> {
    const response = await axios.get(`${API_BASE_URL}/connections/mcp/status`)
    return response.data
  },
}

// Export the connections API in the format expected by the client code
export const connections = {
  list: connectionApi.listConnections,
  getTypes: connectionApi.getConnectionTypes,
  get: connectionApi.getConnection,
  create: connectionApi.createConnection,
  update: connectionApi.updateConnection,
  delete: connectionApi.deleteConnection,
  test: connectionApi.testConnection,
  getMcpStatus: connectionApi.getMcpStatus,
  startMcp: connectionApi.startMcp,
  stopMcp: connectionApi.stopMcp,
  getAllMcpStatus: connectionApi.getAllMcpStatus,
}

// Also export the API client for direct use
export default { connections }
