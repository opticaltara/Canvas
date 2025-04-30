// API client for data connections
import axios from "axios"
import { BACKEND_URL } from "@/config/api-config"
// Import the canonical Connection type
import type { Connection } from "../store/types"
// Import MCPToolInfo type
import type { MCPToolInfo } from "../store/types";

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
    const { type, name, config } = connectionData

    if (!name || !type) {
      throw new Error("Connection name and type are required.")
    }

    let response
    try {
      if (type === "github") {
        if (!config?.github_personal_access_token) {
          throw new Error("GitHub Personal Access Token is required in config.")
        }
        response = await axios.post(`${API_BASE_URL}/connections/github`, {
          name: name,
          github_personal_access_token: config.github_personal_access_token,
        })
      } else if (type === "grafana") {
        if (!config?.url || !config?.api_key) {
          throw new Error("Grafana URL and API Key are required in config.")
        }
        response = await axios.post(`${API_BASE_URL}/connections/grafana`, {
          name: name,
          url: config.url,
          api_key: config.api_key,
        })
      } else if (type === "python") {
        // Assuming python connection takes config directly
        response = await axios.post(`${API_BASE_URL}/connections/python`, {
          name: name,
          ...(config || {}), // Spread the config object
        })
      } else {
        // Fallback to generic endpoint - This might need adjustment depending on backend
        // Consider logging a warning if this generic endpoint is used
        console.warn(`Using generic connection creation endpoint for type: ${type}`)
        response = await axios.post(`${API_BASE_URL}/connections`, {
          name: name,
          type: type,
          config: config || {},
        })
      }
      // Ensure the response data conforms to the Connection type as much as possible
      // Add default 'is_default' if missing
      const connectionResult = response.data as Connection;
      if (connectionResult.is_default === undefined) {
          connectionResult.is_default = false;
      }
      return connectionResult;

    } catch (error) {
      console.error(`Failed to create ${type} connection '${name}':`, error)
      // Re-throw or handle error appropriately
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.detail || `Failed to create connection: ${error.message}`)
      } else if (error instanceof Error) {
          throw error; // Re-throw known errors (like validation errors)
      }
      throw new Error(`An unexpected error occurred while creating the connection.`)
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

  async setDefaultConnection(connectionId: string): Promise<boolean> {
    // Assuming a POST request to a /default endpoint
    await axios.post(`${API_BASE_URL}/connections/${connectionId}/default`)
    return true // Assume success if no error
  },

  async getToolsForConnection(connectionType: string): Promise<MCPToolInfo[]> {
    try {
      const response = await axios.get(`${API_BASE_URL}/connections/${connectionType}/tools`);
      // TODO: Add validation logic here if necessary (e.g., using Zod)
      return response.data as MCPToolInfo[];
    } catch (error) {
      console.error(`Failed to fetch tools for connection type ${connectionType}:`, error);
      // Re-throw or return empty array based on desired error handling
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        // Treat 404 (Not Found) as no tools defined, not necessarily an error
        return [];
      }
      // For other errors, re-throw to be caught by the store action
      throw error;
    }
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
}

// Export the connections API in the format expected by the client code
export const connections = {
  list: connectionApi.listConnections,
  getTypes: connectionApi.getConnectionTypes,
  get: connectionApi.getConnection,
  create: connectionApi.createConnection,
  update: connectionApi.updateConnection,
  delete: connectionApi.deleteConnection,
  setDefault: connectionApi.setDefaultConnection,
  test: connectionApi.testConnection,
  getTools: connectionApi.getToolsForConnection,
}

// Also export the API client for direct use
export default { connections }
