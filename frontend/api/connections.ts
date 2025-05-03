// API client for data connections
import axios from "axios"
import { BACKEND_URL } from "@/config/api-config"
// Import the canonical Connection type and form data types
import type { Connection, JiraConnectionCreateFormData, GithubConnectionCreateFormData } from "../store/types"
// Import MCPToolInfo type
import type { MCPToolInfo } from "../store/types";

export interface ConnectionType {
  id: string
  name: string
}

const API_BASE_URL = `${BACKEND_URL}/api`

// Helper function to handle the final response processing
const processConnectionResponse = (response: any): Connection => {
  const connectionResult = response.data as Connection;
  if (connectionResult.is_default === undefined) {
      connectionResult.is_default = false;
  }
  return connectionResult;
}

// Helper function for error handling
const handleApiError = (error: any, type: string | undefined, name: string | undefined) => {
   console.error(`Failed to create ${type || 'unknown type'} connection '${name || 'unnamed'}':`, error)
   if (axios.isAxiosError(error) && error.response) {
     throw new Error(error.response.data.detail || `Failed to create connection: ${error.message}`)
   } else if (error instanceof Error) {
       throw error; // Re-throw known errors (like validation errors)
   }
   throw new Error(`An unexpected error occurred while creating the connection.`)
}

// Type definition for the new Create request payload
interface CreateConnectionPayload {
  name: string;
  payload: { 
    type: string; 
    [key: string]: any; // for type-specific fields
  };
}

// Type definition for the new Update request payload
interface UpdateConnectionPayload {
  name?: string;
  payload?: { 
    type: string; 
    [key: string]: any; // for type-specific fields
  } | null; // Allow null or undefined payload
}

// Type definition for the new Test request payload
interface TestConnectionPayload {
  payload: { 
    type: string; 
    [key: string]: any; // for type-specific fields
  };
}

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

  // --- Refactored createConnection method --- 
  async createConnection(connectionData: { name: string; type: string; [key: string]: any }): Promise<Connection> {
    const { name, type, ...configFields } = connectionData;

    if (!name || !type) {
      throw new Error("Connection name and type are required.");
    }

    // Construct the payload required by the backend
    const requestPayload: CreateConnectionPayload = {
      name: name,
      payload: {
        type: type,
        ...configFields // Spread the rest of the fields into the payload
      }
    };

    try {
        // Send to the unified POST /connections endpoint
        const response = await axios.post(`${API_BASE_URL}/connections`, requestPayload);
        // Use helper for response processing
        return processConnectionResponse(response);
    } catch (error) {
        // Use helper for error handling
        handleApiError(error, type, name);
        // Need to return/throw something here to satisfy TypeScript
        throw new Error("Connection creation failed."); 
    }
  },
  // --- End Refactored createConnection method ---

  // --- Refactored updateConnection method --- 
  async updateConnection(connectionId: string, connectionData: { name?: string; type: string; [key: string]: any }): Promise<Connection> {
    const { name, type, ...configFields } = connectionData;

    // Construct the payload required by the backend
    let requestPayload: UpdateConnectionPayload = {};
    
    if (name) {
        requestPayload.name = name;
    }
    
    // Check if there are any config fields to update
    // We need to filter out 'type' and potentially 'name' if it was passed in configFields accidentally
    const configUpdateFields = Object.keys(configFields)
        .filter(key => key !== 'type' && key !== 'name')
        .reduce((obj, key) => { 
            obj[key] = configFields[key]; 
            return obj; 
        }, {} as {[key: string]: any});

    if (Object.keys(configUpdateFields).length > 0) {
        requestPayload.payload = {
            type: type, // Type is required in the payload for discrimination
            ...configUpdateFields
        };
    }

    if (!requestPayload.name && !requestPayload.payload) {
         throw new Error("Update requires at least a name or config field changes.");
    }

    try {
        // Send to the unified PUT /connections/{id} endpoint
        const response = await axios.put(`${API_BASE_URL}/connections/${connectionId}`, requestPayload);
        return processConnectionResponse(response);
    } catch (error) {
         handleApiError(error, type, name || 'existing connection');
         throw new Error("Connection update failed.");
    }
  },
   // --- End Refactored updateConnection method --- 

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
      return response.data as MCPToolInfo[];
    } catch (error) {
      console.error(`Failed to fetch tools for connection type ${connectionType}:`, error);
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        return [];
      }
      throw error;
    }
  },

  // --- Refactored testConnection method --- 
  async testConnection(connectionData: { type: string; [key: string]: any }): Promise<{ valid: boolean; message: string }> {
     const { type, ...configFields } = connectionData;
     
     if (!type) {
         throw new Error("Connection type is required for testing.");
     }

     // Construct the payload required by the backend
     const requestPayload: TestConnectionPayload = {
         payload: {
             type: type,
             ...configFields // Spread the rest of the fields into the payload
         }
     };

     try {
         // Send to the unified POST /connections/test endpoint
         const response = await axios.post(`${API_BASE_URL}/connections/test`, requestPayload);
         return response.data;
     } catch (error) {
         // Handle potential errors (e.g., validation errors from backend)
         console.error(`Failed to test ${type} connection:`, error);
         if (axios.isAxiosError(error) && error.response) {
            // Return the backend validation error message
            return { valid: false, message: error.response.data.detail || `Testing failed: ${error.message}` };
         } else if (error instanceof Error) {
             return { valid: false, message: error.message };
         }
         return { valid: false, message: "An unexpected error occurred during testing." };
     }
  }
   // --- End Refactored testConnection method --- 

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
