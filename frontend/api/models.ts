import axios from "axios"
import { BACKEND_URL } from "@/config/api-config"

const API_BASE_URL = `${BACKEND_URL}/api`

export interface Model {
  id: string
  name: string
  provider: string
  description: string
}

export interface ModelConfig {
  model_id: string
}

export const modelsApi = {
  async listModels(): Promise<Model[]> {
    const response = await axios.get(`${API_BASE_URL}/models`)
    return response.data
  },

  async getCurrentModel(): Promise<Model> {
    const response = await axios.get(`${API_BASE_URL}/models/current`)
    return response.data
  },

  async setCurrentModel(modelId: string): Promise<Model> {
    const response = await axios.post(`${API_BASE_URL}/models/current`, {
      model_id: modelId,
    })
    return response.data
  },
}

// Add the named export for compatibility
export const models = modelsApi

// Add default export for compatibility
export default { models }
