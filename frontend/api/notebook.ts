// API client for notebook operations
import axios from "axios"

export interface Notebook {
  id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
}

export interface Cell {
  id: string
  notebook_id: string
  type: "markdown" | "sql" | "code" | "grafana"
  content: string
  result?: any
  status: "idle" | "running" | "success" | "error"
  error?: string
  created_at: string
  updated_at: string
  metadata?: Record<string, any>
}

export interface Dependency {
  source_cell_id: string
  target_cell_id: string
}

const API_BASE_URL = "/api"

export const notebookApi = {
  // Notebook operations
  async listNotebooks(): Promise<Notebook[]> {
    const response = await axios.get(`${API_BASE_URL}/notebooks`)
    return response.data
  },

  async getNotebook(notebookId: string): Promise<Notebook> {
    const response = await axios.get(`${API_BASE_URL}/notebooks/${notebookId}`)
    return response.data
  },

  async createNotebook(name: string, description?: string): Promise<Notebook> {
    const response = await axios.post(`${API_BASE_URL}/notebooks`, { name, description })
    return response.data
  },

  async updateNotebook(notebookId: string, data: Partial<Notebook>): Promise<Notebook> {
    const response = await axios.put(`${API_BASE_URL}/notebooks/${notebookId}`, data)
    return response.data
  },

  async deleteNotebook(notebookId: string): Promise<void> {
    await axios.delete(`${API_BASE_URL}/notebooks/${notebookId}`)
  },

  // Cell operations
  async listCells(notebookId: string): Promise<Cell[]> {
    const response = await axios.get(`${API_BASE_URL}/notebooks/${notebookId}/cells`)
    return response.data
  },

  async getCell(notebookId: string, cellId: string): Promise<Cell> {
    const response = await axios.get(`${API_BASE_URL}/notebooks/${notebookId}/cells/${cellId}`)
    return response.data
  },

  async createCell(notebookId: string, cellData: Partial<Cell>): Promise<Cell> {
    const response = await axios.post(`${API_BASE_URL}/notebooks/${notebookId}/cells`, cellData)
    return response.data
  },

  async updateCell(notebookId: string, cellId: string, cellData: Partial<Cell>): Promise<Cell> {
    const response = await axios.put(`${API_BASE_URL}/notebooks/${notebookId}/cells/${cellId}`, cellData)
    return response.data
  },

  async deleteCell(notebookId: string, cellId: string): Promise<void> {
    await axios.delete(`${API_BASE_URL}/notebooks/${notebookId}/cells/${cellId}`)
  },

  async executeCell(notebookId: string, cellId: string): Promise<Cell> {
    const response = await axios.post(`${API_BASE_URL}/notebooks/${notebookId}/cells/${cellId}/execute`)
    return response.data
  },

  // Dependency operations
  async addDependency(notebookId: string, sourceCellId: string, targetCellId: string): Promise<Dependency> {
    const response = await axios.post(`${API_BASE_URL}/notebooks/${notebookId}/dependencies`, {
      source_cell_id: sourceCellId,
      target_cell_id: targetCellId,
    })
    return response.data
  },

  async removeDependency(notebookId: string, sourceCellId: string, targetCellId: string): Promise<void> {
    await axios.delete(`${API_BASE_URL}/notebooks/${notebookId}/dependencies`, {
      data: {
        source_cell_id: sourceCellId,
        target_cell_id: targetCellId,
      },
    })
  },
}
