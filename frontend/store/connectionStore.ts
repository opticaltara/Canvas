import { create } from "zustand"
import { devtools, persist } from "zustand/middleware"
import type { Connection } from "./types"
import { api } from "../api/client"

interface ConnectionState {
  // Connection data
  connections: Connection[]

  // Loading states
  loading: boolean
  error: string | null

  // MCP statuses
  mcpStatuses: Record<string, { status: string; message?: string }>

  // Actions
  loadConnections: () => Promise<void>
  createConnection: (connectionData: Partial<Connection>) => Promise<Connection | null>
  updateConnection: (id: string, connectionData: Partial<Connection>) => Promise<Connection | null>
  deleteConnection: (id: string) => Promise<boolean>
  setDefaultConnection: (id: string) => Promise<boolean>
  testConnection: (connectionData: Partial<Connection>) => Promise<{ success: boolean; message?: string }>

  // MCP actions
  loadMcpStatuses: () => Promise<void>
  startMcpServer: (id: string) => Promise<boolean>
  stopMcpServer: (id: string) => Promise<boolean>

  // Utility
  getConnectionsByType: (type: string) => Connection[]
  getDefaultConnection: (type: string) => Connection | null
}

export const useConnectionStore = create<ConnectionState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        connections: [],
        loading: false,
        error: null,
        mcpStatuses: {},

        // Load all connections
        loadConnections: async () => {
          set({ loading: true, error: null })

          try {
            const connections = await api.connections.list()
            set({ connections, loading: false })
          } catch (err) {
            console.error("Failed to load connections:", err)
            set({
              error: "Failed to load connections. Please try again.",
              loading: false,
            })
          }
        },

        // Create a new connection
        createConnection: async (connectionData) => {
          try {
            const newConnection = await api.connections.create(connectionData)
            set((state) => ({
              connections: [...state.connections, newConnection],
            }))
            return newConnection
          } catch (err) {
            console.error("Failed to create connection:", err)
            return null
          }
        },

        // Update an existing connection
        updateConnection: async (id, connectionData) => {
          try {
            const updatedConnection = await api.connections.update(id, connectionData)
            set((state) => ({
              connections: state.connections.map((conn) => (conn.id === id ? updatedConnection : conn)),
            }))
            return updatedConnection
          } catch (err) {
            console.error("Failed to update connection:", err)
            return null
          }
        },

        // Delete a connection
        deleteConnection: async (id) => {
          try {
            await api.connections.delete(id)
            set((state) => ({
              connections: state.connections.filter((conn) => conn.id !== id),
            }))
            return true
          } catch (err) {
            console.error("Failed to delete connection:", err)
            return false
          }
        },

        // Set a connection as default
        setDefaultConnection: async (id) => {
          try {
            await api.connections.setDefault(id)
            set((state) => ({
              connections: state.connections.map((conn) => ({
                ...conn,
                is_default: conn.id === id,
              })),
            }))
            return true
          } catch (err) {
            console.error("Failed to set default connection:", err)
            return false
          }
        },

        // Test a connection
        testConnection: async (connectionData) => {
          try {
            return await api.connections.test(connectionData)
          } catch (err) {
            console.error("Failed to test connection:", err)
            return { success: false, message: "Failed to test connection" }
          }
        },

        // Load MCP statuses for all connections
        loadMcpStatuses: async () => {
          try {
            const statuses = await Promise.all(
              get().connections.map(async (conn) => {
                try {
                  const status = await api.connections.getMcpStatus(conn.id)
                  return { id: conn.id, status }
                } catch (err) {
                  return {
                    id: conn.id,
                    status: { status: "error", message: "Failed to get status" },
                  }
                }
              }),
            )

            const statusMap: Record<string, { status: string; message?: string }> = {}
            statuses.forEach((item) => {
              statusMap[item.id] = item.status
            })

            set({ mcpStatuses: statusMap })
          } catch (err) {
            console.error("Failed to load MCP statuses:", err)
          }
        },

        // Start MCP server for a connection
        startMcpServer: async (id) => {
          try {
            await api.connections.startMcp(id)
            const status = await api.connections.getMcpStatus(id)
            set((state) => ({
              mcpStatuses: {
                ...state.mcpStatuses,
                [id]: status,
              },
            }))
            return true
          } catch (err) {
            console.error("Failed to connect:", err)
            return false
          }
        },

        // Stop MCP server for a connection
        stopMcpServer: async (id) => {
          try {
            await api.connections.stopMcp(id)
            const status = await api.connections.getMcpStatus(id)
            set((state) => ({
              mcpStatuses: {
                ...state.mcpStatuses,
                [id]: status,
              },
            }))
            return true
          } catch (err) {
            console.error("Failed to disconnect:", err)
            return false
          }
        },

        // Get connections by type
        getConnectionsByType: (type) => {
          return get().connections.filter((conn) => conn.type === type)
        },

        // Get default connection by type
        getDefaultConnection: (type) => {
          const connections = get().getConnectionsByType(type)
          return connections.find((conn) => conn.is_default) || connections[0] || null
        },
      }),
      {
        name: "connection-store",
        partialize: (state) => ({}), // Don't persist any state
      },
    ),
  ),
)
