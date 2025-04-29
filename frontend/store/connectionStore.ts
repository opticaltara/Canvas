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

  // Actions
  loadConnections: () => Promise<void>
  createConnection: (connectionData: Partial<Connection>) => Promise<Connection | null>
  updateConnection: (id: string, connectionData: Partial<Connection>) => Promise<Connection | null>
  deleteConnection: (id: string) => Promise<boolean>
  setDefaultConnection: (id: string) => Promise<boolean>
  testConnection: (connectionData: Partial<Connection>) => Promise<{ valid: boolean; message: string }>

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

        // Load all connections
        loadConnections: async () => {
          set({ loading: true, error: null })

          try {
            // Assume api.connections.list() returns Connection[] as typed
            const connectionsFromApi = await api.connections.list()

            // Ensure each connection has the is_default field (default to false)
            const connections = connectionsFromApi.map(conn => ({
              ...conn,
              is_default: conn.is_default ?? false,
            }))

            // The type system should handle if the response isn't Connection[]
            // Explicitly assert the type to match the state definition
            set({ connections: connections as Connection[], loading: false })
          } catch (err) {
            console.error("Failed to load connections:", err)
            set({
              error: "Failed to load connections. Please try again.",
              loading: false,
              connections: [], // Ensure connections is reset on error
            })
          }
        },

        // Create a new connection
        createConnection: async (connectionData) => {
          try {
            const newConnection = await api.connections.create(connectionData)
            // Only update state if creation was successful (newConnection is not null)
            if (newConnection) {
              set((state) => ({
                // Assert the type of newConnection to satisfy the state type
                connections: [...state.connections, newConnection as Connection],
              }))
            }
            return newConnection // Return the actual result (Connection or null)
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
            // Return type from api.connections.test is { valid: boolean; message: string }
            return await api.connections.test(connectionData)
          } catch (err) {
            console.error("Failed to test connection:", err)
            // Return the correct structure on error
            const message = err instanceof Error ? err.message : "Failed to test connection"
            return { valid: false, message }
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
