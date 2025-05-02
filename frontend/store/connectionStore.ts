import { create } from "zustand"
import { devtools, persist } from "zustand/middleware"
import type { Connection } from "./types"
import { api } from "../api/client"
import type { MCPToolInfo } from "./types"

interface ConnectionState {
  // Connection data
  connections: Connection[]

  // Tool definitions per connection type
  toolDefinitions: Record<string, MCPToolInfo[] | undefined>
  toolLoadingStatus: Record<string, 'idle' | 'loading' | 'success' | 'error'>

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
  fetchToolsForConnection: (connectionType: string) => Promise<void>

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
        toolDefinitions: {},
        toolLoadingStatus: {},

        // Load all connections
        loadConnections: async () => {
          console.log("useConnectionStore: loadConnections called")
          set({ loading: true, error: null })

          try {
            // Assume api.connections.list() returns Connection[] as typed
            const connectionsFromApi = await api.connections.list()

            // Ensure connectionsFromApi is actually an array before proceeding
            if (!Array.isArray(connectionsFromApi)) {
              console.error("API response for connections is not an array:", connectionsFromApi)
              throw new Error("Invalid data received from server.") // Trigger the catch block
            }

            // Ensure each connection has the is_default field (default to false)
            const connections = connectionsFromApi.map(conn => ({
              ...conn,
              is_default: conn.is_default ?? false,
            }))

            // The type system should handle if the response isn't Connection[]
            // Explicitly assert the type to match the state definition
            set({ connections: connections as Connection[], loading: false })

            // After loading connections, fetch tools for each unique type
            const connectionTypes = [...new Set(connections.map(conn => conn.type))];
            connectionTypes.forEach(type => {
              // Don't re-fetch if already loading or loaded
              if (!get().toolLoadingStatus[type] || get().toolLoadingStatus[type] === 'idle' || get().toolLoadingStatus[type] === 'error') {
                 get().fetchToolsForConnection(type);
              }
            });

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

        // Fetch tools for a specific connection type
        fetchToolsForConnection: async (connectionType) => {
          console.log(`useConnectionStore: fetchToolsForConnection called for ${connectionType}`);
          const currentStatus = get().toolLoadingStatus[connectionType];
          // Avoid redundant fetches
          if (currentStatus === 'loading' || currentStatus === 'success') {
            console.log(`Skipping fetch for ${connectionType}, status: ${currentStatus}`);
            return;
          }

          set(state => ({
            toolLoadingStatus: { ...state.toolLoadingStatus, [connectionType]: 'loading' },
          }));

          try {
            console.log(`useConnectionStore: Attempting to fetch tools for ${connectionType}...`);
            const tools = await api.connections.getTools(connectionType);
            console.log(`useConnectionStore: Received tools for ${connectionType}:`, tools);
            set(state => {
              console.log(`useConnectionStore: Setting definitions and status=success for ${connectionType}`);
              return {
                toolDefinitions: { ...state.toolDefinitions, [connectionType]: tools },
                toolLoadingStatus: { ...state.toolLoadingStatus, [connectionType]: 'success' },
              }
            });
          } catch (err) {
            console.error(`Failed to fetch tools for ${connectionType}:`, err);
            set(state => {
              console.log(`useConnectionStore: Setting status=error for ${connectionType}`);
              return {
                 toolLoadingStatus: { ...state.toolLoadingStatus, [connectionType]: 'error' },
                // Optionally clear definitions on error, or keep stale data
                // toolDefinitions: { ...state.toolDefinitions, [connectionType]: undefined }, 
              }
            });
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
