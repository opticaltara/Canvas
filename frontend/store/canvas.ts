import { create } from "zustand"
import { devtools, persist } from "zustand/middleware"
import type { Cell, Notebook, WebSocketStatus } from "./types"

interface CanvasState {
  // Current notebook data
  notebook: Notebook | null
  cells: Cell[]
  activeNotebookId: string | null

  // Loading states
  loading: boolean
  error: string | null
  executingCells: Set<string>

  // WebSocket state
  wsStatus: WebSocketStatus
  wsMessages: any[]

  // Actions
  setActiveNotebook: (notebookId: string) => void
  loadNotebook: (notebookId: string) => Promise<void>
  executeCell: (cellId: string) => Promise<void>
  updateCell: (cellId: string, content: string, metadata?: Record<string, any>) => Promise<void>
  deleteCell: (cellId: string) => Promise<void>
  createCell: (type: "markdown" | "sql" | "code" | "grafana", initialContent?: string) => Promise<Cell | null>

  // WebSocket actions
  updateWsStatus: (status: WebSocketStatus) => void
  addWsMessage: (message: any) => void
  handleCellUpdate: (cellData: Cell) => void
  handleCellExecutionStarted: (cellId: string) => void
  handleCellExecutionCompleted: (cellId: string) => void
  handleNotebookUpdate: (notebookData: Notebook) => void

  // Utility
  isExecuting: (cellId: string) => boolean
}

export const useCanvasStore = create<CanvasState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        notebook: null,
        cells: [],
        activeNotebookId: null,
        loading: false,
        error: null,
        executingCells: new Set<string>(),
        wsStatus: "disconnected",
        wsMessages: [],

        // Set active notebook
        setActiveNotebook: (notebookId) => {
          set({ activeNotebookId: notebookId })
        },

        // Load notebook data
        loadNotebook: async (notebookId) => {
          if (!notebookId) return

          set({ loading: true, error: null })
        },

        // Execute a cell
        executeCell: async (cellId) => {
          const { activeNotebookId } = get()
          if (!activeNotebookId) return
        },

        // Update cell content
        updateCell: async (cellId, content, metadata) => {
          const { activeNotebookId } = get()
          if (!activeNotebookId) return
        },

        // Delete a cell
        deleteCell: async (cellId) => {
          const { activeNotebookId } = get()
          if (!activeNotebookId) return
        },

        // Create a new cell
        createCell: async (type, initialContent) => {
          const { activeNotebookId } = get()
          if (!activeNotebookId) return null
          return null
        },

        // WebSocket status update
        updateWsStatus: (status) => {
          set({ wsStatus: status })
        },

        // Add WebSocket message
        addWsMessage: (message) => {
          set((state) => ({
            wsMessages: [...state.wsMessages, message],
          }))
        },

        // Handle cell update from WebSocket
        handleCellUpdate: (cellData) => {
          set((state) => {
            const index = state.cells.findIndex((c) => c.id === cellData.id)

            if (index >= 0) {
              const updatedCells = [...state.cells]
              updatedCells[index] = cellData
              return { cells: updatedCells }
            } else {
              return { cells: [...state.cells, cellData] }
            }
          })
        },

        // Handle cell execution started from WebSocket
        handleCellExecutionStarted: (cellId) => {
          set((state) => ({
            executingCells: new Set(state.executingCells).add(cellId),
          }))
        },

        // Handle cell execution completed from WebSocket
        handleCellExecutionCompleted: (cellId) => {
          set((state) => {
            const updated = new Set(state.executingCells)
            updated.delete(cellId)
            return { executingCells: updated }
          })
        },

        // Handle notebook update from WebSocket
        handleNotebookUpdate: (notebookData) => {
          set({ notebook: notebookData })
        },

        // Check if a cell is currently executing
        isExecuting: (cellId) => {
          return get().executingCells.has(cellId)
        },
      }),
      {
        name: "canvas-store",
        partialize: (state) => ({
          activeNotebookId: state.activeNotebookId,
        }),
      },
    ),
  ),
)
