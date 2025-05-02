import { create } from "zustand"
import { devtools, persist } from "zustand/middleware"
import type { Cell, Notebook, WebSocketStatus } from "./types"
import { api } from "../api/client"

interface NotebookState {
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
  createCell: (type: "markdown" | "github" | "summarization", initialContent?: string) => Promise<Cell | null>

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

export const useCanvasStore = create<NotebookState>()(
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

          try {
            const [notebookData, cellsData] = await Promise.all([
              api.notebooks.get(notebookId),
              api.cells.list(notebookId),
            ])

            set({
              notebook: notebookData,
              cells: cellsData,
              activeNotebookId: notebookId,
              loading: false,
            })
          } catch (err) {
            console.error("Failed to load notebook:", err)
            set({
              error: "Failed to load notebook. Please try again.",
              loading: false,
            })
          }
        },

        // Execute a cell
        executeCell: async (cellId) => {
          const { activeNotebookId } = get()
          if (!activeNotebookId) return

          try {
            set((state) => ({
              executingCells: new Set(state.executingCells).add(cellId),
            }))

            await api.cells.execute(activeNotebookId, cellId)
            // The execution status will be updated via WebSocket
          } catch (err) {
            console.error("Failed to execute cell:", err)
            set((state) => {
              const updated = new Set(state.executingCells)
              updated.delete(cellId)
              return { executingCells: updated }
            })
          }
        },

        // Update cell content
        updateCell: async (cellId, content, metadata) => {
          const { activeNotebookId } = get()
          if (!activeNotebookId) return

          try {
            await api.cells.update(activeNotebookId, cellId, { content, metadata })
          } catch (err) {
            console.error("Failed to update cell:", err)
          }
        },

        // Delete a cell
        deleteCell: async (cellId) => {
          const { activeNotebookId } = get()
          if (!activeNotebookId) return

          try {
            await api.cells.delete(activeNotebookId, cellId)
            set((state) => ({
              cells: state.cells.filter((cell) => cell.id !== cellId),
            }))
          } catch (err) {
            console.error("Failed to delete cell:", err)
          }
        },

        // Create a new cell
        createCell: async (type: "markdown" | "github" | "summarization", initialContent?: string) => {
          const { activeNotebookId } = get()
          if (!activeNotebookId) return null

          try {
            let defaultContent = initialContent || ""

            if (!defaultContent) {
              switch (type) {
                case "markdown":
                  defaultContent = "# New Markdown Cell\n\nEnter your markdown content here."
                  break
                case "github":
                  defaultContent = "GitHub Tool Cell - Configure in UI"
                  break
                case "summarization":
                  defaultContent = "Summarization results will appear here."
                  break
              }
            }

            const newCell = await api.cells.create(activeNotebookId, {
              type,
              content: defaultContent,
            })

            set((state) => ({
              cells: [...state.cells, newCell],
            }))

            return newCell
          } catch (err) {
            console.error("Failed to create cell:", err)
            return null
          }
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

          // Process the message based on its type
          switch (message.type) {
            case "cell_update":
              get().handleCellUpdate(message.data)
              break
            case "cell_execution_started":
              get().handleCellExecutionStarted(message.data.cell_id)
              break
            case "cell_execution_completed":
              get().handleCellExecutionCompleted(message.data.cell_id)
              break
            case "notebook_update":
              get().handleNotebookUpdate(message.data)
              break
          }
        },

        // Handle cell update from WebSocket
        handleCellUpdate: (cellData) => {
          console.log("🔧 handleCellUpdate called with cell:", cellData.id, cellData.type)
          console.log("🔧 Cell notebook_id:", cellData.notebook_id)
          console.log("🔧 Current active notebook ID in store:", get().activeNotebookId)

          // Log current cells before update
          const currentCells = get().cells
          console.log(
            "🔧 Current cells in store before update:",
            currentCells.map((c) => ({ id: c.id, type: c.type })),
          )

          // Create a new cells array to ensure reference change
          const updatedCells = [...currentCells]
          const index = updatedCells.findIndex((c) => c.id === cellData.id)
          console.log("🔧 Found cell at index:", index)

          if (index >= 0) {
            console.log("🔧 Updating existing cell in store:", cellData.id)
            updatedCells[index] = { ...cellData }
          } else {
            console.log("🔧 Adding new cell to store:", cellData.id, cellData.type)
            // Add animation flag for new cells
            updatedCells.push({ ...cellData, isNew: true })
          }

          console.log("🔧 New cells array length:", updatedCells.length)
          console.log("🔧 Is cells array reference changed:", updatedCells !== currentCells)

          // Update the state with the new cells array
          set({ cells: updatedCells })

          // Log cells after update
          setTimeout(() => {
            const updatedStoreCells = get().cells
            console.log(
              "🔧 Cells in store after update:",
              updatedStoreCells.map((c) => ({ id: c.id, type: c.type })),
            )
            console.log(
              "🔧 Cell we just added/updated is in store:",
              updatedStoreCells.some((c) => c.id === cellData.id),
            )
          }, 0)

          // Remove the isNew flag after animation completes
          setTimeout(() => {
            console.log("🔧 Removing isNew flag from cell:", cellData.id)
            set((state) => {
              const cells = state.cells.map((cell) => (cell.id === cellData.id ? { ...cell, isNew: false } : cell))
              console.log("🔧 Is cells array reference changed after removing isNew:", cells !== state.cells)
              return { cells }
            })
          }, 500)
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
