import { create } from "zustand"
import { devtools, persist } from "zustand/middleware"
import type { Cell, Notebook, WebSocketStatus } from "./types"
import { api } from "../api/client"

export interface NotebookState {
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

  // Chat Panel Interaction State - ADDED
  sendChatMessageFunction: ((message: string) => Promise<void>) | null
  isSendingChatMessage: boolean // ADDED to prevent race conditions

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

  // Chat Panel Interaction Actions - ADDED
  registerSendChatMessageFunction: (func: ((message: string) => Promise<void>) | null) => void
  sendSuggestedStepToChat: (message: string) => Promise<void>

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
        sendChatMessageFunction: null, // ADDED Initial state
        isSendingChatMessage: false, // ADDED Initial state

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
          // Defensive check: Ensure the cell update is for the active notebook
          if (cellData.notebook_id && cellData.notebook_id !== get().activeNotebookId) {
            console.warn(
              `[CanvasStore] Received cell_update for cell ${cellData.id} (event notebook_id: ${cellData.notebook_id}) ` +
              `that does not match active notebook ${get().activeNotebookId}. Ignoring.`
            );
            return;
          }
          
          // If the update indicates the cell is deleted, filter it out directly.
          // Ensure cellData is properly typed, assuming it can have a 'deleted' property.
          if ((cellData as any).deleted === true) { 
            console.log(`🔧 Cell ${(cellData as any).id} marked as deleted via WebSocket, removing from store.`);
            set((state) => ({
              cells: state.cells.filter((cell) => cell.id !== (cellData as any).id),
            }));
            return; // Stop further processing for this cell
          }

          // Utility function for comparing cells, with special handling for tool_arguments
          const areCellsEffectivelyEqual = (cellA: Cell, cellB: Cell): boolean => {
            if (!cellA || !cellB) return false;

            const keysToIgnore: Array<keyof Cell> = ['created_at', 'updated_at'];

            const keysA = (Object.keys(cellA) as Array<keyof Cell>).filter(k => !keysToIgnore.includes(k));
            const keysB = (Object.keys(cellB) as Array<keyof Cell>).filter(k => !keysToIgnore.includes(k));

            if (keysA.length !== keysB.length) {
                console.log("🔧 [areCellsEffectivelyEqual] Length mismatch after filtering ignored keys. A:", keysA.length, "B:", keysB.length);
                return false;
            }

            for (const key of keysA) {
              if (key === 'tool_arguments') {
                // Deep compare tool_arguments by stringifying them
                if (JSON.stringify(cellA.tool_arguments || {}) !== JSON.stringify(cellB.tool_arguments || {})) {
                  console.log(`🔧 [areCellsEffectivelyEqual] Difference in tool_arguments for cell ${cellA.id}`);
                  return false;
                }
              } else if (key === 'metadata' || key === 'settings' || key === 'result') {
                // Deep compare metadata, settings, and result objects
                if (JSON.stringify(cellA[key] || {}) !== JSON.stringify(cellB[key] || {})) {
                  console.log(`🔧 [areCellsEffectivelyEqual] Difference in ${key} for cell ${cellA.id}`);
                  return false;
                }
              } else if (cellA[key] !== cellB[key]) {
                console.log(`🔧 [areCellsEffectivelyEqual] Difference in primitive key '${key}' for cell ${cellA.id}: A='${cellA[key]}', B='${cellB[key]}'`);
                return false;
              }
            }
            return true;
          };

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
          const index = currentCells.findIndex((c) => c.id === cellData.id) // Find in a non-mutated array
          console.log("🔧 Found cell at index:", index)

          if (index >= 0) {
            const updatedCells = [...currentCells]; // Create a new array for modification if cell found
            // Merge while ignoring undefined values from partial updates so that
            // we never overwrite an existing field with `undefined`.
            const mergedCell = {
              ...updatedCells[index],
              ...Object.fromEntries(
                Object.entries(cellData).filter(([_, v]) => v !== undefined),
              ),
            };

            // Skip state update if nothing actually changed to prevent unnecessary re-renders
            if (areCellsEffectivelyEqual(updatedCells[index], mergedCell)) {
              console.log("🔧 No effective changes detected for cell", cellData.id, "— skipping state update.");
              return;
            }

            console.log("🔧 Updating existing cell in store:", cellData.id)
            // Merge incoming data with existing cell data after change detection
            updatedCells[index] = mergedCell;
            set({ cells: updatedCells }); // Set the modified array
          } else {
            // If cellData.deleted was not true (handled above) and the cell isn't found,
            // then it's a new cell (or an update for a cell not yet in store).
            console.log("🔧 Adding new cell to store (or cell not found for update):", cellData.id, cellData.type)
            // updatedCells.push({ ...cellData }) // This was problematic if updatedCells was based on a filtered list
            set((state) => ({ cells: [...state.cells, { ...cellData }] }));
          }

          // These logs might be misleading if we returned early for deletion.
          // Consider moving them or making them conditional.
          // console.log("🔧 New cells array length:", get().cells.length) 
          // console.log("🔧 Is cells array reference changed:", updatedCells !== currentCells) // 'updatedCells' might not be defined here if returned early

          // set({ cells: updatedCells }) // This was the original problematic line if cell was re-added.
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

        // Chat Panel Interaction Actions - ADDED IMPLEMENTATIONS
        registerSendChatMessageFunction: (func) => {
          set({ sendChatMessageFunction: func })
        },

        sendSuggestedStepToChat: async (message) => {
          const { sendChatMessageFunction, isSendingChatMessage } = get()
          if (sendChatMessageFunction && !isSendingChatMessage) {
            set({ isSendingChatMessage: true });
            try {
                console.log(`[CanvasStore] Sending suggested step to chat: "${message}"`);
                await sendChatMessageFunction(message);
            } catch (error) {
                console.error("[CanvasStore] Error sending suggested step to chat:", error);
                // Optionally update state with an error message for the UI
                set({ error: "Failed to send suggested step to chat." });
            } finally {
                set({ isSendingChatMessage: false });
            }
          } else if (isSendingChatMessage) {
             console.warn("[CanvasStore] Chat message sending already in progress. Ignoring suggested step:", message);
          }
           else {
            console.warn("[CanvasStore] No chat message function registered. Cannot send suggested step:", message);
          }
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
