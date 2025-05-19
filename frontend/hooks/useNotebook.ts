"use client"

import { useState, useEffect, useCallback } from "react"
import { api, type Notebook, type Cell } from "../api/client"
import { useWebSocket, type WebSocketMessage } from "./useWebSocket"

export function useNotebook(notebookId: string) {
  console.log('[useNotebook] Initializing hook with notebookId:', notebookId);

  const [notebook, setNotebook] = useState<Notebook | null>(null)
  const [cells, setCells] = useState<Cell[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [executingCells, setExecutingCells] = useState<Set<string>>(new Set())
  const [latestMessage, setLatestMessage] = useState<WebSocketMessage | null>(null);


  // WebSocket onMessage handler
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    console.log(`[useNotebook ${notebookId}] handleWebSocketMessage received:`, message.type, message.data);
    setLatestMessage(message);
  }, [notebookId]);

  // WebSocket connection for real-time updates
  const { status: wsStatus } = useWebSocket(notebookId, handleWebSocketMessage)
  console.log('[useNotebook] Called useWebSocket. Initial wsStatus:', wsStatus);

  // Load notebook and cells
  useEffect(() => {
    const loadNotebook = async () => {
      if (!notebookId) return

      setLoading(true)
      setError(null)

      try {
        const [notebookData, cellsData] = await Promise.all([api.notebooks.get(notebookId), api.cells.list(notebookId)])

        setNotebook(notebookData)
        setCells(cellsData)
      } catch (err) {
        console.error("Failed to load notebook:", err)
        setError("Failed to load notebook. Please try again.")
      } finally {
        setLoading(false)
      }
    }

    loadNotebook()
  }, [notebookId])

  // Handle WebSocket messages
  useEffect(() => {
    if (!latestMessage) return

    // Process the latest message
    // const message = wsMessages[wsMessages.length - 1] // Old logic

    switch (latestMessage.type) {
      case "cell_update":
        handleCellUpdate(latestMessage.data)
        break
      case "cell_execution_started":
        handleCellExecutionStarted(latestMessage.data.cell_id)
        break
      case "cell_execution_completed":
        handleCellExecutionCompleted(latestMessage.data.cell_id)
        break
      case "notebook_update":
        handleNotebookUpdate(latestMessage.data)
        break
    }
  }, [latestMessage]) // Depend on latestMessage

  // Handle cell update from WebSocket
  const handleCellUpdate = useCallback((cellData: Cell) => {
    console.log(`[useNotebook ${notebookId}] handleCellUpdate for cell ID:`, cellData.id, "New status:", cellData.status, "Result:", cellData.result ? "Exists" : "None");
    setCells((prevCells) => {
      const index = prevCells.findIndex((c) => c.id === cellData.id)
      if (index >= 0) {
        const updatedCells = [...prevCells]
        updatedCells[index] = cellData
        return updatedCells
      } else {
        return [...prevCells, cellData]
      }
    })
  }, [])

  // Handle cell execution started from WebSocket
  const handleCellExecutionStarted = useCallback((cellId: string) => {
    console.log(`[useNotebook ${notebookId}] handleCellExecutionStarted for cell ID:`, cellId);
    setExecutingCells((prev) => {
      const newSet = new Set(prev).add(cellId);
      console.log(`[useNotebook ${notebookId}] executingCells updated (added ${cellId}):`, Array.from(newSet));
      return newSet;
    });
  }, [notebookId])

  // Handle cell execution completed from WebSocket
  const handleCellExecutionCompleted = useCallback((cellId: string) => {
    console.log(`[useNotebook ${notebookId}] handleCellExecutionCompleted for cell ID:`, cellId);
    setExecutingCells((prev) => {
      const updated = new Set(prev)
      updated.delete(cellId)
      console.log(`[useNotebook ${notebookId}] executingCells updated (deleted ${cellId}):`, Array.from(updated));
      return updated
    })
  }, [notebookId])

  // Handle notebook update from WebSocket
  const handleNotebookUpdate = useCallback((notebookData: Notebook) => {
    setNotebook(notebookData)
  }, [])

  // Execute a cell
  const executeCell = useCallback(
    async (cellId: string) => {
      if (!notebookId) return;
      console.log(`[useNotebook ${notebookId}] executeCell called for cell ID:`, cellId);

      try {
        // Optimistically set executing state, though WS should confirm
        setExecutingCells((prev) => {
          const newSet = new Set(prev).add(cellId);
          console.log(`[useNotebook ${notebookId}] executingCells updated (optimistically added ${cellId} for executeCell):`, Array.from(newSet));
          return newSet;
        });
        await api.cells.execute(notebookId, cellId);
        console.log(`[useNotebook ${notebookId}] api.cells.execute successful for cell ID:`, cellId);
        // The execution status will be updated via WebSocket (cell_execution_started, cell_update with status QUEUED)
      } catch (err) {
        console.error(`[useNotebook ${notebookId}] Failed to execute cell ${cellId}:`, err);
        setExecutingCells((prev) => {
          const updated = new Set(prev);
          updated.delete(cellId);
          console.log(`[useNotebook ${notebookId}] executingCells updated (removed ${cellId} due to executeCell error):`, Array.from(updated));
          return updated;
        });
        // Optionally, update cell state to error locally here too, or rely on WS
      }
    },
    [notebookId],
  );

  // Update cell content
  const updateCell = useCallback(
    async (cellId: string, content: string, metadata?: Record<string, any>) => {
      // TEMPORARY DEBUGGING: Log a unique error to confirm this code path is hit
      console.error(`[USE_NOTEBOOK_UPDATE_CELL_DEBUG_CLINE] THIS IS THE REAL updateCell in useNotebook.ts for cell ID: ${cellId}. Notebook ID: ${notebookId}`);
      
      // Original code (commented out for now):
      // if (!notebookId) return;
      // console.log(`[useNotebook ${notebookId}] updateCell called for cell ID:`, cellId, "Content length:", content.length, "Metadata:", metadata);
      // try {
      //   await api.cells.update(notebookId, cellId, { content, metadata });
      //   console.log(`[useNotebook ${notebookId}] api.cells.update successful for cell ID:`, cellId);
      //   // Cell state will be updated via WebSocket (cell_update)
      // } catch (err) {
      //   console.error(`[useNotebook ${notebookId}] Failed to update cell ${cellId}:`, err);
      // }
    },
    [notebookId],
  );

  // Delete a cell
  const deleteCell = useCallback(
    async (cellId: string) => {
      if (!notebookId) return

      try {
        await api.cells.delete(notebookId, cellId)
        setCells((prevCells) => prevCells.filter((cell) => cell.id !== cellId))
      } catch (err) {
        console.error("Failed to delete cell:", err)
      }
    },
    [notebookId],
  )

  // Create a new cell
  const createCell = useCallback(
    async (type: "markdown" | "sql" | "code" | "grafana", initialContent?: string) => {
      if (!notebookId) return

      try {
        let defaultContent = initialContent || ""

        if (!defaultContent) {
          switch (type) {
            case "markdown":
              defaultContent = "# New Markdown Cell\n\nEnter your markdown content here."
              break
            case "sql":
              defaultContent = "SELECT * FROM your_table LIMIT 10;"
              break
            case "code":
              defaultContent = '# Python code\nprint("Hello, world!")'
              break
            case "grafana":
              defaultContent = '{\n  "dashboard": "your-dashboard-uid",\n  "from": "now-6h",\n  "to": "now"\n}'
              break
          }
        }

        const newCell = await api.cells.create(notebookId, {
          type,
          content: defaultContent,
          status: "idle",
        })

        setCells((prevCells) => [...prevCells, newCell])
        return newCell
      } catch (err) {
        console.error("Failed to create cell:", err)
        return null
      }
    },
    [notebookId],
  )

  return {
    notebook,
    cells,
    loading,
    error,
    wsStatus,
    executingCells,
    executeCell,
    updateCell,
    deleteCell,
    createCell,
    isExecuting: (cellId: string) => executingCells.has(cellId),
  }
}
