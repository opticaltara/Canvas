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
    setLatestMessage(message);
  }, []);

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
    setExecutingCells((prev) => new Set(prev).add(cellId))
  }, [])

  // Handle cell execution completed from WebSocket
  const handleCellExecutionCompleted = useCallback((cellId: string) => {
    setExecutingCells((prev) => {
      const updated = new Set(prev)
      updated.delete(cellId)
      return updated
    })
  }, [])

  // Handle notebook update from WebSocket
  const handleNotebookUpdate = useCallback((notebookData: Notebook) => {
    setNotebook(notebookData)
  }, [])

  // Execute a cell
  const executeCell = useCallback(
    async (cellId: string) => {
      if (!notebookId) return

      try {
        setExecutingCells((prev) => new Set(prev).add(cellId))
        await api.cells.execute(notebookId, cellId)
        // The execution status will be updated via WebSocket
      } catch (err) {
        console.error("Failed to execute cell:", err)
        setExecutingCells((prev) => {
          const updated = new Set(prev)
          updated.delete(cellId)
          return updated
        })
      }
    },
    [notebookId],
  )

  // Update cell content
  const updateCell = useCallback(
    async (cellId: string, content: string, metadata?: Record<string, any>) => {
      if (!notebookId) return

      try {
        await api.cells.update(notebookId, cellId, { content, metadata })
      } catch (err) {
        console.error("Failed to update cell:", err)
      }
    },
    [notebookId],
  )

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
