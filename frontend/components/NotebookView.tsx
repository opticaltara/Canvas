"use client"

import type React from "react"
import { useEffect } from "react"
import { useParams } from "react-router-dom"
import { useNotebookStore } from "../store/notebookStore"
import { useWebSocket } from "../hooks/useWebSocket"
import CellFactory from "./cells/CellFactory"
import { Button } from "@/components/ui/button"
import { Plus } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

const NotebookView: React.FC = () => {
  const { notebookId } = useParams<{ notebookId: string }>()

  // Initialize WebSocket connection
  useWebSocket(notebookId || "")

  // Get state and actions from the store
  const {
    notebook,
    cells,
    loading,
    error,
    wsStatus,
    loadNotebook,
    executeCell,
    updateCell,
    deleteCell,
    createCell,
    isExecuting,
  } = useNotebookStore()

  // Load notebook data when the component mounts or notebookId changes
  useEffect(() => {
    if (notebookId) {
      loadNotebook(notebookId)
    }
  }, [notebookId, loadNotebook])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-2">Loading notebook...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="bg-red-50 text-red-800 p-4 rounded-md">
          <p className="font-semibold">Error</p>
          <p>{error}</p>
          <Button variant="outline" className="mt-4" onClick={() => loadNotebook(notebookId || "")}>
            Retry
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-6 px-4">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">{notebook?.name || "Untitled Notebook"}</h1>
        {notebook?.description && <p className="text-gray-600 mt-1">{notebook.description}</p>}
        <div className="flex items-center mt-4">
          <span
            className={`inline-block w-3 h-3 rounded-full mr-2 ${
              wsStatus === "connected" ? "bg-green-500" : wsStatus === "connecting" ? "bg-yellow-500" : "bg-red-500"
            }`}
          ></span>
          <span className="text-sm text-gray-500">
            {wsStatus === "connected" ? "Connected" : wsStatus === "connecting" ? "Connecting..." : "Disconnected"}
          </span>
        </div>
      </div>

      <div className="space-y-4">
        {cells.map((cell) => (
          <CellFactory
            key={cell.id}
            cell={cell}
            onExecute={executeCell}
            onUpdate={updateCell}
            onDelete={deleteCell}
            isExecuting={isExecuting(cell.id)}
          />
        ))}
      </div>

      <div className="mt-6 flex justify-center">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline">
              <Plus className="mr-2 h-4 w-4" />
              Add Cell
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => createCell("markdown")}>Markdown</DropdownMenuItem>
            <DropdownMenuItem onClick={() => createCell("sql")}>SQL</DropdownMenuItem>
            <DropdownMenuItem onClick={() => createCell("code")}>Code</DropdownMenuItem>
            <DropdownMenuItem onClick={() => createCell("grafana")}>Grafana</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}

export default NotebookView
