"use client"

import type React from "react"
import { useState, useEffect } from "react"
import type { Cell } from "../../store/types"
import { useConnectionStore } from "../../store/connectionStore"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Play, Trash } from "lucide-react"
import { Loader2 } from "lucide-react"
import Editor from "@monaco-editor/react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import SQLResultView from "../results/SQLResultView"

interface SQLCellProps {
  cell: Cell
  onExecute: (cellId: string) => void
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void
  onDelete: (cellId: string) => void
  isExecuting: boolean
}

const SQLCell: React.FC<SQLCellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }) => {
  const [content, setContent] = useState(cell.content)
  const [selectedConnection, setSelectedConnection] = useState<string | null>(cell.metadata?.connection_id || null)

  // Get connections from the store
  const {
    connections,
    loading: isLoadingConnections,
    loadConnections,
    getConnectionsByType,
    getDefaultConnection,
  } = useConnectionStore()

  // Get PostgreSQL connections
  const pgConnections = getConnectionsByType("postgresql")

  // Load connections when the component mounts
  useEffect(() => {
    loadConnections()
  }, [loadConnections])

  // Set default connection if none is selected
  useEffect(() => {
    if (!selectedConnection && pgConnections.length > 0) {
      const defaultConn = getDefaultConnection("postgresql")
      if (defaultConn) {
        setSelectedConnection(defaultConn.id)
      }
    }
  }, [pgConnections, selectedConnection, getDefaultConnection])

  const handleExecute = () => {
    // Update the cell content and metadata before executing
    const updatedMetadata = {
      ...cell.metadata,
      connection_id: selectedConnection,
    }

    // First update the cell content and metadata
    onUpdate(cell.id, content, updatedMetadata)

    // Then execute the cell
    onExecute(cell.id)
  }

  return (
    <Card className="mb-4">
      <CardHeader className="py-2 px-4 bg-gray-50 flex flex-row justify-between items-center">
        <div className="text-sm font-medium">SQL Query</div>
        <div className="flex space-x-2">
          <Select
            value={selectedConnection || ""}
            onValueChange={setSelectedConnection}
            disabled={isLoadingConnections}
          >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select database" />
            </SelectTrigger>
            <SelectContent>
              {pgConnections.map((conn) => (
                <SelectItem key={conn.id} value={conn.id}>
                  {conn.name} {conn.is_default && "(Default)"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleExecute}
            disabled={isExecuting || !selectedConnection}
            title="Run Query"
          >
            {isExecuting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          </Button>

          <Button variant="ghost" size="sm" onClick={() => onDelete(cell.id)} title="Delete">
            <Trash className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="border-b">
          <Editor
            height="200px"
            language="sql"
            value={content}
            onChange={(value) => setContent(value || "")}
            options={{
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              fontSize: 14,
              lineNumbers: "on",
              automaticLayout: true,
            }}
          />
        </div>

        {cell.result && (
          <div className="p-4">
            <SQLResultView result={cell.result} />
          </div>
        )}

        {cell.status === "error" && cell.error && (
          <div className="p-4 bg-red-50 text-red-800 rounded-b-md">
            <p className="font-semibold">Error:</p>
            <pre className="text-sm overflow-auto">{cell.error}</pre>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default SQLCell
