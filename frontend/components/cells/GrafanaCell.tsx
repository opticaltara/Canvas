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
import GrafanaResultView from "../results/GrafanaResultView"

interface GrafanaCellProps {
  cell: Cell
  onExecute: (cellId: string) => void
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void
  onDelete: (cellId: string) => void
  isExecuting: boolean
}

const GrafanaCell: React.FC<GrafanaCellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }) => {
  const [content, setContent] = useState(cell.content)
  const [selectedConnection, setSelectedConnection] = useState<string | null>(cell.metadata?.connection_id || null)
  const [queryType, setQueryType] = useState<string>(
    cell.metadata?.query_type || (cell.type === "log" ? "logs" : "dashboard"),
  )

  // Get connections from the store
  const {
    connections,
    loading: isLoadingConnections,
    loadConnections,
    getConnectionsByType,
    getDefaultConnection,
  } = useConnectionStore()

  // Get Grafana connections
  const grafanaConnections = getConnectionsByType("grafana")

  // Load connections when the component mounts
  useEffect(() => {
    loadConnections()
  }, [loadConnections])

  // Set default connection if none is selected
  useEffect(() => {
    if (!selectedConnection && grafanaConnections.length > 0) {
      const defaultConn = getDefaultConnection("grafana")
      if (defaultConn) {
        setSelectedConnection(defaultConn.id)
      }
    }
  }, [grafanaConnections, selectedConnection, getDefaultConnection])

  const handleExecute = () => {
    // Update the cell content and metadata before executing
    const updatedMetadata = {
      ...cell.metadata,
      connection_id: selectedConnection,
      query_type: queryType,
    }

    // First update the cell content and metadata
    onUpdate(cell.id, content, updatedMetadata)

    // Then execute the cell
    onExecute(cell.id)
  }

  return (
    <Card className="mb-4">
      <CardHeader className="py-2 px-4 bg-gray-50 flex flex-row justify-between items-center">
        <div className="text-sm font-medium">{cell.type === "log" ? "Log Query" : "Grafana Query"}</div>
        <div className="flex space-x-2">
          <Select
            value={selectedConnection || ""}
            onValueChange={setSelectedConnection}
            disabled={isLoadingConnections}
          >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select Grafana connection" />
            </SelectTrigger>
            <SelectContent>
              {grafanaConnections.map((conn) => (
                <SelectItem key={conn.id} value={conn.id}>
                  {conn.name} {conn.is_default && "(Default)"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={queryType} onValueChange={setQueryType}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Query Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="dashboard">Dashboard</SelectItem>
              <SelectItem value="metrics">Metrics</SelectItem>
              <SelectItem value="logs">Logs</SelectItem>
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
            language={queryType === "metrics" ? "promql" : queryType === "logs" ? "logql" : "json"}
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
            <GrafanaResultView result={cell.result} queryType={queryType} />
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

export default GrafanaCell
