"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useConnectionStore } from "../store/connectionStore"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import {
  AlertCircle,
  CheckCircle,
  Database,
  LineChart,
  Plus,
  Trash,
  Loader2,
  ArrowLeft,
  Github,
} from "lucide-react"
import type { ConnectionType } from "../store/types"
import { useToast } from "@/hooks/use-toast"

interface DataConnectionsDialogProps {
  isOpen: boolean
  onClose: () => void
  initialPage?: "list" | "add"
}

const DataConnectionsDialog: React.FC<DataConnectionsDialogProps> = ({ isOpen, onClose, initialPage = "list" }) => {
  const [page, setPage] = useState<"list" | "add">(initialPage)
  const [newConnection, setNewConnection] = useState<{
    name: string
    type: ConnectionType
    config: Record<string, any>
  }>({
    name: "",
    type: "github",
    config: {},
  })
  const [testResult, setTestResult] = useState<{ valid: boolean; message: string } | null>(null)
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [isCreatingConnection, setIsCreatingConnection] = useState(false)
  const [dialogError, setDialogError] = useState<string | null>(null)
  const { toast } = useToast()

  // Get state and actions from the store
  const {
    connections,
    loading,
    error,
    loadConnections,
    createConnection,
    deleteConnection,
    testConnection,
  } = useConnectionStore()

  // Load connections when the component mounts or dialog opens
  useEffect(() => {
    if (isOpen) {
      console.log("DataConnectionsDialog: isOpen is true, calling loadConnections...")
      loadConnections()
    }
  }, [isOpen, loadConnections])

  // Update page state if initialPage prop changes while dialog is open
  useEffect(() => {
    if (isOpen) {
      setPage(initialPage);
    }
  }, [initialPage, isOpen]);

  // Reset state when dialog closes
  useEffect(() => {
    if (!isOpen) {
      setPage("list")
      resetForm()
    }
  }, [isOpen])

  const resetForm = () => {
    setNewConnection({
      name: "",
      type: "github",
      config: {},
    })
    setTestResult(null)
    setDialogError(null)
    setIsTestingConnection(false)
    setIsCreatingConnection(false)
  }

  const handleInputChange = (field: string, value: string) => {
    setNewConnection((prev) => ({ ...prev, [field]: value }))
    setDialogError(null)
  }

  const handleConfigChange = (field: string, value: string) => {
    setNewConnection((prev) => ({
      ...prev,
      config: {
        ...prev.config,
        [field]: value,
      },
    }))
    setDialogError(null)
  }

  const handleTestConnection = async () => {
    setTestResult(null)
    setDialogError(null)
    setIsTestingConnection(true)

    try {
      const result = await testConnection(newConnection)
      setTestResult(result)
    } catch (err) {
      console.error("Failed to test connection:", err)
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred during testing."
      setDialogError(errorMessage)
      setTestResult({
        valid: false,
        message: errorMessage,
      })
    } finally {
      setIsTestingConnection(false)
    }
  }

  const handleCreateConnection = async () => {
    if (!newConnection.name.trim()) {
      setDialogError("Connection name is required")
      return
    }

    setDialogError(null)
    setIsCreatingConnection(true)

    try {
      const result = await createConnection(newConnection)
      if (result) {
        resetForm()
        setPage("list")
        toast({
          title: "Connection created",
          description: `${result.name} has been successfully created.`,
        })
      }
    } catch (err) {
      console.error("Failed to create connection:", err)
      const errorMessage = err instanceof Error ? err.message : "An unknown error occurred during creation."
      setDialogError(errorMessage)
    } finally {
      setIsCreatingConnection(false)
    }
  }

  const handleDeleteConnection = async (id: string, name: string) => {
    try {
      const success = await deleteConnection(id)
      if (success) {
        toast({
          title: "Connection deleted",
          description: `${name} has been successfully deleted.`,
        })
      }
    } catch (err) {
      console.error("Failed to delete connection:", err)
      toast({
        variant: "destructive",
        title: "Delete failed",
        description: `Could not delete ${name}. Please try again.`,
      })
    }
  }

  const getConnectionTypeIcon = (type: string) => {
    switch (type) {
      case "grafana":
        return <LineChart className="h-5 w-5 text-primary" />
      case "github":
        return <Github className="h-5 w-5 text-primary" />
      default:
        return <Database className="h-5 w-5 text-muted-foreground" />
    }
  }

  const renderConnectionsList = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center h-40">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <span className="ml-2 text-primary">Loading connections...</span>
        </div>
      )
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-40">
          <div className="p-4 bg-red-50 text-red-800 rounded-md max-w-md">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 mr-2" />
              <p>{error}</p>
            </div>
            <Button variant="outline" className="mt-4" onClick={() => loadConnections()}>
              Retry
            </Button>
          </div>
        </div>
      )
    }

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Data Connections</h2>
          <div className="flex space-x-2">
            <Button size="sm" onClick={() => setPage("add")}>
              <Plus className="mr-2 h-4 w-4" />
              Add Connection
            </Button>
          </div>
        </div>

        <Card>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {connections.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={3} className="text-center py-4 text-muted-foreground">
                      No connections available. Add a connection to get started.
                    </TableCell>
                  </TableRow>
                ) : (
                  Array.isArray(connections) && connections.map((connection) => (
                    <TableRow key={connection.id} className="transition-all duration-200 hover:bg-muted">
                      <TableCell className="font-medium">{connection.name}</TableCell>
                      <TableCell>
                        <div className="flex items-center">
                          {getConnectionTypeIcon(connection.type)}
                          <span className="ml-2">{connection.type}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex space-x-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDeleteConnection(connection.id, connection.name)}
                            className="transition-all duration-200 hover:bg-red-50"
                          >
                            <Trash className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    )
  }

  const renderAddConnectionForm = () => {
    return (
      <div className="space-y-4">
        <div className="flex items-center">
          {initialPage === "list" && (
            <Button variant="ghost" size="sm" onClick={() => setPage("list")} className="mr-2">
              <ArrowLeft className="h-4 w-4 mr-1" />
              Back
            </Button>
          )}
          <h2 className="text-xl font-semibold">Add New Connection</h2>
        </div>

        <div className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="name">Connection Name</Label>
            <Input
              id="name"
              value={newConnection.name}
              onChange={(e) => handleInputChange("name", e.target.value)}
              placeholder="Enter connection name"
              className="w-full"
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="type">Connection Type</Label>
            <select
              id="type"
              value={newConnection.type}
              onChange={(e) => handleInputChange("type", e.target.value as ConnectionType)}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 bg-white"
            >
              <option value="grafana">Grafana</option>
              <option value="github">GitHub</option>
            </select>
          </div>

          {/* Dynamic config fields based on connection type */}
          {newConnection.type === "grafana" && (
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="url">URL</Label>
                <Input
                  id="url"
                  value={newConnection.config?.url || ""}
                  onChange={(e) => handleConfigChange("url", e.target.value)}
                  placeholder="https://your-grafana-instance.com"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="api_key">API Key</Label>
                <Input
                  id="api_key"
                  type="password"
                  value={newConnection.config?.api_key || ""}
                  onChange={(e) => handleConfigChange("api_key", e.target.value)}
                  placeholder="Enter API key"
                />
              </div>
            </div>
          )}

          {newConnection.type === "github" && (
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="github_personal_access_token">Personal Access Token</Label>
                <Input
                  id="github_personal_access_token"
                  type="password"
                  value={newConnection.config?.github_personal_access_token || ""}
                  onChange={(e) => handleConfigChange("github_personal_access_token", e.target.value)}
                  placeholder="ghp_..."
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Create a token with repo and workflow permissions at{" "}
                  <a
                    href="https://github.com/settings/tokens"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    github.com/settings/tokens
                  </a>
                </p>
              </div>
            </div>
          )}

          {testResult && (
            <div
              className={`p-3 rounded-md ${
                testResult.valid ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"
              }`}
            >
              <div className="flex items-center">
                {testResult.valid ? (
                  <CheckCircle className="h-5 w-5 mr-2" />
                ) : (
                  <AlertCircle className="h-5 w-5 mr-2" />
                )}
                <p>{testResult.message || (testResult.valid ? "Connection successful!" : "Connection failed!")}</p>
              </div>
            </div>
          )}

          {dialogError && (
            <div className="p-3 rounded-md bg-red-50 text-red-800">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 mr-2" />
                <p>{dialogError}</p>
              </div>
            </div>
          )}

          <div className="flex justify-end space-x-2 mt-4">
            <Button
              variant="outline"
              onClick={handleTestConnection}
              disabled={isTestingConnection || isCreatingConnection}
            >
              {isTestingConnection ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Testing...
                </>
              ) : (
                "Test Connection"
              )}
            </Button>
            <Button
              onClick={handleCreateConnection}
              disabled={!newConnection.name || !newConnection.type || isTestingConnection || isCreatingConnection}
            >
              {isCreatingConnection ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create Connection"
              )}
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-white max-w-3xl">
        {page === "list" ? renderConnectionsList() : renderAddConnectionForm()}
      </DialogContent>
    </Dialog>
  )
}

export default DataConnectionsDialog
