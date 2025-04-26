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
  RefreshCw,
  Trash,
  XCircle,
  HardDrive,
  Loader2,
  ArrowLeft,
  Github,
  Code,
} from "lucide-react"
import type { ConnectionType } from "../store/types"
import { useToast } from "@/hooks/use-toast"

interface DataConnectionsDialogProps {
  isOpen: boolean
  onClose: () => void
}

const DataConnectionsDialog: React.FC<DataConnectionsDialogProps> = ({ isOpen, onClose }) => {
  const [page, setPage] = useState<"list" | "add">("list")
  const [newConnection, setNewConnection] = useState<{
    name: string
    type: ConnectionType
    config: Record<string, any>
  }>({
    name: "",
    type: "grafana",
    config: {},
  })
  const [testResult, setTestResult] = useState<{ success: boolean; message?: string } | null>(null)
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [isCreatingConnection, setIsCreatingConnection] = useState(false)
  const [dialogError, setDialogError] = useState<string | null>(null)
  const [connectingIds, setConnectingIds] = useState<Record<string, boolean>>({})
  const [disconnectingIds, setDisconnectingIds] = useState<Record<string, boolean>>({})
  const [connectionErrors, setConnectionErrors] = useState<Record<string, string>>({})
  const { toast } = useToast()

  // Get state and actions from the store
  const {
    connections,
    loading,
    error,
    mcpStatuses,
    loadConnections,
    createConnection,
    deleteConnection,
    testConnection,
    loadMcpStatuses,
    startMcpServer,
    stopMcpServer,
  } = useConnectionStore()

  // Load connections and MCP statuses when the component mounts or dialog opens
  useEffect(() => {
    if (isOpen) {
      loadConnections()
      loadMcpStatuses()
    }
  }, [isOpen, loadConnections, loadMcpStatuses])

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
      type: "grafana",
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
      setDialogError(err.message || "Failed to test connection. Please check your inputs and try again.")
      setTestResult({
        success: false,
        message: err.message || "Failed to test connection",
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
      setDialogError(err.message || "Failed to create connection. Please check your inputs and try again.")
    } finally {
      setIsCreatingConnection(false)
    }
  }

  const handleConnect = async (id: string, name: string) => {
    setConnectionErrors((prev) => ({ ...prev, [id]: "" }))
    setConnectingIds((prev) => ({ ...prev, [id]: true }))

    try {
      const success = await startMcpServer(id)
      if (success) {
        toast({
          title: "Connection established",
          description: `Successfully connected to ${name}.`,
        })
      } else {
        throw new Error("Failed to connect. Please try again.")
      }
    } catch (err) {
      console.error("Failed to connect:", err)
      setConnectionErrors((prev) => ({
        ...prev,
        [id]: err.message || "Failed to connect. Please try again.",
      }))
      toast({
        variant: "destructive",
        title: "Connection failed",
        description: `Could not connect to ${name}. Please try again.`,
      })
    } finally {
      setConnectingIds((prev) => ({ ...prev, [id]: false }))
    }
  }

  const handleDisconnect = async (id: string, name: string) => {
    setConnectionErrors((prev) => ({ ...prev, [id]: "" }))
    setDisconnectingIds((prev) => ({ ...prev, [id]: true }))

    try {
      const success = await stopMcpServer(id)
      if (success) {
        toast({
          title: "Connection closed",
          description: `Successfully disconnected from ${name}.`,
        })
      } else {
        throw new Error("Failed to disconnect. Please try again.")
      }
    } catch (err) {
      console.error("Failed to disconnect:", err)
      setConnectionErrors((prev) => ({
        ...prev,
        [id]: err.message || "Failed to disconnect. Please try again.",
      }))
      toast({
        variant: "destructive",
        title: "Disconnect failed",
        description: `Could not disconnect from ${name}. Please try again.`,
      })
    } finally {
      setDisconnectingIds((prev) => ({ ...prev, [id]: false }))
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
      case "kubernetes":
        return <Database className="h-5 w-5 text-primary" />
      case "grafana":
        return <LineChart className="h-5 w-5 text-primary" />
      case "s3":
        return <HardDrive className="h-5 w-5 text-primary" />
      case "github":
        return <Github className="h-5 w-5 text-primary" />
      case "python":
        return <Code className="h-5 w-5 text-primary" />
      default:
        return <Database className="h-5 w-5 text-muted-foreground" />
    }
  }

  const getIntegrationStatusIcon = (status: string | undefined) => {
    switch (status) {
      case "running":
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case "stopped":
        return <XCircle className="h-5 w-5 text-muted-foreground" />
      case "error":
        return <AlertCircle className="h-5 w-5 text-destructive" />
      default:
        return <RefreshCw className="h-5 w-5 text-muted-foreground" />
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
            <Button variant="outline" size="sm" onClick={loadMcpStatuses}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
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
                  <TableHead>Status</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {connections.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center py-4 text-muted-foreground">
                      No connections available. Add a connection to get started.
                    </TableCell>
                  </TableRow>
                ) : (
                  connections.map((connection) => (
                    <TableRow key={connection.id} className="transition-all duration-200 hover:bg-muted">
                      <TableCell className="font-medium">{connection.name}</TableCell>
                      <TableCell>
                        <div className="flex items-center">
                          {getConnectionTypeIcon(connection.type)}
                          <span className="ml-2">{connection.type}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center">
                          {getIntegrationStatusIcon(mcpStatuses[connection.id]?.status)}
                          <span className="ml-2 capitalize">{mcpStatuses[connection.id]?.status || "unknown"}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-col space-y-2">
                          <div className="flex space-x-2">
                            {mcpStatuses[connection.id]?.status === "running" ? (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleDisconnect(connection.id, connection.name)}
                                disabled={disconnectingIds[connection.id]}
                                className="transition-all duration-200 hover:bg-red-100 hover:text-red-700 hover:border-red-300"
                              >
                                {disconnectingIds[connection.id] ? (
                                  <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Disconnecting...
                                  </>
                                ) : (
                                  "Disconnect"
                                )}
                              </Button>
                            ) : (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleConnect(connection.id, connection.name)}
                                disabled={connectingIds[connection.id]}
                                className="transition-all duration-200 hover:bg-green-100 hover:text-green-700 hover:border-green-300"
                              >
                                {connectingIds[connection.id] ? (
                                  <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Connecting...
                                  </>
                                ) : (
                                  "Connect"
                                )}
                              </Button>
                            )}
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDeleteConnection(connection.id, connection.name)}
                              className="transition-all duration-200 hover:bg-red-50"
                            >
                              <Trash className="h-4 w-4" />
                            </Button>
                          </div>

                          {connectionErrors[connection.id] && (
                            <div className="text-xs text-destructive flex items-center mt-1">
                              <AlertCircle className="h-3 w-3 mr-1" />
                              <span>{connectionErrors[connection.id]}</span>
                            </div>
                          )}
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
          <Button variant="ghost" size="sm" onClick={() => setPage("list")} className="mr-2">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
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
              <option value="python">Python</option>
              <option value="kubernetes">Kubernetes</option>
              <option value="s3">S3</option>
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

          {newConnection.type === "python" && (
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="python_script">Python Script Path</Label>
                <Input
                  id="python_script"
                  value={newConnection.config?.python_script || ""}
                  onChange={(e) => handleConfigChange("python_script", e.target.value)}
                  placeholder="/path/to/script.py"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="python_args">Arguments (Optional)</Label>
                <Input
                  id="python_args"
                  value={newConnection.config?.python_args || ""}
                  onChange={(e) => handleConfigChange("python_args", e.target.value)}
                  placeholder="--arg1 value1 --arg2 value2"
                />
              </div>
            </div>
          )}

          {newConnection.type === "kubernetes" && (
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="kubeconfig">Kubeconfig</Label>
                <Input
                  id="kubeconfig"
                  value={newConnection.config?.kubeconfig || ""}
                  onChange={(e) => handleConfigChange("kubeconfig", e.target.value)}
                  placeholder="Enter kubeconfig path or content"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="context">Context</Label>
                <Input
                  id="context"
                  value={newConnection.config?.context || ""}
                  onChange={(e) => handleConfigChange("context", e.target.value)}
                  placeholder="Enter kubernetes context (optional)"
                />
              </div>
            </div>
          )}

          {newConnection.type === "s3" && (
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="bucket">Bucket</Label>
                <Input
                  id="bucket"
                  value={newConnection.config?.bucket || ""}
                  onChange={(e) => handleConfigChange("bucket", e.target.value)}
                  placeholder="Enter S3 bucket name"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="aws_access_key_id">AWS Access Key ID</Label>
                <Input
                  id="aws_access_key_id"
                  value={newConnection.config?.aws_access_key_id || ""}
                  onChange={(e) => handleConfigChange("aws_access_key_id", e.target.value)}
                  placeholder="Enter AWS access key ID"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="aws_secret_access_key">AWS Secret Access Key</Label>
                <Input
                  id="aws_secret_access_key"
                  type="password"
                  value={newConnection.config?.aws_secret_access_key || ""}
                  onChange={(e) => handleConfigChange("aws_secret_access_key", e.target.value)}
                  placeholder="Enter AWS secret access key"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="region">Region</Label>
                <Input
                  id="region"
                  value={newConnection.config?.region || ""}
                  onChange={(e) => handleConfigChange("region", e.target.value)}
                  placeholder="Enter AWS region (e.g., us-east-1)"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="endpoint">Endpoint (Optional)</Label>
                <Input
                  id="endpoint"
                  value={newConnection.config?.endpoint || ""}
                  onChange={(e) => handleConfigChange("endpoint", e.target.value)}
                  placeholder="Enter custom endpoint URL (for MinIO, etc.)"
                />
              </div>
            </div>
          )}

          {testResult && (
            <div
              className={`p-3 rounded-md ${
                testResult.success ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"
              }`}
            >
              <div className="flex items-center">
                {testResult.success ? (
                  <CheckCircle className="h-5 w-5 mr-2" />
                ) : (
                  <AlertCircle className="h-5 w-5 mr-2" />
                )}
                <p>{testResult.message || (testResult.success ? "Connection successful!" : "Connection failed!")}</p>
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
      <DialogContent className="bg-white max-w-3xl max-h-[80vh] overflow-y-auto">
        {page === "list" ? renderConnectionsList() : renderAddConnectionForm()}
      </DialogContent>
    </Dialog>
  )
}

export default DataConnectionsDialog
