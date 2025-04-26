"use client"

import type React from "react"
import { useEffect, useState } from "react"
import { useConnectionStore } from "../store/connectionStore"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
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
  Github,
  Code,
} from "lucide-react"
import type { ConnectionType } from "../store/types"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { useToast } from "@/hooks/use-toast"

const ConnectionManager: React.FC = () => {
  // State definitions remain the same
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
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)
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
    setDefaultConnection,
    testConnection,
    loadMcpStatuses,
    startMcpServer,
    stopMcpServer,
  } = useConnectionStore()

  // All the useEffect hooks and handler functions remain the same
  // Load connections and MCP statuses when the component mounts
  useEffect(() => {
    loadConnections()
    loadMcpStatuses()
  }, [loadConnections, loadMcpStatuses])

  // Reset dialog state when it opens/closes
  useEffect(() => {
    if (!isAddDialogOpen) {
      // Reset form state when dialog closes
      setTimeout(() => {
        setTestResult(null)
        setDialogError(null)
        setIsTestingConnection(false)
        setIsCreatingConnection(false)
        setNewConnection({
          name: "",
          type: "grafana",
          config: {},
        })
      }, 300) // Wait for dialog close animation
    }
  }, [isAddDialogOpen])

  // Handler functions remain the same
  const handleInputChange = (field: string, value: string) => {
    setNewConnection((prev) => ({ ...prev, [field]: value }))
    // Clear error when user makes changes
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
    // Clear error when user makes changes
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
        setIsAddDialogOpen(false)
        setNewConnection({
          name: "",
          type: "grafana",
          config: {},
        })
        setTestResult(null)
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
    // Clear any previous errors for this connection
    setConnectionErrors((prev) => ({ ...prev, [id]: "" }))

    // Set connecting state
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
    // Clear any previous errors for this connection
    setConnectionErrors((prev) => ({ ...prev, [id]: "" }))

    // Set disconnecting state
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

  // Icon functions remain the same
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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        <span className="ml-2 text-primary">Loading connections...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive" className="animate-in fade-in-0 slide-in-from-top-5 max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
          <Button variant="outline" className="mt-4" onClick={() => loadConnections()}>
            Retry
          </Button>
        </Alert>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-6 px-4">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-foreground">Data Connections</h1>
        <div className="flex space-x-2">
          <Button
            variant="outline"
            onClick={loadMcpStatuses}
            className="transition-all duration-200 hover:bg-primary/10 hover:text-primary btn-press"
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh Status
          </Button>
          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button className="bg-primary hover:bg-primary/90 text-primary-foreground transition-all duration-200 btn-press btn-glow">
                <Plus className="mr-2 h-4 w-4" />
                <span>Add Connection</span>
              </Button>
            </DialogTrigger>
            <DialogContent className="DialogContent sm:max-w-[500px] dialog-enter">
              <DialogHeader>
                <DialogTitle className="text-xl font-semibold">Add New Connection</DialogTitle>
                <DialogDescription className="text-muted-foreground">
                  Create a new data source connection for your notebooks.
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid gap-2">
                  <Label htmlFor="name">Connection Name</Label>
                  <Input
                    id="name"
                    value={newConnection.name}
                    onChange={(e) => handleInputChange("name", e.target.value)}
                    placeholder="Enter connection name"
                    className="w-full input-focus-animation"
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="type">Connection Type</Label>
                  <select
                    id="type"
                    value={newConnection.type}
                    onChange={(e) => handleInputChange("type", e.target.value as ConnectionType)}
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 bg-white input-focus-animation"
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
                        className="input-focus-animation"
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
                        className="input-focus-animation"
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
                        className="input-focus-animation"
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

                {newConnection.type === "kubernetes" && (
                  <div className="space-y-4">
                    <div className="grid gap-2">
                      <Label htmlFor="kubeconfig">Kubeconfig</Label>
                      <Input
                        id="kubeconfig"
                        value={newConnection.config?.kubeconfig || ""}
                        onChange={(e) => handleConfigChange("kubeconfig", e.target.value)}
                        placeholder="Enter kubeconfig path or content"
                        className="input-focus-animation"
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="context">Context</Label>
                      <Input
                        id="context"
                        value={newConnection.config?.context || ""}
                        onChange={(e) => handleConfigChange("context", e.target.value)}
                        placeholder="Enter kubernetes context (optional)"
                        className="input-focus-animation"
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
                        className="input-focus-animation"
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="aws_access_key_id">AWS Access Key ID</Label>
                      <Input
                        id="aws_access_key_id"
                        value={newConnection.config?.aws_access_key_id || ""}
                        onChange={(e) => handleConfigChange("aws_access_key_id", e.target.value)}
                        placeholder="Enter AWS access key ID"
                        className="input-focus-animation"
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
                        className="input-focus-animation"
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="region">Region</Label>
                      <Input
                        id="region"
                        value={newConnection.config?.region || ""}
                        onChange={(e) => handleConfigChange("region", e.target.value)}
                        placeholder="Enter AWS region (e.g., us-east-1)"
                        className="input-focus-animation"
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="endpoint">Endpoint (Optional)</Label>
                      <Input
                        id="endpoint"
                        value={newConnection.config?.endpoint || ""}
                        onChange={(e) => handleConfigChange("endpoint", e.target.value)}
                        placeholder="Enter custom endpoint URL (for MinIO, etc.)"
                        className="input-focus-animation"
                      />
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
                        className="input-focus-animation"
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="python_args">Arguments (Optional)</Label>
                      <Input
                        id="python_args"
                        value={newConnection.config?.python_args || ""}
                        onChange={(e) => handleConfigChange("python_args", e.target.value)}
                        placeholder="--arg1 value1 --arg2 value2"
                        className="input-focus-animation"
                      />
                    </div>
                  </div>
                )}

                {testResult && (
                  <div
                    className={`p-3 rounded-md transition-all duration-300 animate-in fade-in-0 slide-in-from-top-5 ${
                      testResult.success ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"
                    }`}
                  >
                    <div className="flex items-center">
                      {testResult.success ? (
                        <CheckCircle className="h-5 w-5 mr-2" />
                      ) : (
                        <AlertCircle className="h-5 w-5 mr-2" />
                      )}
                      <p>
                        {testResult.message || (testResult.success ? "Connection successful!" : "Connection failed!")}
                      </p>
                    </div>
                  </div>
                )}

                {dialogError && (
                  <div className="p-3 rounded-md bg-red-50 text-red-800 transition-all duration-300 animate-in fade-in-0 slide-in-from-top-5">
                    <div className="flex items-center">
                      <AlertCircle className="h-5 w-5 mr-2" />
                      <p>{dialogError}</p>
                    </div>
                  </div>
                )}
              </div>
              <DialogFooter className="flex justify-end space-x-2 mt-6">
                <Button
                  variant="outline"
                  onClick={handleTestConnection}
                  disabled={isTestingConnection || isCreatingConnection}
                  className="transition-all duration-200 hover:bg-secondary btn-press"
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
                  className="bg-primary hover:bg-primary/90 text-primary-foreground transition-colors duration-200 btn-press"
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
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <Card className="transition-all duration-300 hover:shadow-md">
        <CardHeader>
          <CardTitle>Available Connections</CardTitle>
          <CardDescription>Manage your data source connections for notebooks.</CardDescription>
        </CardHeader>
        <CardContent>
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
                  <TableCell colSpan={5} className="text-center py-4 text-muted-foreground">
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
                              className="transition-all duration-200 hover:bg-red-100 hover:text-red-700 hover:border-red-300 btn-press"
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
                              className="transition-all duration-200 hover:bg-green-100 hover:text-green-700 hover:border-green-300 btn-press"
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
                            className="transition-all duration-200 hover:bg-red-50 btn-press"
                          >
                            <Trash className="h-4 w-4" />
                          </Button>
                        </div>

                        {/* Error message */}
                        {connectionErrors[connection.id] && (
                          <div className="text-xs text-destructive flex items-center mt-1 animate-in fade-in-0 slide-in-from-top-5">
                            <AlertCircle className="h-3 w-3 mr-1" />
                            <span>{connectionErrors[connection.id]}</span>
                            <Button
                              variant="link"
                              size="sm"
                              className="text-xs p-0 h-auto ml-1 text-primary"
                              onClick={() =>
                                mcpStatuses[connection.id]?.status === "running"
                                  ? handleDisconnect(connection.id, connection.name)
                                  : handleConnect(connection.id, connection.name)
                              }
                            >
                              Retry
                            </Button>
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

export default ConnectionManager
