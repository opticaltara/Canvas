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
  Plus,
  Trash,
  Loader2,
  ArrowLeft,
  Github,
  Folder,
} from "lucide-react"
import type { ConnectionType } from "../store/types"
import { useToast } from "@/hooks/use-toast"
import GitHubConnectionForm from "./connection-forms/GitHubConnectionForm"
import JiraConnectionForm from "./connection-forms/JiraConnectionForm"
import FileSystemConnectionForm from "./connection-forms/FileSystemConnectionForm"

interface DataConnectionsDialogProps {
  isOpen: boolean
  onClose: () => void
  initialPage?: "list" | "add"
}

const DataConnectionsDialog: React.FC<DataConnectionsDialogProps> = ({ isOpen, onClose, initialPage = "list" }) => {
  const [page, setPage] = useState<"list" | "add">(initialPage)
  const [newConnection, setNewConnection] = useState<{
    name: string
    type: ConnectionType | ""
    config: Record<string, any>
  }>({
    name: "",
    type: "",
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
    availableTypes,
    loadConnections,
    createConnection,
    deleteConnection,
    testConnection,
  } = useConnectionStore()

  // Effect to load connections (which now also loads types via the store action)
  useEffect(() => {
    if (isOpen) {
      console.log("DataConnectionsDialog: isOpen is true, calling loadConnections...")
      loadConnections()
    }
  }, [isOpen, loadConnections])

  // Effect to set default type for the form *after* types load from store
  useEffect(() => {
    if (isOpen && availableTypes.length > 0 && !newConnection.type) {
        setNewConnection(prev => ({ ...prev, type: availableTypes[0] as ConnectionType }));
    }
    // Only run when types array changes or dialog opens
  }, [isOpen, availableTypes, newConnection.type]) 

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
      // Reset using types from store
      type: availableTypes.length > 0 ? availableTypes[0] as ConnectionType : "",
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

  const handleConfigChange = (field: string, value: string | boolean | string[]) => {
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

    // Check if a valid type is selected
    if (!newConnection.type) {
      setDialogError("Please select a connection type.");
      return; 
    }

    setIsTestingConnection(true)

    try {
      // Type is guaranteed to be valid ConnectionType here
      const result = await testConnection(newConnection as { type: ConnectionType; [key: string]: any })
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

    // Check if a valid type is selected
    if (!newConnection.type) {
      setDialogError("Please select a connection type.");
      return; 
    }

    setDialogError(null)
    setIsCreatingConnection(true)

    try {
      // Type is guaranteed to be valid ConnectionType here
      const result = await createConnection(newConnection as { name: string; type: ConnectionType; [key: string]: any })
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
      let errorMessage = "An unknown error occurred during creation.";
      if (err instanceof Error) {
          // Check if the error message is JSON (like FastAPI validation error)
          try {
            // Attempt to parse the message as JSON
            const errorDetail = JSON.parse(err.message); 
            // If successful, format a user-friendly message
            if (Array.isArray(errorDetail.detail)) {
              errorMessage = errorDetail.detail.map((d: any) => 
                `Field '${d.loc.slice(1).join('.')}': ${d.msg}` // Join location path, skip 'body'
              ).join('; ');
            } else if (errorDetail.detail) {
              // Handle cases where detail might be a string
              errorMessage = String(errorDetail.detail);
            } else {
                errorMessage = err.message; // Use original message if parsing fails or format is unexpected
            }
          } catch (parseError) {
            // If parsing fails, it's likely a regular string error message
            errorMessage = err.message; 
          }
      }
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
      case "github":
        return <Github className="h-5 w-5 text-primary" />
      case "jira":
        return <Database className="h-5 w-5 text-primary" />
      case "filesystem":
        return <Folder className="h-5 w-5 text-primary" />
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
            <Label htmlFor="type">Connection Type</Label>
            <select
              id="type"
              value={newConnection.type}
              onChange={(e) => {
                setNewConnection((prev) => ({ 
                  ...prev, 
                  type: e.target.value as ConnectionType, 
                  config: {} 
                }));
                setTestResult(null);
                setDialogError(null);
              }}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 bg-white"
              disabled={availableTypes.length === 0}
            >
              <option value="" disabled hidden>Select Type...</option>
              {availableTypes.map(type => (
                <option key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </select>
          </div>
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

          {newConnection.type === "github" && (
            <GitHubConnectionForm 
              config={newConnection.config}
              onConfigChange={handleConfigChange}
            />
          )}

          {newConnection.type === "jira" && (
             <JiraConnectionForm 
              config={newConnection.config}
              onConfigChange={handleConfigChange}
            />
          )}

          {newConnection.type === "filesystem" && (
            <FileSystemConnectionForm
              config={newConnection.config}
              onConfigChange={handleConfigChange}
            />
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
            {/* <Button
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
            </Button> */}
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
