"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Plus, Search, MoreVertical, Trash2, FileText, LineChart, Github, Database, Loader2, AlertCircle } from "lucide-react"
import { api, Notebook } from "@/api/client"
import { useToast } from "@/hooks/use-toast"
import { useConnectionStore } from "@/store/connectionStore"
import dynamic from 'next/dynamic'
// import DataConnectionsDialog from "@/components/DataConnectionsDialog" // Will be dynamically imported
import type { Connection } from "@/store/types"

const DataConnectionsDialog = dynamic(() => import('@/components/DataConnectionsDialog'), {
  ssr: false,
  loading: () => null, // Or a proper loading skeleton/spinner
});

// Define interfaces based on usage if not available from API client
interface Canvas {
  id: string;
  name: string;
  description?: string;
  updated_at?: string;
}

// Remove local Connection interface

// --- New DataConnectionsView Component ---
interface DataConnectionsViewProps {
  connections: Connection[];
  loading: boolean;
  error: string | null;
  onAddConnection: () => void;
  onDeleteConnection: (id: string, name: string) => Promise<void>;
}

const DataConnectionsView: React.FC<DataConnectionsViewProps> = ({
  connections,
  loading,
  error,
  onAddConnection,
  onDeleteConnection,
}) => {

  const getConnectionTypeIcon = (type: string) => {
    switch (type) {
      case "grafana":
        return <LineChart className="h-5 w-5 text-orange-500" />; // Adjusted color
      case "github":
        return <Github className="h-5 w-5 text-gray-800" />; // Adjusted color
      default:
        return <Database className="h-5 w-5 text-gray-500" />;
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-primary">Loading connections...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-64">
        <Card className="w-full max-w-md border-destructive bg-destructive/10">
          <CardHeader>
            <CardTitle className="text-destructive flex items-center">
              <AlertCircle className="h-5 w-5 mr-2" /> Error Loading Connections
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-destructive">{error}</p>
            {/* Optionally add a retry button if the store supports it easily */}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
           <h2 className="text-2xl font-semibold tracking-tight">Data Connections</h2>
           <p className="text-muted-foreground">Manage your data source connections here.</p>
        </div>
        <Button onClick={onAddConnection}>
          <Plus className="mr-2 h-4 w-4" /> Add Connection
        </Button>
      </div>

      <Card>
        <CardContent className="p-0">
            {/* Using a simple list for now, could use Table component later */}
            {connections.length === 0 ? (
              <div className="p-6 text-center text-muted-foreground">
                 No connections found. Add your first connection to link data sources.
              </div>
            ) : (
              <ul className="divide-y divide-border">
                {connections.map((connection) => (
                  <li key={connection.id} className="flex items-center justify-between p-4 hover:bg-muted/50">
                    <div className="flex items-center space-x-3">
                       {getConnectionTypeIcon(connection.type)}
                       <span className="font-medium">{connection.name}</span>
                       <span className="text-xs text-muted-foreground capitalize">({connection.type})</span>
                    </div>
                    <Button
                       variant="ghost"
                       size="sm"
                       className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                       onClick={() => onDeleteConnection(connection.id, connection.name)}
                    >
                       <Trash2 className="h-4 w-4" />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
        </CardContent>
      </Card>
    </div>
  );
};
// --- End of DataConnectionsView Component ---

// --- Create Canvas Dialog Content Component ---
interface CreateCanvasDialogContentProps {
  newCanvasName: string;
  setNewCanvasName: (value: string) => void;
  newCanvasDescription: string;
  setNewCanvasDescription: (value: string) => void;
  handleCreateCanvas: () => void;
  setIsCreateDialogOpen: (isOpen: boolean) => void;
  isLoading: boolean;
}

const CreateCanvasDialogContent: React.FC<CreateCanvasDialogContentProps> = ({
  newCanvasName,
  setNewCanvasName,
  newCanvasDescription,
  setNewCanvasDescription,
  handleCreateCanvas,
  setIsCreateDialogOpen,
  isLoading,
}) => {
  return (
    <DialogContent className="bg-white">
      <DialogHeader>
        <DialogTitle>Create New Canvas</DialogTitle>
        <DialogDescription>Create a new canvas to start your analysis.</DialogDescription>
      </DialogHeader>
      <div className="space-y-4 py-4">
        <div className="space-y-2">
          <label htmlFor="name" className="text-sm font-medium">Name</label>
          <Input
            id="name"
            placeholder="Enter canvas name"
            value={newCanvasName}
            onChange={(e) => setNewCanvasName(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <label htmlFor="description" className="text-sm font-medium">Description (optional)</label>
          <Input
            id="description"
            placeholder="Enter canvas description"
            value={newCanvasDescription}
            onChange={(e) => setNewCanvasDescription(e.target.value)}
          />
        </div>
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>Cancel</Button>
        <Button onClick={handleCreateCanvas} disabled={!newCanvasName.trim() || isLoading}>Create</Button>
      </DialogFooter>
    </DialogContent>
  );
};
// --- End Create Canvas Dialog Content Component ---

export default function CanvasesPage() {
  const { toast } = useToast()
  const [canvases, setCanvases] = useState<Canvas[]>([])
  const [filteredCanvases, setFilteredCanvases] = useState<Canvas[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [loadingCanvases, setLoadingCanvases] = useState(true)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [isConnectionsDialogOpen, setIsConnectionsDialogOpen] = useState(false)
  const [newCanvasName, setNewCanvasName] = useState("")
  const [newCanvasDescription, setNewCanvasDescription] = useState("")

  const {
    connections,
    loading: loadingConnections,
    error: connectionsError,
    loadConnections,
    deleteConnection,
  } = useConnectionStore()

  const hasConnections = connections.length > 0;

  // Fetch canvases
  useEffect(() => {
    const fetchCanvases = async () => {
      try {
        setLoadingCanvases(true)
        console.log("Fetching notebooks with api:", api.notebooks)
        const notebooks: Notebook[] = await api.notebooks.list()
        console.log("Fetched notebooks:", notebooks)
        const formattedCanvases: Canvas[] = notebooks.map(nb => ({
          id: nb.id,
          name: nb.metadata?.title || 'Untitled Canvas',
          description: nb.metadata?.description || '',
          updated_at: nb.metadata?.updated_at
        }));
        setCanvases(formattedCanvases)
        setFilteredCanvases(formattedCanvases)
      } catch (error) {
        console.error("Failed to fetch canvases:", error)
        const description = error instanceof Error ? error.message : "Failed to load canvases. Please try again.";
        toast({
          variant: "destructive",
          title: "Error Loading Canvases",
          description,
        })
      } finally {
        setLoadingCanvases(false)
      }
    }

    fetchCanvases()
  }, [toast])

  // Fetch connections using the store action
  useEffect(() => {
    console.log("CanvasesPage: Firing loadConnections effect")
    loadConnections();
  }, [loadConnections]);

  // Filter canvases based on search query
  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredCanvases(canvases)
    } else {
      const filtered = canvases.filter((canvas) => canvas.name.toLowerCase().includes(searchQuery.toLowerCase()))
      setFilteredCanvases(filtered)
    }
  }, [searchQuery, canvases])

  // Create a new canvas
  const handleCreateCanvas = async () => {
    if (!newCanvasName.trim() || !hasConnections) return

    try {
      const newCanvas = await api.notebooks.create({
        title: newCanvasName,
        description: newCanvasDescription,
      })

      const formattedNewCanvas: Canvas = {
        id: newCanvas.id,
        name: newCanvas.metadata?.title || 'Untitled Canvas',
        description: newCanvas.metadata?.description || '',
        updated_at: newCanvas.metadata?.updated_at
      };

      setCanvases((prev) => [...prev, formattedNewCanvas])
      setNewCanvasName("")
      setNewCanvasDescription("")
      setIsCreateDialogOpen(false)

      toast({
        title: "Canvas Created",
        description: "Your new canvas has been created successfully.",
      })
    } catch (error) {
      console.error("Failed to create canvas:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to create canvas. Please try again.",
      })
    }
  }

  // Delete a canvas
  const handleDeleteCanvas = async (id: string, name: string) => {
    try {
      await api.notebooks.delete(id)
      setCanvases((prev) => prev.filter((canvas) => canvas.id !== id))

      toast({
        title: "Canvas Deleted",
        description: `"${name}" has been deleted successfully.`,
      })
    } catch (error) {
      console.error("Failed to delete canvas:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete canvas. Please try again.",
      })
    }
  }

  // Delete a connection using the store action
  const handleDeleteConnection = async (id: string, name: string) => {
    try {
      const success = await deleteConnection(id);
      if (success) {
        toast({
          title: "Connection Deleted",
          description: `"${name}" connection has been deleted successfully.`,
        });
      } else {
        throw new Error("Deletion action returned false");
      }
    } catch (error) {
      console.error("Failed to delete connection:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: `Failed to delete "${name}" connection. ${error instanceof Error ? error.message : ''}`.trim(),
      });
    }
  };

  // Format date for display
  const formatDate = (dateString: string | undefined): string => {
    if (!dateString) return ""
    try {
      return new Date(dateString).toLocaleDateString()
    } catch (error) {
      console.warn("Invalid date string for formatting:", dateString);
      return dateString;
    }
  }

  const isLoading = loadingCanvases || loadingConnections;

  const CreateCanvasButton = ({ isIconOnly = false }) => (
    <Button
      onClick={() => hasConnections && setIsCreateDialogOpen(true)}
      disabled={!hasConnections || isLoading || !!connectionsError}
      variant={isIconOnly ? "ghost" : "default"}
      size={isIconOnly ? "icon" : "default"}
    >
      <Plus className={`h-4 w-4 ${!isIconOnly ? 'mr-2' : ''}`} />
      {!isIconOnly && "New Canvas"}
    </Button>
  );

  return (
    <>
      <div className="w-full space-y-8">
        <section>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-semibold tracking-tight">Your Canvases</h2>
            <div className="flex items-center space-x-4">
              <div className="relative w-64">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search canvases..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8"
                />
              </div>
              <CreateCanvasButton />
            </div>
          </div>

          {loadingCanvases ? (
            <div className="flex justify-center items-center h-64">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : filteredCanvases.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredCanvases.map((canvas: Canvas) => (
                <Card key={canvas.id} className="group hover:shadow-md transition-shadow">
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-start">
                      <CardTitle className="text-lg">
                        <Link href={`/canvas/${canvas.id}`} className="hover:text-primary transition-colors">
                          {canvas.name}
                        </Link>
                      </CardTitle>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100">
                            <MoreVertical className="h-4 w-4" />
                            <span className="sr-only">Open menu</span>
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            className="text-destructive focus:text-destructive"
                            onClick={() => handleDeleteCanvas(canvas.id, canvas.name)}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    <CardDescription>{canvas.description || "No description"}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center text-xs text-muted-foreground">
                      <FileText className="h-3 w-3 mr-1" />
                      <span>Last updated: {formatDate(canvas.updated_at)}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {searchQuery ? "No canvases match your search" : "No Canvases Yet"}
              </h3>
              <p className="text-gray-500 mb-4">
                {searchQuery
                   ? "Try a different search term."
                   : hasConnections && !connectionsError
                   ? "Create your first canvas to get started."
                   : connectionsError
                   ? "Could not load connections. Cannot create a canvas."
                   : "Add a Data Connection before creating your first canvas."
                }
              </p>
              {!searchQuery && !connectionsError && (
                <CreateCanvasButton />
              )}
            </div>
          )}
        </section>

        <hr className="my-8" />

        <section>
          <DataConnectionsView
            connections={connections}
            loading={loadingConnections}
            error={connectionsError}
            onAddConnection={() => setIsConnectionsDialogOpen(true)}
            onDeleteConnection={handleDeleteConnection}
          />
        </section>
      </div>

      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <CreateCanvasDialogContent 
           newCanvasName={newCanvasName}
           setNewCanvasName={setNewCanvasName}
           newCanvasDescription={newCanvasDescription}
           setNewCanvasDescription={setNewCanvasDescription}
           handleCreateCanvas={handleCreateCanvas}
           setIsCreateDialogOpen={setIsCreateDialogOpen}
           isLoading={isLoading}
        />
      </Dialog>

      <DataConnectionsDialog
        isOpen={isConnectionsDialogOpen}
        onClose={() => setIsConnectionsDialogOpen(false)}
        initialPage="add"
      />
    </>
  )
}
