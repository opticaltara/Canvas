"use client"

import { useState, useEffect } from "react"
import { useRouter, useParams } from "next/navigation"
import Link from "next/link"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Search, Plus, MoreVertical, Trash2, FileText, ChevronLeft, ChevronRight, Database } from "lucide-react"
import { api } from "@/api/client"
import { useToast } from "@/hooks/use-toast"

interface Canvas {
  id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
}

export default function CanvasSidebar() {
  const router = useRouter()
  const params = useParams()
  const currentCanvasId = params?.id as string
  const { toast } = useToast()

  const [canvases, setCanvases] = useState<Canvas[]>([])
  const [filteredCanvases, setFilteredCanvases] = useState<Canvas[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [loading, setLoading] = useState(true)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [newCanvasName, setNewCanvasName] = useState("")
  const [newCanvasDescription, setNewCanvasDescription] = useState("")
  // Add state for sidebar collapse
  const [isCollapsed, setIsCollapsed] = useState(false)

  // Fetch canvases
  useEffect(() => {
    const fetchCanvases = async () => {
      try {
        setLoading(true)
        console.log("Fetching notebooks with api:", api.notebooks)
        const notebooks = await api.notebooks.list()
        console.log("Fetched notebooks:", notebooks)
        setCanvases(notebooks)
        setFilteredCanvases(notebooks)
      } catch (error) {
        console.error("Failed to fetch canvases:", error)
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to load canvases. Please try again.",
        })
      } finally {
        setLoading(false)
      }
    }

    fetchCanvases()
  }, [toast])

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
    if (!newCanvasName.trim()) return

    try {
      const newCanvas = await api.notebooks.create({
        name: newCanvasName,
        description: newCanvasDescription,
      })

      setCanvases((prev) => [...prev, newCanvas])
      setNewCanvasName("")
      setNewCanvasDescription("")
      setIsCreateDialogOpen(false)

      // Navigate to the new canvas
      router.push(`/canvas/${newCanvas.id}`)

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

      // If the current canvas was deleted, navigate to the home page
      if (id === currentCanvasId) {
        router.push("/")
      }
    } catch (error) {
      console.error("Failed to delete canvas:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete canvas. Please try again.",
      })
    }
  }

  // Format date for display
  const formatDate = (dateString: string) => {
    if (!dateString) return ""
    try {
      return new Date(dateString).toLocaleDateString()
    } catch (error) {
      return dateString
    }
  }

  // Toggle sidebar collapse
  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed)
  }

  return (
    <div
      className={`h-screen border-r bg-background flex flex-col transition-all duration-300 ${
        isCollapsed ? "w-16" : "w-64"
      }`}
    >
      {/* Collapse Toggle Button */}
      <div className="absolute -right-3 top-4 z-10">
        <Button
          variant="outline"
          size="icon"
          className="h-6 w-6 rounded-full border-gray-200 bg-white shadow-md"
          onClick={toggleSidebar}
        >
          {isCollapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
        </Button>
      </div>

      {/* Sidebar Header */}
      <div className={`p-4 border-b ${isCollapsed ? "flex justify-center" : ""}`}>
        {!isCollapsed && (
          <>
            <h2 className="text-lg font-semibold mb-4">Canvases</h2>
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search canvases..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8"
              />
            </div>
          </>
        )}
        {isCollapsed && <FileText className="h-5 w-5 text-muted-foreground" />}
      </div>

      {/* Canvas List */}
      <ScrollArea className="flex-1">
        <div className="p-2">
          {loading ? (
            <div className="flex justify-center items-center h-20">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary"></div>
            </div>
          ) : filteredCanvases.length > 0 ? (
            <div className="space-y-1">
              {filteredCanvases.map((canvas) => (
                <div
                  key={canvas.id}
                  className={`group flex items-center justify-between p-2 rounded-md hover:bg-accent transition-colors ${
                    canvas.id === currentCanvasId ? "bg-accent" : ""
                  }`}
                >
                  <Link href={`/canvas/${canvas.id}`} className="flex items-center flex-1 min-w-0">
                    <FileText className="h-4 w-4 mr-2 text-muted-foreground" />
                    {!isCollapsed && (
                      <div className="truncate">
                        <div className="font-medium truncate">{canvas.name}</div>
                        <div className="text-xs text-muted-foreground">{formatDate(canvas.updated_at)}</div>
                      </div>
                    )}
                  </Link>
                  {!isCollapsed && (
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
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className={`text-center py-4 text-muted-foreground ${isCollapsed ? "px-0" : ""}`}>
              {searchQuery ? (isCollapsed ? "..." : "No canvases found") : isCollapsed ? "..." : "No canvases yet"}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Footer Buttons */}
      <div className={`p-4 border-t ${isCollapsed ? "flex flex-col items-center space-y-4" : ""}`}>
        {/* Data Connections Button */}
        <Link href="/connections">
          <Button
            variant="outline"
            className={`${isCollapsed ? "w-10 h-10 p-0" : "w-full mb-2"}`}
            title="Data Connections"
          >
            <Database className={`h-4 w-4 ${isCollapsed ? "" : "mr-2"}`} />
            {!isCollapsed && "Data Connections"}
          </Button>
        </Link>

        {/* Create Canvas Button */}
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button className={`${isCollapsed ? "w-10 h-10 p-0" : "w-full"}`} title="New Canvas">
              <Plus className={`h-4 w-4 ${isCollapsed ? "" : "mr-2"}`} />
              {!isCollapsed && "New Canvas"}
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Canvas</DialogTitle>
              <DialogDescription>Create a new canvas to start your analysis.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <label htmlFor="name" className="text-sm font-medium">
                  Name
                </label>
                <Input
                  id="name"
                  placeholder="Enter canvas name"
                  value={newCanvasName}
                  onChange={(e) => setNewCanvasName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="description" className="text-sm font-medium">
                  Description (optional)
                </label>
                <Input
                  id="description"
                  placeholder="Enter canvas description"
                  value={newCanvasDescription}
                  onChange={(e) => setNewCanvasDescription(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateCanvas} disabled={!newCanvasName.trim()}>
                Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  )
}
