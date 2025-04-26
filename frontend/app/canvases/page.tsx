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
  DialogTrigger,
} from "@/components/ui/dialog"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Plus, Search, MoreVertical, Trash2, FileText } from "lucide-react"
import { api } from "@/api/client"
import { useToast } from "@/hooks/use-toast"

export default function CanvasesPage() {
  const { toast } = useToast()
  const [canvases, setCanvases] = useState([])
  const [filteredCanvases, setFilteredCanvases] = useState([])
  const [searchQuery, setSearchQuery] = useState("")
  const [loading, setLoading] = useState(true)
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [newCanvasName, setNewCanvasName] = useState("")
  const [newCanvasDescription, setNewCanvasDescription] = useState("")

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
  const handleDeleteCanvas = async (id, name) => {
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

  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return ""
    try {
      return new Date(dateString).toLocaleDateString()
    } catch (error) {
      return dateString
    }
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Your Canvases</h1>
        <div className="flex space-x-4">
          <div className="relative w-64">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search canvases..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8"
            />
          </div>
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                New Canvas
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

      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      ) : filteredCanvases.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredCanvases.map((canvas) => (
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
          <h3 className="text-lg font-medium text-gray-900 mb-2">No canvases found</h3>
          <p className="text-gray-500 mb-4">
            {searchQuery ? "Try a different search term" : "Create your first canvas to get started"}
          </p>
          {!searchQuery && (
            <Button onClick={() => setIsCreateDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Canvas
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
