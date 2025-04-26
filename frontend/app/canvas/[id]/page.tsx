"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Play, Save, Share, Download, MoreHorizontal } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import Cell from "@/components/canvas/Cell"
import CellCreationPills from "@/components/canvas/CellCreationPills"
import AIChatPanel from "@/components/AIChatPanel"
import AIChatToggle from "@/components/AIChatToggle"
import { api } from "@/api/client"
import { useToast } from "@/hooks/use-toast"
import { useInvestigationEvents, type CellCreationParams } from "@/hooks/useInvestigationEvents"
// Update the imports at the top to include useCanvasStore
import { useCanvasStore } from "@/store/canvasStore"

export default function CanvasPage() {
  const params = useParams()
  const notebookId = params.id as string
  const { toast } = useToast()

  const [notebook, setNotebook] = useState(null)
  const [cells, setCells] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [isEditing, setIsEditing] = useState(false)
  const [isEditingDesc, setIsEditingDesc] = useState(false)
  const [editedName, setEditedName] = useState("")
  const [editedDescription, setEditedDescription] = useState("")
  const [deletingCellId, setDeletingCellId] = useState(null)
  const [isChatPanelOpen, setIsChatPanelOpen] = useState(false)

  // Add a ref to track initialization
  const initialized = useRef(false)

  // Set the active notebook ID in the store when the component mounts
  useEffect(() => {
    if (notebookId) {
      console.log("🔑 Setting active notebook ID in store:", notebookId)
      useCanvasStore.getState().setActiveNotebook(notebookId)
    }
  }, [notebookId])

  // Handle cell creation from investigation events
  const handleCreateCell = useCallback(
    (params: CellCreationParams) => {
      console.log("🔍 handleCreateCell called with params:", params)
      // Check if cell already exists
      const existingCellIndex = cells.findIndex((cell) => cell.id === params.id)

      if (existingCellIndex >= 0) {
        // Update existing cell
        const updatedCells = [...cells]
        updatedCells[existingCellIndex] = {
          ...updatedCells[existingCellIndex],
          ...params,
        }
        setCells(updatedCells)
      } else {
        // Create new cell
        setCells((prevCells) => [...prevCells, { ...params, isNew: true }])

        // Remove the isNew flag after animation completes
        setTimeout(() => {
          setCells((prevCells) => prevCells.map((cell) => (cell.id === params.id ? { ...cell, isNew: false } : cell)))
        }, 500)
      }
    },
    [cells],
  )

  // Handle cell updates from investigation events
  const handleUpdateCell = useCallback((cellId: string, updates: Partial<CellCreationParams>) => {
    setCells((prevCells) => prevCells.map((cell) => (cell.id === cellId ? { ...cell, ...updates } : cell)))
  }, [])

  // Handle errors from investigation events
  const handleInvestigationError = useCallback(
    (message: string) => {
      setError(message)
      toast({
        variant: "destructive",
        title: "Investigation Error",
        description: message,
      })
    },
    [toast],
  )

  // Initialize the investigation events hook
  const { wsStatus, isInvestigationRunning, currentPlan, currentStatus } = useInvestigationEvents({
    notebookId,
    onCreateCell: handleCreateCell,
    onUpdateCell: handleUpdateCell,
    onError: handleInvestigationError,
  })

  // Fetch notebook and cells data
  useEffect(() => {
    const fetchNotebookData = async () => {
      if (!notebookId) return

      setLoading(true)
      setError(null)

      try {
        // Fetch notebook details
        const notebookData = await api.notebooks.get(notebookId)
        setNotebook(notebookData)
        setEditedName(notebookData.name || notebookData.metadata?.title || "Untitled Notebook")
        setEditedDescription(notebookData.description || notebookData.metadata?.description || "")

        // Check if api.cells exists before trying to use it
        if (api.cells && typeof api.cells.list === "function") {
          try {
            // Fetch cells for this notebook
            const cellsData = await api.cells.list(notebookId)
            // Handle the case where cellsData might be an object instead of an array
            if (Array.isArray(cellsData)) {
              setCells(cellsData)
            } else if (notebookData.cells && typeof notebookData.cells === "object") {
              // If the notebook response includes cells as an object, convert to array
              const cellsArray = Object.values(notebookData.cells)
              setCells(cellsArray)
            } else {
              // Default to empty array if no cells found
              setCells([])
            }
          } catch (cellError) {
            console.error("Failed to fetch cells:", cellError)
            // If cells can't be fetched, check if they're included in the notebook response
            if (notebookData.cells && typeof notebookData.cells === "object") {
              const cellsArray = Object.values(notebookData.cells)
              setCells(cellsArray)
            } else {
              setCells([])
            }
          }
        } else {
          console.warn("api.cells.list is not available, checking for cells in notebook response")
          // If api.cells.list doesn't exist, check if cells are included in the notebook response
          if (notebookData.cells && typeof notebookData.cells === "object") {
            const cellsArray = Object.values(notebookData.cells)
            setCells(cellsArray)
          } else {
            setCells([])
          }
        }

        // Set the active notebook in the store
        useCanvasStore.getState().setActiveNotebook(notebookId)

        setLoading(false)
        initialized.current = true
      } catch (err) {
        console.error("Failed to fetch notebook data:", err)
        setError(err.message || "Failed to load canvas")
        setLoading(false)
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to load canvas data. Please try again.",
        })
      }
    }

    fetchNotebookData()
  }, [notebookId, toast])

  // Subscribe to the canvasStore to keep cells in sync
  useEffect(() => {
    console.log("🔄 Setting up canvasStore subscription for notebook:", notebookId)
    console.log(
      "🔄 Current cells in component state:",
      cells.map((c) => ({ id: c.id, type: c.type })),
    )

    // Get initial state from store
    const storeCells = useCanvasStore.getState().cells
    const activeNotebookId = useCanvasStore.getState().activeNotebookId
    console.log(
      "🔄 Initial store cells:",
      storeCells.map((c) => ({ id: c.id, type: c.type, notebook_id: c.notebook_id })),
    )
    console.log("🔄 Active notebook ID in store:", activeNotebookId)

    // Apply any cells from store that aren't in our local state
    if (storeCells.length > 0) {
      const relevantCells = storeCells.filter((cell) => cell.notebook_id === notebookId)
      if (relevantCells.length > 0) {
        console.log("🔄 Found relevant cells in store on init:", relevantCells.length)
        const existingIds = new Set(cells.map((cell) => cell.id))
        const newCells = relevantCells.filter((cell) => !existingIds.has(cell.id))

        if (newCells.length > 0) {
          console.log(
            "🔄 Adding cells from store on init:",
            newCells.map((c) => c.id),
          )
          setCells((prevCells) => [...prevCells, ...newCells])
        }
      }
    }

    // Setup subscription to canvasStore for cells
    const unsubscribe = useCanvasStore.subscribe(
      (state) => state.cells,
      (newCells, previousCells) => {
        console.log("🔔 Canvas store cells updated - callback triggered")
        console.log("🔔 Previous cells count:", previousCells?.length || 0)
        console.log("🔔 New cells count:", newCells.length)
        console.log("🔔 Active notebook ID in store:", useCanvasStore.getState().activeNotebookId)

        if (newCells.length && notebookId) {
          // Only update if we have cells in the store and they match our notebook
          const relevantCells = newCells.filter((cell) => cell.notebook_id === notebookId)
          console.log(
            "🔔 Relevant cells for this notebook:",
            relevantCells.map((c) => ({ id: c.id, type: c.type })),
          )
          console.log("🔔 Notebook ID we're filtering for:", notebookId)

          if (relevantCells.length) {
            // Merge new cells with existing ones, avoiding duplicates
            setCells((prevCells) => {
              const existingIds = new Set(prevCells.map((cell) => cell.id))
              console.log("🔔 Existing cell IDs in component:", Array.from(existingIds))

              const newUniqueItems = relevantCells.filter((cell) => !existingIds.has(cell.id))
              console.log(
                "🔔 New unique cells to add:",
                newUniqueItems.map((c) => ({ id: c.id, type: c.type })),
              )

              if (newUniqueItems.length > 0) {
                console.log("🔔 Adding new cells to component state")
                return [...prevCells, ...newUniqueItems]
              }
              console.log("🔔 No new cells to add")
              return prevCells
            })
          } else {
            console.log("🔔 No relevant cells found for this notebook")
          }
        } else {
          console.log("🔔 No cells in store or no notebook ID")
        }
      },
    )

    return () => {
      console.log("🧹 Cleaning up canvasStore subscription")
      unsubscribe()
    }
  }, [notebookId, cells])

  // Add a direct sync effect to pull cells from the store
  useEffect(() => {
    // Only run this after initial data load to avoid overwriting API data
    if (initialized.current && notebookId) {
      console.log("🔄 Syncing component with store cells")
      const storeCells = useCanvasStore.getState().cells
      const relevantCells = storeCells.filter((cell) => cell.notebook_id === notebookId)

      if (relevantCells.length > 0) {
        console.log("🔄 Found cells in store to sync:", relevantCells.length)

        // Merge store cells with component cells, avoiding duplicates
        setCells((prevCells) => {
          const existingIds = new Set(prevCells.map((cell) => cell.id))
          const newCells = relevantCells.filter((cell) => !existingIds.has(cell.id))

          if (newCells.length > 0) {
            console.log("🔄 Adding new cells from store sync:", newCells.length)
            return [...prevCells, ...newCells]
          }

          return prevCells
        })
      }
    }
  }, [useCanvasStore.getState().cells, notebookId])

  // Add a debug effect to log cells changes
  useEffect(() => {
    console.log("📊 Cells state changed, current count:", cells.length)
    console.log(
      "📊 Current cells:",
      cells.map((c) => ({ id: c.id, type: c.type })),
    )
  }, [cells])

  // Update notebook name
  const handleNameChange = async () => {
    if (!notebook || !editedName.trim()) return

    try {
      const updatedNotebook = await api.notebooks.update(notebookId, {
        name: editedName,
      })

      setNotebook(updatedNotebook)
      setIsEditing(false)
      toast({
        title: "Success",
        description: "Canvas name updated successfully.",
      })
    } catch (err) {
      console.error("Failed to update canvas name:", err)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to update canvas name. Please try again.",
      })
    }
  }

  // Update notebook description
  const handleDescriptionChange = async () => {
    if (!notebook) return

    try {
      const updatedNotebook = await api.notebooks.update(notebookId, {
        description: editedDescription,
      })

      setNotebook(updatedNotebook)
      setIsEditingDesc(false)
      toast({
        title: "Success",
        description: "Canvas description updated successfully.",
      })
    } catch (err) {
      console.error("Failed to update canvas description:", err)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to update canvas description. Please try again.",
      })
    }
  }

  // Add a new cell
  const handleAddCell = async (type, index) => {
    if (!notebookId) return

    try {
      // Set default content based on cell type
      let defaultContent = ""
      switch (type) {
        case "markdown":
          defaultContent = "# New markdown cell"
          break
        case "sql":
          defaultContent = "SELECT * FROM table LIMIT 10;"
          break
        case "python":
          defaultContent = "# Python code\nprint('Hello, world!')"
          break
        case "ai":
          defaultContent = ""
          break
        case "log":
          defaultContent = '{level=~"ERROR|CRITICAL"} [10m]'
          break
      }

      // Map "ai" type to "ai_query" for the API
      const apiCellType = type === "ai" ? "ai_query" : type

      // Create the cell via API
      const newCell = await api.cells.create(notebookId, {
        type: apiCellType, // Use the mapped cell type for the API
        content: defaultContent,
        status: "idle",
      })

      // Update local state with animation
      const updatedCells = [...cells]
      updatedCells.splice(index, 0, { ...newCell, isNew: true })
      setCells(updatedCells)

      // Remove the isNew flag after animation completes
      setTimeout(() => {
        setCells((cells) => cells.map((cell) => (cell.id === newCell.id ? { ...cell, isNew: false } : cell)))
      }, 500)

      toast({
        title: "Success",
        description: `New ${type} cell added.`,
      })
    } catch (err) {
      console.error("Failed to create cell:", err)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to create cell. Please try again.",
      })
    }
  }

  // Delete a cell
  const handleDeleteCell = async (cellId) => {
    if (!notebookId) return

    try {
      // Set the deleting cell ID to trigger animation
      setDeletingCellId(cellId)

      // Wait for animation to complete
      setTimeout(async () => {
        // Delete the cell via API
        await api.cells.delete(notebookId, cellId)

        // Update local state
        setCells(cells.filter((cell) => cell.id !== cellId))
        setDeletingCellId(null)

        toast({
          title: "Success",
          description: "Cell deleted successfully.",
        })
      }, 300)
    } catch (err) {
      console.error("Failed to delete cell:", err)
      setDeletingCellId(null)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete cell. Please try again.",
      })
    }
  }

  // Update cell content
  const handleCellContentChange = async (cellId, newContent) => {
    if (!notebookId) return

    try {
      // Find the cell to update
      const cellIndex = cells.findIndex((cell) => cell.id === cellId)
      if (cellIndex === -1) return

      // Update local state immediately for responsiveness
      const updatedCells = [...cells]
      updatedCells[cellIndex] = {
        ...updatedCells[cellIndex],
        content: newContent,
      }
      setCells(updatedCells)

      // Update the cell via API
      await api.cells.update(notebookId, cellId, {
        content: newContent,
      })
    } catch (err) {
      console.error("Failed to update cell content:", err)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to update cell content. Please try again.",
      })
    }
  }

  // Cell execution is handled by the API
  const handleCellRun = async (cellId) => {
    if (!notebookId) return

    try {
      // Find the cell to execute
      const cellIndex = cells.findIndex((cell) => cell.id === cellId)
      if (cellIndex === -1) return

      // Update local state to show running status
      const updatedCells = [...cells]
      updatedCells[cellIndex] = {
        ...updatedCells[cellIndex],
        status: "running",
      }
      setCells(updatedCells)

      // Execute the cell via API
      const result = await api.cells.execute(notebookId, cellId)

      // Update local state with the result
      updatedCells[cellIndex] = result
      setCells(updatedCells)

      toast({
        title: "Success",
        description: "Cell executed successfully.",
      })
    } catch (err) {
      console.error("Failed to execute cell:", err)

      // Update local state to show error
      const cellIndex = cells.findIndex((cell) => cell.id === cellId)
      if (cellIndex !== -1) {
        const updatedCells = [...cells]
        updatedCells[cellIndex] = {
          ...updatedCells[cellIndex],
          status: "error",
          error: err.message || "Execution failed",
        }
        setCells(updatedCells)
      }

      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to execute cell. Please try again.",
      })
    }
  }

  // Send a message to an AI cell
  const handleSendMessage = async (cellId, message) => {
    if (!notebookId) return

    try {
      // Find the cell
      const cellIndex = cells.findIndex((cell) => cell.id === cellId)
      if (cellIndex === -1) return

      const cell = cells[cellIndex]

      // Update local state to add user message and show running status
      const updatedCells = [...cells]
      const messages = cell.messages || []

      updatedCells[cellIndex] = {
        ...cell,
        messages: [...messages, { role: "user", content: message }],
        status: "running",
      }
      setCells(updatedCells)

      // Send the message to the API
      const result = await api.cells.sendMessage(notebookId, cellId, message)

      // Update local state with the result
      updatedCells[cellIndex] = result
      setCells(updatedCells)
    } catch (err) {
      console.error("Failed to send message:", err)

      // Update local state to show error
      const cellIndex = cells.findIndex((cell) => cell.id === cellId)
      if (cellIndex !== -1) {
        const updatedCells = [...cells]
        updatedCells[cellIndex] = {
          ...updatedCells[cellIndex],
          status: "error",
          error: err.message || "Failed to send message",
        }
        setCells(updatedCells)
      }

      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to send message. Please try again.",
      })
    }
  }

  // Run all cells sequentially through the API
  const handleRunAll = async () => {
    if (!notebookId || !cells.length) return

    try {
      // Execute cells sequentially
      for (const cell of cells) {
        if (cell.type !== "markdown" && cell.type !== "ai") {
          await handleCellRun(cell.id)
        }
      }

      toast({
        title: "Success",
        description: "All cells executed successfully.",
      })
    } catch (err) {
      console.error("Failed to run all cells:", err)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to run all cells. Some cells may have errors.",
      })
    }
  }

  // Toggle the chat panel
  const toggleChatPanel = () => {
    setIsChatPanelOpen(!isChatPanelOpen)
  }

  return (
    <div className={`p-6 pb-32 ${isChatPanelOpen ? "pr-96" : ""} transition-all duration-300`}>
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      ) : error ? (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : notebook ? (
        <>
          <div className="mb-8">
            {/* Notebook Title Section */}
            <div className="flex justify-between items-center border-b pb-4 mb-4">
              <div className="flex-1">
                {isEditing ? (
                  <div className="flex items-center">
                    <Input
                      value={editedName}
                      onChange={(e) => setEditedName(e.target.value)}
                      className="text-3xl font-bold max-w-md"
                      onBlur={handleNameChange}
                      onKeyDown={(e) => e.key === "Enter" && handleNameChange()}
                      autoFocus
                    />
                  </div>
                ) : (
                  <h1
                    className="text-3xl font-bold cursor-pointer hover:text-primary transition-colors duration-200"
                    onClick={() => setIsEditing(true)}
                  >
                    {notebook?.name || notebook?.metadata?.title || "Untitled Notebook"}
                  </h1>
                )}
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRunAll}
                  className="bg-blue-50 hover:bg-blue-100 border-blue-200"
                >
                  <Play className="h-4 w-4 mr-1 text-blue-600" />
                  Run All
                </Button>
                <Button variant="outline" size="sm" className="bg-green-50 hover:bg-green-100 border-green-200">
                  <Save className="h-4 w-4 mr-1 text-green-600" />
                  Save
                </Button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="bg-white border shadow-lg">
                    <DropdownMenuItem className="cursor-pointer">
                      <Share className="h-4 w-4 mr-2" />
                      Share
                    </DropdownMenuItem>
                    <DropdownMenuItem className="cursor-pointer">
                      <Download className="h-4 w-4 mr-2" />
                      Export
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>

            {/* WebSocket Status and Investigation Status */}
            <div className="mt-2 flex items-center">
              <div
                className={`w-2 h-2 rounded-full mr-2 ${
                  wsStatus === "connected" ? "bg-green-500" : wsStatus === "connecting" ? "bg-yellow-500" : "bg-red-500"
                }`}
              ></div>
              <span className="text-xs text-gray-500">
                {wsStatus === "connected" ? "Connected" : wsStatus === "connecting" ? "Connecting..." : "Disconnected"}
              </span>

              {isInvestigationRunning && (
                <div className="ml-4 flex items-center">
                  <div className="animate-spin h-3 w-3 border-2 border-purple-500 rounded-full border-t-transparent mr-2"></div>
                  <span className="text-xs text-purple-700">{currentStatus || "Investigation in progress..."}</span>
                </div>
              )}
            </div>

            {/* Notebook Description Section */}
            <div className="mt-2 mb-4">
              {isEditingDesc ? (
                <div>
                  <Textarea
                    value={editedDescription}
                    onChange={(e) => setEditedDescription(e.target.value)}
                    className="text-sm text-gray-500 min-h-[60px]"
                    onBlur={handleDescriptionChange}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleDescriptionChange()}
                    autoFocus
                    placeholder="Add a description..."
                  />
                </div>
              ) : (
                <p
                  className="text-gray-500 text-sm cursor-pointer hover:text-gray-700"
                  onClick={() => setIsEditingDesc(true)}
                >
                  {notebook.description || notebook.metadata?.description || "Add a description..."}
                </p>
              )}
            </div>
          </div>

          <div className="space-y-6">
            {cells.map((cell, index) => (
              <div
                key={cell.id}
                className={`cell-container ${deletingCellId === cell.id ? "deleted" : ""} ${
                  cell.isNew ? "cell-enter" : ""
                }`}
              >
                <Cell
                  cell={cell}
                  onDelete={() => handleDeleteCell(cell.id)}
                  onContentChange={(newContent) => handleCellContentChange(cell.id, newContent)}
                  onRun={() => handleCellRun(cell.id)}
                  onSendMessage={handleSendMessage}
                />
                {/* Only show CellCreationPills after the last cell */}
                {index === cells.length - 1 && (
                  <CellCreationPills onAddCell={(type) => handleAddCell(type, index + 1)} />
                )}
              </div>
            ))}
            {cells.length === 0 && (
              <div className="text-center py-12 bg-gray-50 rounded-lg">
                <h3 className="text-lg font-medium text-gray-900 mb-2">This canvas is empty</h3>
                <p className="text-gray-500 mb-4">Add your first cell to get started</p>
                <CellCreationPills onAddCell={(type) => handleAddCell(type, 0)} />
              </div>
            )}
          </div>
        </>
      ) : null}

      {/* AI Chat Panel */}
      <AIChatPanel isOpen={isChatPanelOpen} onToggle={toggleChatPanel} notebookId={notebookId} />

      {/* AI Chat Toggle Button */}
      <AIChatToggle isOpen={isChatPanelOpen} onToggle={toggleChatPanel} notebookId={notebookId} />
    </div>
  )
}
