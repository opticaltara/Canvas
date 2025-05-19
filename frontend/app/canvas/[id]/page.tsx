"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Play } from "lucide-react"
import dynamic from 'next/dynamic'
import CellCreationPills from "@/components/canvas/CellCreationPills"
// import AIChatPanel from "@/components/AIChatPanel" // Will be dynamically imported
import AIChatToggle from "@/components/AIChatToggle"
import { api } from "@/api/client"
import { useToast } from "@/hooks/use-toast"
import { useInvestigationEvents, type CellCreationParams } from "@/hooks/useInvestigationEvents"
import { useCanvasStore, type NotebookState } from "@/store/canvasStore"
import { type Cell, type CellType } from "@/store/types"
import CellFactory from "@/components/cells/CellFactory"

const AIChatPanel = dynamic(() => import('@/components/AIChatPanel'), {
  ssr: false, // Typically, chat panels are client-side heavy and don't need SSR
  loading: () => <div className="fixed bottom-4 right-4 p-4 bg-gray-700 text-white rounded-lg shadow-lg">Loading Chat...</div>,
});

type DisplayCell = Cell & { isNew?: boolean }

export default function CanvasPage() {
  const params = useParams()
  const notebookId = params.id as string
  const { toast } = useToast()

  const [notebook, setNotebook] = useState<any | null>(null)
  const [cells, setCells] = useState<DisplayCell[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [isEditingDesc, setIsEditingDesc] = useState(false)
  const [editedName, setEditedName] = useState("")
  const [editedDescription, setEditedDescription] = useState("")
  const [deletingCellId, setDeletingCellId] = useState<string | null>(null)
  const [isChatPanelOpen, setIsChatPanelOpen] = useState(false)
  const [isFilteredView, setIsFilteredView] = useState(false)

  const storeCells = useCanvasStore((state) => state.cells)
  const setStoreActiveNotebook = useCanvasStore((state) => state.setActiveNotebook)

  const initialized = useRef(false)

  useEffect(() => {
    if (notebookId) {
      console.log("🔑 Setting active notebook ID in store:", notebookId)
      setStoreActiveNotebook(notebookId)
    }
  }, [notebookId, setStoreActiveNotebook])

  // Effect to scroll to new cells
  useEffect(() => {
    // Find the *last* cell in the array that is marked as new.
    const newCells = cells.filter(c => c.isNew);
    if (newCells.length > 0) {
      const cellToScrollTo = newCells[newCells.length - 1]; // Scroll to the latest new cell
      console.log(`[CanvasPage] Attempting to scroll to new cell: ${cellToScrollTo.id}`);
      setTimeout(() => { // Timeout to ensure DOM is updated
        const element = document.getElementById(`cell-${cellToScrollTo.id}`);
        if (element) {
          console.log(`[CanvasPage] Scrolling to element:`, element);
          element.scrollIntoView({ behavior: "smooth", block: "end" });
        } else {
          console.warn(`[CanvasPage] Could not find element for new cell ${cellToScrollTo.id} to scroll to.`);
        }
      }, 100); // A small delay for rendering.
    }
  }, [cells]); // Trigger when cells array changes

  const handleCreateCell = useCallback(
    (params: CellCreationParams) => {
      console.log(`🔍 handleCreateCell called. Step ID: ${params.step_id}, Cell ID: ${params.id}. Full params:`, params);
      
      setCells((prevCells) => {
        const existingCellIndex = prevCells.findIndex((cell) => cell.metadata?.step_id === params.step_id);
        if (existingCellIndex >= 0) {
          console.log(`🔄 Updating existing cell linked to step ID: ${params.step_id} with definitive Cell ID: ${params.id}`);
          const updatedCells = [...prevCells];
          const existingCell = updatedCells[existingCellIndex];
          const { id, step_id, metadata, ...restParams } = params;
          updatedCells[existingCellIndex] = {
            ...existingCell,
            ...restParams,
            id: params.id,
            type: params.type as CellType,
            metadata: {
              ...existingCell.metadata,
              ...(metadata || {}),
              step_id: params.step_id,
            },
            updated_at: new Date().toISOString(),
            isNew: false, 
          };
          return updatedCells;
        } else {
          console.log(`✨ Creating new cell for step ID: ${params.step_id} with definitive Cell ID: ${params.id}`);
          const now = new Date().toISOString();
          const { id, step_id, metadata, ...restParamsWithoutMeta } = params;
          const newCell: DisplayCell = {
            id: params.id,
            ...restParamsWithoutMeta,
            type: params.type as CellType,
            notebook_id: notebookId, // notebookId is stable from useParams
            created_at: now,
            updated_at: now,
            metadata: {
              ...(metadata || {}),
              step_id: params.step_id,
            },
            isNew: true,
          };
          return [...prevCells, newCell];
        }
      });

      // Animation timeout uses the correct UUID (params.id)
      // This needs to be handled carefully if setCells is async or batched.
      // For now, let's assume it's okay, but if animation is glitchy, this might need adjustment.
      setTimeout(() => {
        setCells((prevCells) =>
          prevCells.map((cell) =>
            cell.id === params.id ? { ...cell, isNew: false } : cell
          )
        );
      }, 500);
    },
    [notebookId] // `notebookId` from `useParams` is stable for the page's lifetime.
  );

  const handleUpdateCell = useCallback((cellId: string, updates: Partial<CellCreationParams>) => {
    console.log(`🔄 handleUpdateCell called for Cell ID: ${cellId} with updates:`, updates);
    setCells((prevCells) =>
      prevCells.map((cell): DisplayCell => {
        if (cell.id === cellId) {
          const updatedCell: DisplayCell = {
            ...cell,
            content: updates.content !== undefined ? updates.content : cell.content,
            status: updates.status !== undefined ? updates.status : cell.status,
            result: updates.result !== undefined ? updates.result : cell.result,
            error: updates.error !== undefined ? updates.error : cell.error,
            type: updates.type !== undefined ? (updates.type as CellType) : cell.type,
            id: cell.id,
            notebook_id: cell.notebook_id,
            created_at: cell.created_at,
            updated_at: new Date().toISOString(),
            metadata: {
              ...cell.metadata,
              ...updates.metadata,
            },
          }
          return updatedCell
        }
        return cell
      }),
    )
  }, [])

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

  const { wsStatus, isInvestigationRunning, currentStatus } = useInvestigationEvents({
    notebookId,
    onCreateCell: handleCreateCell,
    onUpdateCell: handleUpdateCell,
    onError: handleInvestigationError,
  })

  useEffect(() => {
    const fetchNotebookData = async () => {
      if (!notebookId) return

      setLoading(true)
      setError(null)

      try {
        const notebookData = await api.notebooks.get(notebookId)
        setNotebook(notebookData)
        setEditedName(notebookData.name || notebookData.metadata?.title || "Untitled Notebook")
        setEditedDescription(notebookData.description || notebookData.metadata?.description || "")

        if (api.cells && typeof api.cells.list === "function") {
          try {
            const cellsData = await api.cells.list(notebookId)
            if (Array.isArray(cellsData)) {
              console.log("[CanvasPage] Cells loaded from API:", JSON.stringify(cellsData.map(c => ({ id: c.id, type: c.type, result: !!c.result, tool_name: c.tool_name, status: c.status })), null, 2)); // Log API cell data
              setCells(cellsData)
            } else if (notebookData.cells && typeof notebookData.cells === "object") {
              const cellsArray = Object.values(notebookData.cells)
              setCells(cellsArray)
            } else {
              setCells([])
            }
          } catch (cellError) {
            console.error("Failed to fetch cells:", cellError)
            if (notebookData.cells && typeof notebookData.cells === "object") {
              const cellsArray = Object.values(notebookData.cells)
              setCells(cellsArray)
            } else {
              setCells([])
            }
          }
        } else {
          console.warn("api.cells.list is not available, checking for cells in notebook response")
          if (notebookData.cells && typeof notebookData.cells === "object") {
            const cellsArray = Object.values(notebookData.cells)
            setCells(cellsArray)
          } else {
            setCells([])
          }
        }

        setStoreActiveNotebook(notebookId)

        setLoading(false)
        initialized.current = true
      } catch (err: unknown) {
        console.error("Failed to fetch notebook data:", err)
        setError(err instanceof Error ? err.message : "Failed to load canvas")
        setLoading(false)
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to load canvas data. Please try again.",
        })
      }
    }

    fetchNotebookData()
  }, [notebookId, toast, setStoreActiveNotebook])

  useEffect(() => {
    console.log("🔄 Setting up canvasStore subscription for notebook:", notebookId)
    console.log(
      "🔄 Current cells in component state:",
      cells.map((c) => ({ id: c.id, type: c.type })),
    )

    const storeCells = useCanvasStore.getState().cells
    const activeNotebookId = useCanvasStore.getState().activeNotebookId
    console.log(
      "🔄 Initial store cells:",
      storeCells.map((c) => ({ id: c.id, type: c.type, notebook_id: c.notebook_id })),
    )
    console.log("🔄 Active notebook ID in store:", activeNotebookId)

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

    const unsubscribe = useCanvasStore.subscribe(
      (state, prevState) => {
        const newCells = state.cells
        const previousCells = prevState.cells
        console.log("🔔 Canvas store cells updated - callback triggered")
        console.log("🔔 Previous cells count:", previousCells?.length || 0)
        console.log("🔔 New cells count:", newCells.length)
        console.log("🔔 Active notebook ID in store:", useCanvasStore.getState().activeNotebookId)

        if (newCells.length && notebookId) {
          const relevantCells = newCells.filter((cell: Cell) => cell.notebook_id === notebookId)
          console.log(
            "🔔 Relevant cells for this notebook:",
            relevantCells.map((c: Cell) => ({ id: c.id, type: c.type })),
          )
          console.log("🔔 Notebook ID we're filtering for:", notebookId)

          if (relevantCells.length) {
            setCells((prevCells) => {
              const existingIds = new Set(prevCells.map((cell: DisplayCell) => cell.id))
              console.log("🔔 Existing cell IDs in component:", Array.from(existingIds))

              const newUniqueItems = relevantCells.filter((cell: Cell) => !existingIds.has(cell.id))
              console.log(
                "🔔 New unique cells to add:",
                newUniqueItems.map((c: Cell) => ({ id: c.id, type: c.type })),
              )

              if (newUniqueItems.length === 0) {
                // No new cells; avoid triggering a state update to prevent redundant renders.
                console.log("🔔 No new cells to add — skipping setCells call")
                return prevCells
              }

              console.log("🔔 Adding new cells to component state")
              return [...prevCells, ...newUniqueItems]
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
  }, [notebookId])

  useEffect(() => {
    if (initialized.current && notebookId) {
      console.log("🔄 Syncing component with store cells")
      const storeCells = useCanvasStore.getState().cells
      const relevantCells = storeCells.filter((cell) => cell.notebook_id === notebookId)

      if (relevantCells.length > 0) {
        console.log("🔄 Found cells in store to sync:", relevantCells.length)

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
  }, [notebookId])

  useEffect(() => {
    console.log("📊 Cells state changed, current count:", cells.length)
    console.log(
      "📊 Current cells:",
      cells.map((c) => ({ id: c.id, type: c.type })),
    )
  }, [cells])

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

  const handleAddCell = async (type: CellType, index: number) => {
    if (!notebookId) return

    try {
      let defaultContent = ""
      switch (type) {
        case "markdown":
          defaultContent = "# New markdown cell"
          break
        case "github":
          defaultContent = "{}"
          break
        case "filesystem":
          defaultContent = "list_dir ."
          break
      }

      const apiCellType = type

      const newCell = await api.cells.create(notebookId, {
        type: apiCellType,
        content: defaultContent,
        status: "idle",
      })

      const updatedCells = [...cells]
      updatedCells.splice(index, 0, { ...newCell, isNew: true })
      setCells(updatedCells)

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

  const handleDeleteCell = async (cellId: string) => {
    if (!notebookId) return

    try {
      setDeletingCellId(cellId)

      setTimeout(async () => {
        await api.cells.delete(notebookId, cellId)

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

  const handleCellContentChange = async (cellId: string, newContent: string, metadata?: Record<string, any>) => {
    if (!notebookId) return

    try {
      const cellIndex = cells.findIndex((cell) => cell.id === cellId)
      if (cellIndex === -1) return

      const updatedCells = [...cells]
      updatedCells[cellIndex] = {
        ...updatedCells[cellIndex],
        content: newContent,
        metadata: {
          ...updatedCells[cellIndex].metadata,
          ...metadata,
        },
      }
      setCells(updatedCells)

      // ⚠️ Removed immediate backend update on every keystroke.
      // The latest content/metadata will be persisted when the cell is executed
      // or blurred (future improvement).
    } catch (err) {
      console.error("Failed to update cell content:", err)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to update cell content. Please try again.",
      })
    }
  }

  const handleCellRun = async (cellId: string) => {
    if (!notebookId) return

    try {
      const cellIndex = cells.findIndex((cell) => cell.id === cellId)
      if (cellIndex === -1) return

      const updatedCells = [...cells]
      updatedCells[cellIndex] = {
        ...updatedCells[cellIndex],
        status: "queued",
        result: undefined,
      }
      setCells(updatedCells)

      // --- Flush latest content & metadata before execution ---
      try {
        await api.cells.update(notebookId, cellId, {
          content: updatedCells[cellIndex].content,
          metadata: updatedCells[cellIndex].metadata,
        })
      } catch (updateErr) {
        console.warn("[handleCellRun] Failed to autosave cell before execution:", updateErr)
        // We continue to execute even if autosave failed – backend may still have older version.
      }

      const toolArgsFromState = updatedCells[cellIndex].metadata?.toolArgs

      const requestBody = { tool_arguments: toolArgsFromState || {} }
      const result = await api.cells.execute(notebookId, cellId, requestBody)

      toast({
        title: "Cell Queued",
        description: "Cell has been queued for execution.",
      })
    } catch (err: unknown) {
      console.error("Failed to execute cell:", err)

      const cellIndex = cells.findIndex((cell) => cell.id === cellId)
      if (cellIndex !== -1) {
        const updatedCells = [...cells]
        updatedCells[cellIndex] = {
          ...updatedCells[cellIndex],
          status: "error",
          error: err instanceof Error ? err.message : "Execution failed",
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

  const toggleChatPanel = () => {
    setIsChatPanelOpen(!isChatPanelOpen)
  }

  return (
    <div className={`p-6 pb-32 ${isChatPanelOpen ? "pr-[440px]" : ""} transition-all duration-300`}>
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
            <div className="flex justify-between items-center border-b pb-4 mb-4">
              <div className="flex items-center flex-1 mr-4">
                {isEditing ? (
                  <div className="flex items-center">
                    <Input
                      value={editedName}
                      onChange={(e) => setEditedName(e.target.value)}
                      className="text-3xl font-bold"
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
                <div className="ml-4 flex items-center flex-shrink-0">
                  <div
                    className={`w-2 h-2 rounded-full mr-2 ${
                      wsStatus === "connected" ? "bg-green-500" : wsStatus === "connecting" ? "bg-yellow-500" : "bg-red-500"
                    }`}
                  ></div>
                  <span className="text-xs text-gray-500 mr-4">
                    {wsStatus === "connected" ? "Connected" : wsStatus === "connecting" ? "Connecting..." : "Disconnected"}
                  </span>

                  {isInvestigationRunning && (
                    <div className="flex items-center">
                      <div className="animate-spin h-3 w-3 border-2 border-purple-500 rounded-full border-t-transparent mr-2"></div>
                      <span className="text-xs text-purple-700">{currentStatus || "Investigation in progress..."}</span>
                    </div>
                  )}
                </div>
              </div>
              <div className="flex items-center space-x-2">
                {cells.length > 0 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsFilteredView(!isFilteredView)}
                    className="ml-auto"
                  >
                    {isFilteredView ? "Show All Cells" : "Show Key Cells"}
                  </Button>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            {(isFilteredView
              ? cells.filter(
                  (cell) =>
                    cell.type === "markdown" ||
                    cell.type === "media_timeline" ||
                    cell.type === "investigation_report",
                )
              : cells
            ).map((cell, index, displayedCellsArray) => (
              <div
                key={cell.id}
                id={`cell-${cell.id}`}
                className={`cell-container ${deletingCellId === cell.id ? "deleted" : ""} ${cell.isNew ? "cell-enter" : ""}`}
              >
                <CellFactory
                  cell={cell}
                  onExecute={() => handleCellRun(cell.id)}
                  onUpdate={handleCellContentChange}
                  onDelete={() => handleDeleteCell(cell.id)}
                  isExecuting={useCanvasStore.getState().isExecuting(cell.id)}
                />
                {index === displayedCellsArray.length - 1 && !isFilteredView && (
                  <CellCreationPills
                    onAddCell={(type: string) => {
                      // Calculate the actual index in the original 'cells' array
                      const originalIndex =
                        cells.length > 0
                          ? cells.findIndex((c) => c.id === cell.id) + 1
                          : 0
                      handleAddCell(type as CellType, originalIndex)
                    }}
                  />
                )}
              </div>
            ))}
            {(isFilteredView
              ? cells.filter(
                  (cell) =>
                    cell.type === "markdown" ||
                    cell.type === "media_timeline" ||
                    cell.type === "investigation_report",
                )
              : cells
            ).length === 0 && (
              <div className="text-center py-12 bg-gray-50 rounded-lg">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {isFilteredView ? "No key cells to display" : "This canvas is empty"}
                </h3>
                <p className="text-gray-500 mb-4">
                  {isFilteredView
                    ? "Switch to 'Show All Cells' view or add a Markdown, Mediatimeline, or Investigation Report cell."
                    : "Add your first cell to get started"}
                </p>
                {!isFilteredView && (
                  <CellCreationPills
                    onAddCell={(type: string) => {
                      handleAddCell(type as CellType, 0)
                    }}
                  />
                )}
              </div>
            )}
          </div>
        </>
      ) : null}

      <AIChatPanel isOpen={isChatPanelOpen} onToggle={toggleChatPanel} notebookId={notebookId} />

      <AIChatToggle isOpen={isChatPanelOpen} onToggle={toggleChatPanel} notebookId={notebookId} />
    </div>
  )
}
