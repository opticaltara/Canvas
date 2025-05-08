"use client"

import type React from "react"
import { useState } from "react"
import { ShareIcon, CopyIcon, PlayCircleIcon, Database, ChevronsUpDownIcon } from "lucide-react"
import dynamic from 'next/dynamic'
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
// import DataConnectionsDialog from "./DataConnectionsDialog" // Will be dynamically imported
import { useConnectionStore } from "@/store/connectionStore"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

const DataConnectionsDialog = dynamic(() => import('./DataConnectionsDialog'), {
  ssr: false,
  loading: () => null, // Or a proper loading skeleton/spinner
});

interface CanvasHeaderProps {
  name: string
  onNameChange: (newName: string) => void
}

const CanvasHeader: React.FC<CanvasHeaderProps> = ({ name, onNameChange }) => {
  const [isEditing, setIsEditing] = useState(false)
  const [editedName, setEditedName] = useState(name)
  const [isConnectionsDialogOpen, setIsConnectionsDialogOpen] = useState(false)
  const { connections, mcpStatuses } = useConnectionStore()
  const areAllCellsExpanded = useConnectionStore((state) => state.areAllCellsExpanded)
  const toggleAllCellsExpanded = useConnectionStore((state) => state.toggleAllCellsExpanded)

  // Count active connections
  const activeConnections = connections.filter((conn) => mcpStatuses[conn.id]?.status === "running").length

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditedName(e.target.value)
  }

  const handleNameSubmit = () => {
    onNameChange(editedName)
    setIsEditing(false)
  }

  return (
    <div className="flex justify-between items-center p-4 border-b">
      {isEditing ? (
        <Input
          value={editedName}
          onChange={handleNameChange}
          onBlur={handleNameSubmit}
          onKeyPress={(e) => e.key === "Enter" && handleNameSubmit()}
          className="text-2xl font-bold w-1/2"
          autoFocus
        />
      ) : (
        <h1 className="text-2xl font-bold cursor-pointer" onClick={() => setIsEditing(true)}>
          {name}
        </h1>
      )}
      <div className="flex items-center space-x-4">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsConnectionsDialogOpen(true)}
          className="flex items-center gap-2"
        >
          <Database className="h-4 w-4" />
          <span>Data Connections</span>
          {activeConnections > 0 && (
            <span className="inline-flex items-center justify-center w-5 h-5 text-xs font-medium bg-primary text-primary-foreground rounded-full">
              {activeConnections}
            </span>
          )}
        </Button>
        <div className="flex -space-x-2">
          <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
            NM
          </div>
          <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
            SS
          </div>
        </div>
        <div className="flex space-x-2">
          <TooltipProvider delayDuration={100}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-gray-600 hover:text-gray-800" onClick={toggleAllCellsExpanded}>
                  <ChevronsUpDownIcon className="w-5 h-5" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{areAllCellsExpanded ? "Collapse All Cells" : "Expand All Cells"}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider delayDuration={100}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-gray-600 hover:text-gray-800">
                  <ShareIcon className="w-5 h-5" />
                </button>
              </TooltipTrigger>
              <TooltipContent><p>Share Canvas (soon)</p></TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider delayDuration={100}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-gray-600 hover:text-gray-800">
                  <CopyIcon className="w-5 h-5" />
                </button>
              </TooltipTrigger>
              <TooltipContent><p>Copy Canvas Link (soon)</p></TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider delayDuration={100}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-gray-600 hover:text-gray-800">
                  <PlayCircleIcon className="w-5 h-5" />
                </button>
              </TooltipTrigger>
              <TooltipContent><p>Run All Cells (soon)</p></TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      <DataConnectionsDialog isOpen={isConnectionsDialogOpen} onClose={() => setIsConnectionsDialogOpen(false)} />
    </div>
  )
}

export default CanvasHeader
