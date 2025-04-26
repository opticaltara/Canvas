"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ChevronLeft, ChevronRight, Plus, LogOut, BookOpen, Briefcase, Plug } from "lucide-react"

interface Canvas {
  id: string
  name: string
}

interface LeftSidebarProps {
  workspace: string
  canvases: Canvas[]
  onAddCanvas: (name: string) => void
  onSelectCanvas: (id: string) => void
  onSignOut: () => void
  onNavigateToIntegrations: () => void
}

const LeftSidebar: React.FC<LeftSidebarProps> = ({
  workspace,
  canvases,
  onAddCanvas,
  onSelectCanvas,
  onSignOut,
  onNavigateToIntegrations,
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [newCanvasName, setNewCanvasName] = useState("")
  const [activeCanvasId, setActiveCanvasId] = useState("1") // Assuming "1" is the ID for "Payments Failure Investigation"

  const handleAddCanvas = () => {
    if (newCanvasName.trim()) {
      onAddCanvas(newCanvasName.trim())
      setNewCanvasName("")
    }
  }

  const handleSelectCanvas = (id: string) => {
    setActiveCanvasId(id)
    onSelectCanvas(id)
  }

  return (
    <div className={`bg-gray-100 flex flex-col transition-all duration-300 ${isCollapsed ? "w-20" : "w-72"} border-r`}>
      <div className="p-3 border-b bg-white flex items-center justify-between">
        <div className="flex items-center">
          <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold mr-2">
            A
          </div>
          {!isCollapsed && <h2 className="font-bold">Acme</h2>}
        </div>
        <Button
          variant="outline"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="rounded-full bg-white hover:bg-gray-100"
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </Button>
      </div>
      <div className="flex-grow overflow-y-auto p-4">
        <div className="mb-4">
          <div className="flex items-center mb-2">
            <Briefcase className="w-5 h-5 mr-2" />
            {!isCollapsed && <h3 className="font-semibold text-sm">Canvases</h3>}
          </div>
          {!isCollapsed && (
            <div className="flex space-x-2 mb-2">
              <Input
                placeholder="New canvas name"
                value={newCanvasName}
                onChange={(e) => setNewCanvasName(e.target.value)}
              />
              <Button onClick={handleAddCanvas} size="sm">
                <Plus className="w-4 h-4" />
              </Button>
            </div>
          )}
          <ul className="space-y-1">
            {canvases.map((canvas) => (
              <li key={canvas.id}>
                <Button
                  variant={activeCanvasId === canvas.id ? "secondary" : "ghost"}
                  className={`w-full justify-start ${isCollapsed ? "px-2" : ""} ${
                    activeCanvasId === canvas.id ? "bg-blue-100 text-blue-800" : ""
                  }`}
                  onClick={() => handleSelectCanvas(canvas.id)}
                >
                  <BookOpen className="w-4 h-4 mr-2" />
                  {!isCollapsed && canvas.name}
                </Button>
              </li>
            ))}
          </ul>
        </div>
        <div className="mb-4">
          <div className="flex items-center mb-2">
            <Plug className="w-5 h-5 mr-2" />
            {!isCollapsed && <h3 className="font-semibold text-sm">Integrations</h3>}
          </div>
          <ul className="space-y-1">
            <li>
              <Button
                variant="ghost"
                className={`w-full justify-start ${isCollapsed ? "px-2" : ""}`}
                onClick={onNavigateToIntegrations}
              >
                <Plug className="w-4 h-4 mr-2" />
                {!isCollapsed && "Manage Integrations"}
              </Button>
            </li>
          </ul>
        </div>
      </div>
      <div className="p-4 border-t">
        <Button variant="ghost" className={`w-full justify-start ${isCollapsed ? "px-2" : ""}`} onClick={onSignOut}>
          <LogOut className="w-5 h-5 mr-2" />
          {!isCollapsed && "Sign Out"}
        </Button>
      </div>
    </div>
  )
}

export default LeftSidebar
