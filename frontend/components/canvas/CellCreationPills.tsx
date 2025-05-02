"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Plus, FileText, Github } from "lucide-react"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"

interface CellCreationPillsProps {
  onAddCell: (type: string) => void
}

export default function CellCreationPills({ onAddCell }: CellCreationPillsProps) {
  const [isOpen, setIsOpen] = useState(false)

  const handleAddCell = (type: string) => {
    onAddCell(type)
    setIsOpen(false)
  }

  return (
    <div className="flex justify-center my-2 transition-all duration-300 hover:scale-105">
      <Popover open={isOpen} onOpenChange={setIsOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            className="rounded-full w-8 h-8 p-0 bg-white border border-gray-200 hover:bg-blue-50 hover:border-blue-300 transition-all duration-300 shadow-sm hover:shadow"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-56 p-2 bg-white border shadow-lg animate-suggestion-fade-in">
          <div className="grid gap-1">
            <Button
              variant="ghost"
              className="flex justify-start items-center px-2 py-1 text-sm hover:bg-blue-50 transition-colors duration-200"
              onClick={() => handleAddCell("markdown")}
            >
              <FileText className="h-4 w-4 mr-2 text-blue-500" />
              <span>Markdown</span>
            </Button>
            <Button
              variant="ghost"
              className="flex justify-start items-center px-2 py-1 text-sm hover:bg-blue-50 transition-colors duration-200"
              onClick={() => handleAddCell("github")}
            >
              <Github className="h-4 w-4 mr-2 text-gray-500" />
              <span>Github Tool</span>
            </Button>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}
