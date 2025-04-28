"use client"

import type React from "react"
import { useState } from "react"
import type { Cell } from "../../api/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Edit, Save, Trash } from "lucide-react"
import ReactMarkdown from "react-markdown"

interface MarkdownCellProps {
  cell: Cell
  onExecute: (cellId: string) => void
  onUpdate: (cellId: string, content: string) => void
  onDelete: (cellId: string) => void
  isExecuting: boolean
}

const MarkdownCell: React.FC<MarkdownCellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }) => {
  const [isEditing, setIsEditing] = useState(false)
  const [content, setContent] = useState(cell.content)

  const handleSave = () => {
    onUpdate(cell.id, content)
    setIsEditing(false)
    onExecute(cell.id)
  }

  return (
    <Card className="mb-4">
      <CardHeader className="py-2 px-4 bg-gray-50 flex flex-row justify-between items-center">
        <div className="text-sm font-medium">Markdown</div>
        <div className="flex space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsEditing(!isEditing)}
            title={isEditing ? "Save" : "Edit"}
          >
            {isEditing ? <Save className="h-4 w-4" /> : <Edit className="h-4 w-4" />}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => onDelete(cell.id)} title="Delete">
            <Trash className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-4">
        {isEditing ? (
          <Textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="min-h-[150px] font-mono"
            placeholder="Enter markdown content..."
          />
        ) : (
          <div className="prose text-sm max-w-none max-h-60 overflow-y-auto">
            <ReactMarkdown>{cell.content.replace(/\\n/g, '\n')}</ReactMarkdown>
          </div>
        )}
      </CardContent>
      {isEditing && (
        <CardFooter className="flex justify-end py-2 px-4 bg-gray-50">
          <Button
            variant="outline"
            size="sm"
            className="mr-2"
            onClick={() => {
              setContent(cell.content)
              setIsEditing(false)
            }}
          >
            Cancel
          </Button>
          <Button size="sm" onClick={handleSave}>
            Save
          </Button>
        </CardFooter>
      )}
    </Card>
  )
}

export default MarkdownCell
