"use client"

import type React from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle, AlertCircle, Loader2, Clock } from "lucide-react"
import type { Cell } from "../../store/types" // Use the updated Cell type

interface SummarizationCellProps {
  cell: Cell
  // No onExecute, onUpdate, or isExecuting needed for display-only
  onDelete: (cellId: string) => void
}

const SummarizationCell: React.FC<SummarizationCellProps> = ({ cell, onDelete }) => {
  const renderStatusIcon = () => {
    switch (cell.status) {
      case "success":
        return <CheckCircle className="h-4 w-4 text-green-600" />
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-600" />
      case "running":
      case "queued":
        return <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
      case "stale":
         return <Clock className="h-4 w-4 text-gray-500" />
      case "idle":
      default:
        return null // Or a default icon if preferred
    }
  }

  return (
    <Card className="mb-4 border-l-4 border-purple-400 bg-purple-50/30 mx-8">
      <CardHeader className="py-2 px-4 flex flex-row justify-between items-center">
        <div className="flex items-center space-x-2">
          <CardTitle className="text-sm font-medium text-purple-800">AI Summary</CardTitle>
          {renderStatusIcon()}
        </div>
        {/* Add Delete button if needed */}
        {/* <Button variant="ghost" size="sm" onClick={() => onDelete(cell.id)} title="Delete Summary">
          <Trash2Icon className="h-4 w-4 text-gray-500 hover:text-red-600" />
        </Button> */}
      </CardHeader>
      <CardContent className="p-4 prose prose-sm max-w-none">
        {/* Display error message if present */}
        {cell.result?.error && (
          <div className="mb-2 p-2 border border-red-200 bg-red-50 text-red-700 rounded">
            <p className="font-semibold">Error generating summary:</p>
            <p>{cell.result.error}</p>
          </div>
        )}
        {/* Display the main content using ReactMarkdown */}
        {cell.content ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{cell.content}</ReactMarkdown>
        ) : (
          <p className="text-gray-500 italic">Summary content is empty.</p>
        )}
      </CardContent>
    </Card>
  )
}

export default SummarizationCell 