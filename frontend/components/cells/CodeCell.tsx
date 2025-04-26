"use client"

import type React from "react"
import { useState } from "react"
import type { Cell } from "../../api/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Play, Trash } from "lucide-react"
import { Loader2 } from "lucide-react"
import Editor from "@monaco-editor/react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import CodeResultView from "../results/CodeResultView"

interface CodeCellProps {
  cell: Cell
  onExecute: (cellId: string) => void
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void
  onDelete: (cellId: string) => void
  isExecuting: boolean
}

const CodeCell: React.FC<CodeCellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }) => {
  const [content, setContent] = useState(cell.content)
  const [language, setLanguage] = useState<string>(cell.metadata?.language || "python")

  const handleExecute = () => {
    // Update the cell content and metadata before executing
    const updatedMetadata = {
      ...cell.metadata,
      language,
    }

    // First update the cell content and metadata
    onUpdate(cell.id, content, updatedMetadata)

    // Then execute the cell
    onExecute(cell.id)
  }

  return (
    <Card className="mb-4">
      <CardHeader className="py-2 px-4 bg-gray-50 flex flex-row justify-between items-center">
        <div className="text-sm font-medium">Code</div>
        <div className="flex space-x-2">
          <Select value={language} onValueChange={setLanguage}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Language" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="python">Python</SelectItem>
              <SelectItem value="javascript">JavaScript</SelectItem>
              <SelectItem value="bash">Bash</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="ghost" size="sm" onClick={handleExecute} disabled={isExecuting} title="Run Code">
            {isExecuting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          </Button>

          <Button variant="ghost" size="sm" onClick={() => onDelete(cell.id)} title="Delete">
            <Trash className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="border-b">
          <Editor
            height="200px"
            language={language}
            value={content}
            onChange={(value) => setContent(value || "")}
            options={{
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              fontSize: 14,
              lineNumbers: "on",
              automaticLayout: true,
            }}
          />
        </div>

        {cell.result && (
          <div className="p-4">
            <CodeResultView result={cell.result} language={language} />
          </div>
        )}

        {cell.status === "error" && cell.error && (
          <div className="p-4 bg-red-50 text-red-800 rounded-b-md">
            <p className="font-semibold">Error:</p>
            <pre className="text-sm overflow-auto">{cell.error}</pre>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default CodeCell
