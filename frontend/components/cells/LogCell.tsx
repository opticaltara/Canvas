"use client"

import type React from "react"
import { useState, useEffect } from "react"
import type { Cell } from "../../store/types"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Play, Trash, Maximize2, Minimize2, FileText } from "lucide-react"
import { Loader2 } from "lucide-react"
import Editor from "@monaco-editor/react"

interface LogCellProps {
  cell: Cell
  onExecute: (cellId: string) => void
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void
  onDelete: (cellId: string) => void
  isExecuting: boolean
}

const LogCell: React.FC<LogCellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }) => {
  const [content, setContent] = useState(cell.content || "")
  const [isExpanded, setIsExpanded] = useState(false)

  // Update content when prop changes
  useEffect(() => {
    if (cell.content !== content && cell.content !== undefined) {
      setContent(cell.content)
    }
  }, [cell.content])

  const handleExecute = () => {
    onUpdate(cell.id, content)
    onExecute(cell.id)
  }

  // Define custom theme for log queries
  const editorOptions = {
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    fontSize: 13,
    lineNumbers: "on",
    automaticLayout: true,
    wordWrap: "on",
  }

  // Define custom syntax highlighting for LogQL
  const beforeMount = (monaco: any) => {
    // Register a new language
    monaco.languages.register({ id: "logql" })

    // Define syntax highlighting rules
    monaco.languages.setMonarchTokensProvider("logql", {
      tokenizer: {
        root: [
          // Stream selectors
          [/{/, { token: "delimiter.curly", next: "@streamSelector" }],

          // Duration literals
          [/\b\d+[smhdwy]\b/, "number.duration"],

          // Numbers
          [/\b\d+\.\d+\b/, "number.float"],
          [/\b\d+\b/, "number"],

          // Operators
          [/[=!<>]=|[<>]|=~|!~/, "operator"],
          [/\|/, "operator.pipe"],

          // Keywords
          [/\b(by|without|on|ignoring|group_left|group_right|offset)\b/, "keyword"],
          [/\b(sum|avg|count|min|max|rate|count_over_time|rate_over_time)\b/, "keyword.function"],

          // Pipeline stages
          [/\|\s*(json|logfmt|regexp|pattern|line_format|label_format|unwrap)\b/, "keyword.pipe"],

          // Strings
          [/"([^"\\]|\\.)*"/, "string"],
          [/'([^'\\]|\\.)*'/, "string"],
        ],

        streamSelector: [
          [/}/, { token: "delimiter.curly", next: "@pop" }],
          [/[a-zA-Z_]\w*/, "identifier"],
          [/=|!=|=~|!~/, "operator"],
          [/"([^"\\]|\\.)*"/, "string"],
          [/'([^'\\]|\\.)*'/, "string"],
          [/,/, "delimiter"],
        ],
      },
    })
  }

  return (
    <Card className="mb-3 border-l-4 border-l-amber-500">
      <CardHeader className="py-1.5 px-3 bg-amber-50 text-amber-900 flex flex-row justify-between items-center">
        <div className="text-xs font-medium flex items-center">
          <FileText className="h-3 w-3 mr-1.5 text-amber-600" />
          Log Query
        </div>
        <div className="flex space-x-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? "Minimize" : "Maximize"}
            className="text-amber-700 hover:text-amber-900 hover:bg-amber-100 h-6 w-6 p-0"
          >
            {isExpanded ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleExecute}
            disabled={isExecuting}
            title="Run Query"
            className="text-amber-700 hover:text-amber-900 hover:bg-amber-100 h-6 w-6 p-0"
          >
            {isExecuting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onDelete(cell.id)}
            title="Delete"
            className="text-amber-700 hover:text-red-600 hover:bg-amber-100 h-6 w-6 p-0"
          >
            <Trash className="h-3 w-3" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="border-b bg-gray-900">
          <Editor
            height={isExpanded ? "200px" : "80px"}
            language="logql"
            value={content}
            onChange={(value) => setContent(value || "")}
            options={editorOptions}
            theme="vs-dark"
            beforeMount={beforeMount}
          />
        </div>

        {cell.result && (
          <div className="p-3 bg-gray-50">
            <pre className="text-xs overflow-auto whitespace-pre-wrap">
              {typeof cell.result === "string" ? cell.result : JSON.stringify(cell.result, null, 2)}
            </pre>
          </div>
        )}

        {cell.status === "error" && cell.error && (
          <div className="p-3 bg-red-50 text-red-800 rounded-b-md">
            <p className="font-semibold text-xs">Error:</p>
            <pre className="text-xs overflow-auto">{cell.error}</pre>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default LogCell
