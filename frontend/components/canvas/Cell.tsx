"\"use client"

import type React from "react"
import { useEffect, useRef, useCallback } from "react"
import { PlayIcon, CheckIcon, XIcon, TrashIcon } from "lucide-react"
import { basicSetup } from "codemirror"
import { EditorView, type ViewUpdate } from "@codemirror/view"
import { EditorState } from "@codemirror/state"
import { indentWithTab } from "@codemirror/commands"
import { python } from "@codemirror/lang-python"
import { sql } from "@codemirror/lang-sql"
import { javascript } from "@codemirror/lang-javascript"
import { markdown } from "@codemirror/lang-markdown"
import { autocompletion, type CompletionContext } from "@codemirror/autocomplete"
import { keymap } from "@codemirror/view"
import TremorChart from "../TremorChart"
import ResultTable from "../ResultTable"
import { LogView } from "../LogView"
import AICell from "./AICell"

// Custom completions for different languages
const pythonCompletions = {
  python: [
    { label: "import", type: "keyword" },
    { label: "def", type: "keyword" },
    { label: "class", type: "keyword" },
    { label: "return", type: "keyword" },
    { label: "print", type: "function" },
    { label: "if", type: "keyword" },
    { label: "else", type: "keyword" },
    { label: "elif", type: "keyword" },
    { label: "for", type: "keyword" },
    { label: "while", type: "keyword" },
    { label: "try", type: "keyword" },
    { label: "except", type: "keyword" },
    { label: "finally", type: "keyword" },
    { label: "with", type: "keyword" },
    { label: "as", type: "keyword" },
    { label: "lambda", type: "keyword" },
    // Common libraries
    { label: "pandas", type: "variable", info: "Data manipulation library" },
    { label: "numpy", type: "variable", info: "Numerical computing library" },
    { label: "matplotlib", type: "variable", info: "Plotting library" },
    { label: "plt", type: "variable", info: "Matplotlib pyplot module" },
    { label: "sklearn", type: "variable", info: "Machine learning library" },
    { label: "tensorflow", type: "variable", info: "Deep learning library" },
    { label: "torch", type: "variable", info: "PyTorch deep learning library" },
  ],
}

const sqlCompletions = {
  sql: [
    { label: "SELECT", type: "keyword" },
    { label: "FROM", type: "keyword" },
    { label: "WHERE", type: "keyword" },
    { label: "GROUP BY", type: "keyword" },
    { label: "ORDER BY", type: "keyword" },
    { label: "HAVING", type: "keyword" },
    { label: "JOIN", type: "keyword" },
    { label: "LEFT JOIN", type: "keyword" },
    { label: "RIGHT JOIN", type: "keyword" },
    { label: "INNER JOIN", type: "keyword" },
    { label: "LIMIT", type: "keyword" },
    { label: "OFFSET", type: "keyword" },
    { label: "INSERT INTO", type: "keyword" },
    { label: "UPDATE", type: "keyword" },
    { label: "DELETE FROM", type: "keyword" },
    { label: "CREATE TABLE", type: "keyword" },
    { label: "ALTER TABLE", type: "keyword" },
    { label: "DROP TABLE", type: "keyword" },
  ],
}

interface CellProps {
  type: string
  content: string
  status: "running" | "success" | "error"
  output: any
  onDelete: () => void
  creator: string
  chartData?: {
    data: any[]
    type: "area" | "bar" | "line" | "donut"
    categories: string[]
    index: string
    title: string
  }
  onContentChange: (newContent: string) => void
  onRun: () => void
  hasComments: boolean // Keep this prop but we'll ignore it
  isHighlighted: boolean
  onAddComment: () => void // Keep this prop but we'll use an empty function
  onShareToSlack: () => void
  cellRef: React.Ref<HTMLDivElement>
  id: string
  onSendMessage?: (cellId: string, message: string) => void
}

const Cell: React.FC<CellProps> = ({
  type,
  content,
  status,
  output,
  onDelete,
  creator,
  chartData,
  onContentChange,
  onRun,
  hasComments, // Keep but ignore
  isHighlighted,
  onAddComment, // Keep but ignore
  onShareToSlack,
  cellRef,
  id,
  onSendMessage,
}) => {
  const editorRef = useRef<HTMLDivElement>(null)
  const editorViewRef = useRef<EditorView | null>(null)
  const contentRef = useRef(content)

  const createCompletions = useCallback(
    (context: CompletionContext) => {
      const word = context.matchBefore(/\w*/)
      if (!word || (word.from === word.to && !context.explicit)) return null

      let options = []
      if (type === "python") {
        options = pythonCompletions.python
      } else if (type === "sql" || type === "promql") {
        options = sqlCompletions.sql
      }

      return {
        from: word.from,
        options,
        span: /^\w*$/,
      }
    },
    [type],
  )

  const getLanguageExtension = useCallback((cellType: string) => {
    switch (cellType) {
      case "python":
        return python()
      case "sql":
      case "promql":
        return sql()
      case "javascript":
      case "code":
        return javascript()
      case "markdown":
        return markdown()
      case "ai":
        return javascript() //Added for ai cell type
      default:
        return javascript()
    }
  }, [])

  useEffect(() => {
    if (!editorRef.current || editorViewRef.current) return

    const updateListener = EditorView.updateListener.of((update: ViewUpdate) => {
      if (update.docChanged) {
        const newContent = update.state.doc.toString()
        contentRef.current = newContent
        onContentChange(newContent)
      }
    })

    const state = EditorState.create({
      doc: contentRef.current,
      extensions: [
        basicSetup,
        getLanguageExtension(type),
        autocompletion({ override: [createCompletions] }),
        keymap.of([indentWithTab]),
        updateListener,
        EditorView.theme({
          "&": { maxHeight: "400px" },
          ".cm-scroller": { overflow: "auto" },
          "&.cm-focused": { outline: "none" },
          ".cm-line": { padding: "0 4px" },
          ".cm-matchingBracket": { backgroundColor: "#e9e9e9" },
          ".cm-content": { padding: "8px 0" },
        }),
      ],
    })

    const view = new EditorView({
      state,
      parent: editorRef.current,
    })

    editorViewRef.current = view

    return () => {
      if (editorViewRef.current) {
        editorViewRef.current.destroy()
        editorViewRef.current = null
      }
    }
  }, [type, createCompletions, getLanguageExtension, onContentChange])

  // Update content when prop changes and it's different from our current content
  useEffect(() => {
    if (editorViewRef.current && content !== contentRef.current) {
      contentRef.current = content
      editorViewRef.current.dispatch({
        changes: {
          from: 0,
          to: editorViewRef.current.state.doc.length,
          insert: content,
        },
      })
    }
  }, [content, editorViewRef])

  const renderOutput = () => {
    if (type === "sql" || type === "promql") {
      return <ResultTable data={output} />
    } else if (type === "datadog" || type === "splunk") {
      return <LogView logs={output} />
    } else if (type === "python" && output === "chart" && chartData) {
      return <TremorChart {...chartData} />
    } else {
      return <pre className="text-sm">{JSON.stringify(output, null, 2)}</pre>
    }
  }

  const getCreatorColor = () => {
    switch (creator) {
      case "NM":
        return "bg-blue-500"
      case "SS":
        return "bg-green-500"
      case "Sherlog":
        return "bg-yellow-500"
      default:
        return "bg-gray-500"
    }
  }

  const getCreatorName = () => {
    switch (creator) {
      case "NM":
        return "Navneet"
      case "SS":
        return "Sidharth"
      case "Sherlog":
        return "Sherlog AI"
      default:
        return creator
    }
  }

  const getCreatorInitials = () => {
    return creator === "Sherlog" ? "S" : creator
  }

  return (
    <div
      id={id}
      ref={cellRef}
      className={`mb-8 border rounded-lg overflow-hidden transition-all duration-300 ${
        isHighlighted ? "ring-2 ring-blue-500 bg-blue-50" : ""
      }`}
    >
      <div className={`p-2 ${type !== "markdown" && type !== "ai" ? "bg-gray-100" : "bg-white"}`}>
        <div className="flex justify-between items-center mb-2">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-semibold capitalize">{type} Cell</span>
            <div className="relative group">
              <span
                className={`w-6 h-6 ${getCreatorColor()} rounded-full flex items-center justify-center text-white text-xs font-bold`}
              >
                {getCreatorInitials()}
              </span>
              <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
                Added by {getCreatorName()}
              </span>
            </div>
            {/* Remove comment icon */}
          </div>
          <div className="flex items-center space-x-2">
            {status === "running" && <span className="animate-spin">⏳</span>}
            {status === "success" && <CheckIcon className="w-4 h-4 text-green-500" />}
            {status === "error" && <XIcon className="w-4 h-4 text-red-500" />}
            <button className="bg-blue-500 text-white px-2 py-1 rounded text-sm" onClick={onRun} title="Run cell">
              <PlayIcon className="w-4 h-4" />
            </button>
            <button onClick={onDelete} className="text-red-500 hover:text-red-700" title="Delete cell">
              <TrashIcon className="w-4 h-4" />
            </button>
          </div>
        </div>
        {type === "ai" ? (
          <AICell
            cell={{ id: cell.id, type: cell.type, content: cell.content, status: cell.status }}
            onDelete={onDelete}
            onSendMessage={onSendMessage}
          />
        ) : (
          <div ref={editorRef} className="border rounded" />
        )}
      </div>
      {output && type !== "ai" && <div className="p-2 bg-white border-t">{renderOutput()}</div>}
    </div>
  )
}

export default Cell
