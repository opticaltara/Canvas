import type React from "react"
import MarkdownCell from "./MarkdownCell"
import GitHubCell from "./GitHubCell"
import SummarizationCell from './SummarizationCell'
import type { Cell } from "../../store/types"

interface CellFactoryProps {
  cell: Cell
  onExecute: (cellId: string) => void
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void
  onDelete: (cellId: string) => void
  isExecuting: boolean
}

const CellFactory: React.FC<CellFactoryProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }) => {
  switch (cell.type) {
    case "markdown":
      return <MarkdownCell cell={cell} onUpdate={onUpdate} onDelete={onDelete} />
    case "github":
      return (
        <GitHubCell
          cell={cell}
          onExecute={onExecute}
          onUpdate={onUpdate}
          onDelete={onDelete}
          isExecuting={isExecuting}
        />
      )
    case "summarization":
      return (
        <SummarizationCell
          cell={cell}
          onDelete={onDelete}
        />
      )
    default:
      const _exhaustiveCheck: never = cell.type;
      return (
        <div className="p-4 border rounded-md bg-red-50">
          <p>Unknown or unsupported cell type: {cell.type}</p>
          <pre className="mt-2 p-2 bg-gray-100 rounded text-sm">{JSON.stringify(cell, null, 2)}</pre>
        </div>
      )
  }
}

export default CellFactory
