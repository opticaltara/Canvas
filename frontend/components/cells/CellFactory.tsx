import type React from "react"
import MarkdownCell from "./MarkdownCell"
import SQLCell from "./SQLCell"
import CodeCell from "./CodeCell"
import GrafanaCell from "./GrafanaCell"
import LogCell from "./LogCell"
import GitHubCell from "./GitHubCell"
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
    case "sql":
      return (
        <SQLCell cell={cell} onExecute={onExecute} onUpdate={onUpdate} onDelete={onDelete} isExecuting={isExecuting} />
      )
    case "code":
      return (
        <CodeCell cell={cell} onExecute={onExecute} onUpdate={onUpdate} onDelete={onDelete} isExecuting={isExecuting} />
      )
    case "grafana":
      return (
        <GrafanaCell
          cell={cell}
          onExecute={onExecute}
          onUpdate={onUpdate}
          onDelete={onDelete}
          isExecuting={isExecuting}
        />
      )
    case "log":
      return (
        <LogCell cell={cell} onExecute={onExecute} onUpdate={onUpdate} onDelete={onDelete} isExecuting={isExecuting} />
      )
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
    default:
      return (
        <div className="p-4 border rounded-md bg-gray-50">
          <p>Unknown cell type: {cell.type}</p>
          <pre className="mt-2 p-2 bg-gray-100 rounded text-sm">{JSON.stringify(cell, null, 2)}</pre>
        </div>
      )
  }
}

export default CellFactory
