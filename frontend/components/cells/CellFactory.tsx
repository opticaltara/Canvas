import type React from "react"
import dynamic from 'next/dynamic'
import type { Cell } from "../../store/types"

// Dynamically import cell components
const MarkdownCell = dynamic(() => import('./MarkdownCell'), { loading: () => <p>Loading Markdown Cell...</p> });
const GitHubCell = dynamic(() => import('./GitHubCell'), { loading: () => <p>Loading GitHub Cell...</p> });
const SummarizationCell = dynamic(() => import('./SummarizationCell'), { loading: () => <p>Loading Summarization Cell...</p> });
const InvestigationReportCell = dynamic(() => import('./InvestigationReportCell'), { loading: () => <p>Loading Report Cell...</p> });
const FileSystemCell = dynamic(() => import('./FileSystemCell'), { loading: () => <p>Loading FileSystem Cell...</p> });
const PythonCell = dynamic(() => import('./PythonCell'), { loading: () => <p>Loading Python Cell...</p> });
const MediaTimelineCell = dynamic(() => import('./MediaTimelineCell'), { loading: () => <p>Loading Media Timeline...</p> });

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
    case "investigation_report":
      return (
        <InvestigationReportCell
          cell={cell}
          onDelete={onDelete}
        />
      )
    case "filesystem":
      return (
        <FileSystemCell
          cell={cell}
          onUpdate={onUpdate}
          onExecute={onExecute}
          onDelete={onDelete}
          isExecuting={isExecuting}
        />
      )
    case "python":
      return (
        <PythonCell
          cell={cell}
          onExecute={onExecute}
          onUpdate={onUpdate}
          onDelete={onDelete}
          isExecuting={isExecuting}
        />
      )
    case "media_timeline":
      return (
        <MediaTimelineCell
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
