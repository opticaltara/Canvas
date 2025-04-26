import type React from "react"

interface LogViewProps {
  logs: string[]
}

export const LogView: React.FC<LogViewProps> = ({ logs }) => {
  // Simple null check to prevent the error
  if (!logs) {
    return null
  }

  return (
    <div className="bg-black text-green-400 font-mono text-sm p-4 overflow-x-auto">
      {logs.map((log, index) => (
        <div key={index}>{log}</div>
      ))}
    </div>
  )
}
