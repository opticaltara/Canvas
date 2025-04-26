import type React from "react"
import { Badge } from "@/components/ui/badge"

interface LogEntry {
  timestamp: string
  level: string
  component?: string
  message: string
  [key: string]: any // For any additional fields
}

interface LogViewProps {
  logs: LogEntry[] | string
}

export const LogView: React.FC<LogViewProps> = ({ logs }) => {
  // If logs is a string, try to parse it as JSON
  let logEntries: LogEntry[] = []

  if (typeof logs === "string") {
    try {
      logEntries = JSON.parse(logs)
    } catch (e) {
      // If parsing fails, split by newlines and create simple log entries
      logEntries = logs.split("\n").map((line, index) => ({
        timestamp: "",
        level: "",
        message: line,
        id: index.toString(),
      }))
    }
  } else if (Array.isArray(logs)) {
    logEntries = logs
  }

  const getLevelColor = (level: string) => {
    const levelLower = level.toLowerCase()
    if (levelLower.includes("error") || levelLower.includes("critical") || levelLower.includes("fatal")) {
      return "bg-red-600 text-white"
    } else if (levelLower.includes("warn")) {
      return "bg-amber-500 text-white"
    } else if (levelLower.includes("info")) {
      return "bg-blue-500 text-white"
    } else if (levelLower.includes("debug")) {
      return "bg-gray-500 text-white"
    } else {
      return "bg-gray-300 text-gray-800"
    }
  }

  return (
    <div className="overflow-auto max-h-[500px] font-mono text-sm">
      <table className="min-w-full">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Timestamp
            </th>
            <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Level</th>
            <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Component
            </th>
            <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Message</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {logEntries.map((log, index) => (
            <tr key={log.id || index} className={index % 2 === 0 ? "bg-white" : "bg-gray-50"}>
              <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-500">{log.timestamp}</td>
              <td className="px-2 py-1 whitespace-nowrap">
                {log.level && <Badge className={`text-xs ${getLevelColor(log.level)}`}>{log.level}</Badge>}
              </td>
              <td className="px-2 py-1 whitespace-nowrap text-xs text-gray-800">{log.component || ""}</td>
              <td className="px-2 py-1 text-xs text-gray-800 break-all">{log.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
