"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Loader2 } from "lucide-react"

interface StreamingCellProps {
  content: string
  onComplete?: () => void // Make onComplete optional
  type: "answer" | "query" | "results"
}

const StreamingCell: React.FC<StreamingCellProps> = ({
  content,
  onComplete = () => {}, // Provide default empty function
  type,
}) => {
  const [streamedContent, setStreamedContent] = useState("")
  const [isComplete, setIsComplete] = useState(false)

  useEffect(() => {
    // Check if content is undefined or null
    if (!content) {
      setIsComplete(true)
      onComplete()
      return
    }

    let index = 0
    const interval = setInterval(() => {
      if (index < content.length) {
        setStreamedContent((prev) => prev + content[index])
        index++
      } else {
        clearInterval(interval)
        setIsComplete(true)
        onComplete()
      }
    }, 50)

    return () => clearInterval(interval)
  }, [content, onComplete])

  const renderContent = () => {
    switch (type) {
      case "answer":
        return <div className="text-sm">{streamedContent}</div>
      case "query":
        return <pre className="text-sm bg-gray-100 p-2 rounded">{streamedContent}</pre>
      case "results":
        return <pre className="text-sm bg-black text-green-400 p-2 rounded">{streamedContent}</pre>
    }
  }

  return (
    <div className="mb-4 border rounded-lg overflow-hidden p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold capitalize">{type} (Streaming)</span>
        {!isComplete && <Loader2 className="w-4 h-4 animate-spin text-blue-500" />}
      </div>
      {renderContent()}
    </div>
  )
}

export default StreamingCell
