"use client"

import type React from "react"
import { useState } from "react"
import { useChat } from "ai/react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send } from "lucide-react"

interface AICellProps {
  initialPrompt?: string
  onResult: (result: string) => void
}

const AICell: React.FC<AICellProps> = ({ initialPrompt = "", onResult }) => {
  const [prompt, setPrompt] = useState(initialPrompt)
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "/api/chat", // Ensure this endpoint is set up in your FastAPI backend
    onFinish: (message) => {
      onResult(message.content)
    },
  })

  const handlePromptSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    handleSubmit(e)
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <form onSubmit={handlePromptSubmit} className="mb-4">
        <div className="flex space-x-2">
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask AI for assistance..."
            className="flex-grow"
          />
          <Button type="submit">
            <Send className="w-4 h-4 mr-2" />
            Send
          </Button>
        </div>
      </form>
      <div className="space-y-4">
        {messages.map((message, i) => (
          <div key={i} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`rounded-lg p-2 max-w-3/4 ${
                message.role === "user" ? "bg-blue-100 text-blue-800" : "bg-gray-100 text-gray-800"
              }`}
            >
              <p className="text-sm">{message.content}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default AICell
