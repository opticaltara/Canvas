"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, Trash2, Bot, User, Loader2 } from "lucide-react"

interface AIMessage {
  role: "user" | "assistant"
  content: string
}

interface AICell {
  id: string
  type: string
  content: string
  status: "idle" | "running" | "success" | "error"
  messages?: AIMessage[]
  error?: string
}

interface AICellProps {
  cell: AICell
  onDelete: () => void
  onSendMessage: (cellId: string, message: string) => void
}

export default function AICell({ cell, onDelete, onSendMessage }: AICellProps) {
  const [message, setMessage] = useState("")
  const [messages, setMessages] = useState<AIMessage[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Update messages when cell changes
  useEffect(() => {
    if (cell.messages) {
      setMessages(cell.messages)
    }
  }, [cell.messages])

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (message.trim() && cell.status !== "running") {
      onSendMessage(cell.id, message)
      setMessage("")
    }
  }

  return (
    <Card className="overflow-hidden cell-ai">
      <div className="px-4 py-2 flex justify-between items-center border-b bg-purple-50">
        <div className="flex items-center space-x-2">
          <Bot className="h-5 w-5 text-purple-600" />
          <span className="text-sm font-medium">Sherlog AI Assistant</span>
          {cell.status === "running" && (
            <div className="flex items-center text-purple-600 text-xs">
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
              Thinking...
            </div>
          )}
        </div>
        <Button variant="ghost" size="sm" onClick={onDelete}>
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>

      <div className="p-4 bg-gray-50 min-h-[200px] max-h-[500px] overflow-y-auto">
        {messages && messages.length > 0 ? (
          <div className="space-y-4">
            {messages.map((msg, index) => (
              <div key={index} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`flex items-start space-x-2 max-w-[80%] ${
                    msg.role === "user" ? "flex-row-reverse space-x-reverse" : ""
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      msg.role === "user" ? "bg-blue-100" : "bg-purple-100"
                    }`}
                  >
                    {msg.role === "user" ? (
                      <User className="h-4 w-4 text-blue-600" />
                    ) : (
                      <Bot className="h-4 w-4 text-purple-600" />
                    )}
                  </div>
                  <div
                    className={`p-3 rounded-lg ${
                      msg.role === "user" ? "bg-blue-100 text-blue-900" : "bg-white border text-gray-800"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Bot className="h-12 w-12 text-purple-300 mb-2" />
            <p className="text-center">Ask Sherlog AI a question to get started</p>
            <p className="text-center text-sm text-gray-400 mt-2">
              Sherlog can help analyze data, explain errors, and suggest solutions
            </p>
          </div>
        )}
      </div>

      <div className="p-3 border-t bg-white">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask Sherlog AI a question..."
            className="flex-1"
            disabled={cell.status === "running"}
          />
          <Button
            type="submit"
            size="sm"
            disabled={!message.trim() || cell.status === "running"}
            className="bg-purple-600 hover:bg-purple-700 text-white"
          >
            {cell.status === "running" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </form>
      </div>

      {cell.error && (
        <div className="p-3 border-t bg-red-50">
          <div className="text-red-600 font-medium mb-1">Error:</div>
          <div className="text-red-800 text-sm font-mono">{cell.error}</div>
        </div>
      )}
    </Card>
  )
}
