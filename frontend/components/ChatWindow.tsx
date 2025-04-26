"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { PaperclipIcon, Search } from "lucide-react"
import { useChat } from "ai/react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface ChatWindowProps {
  onGenerateCell: (cellType: string, content: string) => void
}

const ChatWindow: React.FC<ChatWindowProps> = ({ onGenerateCell }) => {
  const [attachments, setAttachments] = useState<FileList | null>(null)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  // Common investigation questions related to payment failures
  const suggestions = [
    "What are the most common error types in the payment service?",
    "Show me the error distribution by device type",
    "When did the payment failures start increasing?",
    "Are there any recent deployments that might have caused this issue?",
    "How many customers are affected by the payment failures?",
    "What's the impact on our top tier customers?",
    "Is there a correlation between payment failures and server load?",
    "Show me the geographic distribution of payment failures",
    "Compare payment failure rates before and after the latest deployment",
    "Check if there are any database connection issues related to payments",
  ]

  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: "/api/chat",
    onFinish: (message) => {
      // Parse the message to extract cell type and content
      const cellInfo = parseCellInfo(message.content)
      if (cellInfo) {
        onGenerateCell(cellInfo.type, cellInfo.content)
      }
    },
  })

  // Filter suggestions based on input
  useEffect(() => {
    if (input.trim() === "") {
      setFilteredSuggestions(suggestions)
    } else {
      const filtered = suggestions.filter((suggestion) => suggestion.toLowerCase().includes(input.toLowerCase()))
      setFilteredSuggestions(filtered)
    }
    setActiveSuggestionIndex(0)
  }, [input])

  // Show suggestions when input is focused
  const handleInputFocus = () => {
    setShowSuggestions(true)
  }

  // Handle keyboard navigation through suggestions
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // If no suggestions or not showing suggestions, return
    if (filteredSuggestions.length === 0 || !showSuggestions) return

    // Arrow up
    if (e.key === "ArrowUp") {
      e.preventDefault()
      setActiveSuggestionIndex((prevIndex) => (prevIndex === 0 ? filteredSuggestions.length - 1 : prevIndex - 1))
    }
    // Arrow down
    else if (e.key === "ArrowDown") {
      e.preventDefault()
      setActiveSuggestionIndex((prevIndex) => (prevIndex === filteredSuggestions.length - 1 ? 0 : prevIndex + 1))
    }
    // Enter key - select the active suggestion
    else if (e.key === "Enter" && showSuggestions && filteredSuggestions.length > 0) {
      e.preventDefault()
      selectSuggestion(filteredSuggestions[activeSuggestionIndex])
    }
    // Escape key - close suggestions
    else if (e.key === "Escape") {
      setShowSuggestions(false)
    }
  }

  // Select a suggestion
  const selectSuggestion = (suggestion: string) => {
    // Use the handleInputChange from useChat hook
    const event = {
      target: { value: suggestion },
    } as React.ChangeEvent<HTMLInputElement>

    handleInputChange(event)
    setShowSuggestions(false)
  }

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (inputRef.current && !inputRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [])

  const handleAttachmentChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setAttachments(e.target.files)
  }

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setShowSuggestions(false)
    handleSubmit(e, { experimental_attachments: attachments || undefined })
    setAttachments(null)
  }

  return (
    <div className="border-t p-4">
      <form onSubmit={handleFormSubmit} className="flex items-center space-x-2">
        <div className="flex-1 relative">
          <Input
            ref={inputRef}
            type="text"
            value={input}
            onChange={handleInputChange}
            onFocus={handleInputFocus}
            onKeyDown={handleKeyDown}
            placeholder="Ask Sherlog..."
            className="flex-1 border rounded-lg px-4 py-2 pl-10 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />

          {/* Suggestions dropdown with animation */}
          {showSuggestions && filteredSuggestions.length > 0 && (
            <div
              className="absolute bottom-full left-0 right-0 mb-2 bg-white border rounded-md shadow-lg max-h-60 overflow-auto z-50 animate-suggestion-fade-in"
              style={{
                transformOrigin: "bottom center",
                animation: "suggestionFadeIn 0.2s ease-out forwards",
              }}
            >
              <div className="p-2 text-xs text-gray-500 border-b">Suggested questions:</div>
              <ul>
                {filteredSuggestions.map((suggestion, index) => (
                  <li
                    key={index}
                    className={`px-4 py-2 text-sm cursor-pointer hover:bg-blue-50 ${
                      index === activeSuggestionIndex ? "bg-blue-50" : ""
                    }`}
                    onClick={() => selectSuggestion(suggestion)}
                  >
                    {suggestion}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <label htmlFor="file-upload" className="cursor-pointer">
          <PaperclipIcon className="w-5 h-5 text-gray-400 hover:text-gray-600" />
          <input id="file-upload" type="file" multiple onChange={handleAttachmentChange} className="hidden" />
        </label>
        <Button type="submit" disabled={isLoading}>
          Send
        </Button>
      </form>
      {attachments && (
        <div className="mt-2">
          <p className="text-sm text-gray-500">
            Attached:{" "}
            {Array.from(attachments)
              .map((file) => file.name)
              .join(", ")}
          </p>
        </div>
      )}
      {isLoading && (
        <div className="mt-4">
          <p className="text-sm text-gray-500">Sherlog is thinking...</p>
        </div>
      )}
    </div>
  )
}

// Helper function to parse cell info from the AI response
const parseCellInfo = (content: string): { type: string; content: string } | null => {
  // This is a placeholder implementation. You'll need to implement
  // the actual parsing logic based on your backend's response format.
  const match = content.match(/^TYPE: (\w+)\nCONTENT:\n(.+)$/s)
  if (match) {
    return { type: match[1], content: match[2] }
  }
  return null
}

export default ChatWindow
