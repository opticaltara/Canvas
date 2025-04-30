"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, Bot, User, Loader2, ChevronRight, AlertCircle, RefreshCw, XCircle, ChevronDown } from "lucide-react"
import { api } from "@/api/client"
import type { ChatMessage, CellCreationEvent } from "@/api/chat"
import type { Model } from "@/api/models"
import { useToast } from "@/hooks/use-toast"
import { ModelSelector } from "@/components/ModelSelector"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"

interface AIChatPanelProps {
  isOpen: boolean
  onToggle: () => void
  notebookId: string // Make this required
}

// Define a type for the parsed status content
interface StatusContent {
  type: string
  content: string
  agent_type: string
}

// Define a type for tracking agent status
interface AgentStatus {
  agentType: string
  messages: StatusContent[]
  isActive: boolean
  isExpanded: boolean
}

// Change to default export
export default function AIChatPanel({ isOpen, onToggle, notebookId }: AIChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()
  // Update the state to track the last failed message
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null)
  // Track if we're retrying a message
  const [isRetrying, setIsRetrying] = useState(false)
  // Track the current model
  const [currentModel, setCurrentModel] = useState<Model | null>(null)
  // State to track active agent statuses
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({})

  // Create a chat session when the panel opens
  useEffect(() => {
    if (isOpen && !sessionId) {
      createChatSession()
    }
  }, [isOpen])

  // Focus input when panel opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isOpen])

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages, agentStatuses])

  // Add this near where messages are processed
  useEffect(() => {
    if (messages.length > 0) {
      console.log("Chat messages updated, current count:", messages.length)

      // Log the last message
      const lastMessage = messages[messages.length - 1]
      if (lastMessage) {
        console.log("Last message role:", lastMessage.role)
        console.log("Last message content type:", typeof lastMessage.content)
        console.log(
          "Last message content preview:",
          typeof lastMessage.content === "string" ? lastMessage.content.substring(0, 100) : "Not a string",
        )
      }
    }
  }, [messages])

  // Create a new chat session
  const createChatSession = async () => {
    try {
      setIsLoading(true)
      setError(null)

      const session = await api.chat.createSession(notebookId)
      setSessionId(session.session_id)

      // Load existing messages if any
      if (session.session_id) {
        const existingMessages = await api.chat.getSessionMessages(session.session_id)
        if (existingMessages.length > 0) {
          // Process existing messages to potentially populate agent statuses
          processMessagesForAgents(existingMessages)
          setMessages(existingMessages)
        }
      }
    } catch (error) {
      console.error("Failed to create chat session:", error)
      setError("Failed to initialize chat. Please try again.")
      toast({
        variant: "destructive",
        title: "Chat Error",
        description: "Failed to initialize chat session. Please try again.",
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Parse error message from API response
  const parseErrorMessage = (error: any): string => {
    let errorMessage = "An unexpected error occurred. Please try again."

    try {
      // Handle when error is undefined or null
      if (!error) {
        return errorMessage
      }

      // Check if error is a string that contains JSON
      if (typeof error === "string") {
        if (error.startsWith("{") || error.includes('{"error":')) {
          try {
            const errorObj = JSON.parse(error)
            if (errorObj.error) {
              errorMessage =
                typeof errorObj.error === "string" ? errorObj.error : errorObj.error.message || "Unknown error"
            }
          } catch (e) {
            // If it's not valid JSON but contains error message patterns, try to extract them
            const errorMatch = error.match(/error[":]\s*"?([^"]+)"?/i)
            if (errorMatch && errorMatch[1]) {
              errorMessage = errorMatch[1]
            } else {
              errorMessage = error // Just use the string as is
            }
          }
        } else {
          // It's a plain string error
          errorMessage = error
        }
      }
      // Check if error is an Error object with message
      else if (error instanceof Error) {
        errorMessage = error.message
      }
      // Check if error.message contains JSON
      else if (error.message) {
        if (error.message.startsWith("{") || error.message.includes('{"error":')) {
          try {
            const errorObj = JSON.parse(error.message)
            if (errorObj.error) {
              errorMessage =
                typeof errorObj.error === "string" ? errorObj.error : errorObj.error.message || "Unknown error"
            }
          } catch (e) {
            errorMessage = error.message
          }
        } else {
          errorMessage = error.message
        }
      }
      // Check if error has a response with data
      else if (error.response && error.response.data) {
        if (typeof error.response.data === "string") {
          errorMessage = error.response.data
        } else if (error.response.data.error) {
          errorMessage =
            typeof error.response.data.error === "string"
              ? error.response.data.error
              : error.response.data.error.message || "API error"
        }
      }
    } catch (e) {
      console.error("Error parsing error message:", e)
    }

    // Make the error message more user-friendly
    if (
      errorMessage.includes("Connection error") ||
      errorMessage.includes("network") ||
      errorMessage.includes("ECONNREFUSED")
    ) {
      return "Connection to the AI service failed. Please check your internet connection and try again later."
    } else if (errorMessage.includes("timeout") || errorMessage.includes("timed out")) {
      return "The request timed out. The server might be busy or experiencing issues. Please try again later."
    } else if (errorMessage.includes("status_code: 500") || errorMessage.includes("Internal Server Error")) {
      return "The server encountered an error. Please try again later."
    } else if (errorMessage.includes("overloaded") || errorMessage.includes("capacity")) {
      return "The AI service is currently overloaded. Please try again in a moment."
    } else if (errorMessage.includes("rate limit") || errorMessage.includes("too many requests")) {
      return "Rate limit exceeded. Please wait a moment before trying again."
    } else if (errorMessage.includes("stream") && errorMessage.includes("error")) {
      return "Error in streaming response. Please try again."
    }

    return errorMessage
  }

  // Handle retrying a failed message
  const handleRetry = async () => {
    if (!lastFailedMessage || isLoading || !sessionId) return

    setIsRetrying(true)
    setError(null)
    setIsLoading(true)

    try {
      // Send message to API and handle streaming response
      await api.chat.sendMessage(sessionId, lastFailedMessage, handleMessageOrEvent)

      // Clear the failed message after successful retry
      setLastFailedMessage(null)
    } catch (error) {
      console.error("Error retrying message:", error)

      // Set a more user-friendly error message
      const errorMessage = parseErrorMessage(error)
      setError(errorMessage)

      toast({
        variant: "destructive",
        title: "Chat Error",
        description: errorMessage,
      })
    } finally {
      setIsLoading(false)
      setIsRetrying(false)
    }
  }

  // Handle model change
  const handleModelChange = (model: Model) => {
    setCurrentModel(model)
    // We don't need to add a message about model change
    // The current model is already visible in the dropdown
  }

  // Handle cell creation events - just show a toast notification
  const handleCellCreation = async (event: CellCreationEvent) => {
    console.log("AIChatPanel received cell_created event:", event)
    // Potentially update agent status if cell creation signifies completion
    if (event.agentType) {
      setAgentStatuses((prev) => ({
        ...prev,
        [event.agentType]: {
          ...prev[event.agentType],
          isActive: false, // Assume cell creation completes the agent task for now
        },
      }))
    }
  }

  // Helper function to safely parse message content
  const parseMessageContent = (content: string): StatusContent | null => {
    try {
      const parsed = JSON.parse(content)
      if (parsed && typeof parsed === "object" && parsed.type === "status_response") {
        return parsed as StatusContent
      }
    } catch (e) {
      // Ignore errors, it's likely not JSON or not the format we expect
    }
    return null
  }

  // Process a batch of messages to update agent statuses
  const processMessagesForAgents = (newMessages: ChatMessage[]) => {
    let updatedStatuses = { ...agentStatuses }
    let statusChanged = false

    newMessages.forEach((msg) => {
      if (msg.role === "model") {
        const statusContent = parseMessageContent(msg.content)
        if (statusContent) {
          const agentType = statusContent.agent_type
          if (!updatedStatuses[agentType]) {
            updatedStatuses[agentType] = {
              agentType: agentType,
              messages: [],
              isActive: true,
              isExpanded: false,
            }
          }
          updatedStatuses[agentType].messages.push(statusContent)
          updatedStatuses[agentType].isActive = true // Mark as active on receiving status
          statusChanged = true
        } else if (msg.agent && updatedStatuses[msg.agent]?.isActive) {
          // If a regular message comes from an agent that was active, mark it as inactive
          updatedStatuses[msg.agent].isActive = false
          statusChanged = true
        }
      }
    })

    if (statusChanged) {
      setAgentStatuses(updatedStatuses)
    }
  }

  // Handle both message and cell creation events
  const handleMessageOrEvent = (messageOrEvent: ChatMessage | CellCreationEvent) => {
    console.log("Received message or event:", messageOrEvent)

    if ("type" in messageOrEvent && messageOrEvent.type === "cell_created") {
      // Handle cell creation event
      handleCellCreation(messageOrEvent)
      // We don't add cell_created events directly to the chat message list
    } else if ("role" in messageOrEvent) {
      // Handle regular chat message
      const message = messageOrEvent as ChatMessage
      const statusContent = parseMessageContent(message.content) // Check if it's a status update

      // If it's a status message, process it for agent status.
      // Only stop processing for the main chat list IF it's NOT from chat_agent.
      if (statusContent && message.role === "model") {
        processMessagesForAgents([message]) // Always process for agent status UI
        if (statusContent.agent_type !== 'chat_agent') {
            return // Stop processing for the main chat list ONLY for non-chat_agent status messages
        }
        // If it IS chat_agent, continue execution to add to setMessages below
      }

      // --- Add this check --- 
      // Try parsing the content to see if it's a cell_response
      try {
          const potentialCellResponse = JSON.parse(message.content);
          if (potentialCellResponse && typeof potentialCellResponse === 'object' && potentialCellResponse.type === 'cell_response') {
              // If it IS a cell_response, don't add it to the chat messages.
              // It's handled by canvasStore via handleCellCreation in chat.ts
              // But we still might need to mark the agent as finished if it's a final step.
              if (message.agent && potentialCellResponse.status_type === 'step_completed') {
                 setAgentStatuses((prev) => {
                    if (prev[message.agent!]?.isActive) {
                      return {
                        ...prev,
                        [message.agent!]: {
                          ...prev[message.agent!],
                          isActive: false,
                        },
                      }
                    }
                    return prev
                  })
              }
              return; // Stop processing this message for the main chat list
          }
      } catch (e) {
          // Not a parsable JSON or not a cell_response, continue processing as a regular message.
          console.log("Error parsing message content as JSON, treating as regular message:", e);
      }
      // --- End of added check ---

      // If not a status message or cell_response, add it to the chat list
      setMessages((prevMessages) => {
        const newMessages = [...prevMessages]
        const lastMessage = newMessages[newMessages.length - 1]

        // Append only if the new message AND the last message are from the USER.
        if (
          lastMessage &&
          lastMessage.role === "user" &&
          message.role === "user" &&
          typeof lastMessage.content === "string" && // Still need this basic type check for safety
          typeof message.content === "string"
        ) {
          // Append user message content
          lastMessage.content += "\n" + message.content // Add newline for clarity
          lastMessage.timestamp = message.timestamp // Update timestamp
        } else {
          // Otherwise, always push the new message as a separate entry
          newMessages.push(message)
        }

        // Process the new message for agent status updates (stores them)
        processMessagesForAgents([message])

        return newMessages
      })

      // If the message marks an agent as inactive, ensure its state reflects that
      if (message.role === "model") {
        const statusContent = parseMessageContent(message.content)
        if (!statusContent && message.agent) {
          // This is a final message from the agent
          setAgentStatuses((prev) => {
            if (prev[message.agent!]?.isActive) {
              return {
                ...prev,
                [message.agent!]: {
                  ...prev[message.agent!],
                  isActive: false,
                },
              }
            }
            return prev
          })
        }
      }
    }
  }

  // Modify the handleSendMessage function to handle errors better
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()

    if (!input.trim() || isLoading || !sessionId) return

    const userMessageContent = input.trim() // Store content
    setInput("")
    setError(null)
    setLastFailedMessage(null) // Clear previous failure on new send

    // --- Optimistic UI Update ---
    const timestamp = new Date().toISOString()
    const userMessage: ChatMessage = {
      role: "user",
      content: userMessageContent,
      timestamp: timestamp,
    }
    // Add user message immediately to the state
    setMessages((prevMessages) => [...prevMessages, userMessage])
    // --- End Optimistic Update ---

    setIsLoading(true)

    try {
      // Send message to API and handle streaming response
      await api.chat.sendMessage(sessionId, userMessageContent, handleMessageOrEvent)

      // If this is the first message, it might trigger an investigation
      if (messages.length <= 1) { 
        toast({
          title: "Investigation Started",
          description: "Your query may trigger an automated investigation. Watch for new cells being created.",
        })
      }
    } catch (error) {
      console.error("Error sending message:", error)

      // Store the failed message for potential retry
      setLastFailedMessage(userMessageContent) // Use captured content

      // Set a more user-friendly error message
      const errorMessage = parseErrorMessage(error)
      setError(errorMessage)

      // Display the error message as a system message in the chat
      const errorChatMessage: ChatMessage = {
        role: "model", // Or maybe 'system' or 'error' role if defined
        content: `Error: ${errorMessage}. Please try again or rephrase your question.`,
        timestamp: new Date().toISOString(),
        agent: 'system' // Assign an agent like 'system'
      }
      setMessages((prev) => [...prev, errorChatMessage])

      toast({
        variant: "destructive",
        title: "Chat Error",
        description: errorMessage,
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Escape to close panel
    if (e.key === "Escape") {
      onToggle()
    }
  }

  // Format timestamp for display
  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    } catch (e) {
      return ""
    }
  }

  // Clear error
  const dismissError = () => {
    setError(null)
    setLastFailedMessage(null)
  }

  const toggleAgentExpansion = (agentType: string) => {
    setAgentStatuses((prev) => ({
      ...prev,
      [agentType]: {
        ...prev[agentType],
        isExpanded: !prev[agentType].isExpanded,
      },
    }))
  }

  // Filter out raw status messages that are handled by the agent status UI,
  // EXCEPT for status messages from chat_agent which should be displayed directly.
  const filteredMessages = messages.filter((msg) => {
    if (msg.role === "model") {
      const statusContent = parseMessageContent(msg.content)
      // Filter out status messages UNLESS they are from chat_agent
      if (statusContent && statusContent.agent_type !== 'chat_agent') {
        return false // Filter out non-chat_agent status messages
      }
    }
    return true // Keep user messages and chat_agent status messages (and regular model messages)
  })

  // --- Add the new helper function here ---
  const renderModelContent = (content: string): React.ReactNode => {
    try {
      const parsedContent = JSON.parse(content)

      // Handle specific known types first
      if (parsedContent && parsedContent.type === "status_response") {
        // Prefer 'content' field within the status_response structure
        return parsedContent.content || "[Status Update]" 
      } 
      // Add explicit handling for other known types if needed
      // else if (parsedContent && parsedContent.type === 'cell_response') { ... }
      
      // --- Generic Handling for Unknown Structured Types ---
      // If it parsed as an object, try common fields before showing raw JSON
      else if (typeof parsedContent === 'object' && parsedContent !== null) {
        if (typeof parsedContent.message === 'string') {
           console.warn("Rendering .message from unknown structure:", parsedContent);
           return parsedContent.message; 
        } else if (typeof parsedContent.content === 'string') {
           // Might be less common if type wasn't status_response, but check anyway
           console.warn("Rendering .content from unknown structure:", parsedContent);
           return parsedContent.content; 
        } else {
          // Fallback for unknown objects: pretty-print JSON
           console.warn("Parsed unknown object structure, showing JSON:", parsedContent);
           return <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-x-auto">{JSON.stringify(parsedContent, null, 2)}</pre>;
        }
      } 
      // If it parsed but wasn't an object (e.g., just a string/number inside JSON), render raw content
      else {
          console.warn("Parsed non-object JSON, showing raw content:", content);
          return content; // Render original string if parsed content isn't helpful object
      }

    } catch (e) {
      // Parsing failed, assume it's plain text
      return content
    }
  }

  return (
    <div
      className={`fixed right-0 top-0 bottom-0 bg-white border-l border-gray-200 shadow-lg transition-all duration-300 ease-in-out z-20 flex flex-col ${
        isOpen ? "w-96" : "w-0 opacity-0"
      }`}
      onKeyDown={handleKeyDown}
    >
      {isOpen && (
        <>
          {/* Header */}
          <div className="p-4 border-b bg-purple-50 flex justify-between items-center">
            <div className="flex items-center">
              <Bot className="h-5 w-5 text-purple-600 mr-2" />
              <h3 className="font-medium">Sherlog AI Assistant</h3>
            </div>
            <div className="flex items-center gap-2">
              <ModelSelector onModelChange={handleModelChange} />
              <Button
                variant="ghost"
                size="sm"
                onClick={onToggle}
                className="h-8 w-8 p-0 hover:bg-purple-100"
                aria-label="Close panel"
              >
                <ChevronRight className="h-5 w-5 text-purple-600" />
              </Button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
            {error && (
              <Alert variant="destructive" className="mb-4 animate-in fade-in-0 slide-in-from-top-5">
                <div className="flex justify-between items-start">
                  <div className="flex items-start">
                    <AlertCircle className="h-4 w-4 mt-0.5 mr-2" />
                    <div>
                      <AlertTitle className="text-sm font-medium">Error</AlertTitle>
                      <AlertDescription className="text-sm">{error}</AlertDescription>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 -mt-1 -mr-1 hover:bg-red-200"
                    onClick={dismissError}
                  >
                    <XCircle className="h-4 w-4" />
                  </Button>
                </div>
                {lastFailedMessage && (
                  <div className="mt-2 flex justify-end">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleRetry}
                      disabled={isRetrying}
                      className="text-xs border-red-300 hover:bg-red-100 hover:text-red-800"
                    >
                      {isRetrying ? (
                        <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                      ) : (
                        <RefreshCw className="h-3 w-3 mr-1" />
                      )}
                      Retry message
                    </Button>
                  </div>
                )}
              </Alert>
            )}

            {filteredMessages.length === 0 && Object.values(agentStatuses).filter((s) => s.isActive).length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <Bot className="h-12 w-12 text-purple-300 mb-3" />
                <h4 className="text-lg font-medium text-gray-700 mb-2">How can I help you?</h4>
                <p className="text-sm text-gray-500 max-w-xs">
                  Ask me anything about your data, or how to analyze it. I can help you create cells, explain results,
                  or suggest next steps.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {filteredMessages.map((message, index) => (
                  <div
                    key={`msg-${index}-${message.timestamp}`}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} message-enter`}
                  >
                    <div
                      className={`flex items-start space-x-2 max-w-[85%] ${
                        message.role === "user" ? "flex-row-reverse space-x-reverse" : ""
                      }`}
                    >
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          message.role === "user" ? "bg-blue-100" : "bg-purple-100"
                        }`}
                      >
                        {message.role === "user" ? (
                          <User className="h-4 w-4 text-blue-600" />
                        ) : (
                          <Bot className="h-4 w-4 text-purple-600" />
                        )}
                      </div>
                      <div className="flex flex-col">
                        <div
                          className={`p-3 rounded-lg ${
                            message.role === "user"
                              ? "bg-blue-100 text-blue-900"
                              : "bg-white border border-gray-200 text-gray-800"
                          }`}
                        >
                          {message.role === "model" ? (
                            message.agent === "chat_agent" ? (
                              // Handle chat_agent specifically: parse and display inner content
                              (() => {
                                try {
                                  const parsedContent = JSON.parse(message.content)
                                  if (parsedContent && parsedContent.type === "status_response" && parsedContent.content) {
                                    // Render the actual clarification message
                                    return <p className="whitespace-pre-wrap text-sm">{parsedContent.content}</p>
                                  }
                                } catch (e) {
                                  // Fallback if parsing fails or structure is wrong
                                  console.error("Failed to parse chat_agent message content:", e, message.content)
                                }
                                // Fallback: Render raw content if parsing failed or it's not the expected format
                                return <p className="whitespace-pre-wrap text-sm italic text-gray-500">{message.content}</p>
                              })()
                            ) : (
                              // For other agents, use the existing renderModelContent
                              renderModelContent(message.content)
                            )
                          ) : (
                            // User message
                            <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                          )}
                        </div>
                        {message.timestamp && (
                          <span className="text-xs text-gray-500 mt-1 px-1">{formatTimestamp(message.timestamp)}</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {/* Render active agent statuses */}
                {Object.values(agentStatuses)
                  // Filter out chat_agent statuses from this specific UI block
                  .filter((status) => (status.isActive || status.messages.length > 0) && status.agentType !== 'chat_agent') 
                  .map((status) => (
                    <div key={status.agentType} className="flex justify-start message-enter">
                      <div className="flex items-start space-x-2 max-w-[85%]">
                        <div className="w-8 h-8 rounded-full flex items-center justify-center bg-purple-100">
                          <Bot className="h-4 w-4 text-purple-600" />
                        </div>
                        <Collapsible
                          open={status.isExpanded}
                          onOpenChange={() => toggleAgentExpansion(status.agentType)}
                          className="w-full"
                        >
                          <div className="flex flex-col w-full">
                            <CollapsibleTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="flex justify-between items-center p-3 rounded-lg bg-white border border-gray-200 text-gray-800 w-full text-left h-auto"
                              >
                                <div className="flex items-center space-x-2">
                                  {status.isActive && <Loader2 className="h-4 w-4 animate-spin" />}
                                  <span>
                                    {status.agentType} agent{" "}
                                    {status.isActive ? "is working..." : "finished."}
                                  </span>
                                </div>
                                {status.isExpanded ? (
                                  <ChevronDown className="h-4 w-4" />
                                ) : (
                                  <ChevronRight className="h-4 w-4" />
                                )}
                              </Button>
                            </CollapsibleTrigger>
                            <CollapsibleContent className="pt-2 pl-3 pr-3 pb-1 rounded-b-lg border border-t-0 border-gray-200 bg-gray-50">
                              <ul className="space-y-1 text-xs text-gray-600 list-disc list-inside">
                                {(() => {
                                  const deduplicatedMessages: StatusContent[] = []
                                  let lastInnerContent: string | null = null
                                  status.messages.forEach((msg) => {
                                    const currentInnerContent = msg.content
                                    if (currentInnerContent !== lastInnerContent) {
                                      deduplicatedMessages.push(msg)
                                      lastInnerContent = currentInnerContent
                                    }
                                  })
                                  return deduplicatedMessages.map((msgContent, msgIndex) => (
                                    <li key={msgIndex}>{msgContent.content}</li>
                                  ))
                                })()}
                                {!status.isActive && status.messages.length > 0 && (
                                   <li className="text-gray-400 italic">End of actions for {status.agentType}</li>
                                )}
                              </ul>
                            </CollapsibleContent>
                            {/* Timestamp for the group */}
                             {status.messages.length > 0 && (
                               <span className="text-xs text-gray-500 mt-1 px-1">
                                 Last update: {formatTimestamp(messages.find(m => m.agent === status.agentType)?.timestamp || new Date().toISOString())}
                               </span>
                             )}
                          </div>
                        </Collapsible>
                      </div>
                    </div>
                  ))}

                {/* Show loading indicator when waiting for response */}
                {isLoading && (
                  <div className="flex justify-center items-center p-4 message-enter">
                    <div className="flex items-center space-x-2 text-gray-500">
                      <Loader2 className="h-5 w-5 animate-spin text-purple-600" />
                      <span>Thinking...</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <div className="p-4 border-t bg-white">
            <form onSubmit={handleSendMessage} className="flex space-x-2">
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask Sherlog AI a question..."
                className="flex-1"
                disabled={isLoading || !sessionId}
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isLoading || !sessionId}
                className="bg-purple-600 hover:bg-purple-700 text-white"
              >
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </form>
          </div>
        </>
      )}
    </div>
  )
}
