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

// Import the hook and event types
import { useInvestigationEvents, type InvestigationEvent } from "@/hooks/useInvestigationEvents"
// Import cell CRUD operations from zustand store
import { useCanvasStore } from '@/store/canvasStore'
// Import the Cell type
import type { Cell } from "@/store/types"

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

// --- Define helper function at the top level --- 
// Helper function to attempt parsing ChatMessage content into an InvestigationEvent
// This needs to be robust as not all messages are events.
const parseChatToInvestigationEvent = (message: ChatMessage): InvestigationEvent | null => {
  if (message.role !== 'model' || !message.content) {
    return null; // Only model messages with content can be events
  }
  try {
    const parsedContent = JSON.parse(message.content);
    // Basic check: Does it look like one of our event structures?
    // Requires 'type' and 'status' fields common to our BaseEvent.
    if (parsedContent && typeof parsedContent === 'object' && parsedContent.type && parsedContent.status) {
      // We could add more specific checks here if needed based on event types
      // For now, assume if it parses and has type/status, it's intended as an event.
      console.log("[AIChatPanel] Parsed chat content as potential InvestigationEvent:", parsedContent.type)
      return parsedContent as InvestigationEvent; // Cast, assuming backend sends valid structure
    }
  } catch (e) {
    // Not JSON or doesn't match basic structure, so not an event
    // console.log("[AIChatPanel] Failed to parse chat content as InvestigationEvent:", message.content.substring(0, 50))
  }
  return null; // Not parsable or not a recognized event structure
};

// Change to default export
export default function AIChatPanel({ isOpen, onToggle, notebookId }: AIChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [investigationEventStream, setInvestigationEventStream] = useState<InvestigationEvent[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [currentModel, setCurrentModel] = useState<Model | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null)
  const [isRetrying, setIsRetrying] = useState(false)
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({})

  const { toast } = useToast()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // --- Get cell handlers from Zustand store --- 
  const handleCellUpdate = useCanvasStore((state) => state.handleCellUpdate);
  const updateCell = useCanvasStore((state) => state.updateCell);
  const registerSendChatMessageFunction = useCanvasStore((state) => state.registerSendChatMessageFunction);

  // --- Instantiate the investigation events hook --- 
  const { 
    isInvestigationRunning, 
    currentStatus, 
    wsStatus // Destructure wsStatus as it's now returned
  } = useInvestigationEvents({
    notebookId: notebookId, 
    // REMOVE: streamedMessages: investigationEventStream, 
    // Provide the actual cell manipulation functions from Zustand
    onCreateCell: (params) => {
        console.log("[AIChatPanel onCreateCell using Zustand] Creating cell:", params.id, params.type);
        // Map InvestigationEvent CellCreationParams to Zustand's Cell format
        // Using handleCellUpdate which should handle adding new cells
        const newCellData = {
            id: params.id,
            notebook_id: notebookId, 
            type: params.type as any, // Cast needed if types mismatch slightly
            content: params.content,
            status: params.status as any,
            result: params.result,
            error: params.error,
            metadata: params.metadata,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            // Default position or logic to determine position might be needed
            position: Date.now(), // Use timestamp for default position? Adjust as needed.
        };
        handleCellUpdate(newCellData); // Use handleCellUpdate from Zustand store
    },
    onUpdateCell: (cellId, updates) => {
        console.log("[AIChatPanel onUpdateCell using Zustand] Updating cell:", cellId, updates);
        // Use handleCellUpdate for updates as well, providing a partial Cell object
        // Construct a Cell-like object with the id and the updates
        const partialCellData = {
            id: cellId,
            ...updates,
            // Ensure type compatibility if handleCellUpdate expects a full Cell
            // For example, add placeholders or fetch existing cell if needed.
            // However, handleCellUpdate in the store seems designed to merge.
             updated_at: new Date().toISOString(), // Ensure timestamp is updated
        } as Partial<Cell> & { id: string }; // Type assertion to guide TS
        
        // Cast to 'any' temporarily if type issues persist with handleCellUpdate signature
        // handleCellUpdate(partialCellData as any);
        handleCellUpdate(partialCellData as Cell); // Try casting to full Cell, assuming merge logic handles missing fields
    },
    onError: (message) => {
        console.error("[AIChatPanel Investigation Error]:", message);
        setError(`Investigation Error: ${message}`);
        toast({
            variant: "destructive",
            title: "Investigation Error",
            description: message,
        });
    },
  });

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
          // Separate initial messages into chat display and events
          const initialChatMessages: ChatMessage[] = [];
          const initialEvents: InvestigationEvent[] = [];
          existingMessages.forEach(msg => {
              const event = parseChatToInvestigationEvent(msg);
              if (event) {
                  // Check if it's a type that should be processed by the hook
                  // Add more types here if needed (e.g., step_completed for markdown)
                  if ([ 
                      "plan_created", "plan_cell_created", "step_completed", 
                      "summary_started", "summary_update", "summary_cell_created", 
                      "summary_cell_error", "github_tool_cell_created", "github_tool_error",
                      "investigation_complete", "error"
                    ].includes(event.type)) { 
                     initialEvents.push(event);
                  }
                  // Decide if the original message should *also* be displayed in chat
                  // For now, let's not display raw event JSON in chat.
              } else {
                  // If it's not an event, add it to chat display
                  initialChatMessages.push(msg);
              }
          });
          setMessages(initialChatMessages);
          setInvestigationEventStream(initialEvents);
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
      await api.chat.unifiedChat(lastFailedMessage, notebookId, sessionId, handleMessageOrEvent)

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
      // This is now handled by useInvestigationEvents via Zustand store
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

      // Check if content contains a cell_response
      try {
          const potentialCellResponse = JSON.parse(message.content);
          if (potentialCellResponse && typeof potentialCellResponse === 'object' && potentialCellResponse.type === 'cell_response') {
              // It IS a cell_response, process it using the Zustand store action
              console.log("[AIChatPanel] Processing cell_response:", potentialCellResponse);
              // Ensure cell_params exists and the TOP-LEVEL cell_id exists
              if (potentialCellResponse.cell_params && potentialCellResponse.cell_id) {
                  // Map the cell_params from the event to the structure expected by handleCellUpdate
                  // handleCellUpdate expects a Cell object (or partial Cell)
                  const cellDataForStore = {
                      id: potentialCellResponse.cell_id,
                      notebook_id: potentialCellResponse.cell_params.notebook_id || notebookId, // Use notebookId from props as fallback
                      type: potentialCellResponse.cell_params.cell_type as any || 'markdown', // Default type if missing, use 'any' temporarily for type flexibility
                      content: potentialCellResponse.cell_params.content || '',
                      status: potentialCellResponse.cell_params.status || 'idle',
                      result: potentialCellResponse.cell_params.result, // Pass result object
                      error: potentialCellResponse.cell_params.error,
                      metadata: potentialCellResponse.cell_params.metadata,
                      // Ensure timestamps exist, default if necessary
                      created_at: potentialCellResponse.cell_params.created_at || new Date().toISOString(), 
                      updated_at: potentialCellResponse.cell_params.updated_at || new Date().toISOString(), 
                      position: potentialCellResponse.cell_params.position ?? Date.now(), // Use ?? for nullish coalescing, provide default position
                      dependencies: potentialCellResponse.cell_params.dependencies || [], // Default to empty array
                      // Add other potential fields from Cell type if needed
                      connection_id: potentialCellResponse.cell_params.connection_id,
                      tool_call_id: potentialCellResponse.cell_params.tool_call_id,
                      tool_name: potentialCellResponse.cell_params.tool_name,
                      tool_arguments: potentialCellResponse.cell_params.tool_arguments,
                      settings: potentialCellResponse.cell_params.settings,
                      step_id: potentialCellResponse.cell_params.step_id, // Ensure step_id is mapped if present
                  };
                  console.log("Calling handleCellUpdate with:", cellDataForStore);
                  // Cast to Cell type; handleCellUpdate in store should handle merging partial data if needed
                  handleCellUpdate(cellDataForStore as Cell); 
              } else {
                  console.error("Received cell_response is missing cell_params or top-level cell_id", potentialCellResponse);
              }

              // Agent status updates based on cell status might be complex if multiple cells per step.
              // Consider relying on dedicated agent status events instead of deactivating here.
              /*
              if (message.agent && potentialCellResponse.cell_params?.status && ['success', 'error'].includes(potentialCellResponse.cell_params.status)) {
                 setAgentStatuses((prev) => { ... });
              }
              */

              return; // Stop processing this message for the main chat display
          }
      } catch (e) {
          // Not a parsable JSON or not a cell_response, continue processing as a regular message.
          // Error is logged inside the catch block, avoid duplicate logging here.
          console.log("[AIChatPanel handleMessageOrEvent] Processing as regular message:", message.content.substring(0, 50))
      }

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
    e?.preventDefault();
    const messageToSend = input.trim();
    if (!messageToSend) return; // Check if message is empty after trimming

    setInput(""); // Clear the input state field immediately

    // Call the programmatic sender function
    await programmaticSendMessage(messageToSend);
  }

  // ADDED: Function to programmatically send a message (used by store action)
  const programmaticSendMessage = async (messageContent: string) => {
    // Check session and loading state
    if (!messageContent || isLoading || !sessionId) {
      console.warn("[AIChatPanel programmaticSendMessage] Aborted: Invalid state (isLoading, no sessionId, or empty message)", { isLoading, sessionId, messageContent });
      return;
    }

    setError(null)
    setLastFailedMessage(null)
    setIsLoading(true)

    // 1. Add user message to the UI immediately
    const userMessage: ChatMessage = {
      role: 'user',
      content: messageContent,
      timestamp: new Date().toISOString(),
      id: `user-${Date.now()}`
    };
    setMessages((prev) => [...prev, userMessage]);

    // Define placeholder for assistant's response
    const assistantPlaceholder: ChatMessage = {
      role: "model",
      content: "", // Initially empty, will be populated by stream
      timestamp: new Date().toISOString(),
      id: `assistant-loading-${Date.now()}`, // Temporary ID
    }

    try {
      // Add a placeholder for the assistant's response
      setMessages((prev) => [...prev, assistantPlaceholder])

      // Call the correct sendMessage endpoint, not unifiedChat
      console.log(`Calling sendMessage with sessionId: ${sessionId}`)
      // await api.chat.unifiedChat(userMessageContent, notebookId, sessionId, handleMessageOrEvent)
      await api.chat.sendMessage(sessionId, messageContent, handleMessageOrEvent)

      // Remove the loading placeholder *after* the stream finishes
      // The stream might have added the actual message, or status updates handled elsewhere
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantPlaceholder.id))

    } catch (error) {
      console.error("Error sending message:", error)

      // Remove the loading placeholder on error
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantPlaceholder.id))

      // Store the failed message for potential retry
      setLastFailedMessage(messageContent) // Use function argument

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

  // --- ADDED: Effect to register/unregister the sending function with the store ---
  useEffect(() => {
    // --- START Logging ---
    console.log(
      `[AIChatPanel Registration Effect] Running. Session ID: ${sessionId}, Investigation Running: ${isInvestigationRunning}`
    )
    // --- END Logging ---
    if (sessionId && !isInvestigationRunning) {
      // Only register if session is ready and not already in a complex investigation
      console.log("[AIChatPanel] Registering programmaticSendMessage with Zustand store.")
      registerSendChatMessageFunction(programmaticSendMessage)
    } else {
      // If session is not ready or investigation is running, ensure function is not registered
      console.log(
        `[AIChatPanel] De-registering programmaticSendMessage (Session ID: ${sessionId}, Investigation Running: ${isInvestigationRunning}).`
      )
      registerSendChatMessageFunction(null)
    }

    // Cleanup function to de-register on unmount or when dependencies change
    return () => {
      console.log("[AIChatPanel] De-registering programmaticSendMessage on cleanup.")
      registerSendChatMessageFunction(null)
    }
    // Dependencies: registration depends on session ID and investigation status
  }, [sessionId, isInvestigationRunning, registerSendChatMessageFunction])

  useEffect(() => {
    // Log the current value of isInvestigationRunning on every render where this effect might run
    console.log(`[Investigation Status Effect] isInvestigationRunning: ${isInvestigationRunning}`);

    // When the investigation stops running, mark all agents as inactive
    if (!isInvestigationRunning) {
      console.log("[Investigation Status Effect] Condition met: !isInvestigationRunning. Attempting to deactivate agents.");
      setAgentStatuses((prevStatuses) => {
        console.log("[Investigation Status Effect] Inside setAgentStatuses. Previous statuses:", JSON.stringify(prevStatuses));
        const updatedStatuses = { ...prevStatuses };
        let changed = false;
        Object.keys(updatedStatuses).forEach((agentType) => {
          // Deactivate non-chat agents that are currently marked as active
          if (agentType !== 'chat_agent' && updatedStatuses[agentType]?.isActive) { // Add null check for safety
             console.log(`[Investigation Status Effect] Deactivating agent: ${agentType}`);
             updatedStatuses[agentType] = {
               ...updatedStatuses[agentType],
               isActive: false,
             };
             changed = true;
          }
        });

        if (changed) {
            console.log("[Investigation Status Effect] Statuses changed. New statuses:", JSON.stringify(updatedStatuses));
            return updatedStatuses;
        } else {
            console.log("[Investigation Status Effect] No status changes needed.");
            return prevStatuses; // Only update state if something changed
        }
      });
    }
    // Dependency array ONLY includes the trigger: isInvestigationRunning
    // and setAgentStatuses which is stable and typically recommended by lint rules
  }, [isInvestigationRunning, setAgentStatuses]);

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Submit on Enter (if not Shift+Enter)
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault(); // Prevent default newline behavior
        handleSendMessage();
    }
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
