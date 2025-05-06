import { BACKEND_URL } from "@/config/api-config"
import { useCanvasStore } from "@/store/canvasStore"
import { CellType, CellStatus } from "../store/types"

export interface ChatMessage {
  role: "user" | "model"
  content: string
  timestamp: string
  id?: string // Optional ID field
  agent?: string // Added optional agent field
}

// Add a new interface for cell creation events
export interface CellCreationEvent {
  type: "cell_created"
  cellParams: {
    id: string
    notebook_id: string
    cell_type: string
    content: string
    metadata?: Record<string, any>
  }
  agentType: string
  _handled?: boolean
}

export interface ChatSession {
  session_id: string
}

// Helper function to handle streaming responses
const handleStreamingResponse = async (
  response: Response,
  onChunk: (message: ChatMessage | CellCreationEvent) => void,
): Promise<ChatMessage[]> => {
  if (!response.ok) {
    const errorText = await response.text()
    let errorMessage = `API error: ${response.status}`

    try {
      // Try to parse the error as JSON
      const errorJson = JSON.parse(errorText)
      if (errorJson.error) {
        if (typeof errorJson.error === "string") {
          errorMessage = errorJson.error
        } else if (errorJson.error.message) {
          errorMessage = errorJson.error.message

          // Add error type if available
          if (errorJson.error.type) {
            errorMessage += ` (${errorJson.error.type})`
          }
        }
      }
    } catch (e) {
      // If parsing fails, use the raw error text
      errorMessage += ` - ${errorText}`
    }

    throw new Error(errorMessage)
  }

  if (!response.body) {
    throw new Error("Response body is null")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  const messages: ChatMessage[] = []
  let currentMessage: ChatMessage | null = null

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      // Decode the chunk and add it to our buffer
      buffer += decoder.decode(value, { stream: true })

      // Process complete lines in the buffer
      const lines = buffer.split("\n")
      buffer = lines.pop() || "" // Keep the last incomplete line in the buffer

      for (const line of lines) {
        if (line.trim()) {
          try {
            // Check if the line contains an error
            if (line.includes('"error":')) {
              const errorJson = JSON.parse(line)
              if (errorJson.error) {
                let errorMessage = "Error in response"

                if (typeof errorJson.error === "string") {
                  errorMessage = errorJson.error
                } else if (errorJson.error.message) {
                  errorMessage = errorJson.error.message

                  // Add error type if available
                  if (errorJson.error.type) {
                    errorMessage += ` (${errorJson.error.type})`
                  }
                }

                throw new Error(errorMessage)
              }
            }

            const chunk = JSON.parse(line) as ChatMessage

            // Pass the UNMODIFIED chunk to the UI callback
            onChunk(chunk)
            messages.push(chunk) // Store the raw chunk
            currentMessage = chunk // Update the 'last' message seen

          } catch (e) {
            console.error("Error parsing JSON:", e, "Line:", line)

            // If this is an error we threw ourselves, rethrow it
            if (e instanceof Error && (e.message.includes("Error in response") || e.message.includes("overloaded"))) {
              throw e
            }

            // Add a more comprehensive error handling for malformed responses
            if (line.includes("error") || line.includes("exception") || line.includes("fail")) {
              // Try to extract error message from potentially malformed JSON
              let errorMessage = "Error in response"
              try {
                // Try to extract just the error part if it's not valid JSON
                const errorMatch = line.match(/"error":\s*"([^"]+)"/)
                if (errorMatch && errorMatch[1]) {
                  errorMessage = errorMatch[1]
                } else if (line.includes("timeout") || line.includes("timed out")) {
                  errorMessage = "Request timed out. The server might be busy."
                }
              } catch (err) {
                // If that fails too, use the raw line
                errorMessage = `Error in response: ${line.substring(0, 100)}${line.length > 100 ? "..." : ""}`
              }

              throw new Error(errorMessage)
            }
          }
        }
      }
    }

    if (buffer.trim()) {
      try {
        const chunk = JSON.parse(buffer) as ChatMessage
        onChunk(chunk)
        messages.push(chunk)
        currentMessage = chunk
      } catch (e) {
         console.error("Error parsing final buffer JSON:", e, "Buffer:", buffer)
         // Decide if this error should be thrown or handled differently
         // For now, let's re-throw but provide context
         throw new Error(`Error parsing final chunk: ${e instanceof Error ? e.message : String(e)} - Buffer: ${buffer.substring(0, 100)}...`)
      }
    }

    return messages
  } catch (error) {
    console.error("Error reading stream:", error)
    throw error
  }
}

export const chatApi = {
  // Create a new chat session
  createSession: async (notebookId: string): Promise<ChatSession> => {
    if (!notebookId) {
      throw new Error("Notebook ID is required to create a chat session")
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat/sessions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          notebook_id: notebookId,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Failed to create chat session: ${response.status} - ${errorText}`)
      }

      return await response.json()
    } catch (error) {
      console.error("Error creating chat session:", error)
      throw error
    }
  },

  // Get all messages for a session
  getSessionMessages: async (sessionId: string): Promise<ChatMessage[]> => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/chat/sessions/${sessionId}/messages`)

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Failed to get session messages: ${response.status} - ${errorText}`)
      }

      const text = await response.text()
      const messages: ChatMessage[] = []
      let currentMessage: ChatMessage | null = null

      // Parse newline-delimited JSON
      for (const line of text.split("\n")) {
        if (line.trim()) {
          try {
            const chunk = JSON.parse(line) as ChatMessage

            // Check if this is a continuation of the current message
            if (currentMessage && currentMessage.role === chunk.role) {
              // Append to the existing message
              currentMessage.content += chunk.content
              currentMessage.timestamp = chunk.timestamp // Update timestamp to the latest
            } else {
              // This is a new message
              if (currentMessage) {
                messages.push(currentMessage)
              }
              currentMessage = { ...chunk }
            }
          } catch (e) {
            console.error("Error parsing message JSON:", e)
          }
        }
      }

      // Add the last message if it exists
      if (currentMessage) {
        messages.push(currentMessage)
      }

      return messages
    } catch (error) {
      console.error("Error getting session messages:", error)
      throw error
    }
  },

  // Send a message and get streaming response
  sendMessage: async (
    sessionId: string,
    prompt: string,
    onChunk: (message: ChatMessage | CellCreationEvent) => void,
  ): Promise<ChatMessage[]> => {
    try {
      const formData = new FormData()
      formData.append("prompt", prompt)

      const response = await fetch(`${BACKEND_URL}/api/chat/sessions/${sessionId}/messages`, {
        method: "POST",
        body: formData,
      })

      return await handleStreamingResponse(response, onChunk)
    } catch (error) {
      console.error("Error sending message:", error)
      throw error
    }
  },

  // Delete a chat session
  deleteSession: async (sessionId: string): Promise<void> => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/chat/sessions/${sessionId}`, {
        method: "DELETE",
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Failed to delete chat session: ${response.status} - ${errorText}`)
      }
    } catch (error) {
      console.error("Error deleting chat session:", error)
      throw error
    }
  },

  // Unified chat endpoint
  unifiedChat: async (
    prompt: string,
    notebookId: string, // Now required
    sessionId?: string,
    onChunk: (message: ChatMessage | CellCreationEvent) => void = () => {},
  ): Promise<ChatMessage[]> => {
    if (!notebookId) {
      throw new Error("Notebook ID is required for chat requests")
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt,
          notebook_id: notebookId, // Always include notebook_id
          session_id: sessionId,
        }),
      })

      return await handleStreamingResponse(response, onChunk)
    } catch (error) {
      console.error("Error in unified chat:", error)
      throw error
    }
  },
}
