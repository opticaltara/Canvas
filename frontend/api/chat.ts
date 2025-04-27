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
  // Helper function to handle cell creation from agent responses
  const handleAgentResponse = (message: ChatMessage) => {
    try {
      // Parse the content as JSON
      const contentObj = JSON.parse(message.content)

      console.log("Parsed agent response content:", contentObj)

      // Check if this is a final response containing successful tool calls to be turned into cells
      if (
        contentObj.type === "cell_response" &&
        contentObj.status_type === "step_completed" && // Or another indicator of final result
        contentObj.result?.tool_calls &&
        Array.isArray(contentObj.result.tool_calls) &&
        contentObj.result.tool_calls.length > 0 &&
        contentObj.agent_type === "github" // Ensure this logic is specific to GitHub agent for now
      ) {
        console.log(
          `Received final GitHub agent response with ${contentObj.result.tool_calls.length} tool calls. Creating individual cells.`,
        )

        const canvasStore = useCanvasStore.getState()
        const notebookId = contentObj.cell_params?.notebook_id || canvasStore.activeNotebookId

        if (!notebookId) {
          console.error("Cannot create cells, missing notebook ID.")
          return // Skip cell creation if no notebook ID
        }
        
        // Ensure active notebook is set
        if (canvasStore.activeNotebookId !== notebookId) {
          console.log(`Setting active notebook ID in store to: ${notebookId}`)
          canvasStore.setActiveNotebook(notebookId)
        }


        contentObj.result.tool_calls.forEach((toolCall: any, index: number) => {
          if (!toolCall.tool_name || !toolCall.tool_args) {
             console.warn("Skipping tool call due to missing name or args:", toolCall)
             return;
          }
          
          const cellId = `github-${toolCall.tool_call_id || Date.now() + index}` // Use tool call ID if available
          const cellContent = `GitHub Tool: ${toolCall.tool_name}` // Simple content for the cell

          const cellData = {
            id: cellId,
            notebook_id: notebookId,
            type: "github" as CellType,
            content: cellContent,
            status: "success" as CellStatus, // Mark as success initially as it came from a successful agent run
            metadata: {
              // Store tool info in metadata for the cell component
              toolName: toolCall.tool_name,
              toolArgs: toolCall.tool_args,
              source_agent: contentObj.agent_type,
              // Optionally add original query or step info if needed
              // original_query: contentObj.result?.query
            },
            result: toolCall.tool_result, // Store the result obtained by the agent
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          }

          console.log(`Adding GitHub cell to canvas store: ${cellId} for tool ${toolCall.tool_name}`)
          canvasStore.handleCellUpdate(cellData) // Use handleCellUpdate which acts like add/update
        })

        // Mark the original message as handled regarding cell creation
        // To prevent any potential duplicate processing if the structure was ambiguous
        // We might need a way to signal that the message's primary purpose (creating cells) is done.
        message.content = JSON.stringify({ type: "status", message: `Generated ${contentObj.result.tool_calls.length} GitHub cells.` })


      } else if (
        contentObj.type === "cell_response" &&
        contentObj.cell_params &&
        ["plan_cell_created", "summary_created"].includes(contentObj.status_type) &&
        contentObj.agent_type !== "github" // Make sure the old logic doesn't apply to github agent final messages
      ) {
        // --- Keep existing logic for non-GitHub agents or other message types ---
        const { notebook_id, cell_type, content, metadata, position } = contentObj.cell_params

        console.log("Applying standard cell creation logic for:", {
          notebook_id,
          cell_type,
          content_length: content?.length || 0,
          metadata: metadata ? "present" : "absent",
          agent: contentObj.agent_type,
        })
        const cellId = `temp-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`

        const cellData = {
          id: cellId,
          notebook_id: notebook_id as string,
          type: cell_type as CellType,
          content: content as string,
          status: "idle" as CellStatus,
          metadata: metadata || {},
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }

        console.log(`Adding cell to canvas store: ${cellId} (${cell_type})`)
        const canvasStore = useCanvasStore.getState()

        if (canvasStore.activeNotebookId !== notebook_id) {
          console.log(`Setting active notebook ID in store to: ${notebook_id}`)
          canvasStore.setActiveNotebook(notebook_id)
        }

        canvasStore.handleCellUpdate(cellData)
        console.log(`Added new ${cell_type} cell from agent ${contentObj.agent_type} directly to canvas store`)
         // Optionally modify the message content if needed after handling
         message.content = JSON.stringify({ type: "status", message: `Generated ${cell_type} cell.` })

      }
    } catch (error) {
      // Log error but don't modify the original message if parsing failed
      console.error("Error processing agent response for cell creation:", error, "Original message:", message)
    }
  }

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

            // NEW: Process chunk for potential cell creation *before* passing to UI callback
            if (chunk.role === 'model' && chunk.content) {
               handleAgentResponse(chunk); // This function might modify chunk.content if it handles cell creation
            }

            // Pass the (potentially modified) chunk to the UI callback
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
        // NEW: Process final chunk for potential cell creation
        if (chunk.role === 'model' && chunk.content) {
            handleAgentResponse(chunk);
        }
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
