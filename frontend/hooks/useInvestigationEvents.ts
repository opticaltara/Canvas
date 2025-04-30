"use client"

import { useState, useEffect, useCallback } from "react"
import { useWebSocket } from "./useWebSocket"
import { useToast } from "@/hooks/use-toast"
import { useCanvasStore } from "@/store/canvas" // Import useCanvasStore
import { type CellStatus } from "@/store/types"; // Import CellStatus type

// Define event types based on the documentation
export interface BaseEvent {
  type: string
  status: string
  timestamp: number
}

export interface PlanCreatedEvent extends BaseEvent {
  type: "plan_created"
  status: "plan_created"
  thinking: string
}

export interface PlanCellCreatedEvent extends BaseEvent {
  type: "plan_cell_created"
  status: "plan_cell_created"
}

export interface StepCompletedEvent extends BaseEvent {
  type: string // "step_${stepId}_completed"
  status: "step_completed"
  step_id: string
  step_type: "log" | "metric" | "markdown" | "sql" | "python" | "ai_query"
  cell_id: string
  result: {
    data: any
    query: string
    error: string | null
    metadata?: Record<string, any>
  }
}

export interface PlanRevisedEvent extends BaseEvent {
  type: "plan_revised"
  status: "plan_revised"
  explanation: string
}

export interface SummaryCreatedEvent extends BaseEvent {
  type: "summary_created"
  status: "summary_created"
}

export interface ErrorEvent extends BaseEvent {
  type: "error"
  status: "error"
  message: string
}

export type InvestigationEvent =
  | PlanCreatedEvent
  | PlanCellCreatedEvent
  | StepCompletedEvent
  | PlanRevisedEvent
  | SummaryCreatedEvent
  | ErrorEvent

// Define cell creation parameters
export interface CellCreationParams {
  id: string
  step_id: string
  type: string
  content: string
  status: "idle" | "running" | "success" | "error"
  result?: any
  error?: string
  metadata?: Record<string, any>
}

interface UseInvestigationEventsProps {
  notebookId: string
  onCreateCell: (params: CellCreationParams) => void
  onUpdateCell: (cellId: string, updates: Partial<CellCreationParams>) => void
  onError: (message: string) => void
}

export function useInvestigationEvents({
  notebookId,
  onCreateCell,
  onUpdateCell,
  onError,
}: UseInvestigationEventsProps) {
  const { status, messages, sendMessage } = useWebSocket(notebookId)
  const [isInvestigationRunning, setIsInvestigationRunning] = useState(false)
  const [currentPlan, setCurrentPlan] = useState<string | null>(null)
  const [currentStatus, setCurrentStatus] = useState<string | null>(null)
  const [steps, setSteps] = useState<Record<string, StepCompletedEvent>>({})
  const { toast } = useToast()

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!messages.length) return

    // Process the latest message
    const message = messages[messages.length - 1]

    try {
      handleEvent(message as InvestigationEvent)
    } catch (error) {
      console.error("Error handling investigation event:", error)
      const errorMessage = error instanceof Error ? error.message : String(error);
      onError(`Failed to process event: ${errorMessage}`)
    }
  }, [messages])

  // Handle different event types
  const handleEvent = useCallback(
    (event: InvestigationEvent) => {
      console.log("Received investigation event:", event)

      switch (event.type) {
        case "plan_created":
          handlePlanCreated(event as PlanCreatedEvent)
          break
        case "plan_cell_created":
          handlePlanCellCreated()
          break
        case "plan_revised":
          handlePlanRevised(event as PlanRevisedEvent)
          break
        case "summary_created":
          handleSummaryCreated(event as SummaryCreatedEvent)
          break
        case "error":
          handleError(event as ErrorEvent)
          break
        default:
          // Check if it's a step completion event
          if (event.type.startsWith("step_") && event.type.endsWith("_completed")) {
            handleStepCompleted(event as StepCompletedEvent)
          } else {
            console.warn("Unknown event type:", event.type)
          }
      }
    },
    [onCreateCell, onUpdateCell, onError],
  )

  // Handle plan_created event - just update status, don't create a cell
  const handlePlanCreated = useCallback(
    (event: PlanCreatedEvent) => {
      setIsInvestigationRunning(true)
      setCurrentPlan(event.thinking)
      setCurrentStatus("Creating investigation plan...")

      toast({
        title: "Investigation Started",
        description: "AI is creating an investigation plan...",
      })
    },
    [toast],
  )

  // Handle plan_cell_created event - just update status, don't create a cell
  const handlePlanCellCreated = useCallback(() => {
    setCurrentStatus("Plan created, executing steps...")
  }, [])

  // Handle step_*_completed events - THIS creates cells
  const handleStepCompleted = useCallback(
    (event: StepCompletedEvent) => {
      // Store the step information
      setSteps((prev) => ({
        ...prev,
        [event.step_id]: event,
      }))

      // Update status
      setCurrentStatus(`Completed step ${event.step_id}`)

      // Create the cell - THIS IS THE ONLY PLACE WE CREATE CELLS
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: event.step_id,
        type: event.step_type,
        content: event.result.query || "",
        status: event.result.error ? "error" : "success",
        result: event.result.data,
        error: event.result.error || undefined,
      }

      // Create the cell
      onCreateCell(cellParams)

      toast({
        title: "Step Completed",
        description: `Investigation step ${event.step_id} completed`,
      })
    },
    [onCreateCell, toast],
  )

  // Handle plan_revised event - just update status, don't create a cell
  const handlePlanRevised = useCallback(
    (event: PlanRevisedEvent) => {
      setCurrentPlan((prev) => (prev ? `${prev}\n\n### Plan Revision\n${event.explanation}` : event.explanation))
      setCurrentStatus("Investigation plan revised")

      toast({
        title: "Plan Revised",
        description: "The investigation plan has been updated based on findings.",
      })
    },
    [currentPlan, toast],
  )

  // Handle summary_created event - just update status, don't create a cell
  const handleSummaryCreated = useCallback(
    (event: SummaryCreatedEvent) => {
      setIsInvestigationRunning(false)
      setCurrentStatus("Investigation completed")

      toast({
        title: "Investigation Complete",
        description: "The investigation has been completed and a summary has been created.",
      })
    },
    [toast],
  )

  // Handle error event
  const handleError = useCallback(
    (event: ErrorEvent) => {
      setCurrentStatus(`Error: ${event.message}`)
      onError(event.message)

      toast({
        variant: "destructive",
        title: "Investigation Error",
        description: event.message,
      })
    },
    [onError, toast],
  )

  const handleCellCreation = useCallback(
    (message: any) => {
      console.log("🔍 handleCellCreation called with message:", message)

      try {
        // Parse the content if it's a string
        let parsedContent
        if (typeof message.content === "string") {
          try {
            parsedContent = JSON.parse(message.content)
            console.log("🔍 Successfully parsed message content:", parsedContent)
          } catch (e) {
            console.error("🔍 Error parsing message content:", e)
            parsedContent = message.content
          }
        } else {
          parsedContent = message.content
        }

        // Check for cell creation patterns
        if (
          (parsedContent.status_type === "plan_cell_created" || parsedContent.type === "cell_response") &&
          parsedContent.cell_params
        ) {
          console.log("🔍 Found cell creation pattern in message")
          console.log("🔍 Cell params:", parsedContent.cell_params)

          const { notebook_id, cell_type, content, position, metadata } = parsedContent.cell_params

          // Generate a temporary ID for the cell
          const tempId = `temp-${Date.now()}`
          console.log("🔍 Generated temporary ID for cell:", tempId)

          // Create cell data object
          const cellData = {
            id: tempId,
            notebook_id,
            type: cell_type,
            content,
            position,
            metadata,
            status: "idle" as CellStatus,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          }

          console.log("🔍 Created cell data object:", {
            id: cellData.id,
            notebook_id: cellData.notebook_id,
            type: cellData.type,
            content_length: cellData.content?.length || 0,
            metadata: cellData.metadata ? "present" : "absent",
          })

          // Log cell parameters for debugging
          console.log("🔍 Cell creation parameters:", {
            notebook_id,
            cell_type,
            content_length: content?.length || 0,
            metadata: metadata ? "present" : "absent",
          })

          // Add the cell directly to the canvas store
          console.log("🔍 Adding cell to canvas store:", tempId, `(${cell_type})`)
          const canvasStore = useCanvasStore.getState()
          console.log("🔍 Current active notebook ID in store:", canvasStore.activeNotebookId)
          console.log("🔍 Current cells count in store:", canvasStore.cells.length)

          canvasStore.handleCellUpdate(cellData)

          console.log(`🔍 Added new ${cell_type} cell from agent ${parsedContent.agent_type} directly to canvas store`)

          // Also call the onCreateCell callback if provided
          if (onCreateCell) {
            console.log("🔍 Calling onCreateCell callback")
            onCreateCell({
              id: tempId,
              step_id: '',
              type: cell_type,
              content,
              status: "idle",
              metadata,
            })
          }
        } else {
          console.log("🔍 Message does not match cell creation pattern:", parsedContent)
        }
      } catch (error) {
        console.error("🔍 Error in handleCellCreation:", error)
      }
    },
    [onCreateCell],
  )

  return {
    wsStatus: status,
    isInvestigationRunning,
    currentPlan,
    currentStatus,
    steps,
  }
}
