"use client"

import { useState, useEffect, useCallback } from "react"
import { useWebSocket, type WebSocketMessage } from "./useWebSocket" // Restore import and add WebSocketMessage
import { useToast } from "@/hooks/use-toast"
import { useCanvasStore } from "@/store/canvasStore" // Import useCanvasStore
import { type CellStatus, CellType } from "@/store/types"; // Import CellStatus & CellType // Ensure CellType is imported

// Define a simple QueryResult type matching backend structure
export interface QueryResult {
  data: any;
  query: string;
  error?: string | null;
  metadata?: Record<string, any>;
  // Add fields specific to subclasses if needed, e.g., tool_calls for GithubQueryResult
  tool_calls?: any[] | null;
}

export interface CodeReference {
  file_path: string;
  line_number?: number | null;
  code_snippet?: string | null;
  url?: string | null;
  component_name?: string | null;
}

export interface Finding {
  summary: string;
  details?: string | null;
  code_reference?: CodeReference | null;
  supporting_quotes?: string[] | null;
}

export interface RelatedItem {
  type: "Pull Request" | "Issue" | "Commit" | "Discussion" | "Documentation" | "Other";
  identifier: string;
  url: string;
  relevance?: string | null;
  status?: string | null;
}

export interface InvestigationReportData {
  query: string;
  title: string;
  status?: string | null;
  status_reason?: string | null;
  estimated_severity?: "Critical" | "High" | "Medium" | "Low" | "Unknown" | null;
  issue_summary: Finding;
  root_cause: Finding;
  root_cause_confidence?: "Low" | "Medium" | "High" | null;
  key_components: CodeReference[];
  related_items: RelatedItem[];
  proposed_fix?: Finding | null;
  proposed_fix_confidence?: "Low" | "Medium" | "High" | null;
  affected_context?: Finding | null;
  suggested_next_steps: string[];
  tags: string[];
  error?: string | null;
}

// Interface for the event carrying the report
export interface InvestigationReportEvent extends BaseEvent {
    type: "investigation_report"; // Specific type identifier
    status: "success"; // Assume success if report is generated
    agent_type: "investigation_reporter";
    cell_id: string; // ID for the new cell
    report: InvestigationReportData; // The actual report payload
    cell_params?: Partial<CellCreationParams>; // Optional backend params
}

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
  step_type: "log" | "metric" | "markdown" | "github" // Removed summarization, sql, python, ai_query (update if others are used)
  cell_id: string
  result: QueryResult // Use the imported QueryResult type from store/types
  cell_params: Partial<CellCreationParams>; // Add cell_params used by backend
  agent_type: string;
  is_single_step_plan?: boolean;
  tool_call_record_id?: string;
}

export interface PlanRevisedEvent extends BaseEvent {
  type: "plan_revised"
  status: "plan_revised"
  explanation: string
}

export interface ErrorEvent extends BaseEvent {
  type: "error"
  status: "error"
  message: string
}

// --- New Summarization Events ---
export interface SummaryStartedEvent extends BaseEvent {
  type: "summary_started";
  status: "summary_started";
  agent_type: "summarization_generator";
}

export interface SummaryUpdateEvent extends BaseEvent {
  type: "summary_update";
  status: "summary_update";
  agent_type: "summarization_generator";
  update_info: {
    status?: string; // e.g., "agent_ready", "attempt_start"
    message?: string;
    error?: string;
    attempt?: number;
    max_attempts?: number;
  };
}

export interface SummaryCellCreatedEvent extends BaseEvent {
  type: "summary_cell_created";
  status: "summary_cell_created";
  agent_type: "summarization_generator";
  cell_params: CellCreationParams; // Assuming backend sends params matching CellCreationParams
  cell_id: string;
  error: string | null; // Error during summary generation
}

export interface SummaryCellErrorEvent extends BaseEvent {
  type: "summary_cell_error";
  status: "summary_cell_error";
  agent_type: "summarization_generator";
  error: string; // Error during cell creation itself
}

// --- ADDED: GitHub Tool Events --- 
export interface GithubToolCellCreatedEvent extends BaseEvent {
  type: "github_tool_cell_created";
  status: "success";
  original_plan_step_id: string;
  cell_id: string;
  tool_call_record_id: string;
  tool_name: string;
  tool_args: Record<string, any>;
  result: any; // The actual result from the tool call
  agent_type: "github_tool";
  cell_params: Partial<CellCreationParams>; // Params used by backend to create cell
}

export interface GithubToolErrorEvent extends BaseEvent {
  type: "github_tool_error";
  status: "error";
  original_plan_step_id?: string; // Optional as it might fail before step ID known
  tool_name?: string;
  tool_args?: Record<string, any>;
  error: string; // Error message from the tool execution/cell creation
  agent_type: "github_tool";
}

// --- Investigation Event Union ---
export type InvestigationEvent =
  | PlanCreatedEvent
  | PlanCellCreatedEvent
  | StepCompletedEvent
  | PlanRevisedEvent
  | ErrorEvent
  // Add new summary events
  | SummaryStartedEvent
  | SummaryUpdateEvent
  | SummaryCellCreatedEvent
  | SummaryCellErrorEvent
  // ADDED: GitHub Tool Events
  | GithubToolCellCreatedEvent
  | GithubToolErrorEvent
  // ADDED: Investigation Report Event
  | InvestigationReportEvent
  // ADDED: Cell Update Event
  | CellUpdateEvent
  // ADDED: Filesystem Tool Events
  | FileSystemToolCellCreatedEvent 
  | FileSystemToolErrorEvent
  // ADDED: Python Tool Events
  | PythonToolCellCreatedEvent
  | PythonToolErrorEvent

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

// --- ADDED: Interface for Cell Update Event --- 
export interface CellUpdateEvent extends BaseEvent {
  type: "cell_update";
  data: Partial<CellCreationParams> & { id: string }; // Ensure ID is always present
}
// --- END: Added Interface --- 

// --- ADDED: Filesystem Tool Event Interfaces (mirroring GitHub) ---
export interface FileSystemToolCellCreatedEvent extends BaseEvent {
  type: "filesystem_tool_cell_created";
  status: "success";
  original_plan_step_id: string;
  cell_id: string;
  tool_name: string;
  tool_args: Record<string, any>;
  result: any;
  agent_type: "filesystem"; // Specific agent type
  cell_params: Partial<CellCreationParams>;
}

export interface FileSystemToolErrorEvent extends BaseEvent {
  type: "filesystem_tool_error";
  status: "error";
  original_plan_step_id?: string;
  tool_name?: string;
  tool_args?: Record<string, any>;
  error: string;
  agent_type: "filesystem"; // Specific agent type
}
// --- END: Added Filesystem Interfaces ---

// --- ADDED: Python Tool Event Interfaces --- 
export interface PythonToolCellCreatedEvent extends BaseEvent {
  type: "python_tool_cell_created";
  status: "success";
  original_plan_step_id: string;
  cell_id: string;
  tool_name: string; // e.g., "run_python_code"
  tool_args: { python_code?: string; [key: string]: any }; // Expect python_code
  result: any; // Structured result from Python execution (stdout, return_value, error)
  agent_type: "python"; // Specific agent type
  cell_params?: Partial<CellCreationParams>; // Use Partial since backend might not send all
}

export interface PythonToolErrorEvent extends BaseEvent {
  type: "python_tool_error";
  status: "error";
  original_plan_step_id?: string;
  tool_call_id?: string; // Added if available from backend
  tool_name?: string;
  tool_args?: { python_code?: string; [key: string]: any };
  error: string;
  agent_type: "python"; // Specific agent type
}
// --- END: Added Python Interfaces ---

interface UseInvestigationEventsProps {
  notebookId: string
  // REMOVED: streamedMessages: InvestigationEvent[] 
  onCreateCell: (params: CellCreationParams) => void
  onUpdateCell: (cellId: string, updates: Partial<CellCreationParams>, metadata?: Record<string, any>) => void
  onError: (message: string) => void
}

export function useInvestigationEvents({
  notebookId,
  // REMOVED: streamedMessages, 
  onCreateCell,
  onUpdateCell,
  onError,
}: UseInvestigationEventsProps) {
  const [latestEvent, setLatestEvent] = useState<InvestigationEvent | null>(null);

  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    // Assuming all messages for investigation are InvestigationEvent
    setLatestEvent(message as InvestigationEvent);
  }, []);

  const { status: wsStatus, sendMessage } = useWebSocket(notebookId, handleWebSocketMessage) // Call useWebSocket with callback
  const [isInvestigationRunning, setIsInvestigationRunning] = useState(false)
  const [currentPlan, setCurrentPlan] = useState<string | null>(null)
  const [currentStatus, setCurrentStatus] = useState<string | null>(null)
  const [steps, setSteps] = useState<Record<string, StepCompletedEvent>>({})
  const { toast } = useToast()
  // REMOVED: const [processedIndices, setProcessedIndices] = useState<Set<number>>(new Set()); 

  // --- START: Define individual event handlers FIRST ---
  // handlePlanCreated
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

  // handlePlanCellCreated
  const handlePlanCellCreated = useCallback(() => {
    setCurrentStatus("Plan created, executing steps...")
  }, [])

  // handleStepCompleted
  const handleStepCompleted = useCallback(
    (event: StepCompletedEvent) => {
      // Store the step information
      setSteps((prev) => ({
        ...prev,
        [event.step_id]: event,
      }))

      // Update status
      setCurrentStatus(`Completed step ${event.step_id}`)

      // Use content directly from cell_params if available, otherwise derive (for non-github)
      let cellContent = event.cell_params?.content || event.result?.query || "";
      if (event.step_type === "markdown" && event.result?.data) {
        cellContent = String(event.result.data);
      }

      // Determine cell type based on step type
      let cellType: CellType = "markdown"; // Default
      if (event.step_type === "github") {
        cellType = "github";
      } else if (event.step_type === "log" || event.step_type === "metric") {
        // Assign a type for logs/metrics if needed, maybe a generic 'output' or 'markdown'
        cellType = "markdown"; // Or a new specific type
      } // Add other mappings if needed

      // Create the cell - THIS IS THE ONLY PLACE WE CREATE STEP CELLS
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: event.step_id,
        type: cellType, // Use the determined cell type
        content: cellContent,
        status: event.result.error ? "error" : "success",
        result: event.result.data,
        error: event.result.error || undefined,
        metadata: { ...(event.result.metadata || {}), ...(event.cell_params?.metadata || {})}, // Merge metadata
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

  // handlePlanRevised
  const handlePlanRevised = useCallback(
    (event: PlanRevisedEvent) => {
      setCurrentPlan((prev) => (prev ? `${prev}\\n\\n### Plan Revision\\n${event.explanation}` : event.explanation))
      setCurrentStatus("Investigation plan revised")

      toast({
        title: "Plan Revised",
        description: "The investigation plan has been updated based on findings.",
      })
    },
    [currentPlan, toast],
  )

  // handleSummaryStarted
  const handleSummaryStarted = useCallback(
    (event: SummaryStartedEvent) => {
      setCurrentStatus("Generating final answer...");
      toast({
        title: "Summarizing Findings",
        description: "AI is generating the final answer...",
      });
    },
    [toast]
  );

  // handleSummaryUpdate
  const handleSummaryUpdate = useCallback(
    (event: SummaryUpdateEvent) => {
      const { update_info } = event;
      let message = "Summarization in progress...";
      if (update_info.message) {
        message = update_info.message;
      } else if (update_info.status) {
        message = `Summarization status: ${update_info.status}`;
      }
      setCurrentStatus(message);

      // Optionally show toast for errors during update
      if (update_info.error) {
        console.error("Summarization update error:", update_info.error);
        toast({
          variant: "destructive",
          title: "Summarization Issue",
          description: `An issue occurred during summary generation: ${update_info.error}`,
        });
      }
    },
    [toast]
  );

  // handleSummaryCellCreated
  const handleSummaryCellCreated = useCallback(
    (event: SummaryCellCreatedEvent) => {
      // Extract cell parameters from the event
      const backendCellParams = event.cell_params;

      // Create the final summary cell
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: backendCellParams.step_id || "final_summary", // Use step_id from backend or default
        type: "markdown", // Explicitly set type to markdown for summary
        content: backendCellParams.content || "",
        status: event.error ? "error" : "success", // Status reflects summary generation error
        result: null, // No direct 'result' data for the summary cell itself
        error: event.error || undefined,
        metadata: backendCellParams.metadata,
      };

      onCreateCell(cellParams);

      // Final status updates
      setIsInvestigationRunning(false); // Investigation is now complete
      setCurrentStatus("Investigation complete.");

      if (event.error) {
        toast({
          title: "Investigation Complete (with Summary Issue)",
          description: `Investigation finished, but there was an issue generating the summary: ${event.error}`,
        });
      } else {
        toast({
          title: "Investigation Complete",
          description: "The investigation has finished and the final answer is ready.",
        });
      }
    },
    [onCreateCell, toast]
  );

  // handleSummaryCellError
  const handleSummaryCellError = useCallback(
    (event: SummaryCellErrorEvent) => {
      const errorMsg = `Failed to create final summary cell: ${event.error}`;
      console.error(errorMsg);
      onError(errorMsg); // Propagate error
      setCurrentStatus("Error creating final summary cell.");
      setIsInvestigationRunning(false); // Mark as complete even if cell creation failed

      toast({
        variant: "destructive",
        title: "Summary Cell Error",
        description: errorMsg,
      });
    },
    [onError, toast]
  );

  // handleGithubToolCellCreated
  const handleGithubToolCellCreated = useCallback(
    (event: GithubToolCellCreatedEvent) => {
      setCurrentStatus(`Completed tool: ${event.tool_name}`);

      const backendCellParams = event.cell_params || {};
      const backendMetadata = backendCellParams.metadata || {};
      
      // Construct CellCreationParams for the frontend store
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: event.original_plan_step_id, // Link back to original plan step
        type: "github", // Explicitly set type to github
        content: backendCellParams.content || `GitHub: ${event.tool_name}`, // Use backend content or generate default
        status: "success", // Tool execution was successful
        result: event.result, // The actual result data from the tool
        error: undefined,
        // Combine metadata from backend params and event details
        metadata: {
          ...backendMetadata,
          tool_name: event.tool_name,
          tool_args: event.tool_args,
          tool_call_record_id: event.tool_call_record_id,
          pydantic_ai_tool_call_id: backendMetadata.pydantic_ai_tool_call_id, // Get original ID if present
          source_agent: event.agent_type,
        },
      };

      console.log(`[handleGithubToolCellCreated] Creating cell ${event.cell_id} with status: success and result.`);
      onCreateCell(cellParams); // Use onCreateCell with the constructed params
      
      // Original call was onUpdateCell:
      /* 
      console.log(`[handleGithubToolCellCreated] Updating cell ${event.cell_id} with status: success and result.`);
      onUpdateCell(event.cell_id, {
          status: "success",
          result: event.result,
          // We might also want to update metadata if it changed, but status/result are key
          metadata: cellParams.metadata, // Pass updated metadata too
          content: cellParams.content // Update content just in case backend changed it
      });
      */

      toast({
        title: "GitHub Tool Completed",
        description: `Tool '${event.tool_name}' executed successfully.`,
      });
    },
    [onCreateCell, toast] // Updated dependency from onUpdateCell to onCreateCell
  );

  // handleGithubToolError
  const handleGithubToolError = useCallback(
    (event: GithubToolErrorEvent) => {
      const errorMsg = `GitHub tool '${event.tool_name || 'unknown'} failed: ${event.error}`;
      console.error(errorMsg);
      onError(errorMsg); // Propagate error
      setCurrentStatus(`Error executing tool: ${event.tool_name || 'unknown'}`);
      // No cell is created for tool errors

      toast({
        variant: "destructive",
        title: "GitHub Tool Error",
        description: errorMsg,
      });
    },
    [onError, toast]
  );

  // --- ADDED: Handler for Filesystem Tool Cell Created ---
  const handleFileSystemToolCellCreated = useCallback(
    (event: FileSystemToolCellCreatedEvent) => {
      setCurrentStatus(`Completed tool: ${event.tool_name}`);

      const backendCellParams = event.cell_params || {};
      const backendMetadata = backendCellParams.metadata || {};
      
      // Construct CellCreationParams for the frontend store
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: event.original_plan_step_id,
        type: "filesystem", // Explicitly set type
        content: backendCellParams.content || `Filesystem: ${event.tool_name}`,
        status: "success",
        result: event.result,
        error: undefined,
        metadata: {
          ...backendMetadata,
          tool_name: event.tool_name,
          tool_args: event.tool_args,
          source_agent: event.agent_type,
        },
      };

      console.log(`[handleFileSystemToolCellCreated] Creating cell ${event.cell_id} with status: success and result.`);
      onCreateCell(cellParams);

      toast({
        title: "Filesystem Tool Completed",
        description: `Tool '${event.tool_name}' executed successfully.`,
      });
    },
    [onCreateCell, toast]
  );
  // --- END: Added Filesystem Handler ---

  // --- ADDED: Handler for Filesystem Tool Error ---
  const handleFileSystemToolError = useCallback(
    (event: FileSystemToolErrorEvent) => {
      const errorMsg = `Filesystem tool '${event.tool_name || 'unknown'} failed: ${event.error}`;
      console.error(errorMsg);
      onError(errorMsg); // Propagate error
      setCurrentStatus(`Error executing tool: ${event.tool_name || 'unknown'}`);
      // No cell is created for tool errors

      toast({
        variant: "destructive",
        title: "Filesystem Tool Error",
        description: errorMsg,
      });
    },
    [onError, toast]
  );
  // --- END: Added Filesystem Handler ---

  // --- ADDED: Handler for Python Tool Cell Created ---
  const handlePythonToolCellCreated = useCallback(
    (event: PythonToolCellCreatedEvent) => {
      setCurrentStatus(`Completed Python execution: ${event.tool_name}`);

      const backendCellParams = event.cell_params || {};
      const backendMetadata = backendCellParams.metadata || {};
      const toolArgs = event.tool_args || {};
      
      // Construct CellCreationParams for the frontend store
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: event.original_plan_step_id,
        type: "python", // Explicitly set type
        // Content should be the Python code itself
        content: toolArgs.python_code || backendCellParams.content || `Python code execution`, 
        status: "success", // Tool execution reported success
        // Result contains structured output { stdout, return_value, error? }
        // Backend sends CellResult, frontend expects raw content
        result: event.result?.content, // Extract content from CellResult if backend sends it
        error: event.result?.error || undefined, // Extract error from CellResult
        metadata: {
          ...backendMetadata,
          tool_name: event.tool_name,
          tool_args: toolArgs, // Store original args
          source_agent: event.agent_type,
          // Extract specific result parts into metadata for potential display?
          // stdout: event.result?.content?.stdout,
          // return_value: event.result?.content?.return_value,
        },
      };

      console.log(`[handlePythonToolCellCreated] Creating cell ${event.cell_id} with type: python, status: success.`);
      onCreateCell(cellParams);

      toast({
        title: "Python Execution Completed",
        description: `Code execution via '${event.tool_name}' successful.`,
      });
    },
    [onCreateCell, toast, setCurrentStatus] // Added setCurrentStatus dependency
  );
  // --- END: Added Python Handler ---

  // --- ADDED: Handler for Python Tool Error ---
  const handlePythonToolError = useCallback(
    (event: PythonToolErrorEvent) => {
      const errorMsg = `Python execution via '${event.tool_name || 'run_python_code'}' failed: ${event.error}`;
      console.error(errorMsg);
      onError(errorMsg); 
      setCurrentStatus(`Error executing Python code: ${event.tool_name || 'run_python_code'}`);
      // No cell is created for tool errors usually, but backend might create one?
      // If backend creates an error cell, a cell_update event might follow.

      toast({
        variant: "destructive",
        title: "Python Execution Error",
        description: errorMsg,
      });
    },
    [onError, toast, setCurrentStatus] // Added setCurrentStatus dependency
  );
  // --- END: Added Python Handler ---

  // handleStepUpdate
  const handleStepUpdate = useCallback((event: any) => {
    // Example: Update cell status visually if needed
    const stepId = event.type.split('_')[1]; // Extract step ID
    const updateInfo = event.update_info;
    console.log(`Received update for step ${stepId}:`, updateInfo);
    // Find the corresponding cell and update its status/display?
    // This might require finding the cellId associated with the stepId
    // and calling onUpdateCell. Needs more logic based on how cell IDs map to step IDs.
    // For now, just log it.
    setCurrentStatus(`Updating step ${stepId}: ${updateInfo?.message || updateInfo?.status || '...'}`);
  }, [onUpdateCell]); // Add dependencies if it interacts with state/props

  // handleError
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

  // --- ADDED: Handler for Investigation Report ---
  const handleInvestigationReport = useCallback(
    (event: InvestigationReportEvent) => {
      setCurrentStatus("Final investigation report generated.");
      setIsInvestigationRunning(false); // Investigation ends with the report

      const backendCellParams = event.cell_params || {};

      // Create the report cell
      const cellParams: CellCreationParams = {
        id: event.cell_id,
        step_id: backendCellParams.step_id || "final_report", // Use step_id or default
        type: "investigation_report", // Use the new cell type string value
        content: event.report.title || "Investigation Report", // Use report title or default
        status: event.report.error ? "error" : "success", // Status based on report error field
        result: event.report, // The full report object is the result
        error: event.report.error || undefined,
        metadata: { ...(backendCellParams.metadata || {}), source_agent: event.agent_type }, // Combine metadata
      };

      console.log(`[handleInvestigationReport] Creating cell ${event.cell_id} with type: "investigation_report"`);
      onCreateCell(cellParams);

      toast({
        title: "Investigation Report Ready",
        description: "The final investigation report has been generated.",
      });
    },
    [onCreateCell, toast]
  );
  // --- END: Added Report Handler ---

  // --- ADDED: Handler for Cell Update --- 
  const handleCellUpdate = useCallback(
    (event: CellUpdateEvent) => {
      const { id, ...updates } = event.data; // Extract ID and the rest are updates
      console.log(`[handleCellUpdate] Updating cell ${id} with data:`, updates);
      onUpdateCell(id, updates);
      // Optionally add a subtle toast or log
      // toast({ title: "Cell Updated", description: `Cell ${id} received updates.` });
    },
    [onUpdateCell] // Dependency: onUpdateCell prop
  );
  // --- END: Added Cell Update Handler --- 

  // --- START: Main event router and message processor ---
  // Handle different event types
  const handleEvent = useCallback(
    (event: InvestigationEvent) => {
      // Check if the event is null or undefined before proceeding
      if (!event || typeof event !== 'object' || !event.type) {
          console.warn("Received invalid or null event in handleEvent, skipping:", event);
          return; 
      }
      
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
        case "summary_started":
          handleSummaryStarted(event as SummaryStartedEvent);
          break;
        case "summary_update":
          handleSummaryUpdate(event as SummaryUpdateEvent);
          break;
        case "summary_cell_created":
          handleSummaryCellCreated(event as SummaryCellCreatedEvent);
          break;
        case "summary_cell_error":
          handleSummaryCellError(event as SummaryCellErrorEvent);
          break;
        case "github_tool_cell_created":
          handleGithubToolCellCreated(event as GithubToolCellCreatedEvent);
          break;
        case "github_tool_error":
          handleGithubToolError(event as GithubToolErrorEvent);
          break;
        case "error":
          handleError(event as ErrorEvent)
          break
        case "investigation_report":
          handleInvestigationReport(event as InvestigationReportEvent);
          break;
        case "cell_update":
          handleCellUpdate(event as CellUpdateEvent);
          break;
        case "filesystem_tool_cell_created":
          handleFileSystemToolCellCreated(event as FileSystemToolCellCreatedEvent);
          break;
        case "filesystem_tool_error":
          handleFileSystemToolError(event as FileSystemToolErrorEvent);
          break;
        case "python_tool_cell_created":
          handlePythonToolCellCreated(event as PythonToolCellCreatedEvent);
          break;
        case "python_tool_error":
          handlePythonToolError(event as PythonToolErrorEvent);
          break;
        default:
          // Check if it's a step completion event
          if (event.type.startsWith("step_") && event.type.endsWith("_completed")) {
            handleStepCompleted(event as StepCompletedEvent)
          } else if (event.type === "investigation_complete") {
            // Handle the final completion signal if necessary (might be redundant with summary_cell_created)
            console.log("Received investigation_complete signal");
            // Optionally ensure UI state reflects completion if not already done by summary handler
            setIsInvestigationRunning(false); // Ensure investigation stops on completion signal
            setCurrentStatus("Investigation finished.");
          } else if (event.type.startsWith("step_") && event.type.endsWith("_event")) {
            // Generic handler for other potential step events (like status updates)
            handleStepUpdate(event); // Placeholder function 
          } else {
            console.warn("Unknown event type:", event.type)
          }
      }
    },
    // Add new handlers to dependency array
    [handlePlanCreated, handlePlanCellCreated, handlePlanRevised, handleSummaryStarted, handleSummaryUpdate, handleSummaryCellCreated, handleSummaryCellError, handleGithubToolCellCreated, handleGithubToolError, handleError, handleStepCompleted, handleStepUpdate, handleInvestigationReport, handleCellUpdate, handleFileSystemToolCellCreated, handleFileSystemToolError, handlePythonToolCellCreated, handlePythonToolError]
  );

  // Process incoming WebSocket messages from the internal hook
  useEffect(() => {
    if (!latestEvent) return;

    try {
        console.log("[useInvestigationEvents] Processing latest event:", latestEvent)
        handleEvent(latestEvent); // latestEvent is already typed as InvestigationEvent | null
    } catch (error) {
        console.error("Error handling investigation event:", error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        onError(`Failed to process event: ${errorMessage}`);
    }
  }, [latestEvent, handleEvent, onError]); // Depend on latestEvent
  // --- END: Main event router and message processor ---

  return {
    wsStatus, // Add wsStatus back
    isInvestigationRunning,
    currentPlan,
    currentStatus,
    steps,
    // REMOVED: No longer returning streamedMessages or related processing logic
  }
}
