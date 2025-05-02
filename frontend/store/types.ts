import { z } from "zod"

// Define schemas with Zod for type safety
export const CellStatusSchema = z.enum([
  "idle",
  "queued",
  "running",
  "success",
  "error",
  "stale",
])
export type CellStatus = z.infer<typeof CellStatusSchema>

// Update the CellTypeSchema to include only supported types from backend
export const CellTypeSchema = z.enum([
  "markdown",
  "github",
  "summarization",
])
export type CellType = z.infer<typeof CellTypeSchema>

// New schema for CellResult matching backend
export const CellResultSchema = z.object({
  content: z.any().optional(),
  error: z.string().optional(),
  execution_time: z.number().optional(),
  timestamp: z.string().datetime().optional(),
})
export type CellResult = z.infer<typeof CellResultSchema>

export const CellSchema = z.object({
  id: z.string(),
  notebook_id: z.string(),
  type: CellTypeSchema,
  content: z.string(),
  result: CellResultSchema.optional(),
  status: CellStatusSchema,
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  metadata: z.record(z.any()).optional(),
  connection_id: z.number().int().optional(),
  error: z.string().optional(),
  dependencies: z.array(z.string()).optional(),
  dependents: z.array(z.string()).optional(),
  tool_name: z.string().optional(),
  tool_arguments: z.record(z.any()).optional(),
  tool_call_id: z.string().uuid().optional(),
  settings: z.record(z.any()).optional(),
  isNew: z.boolean().optional(),
})
export type Cell = z.infer<typeof CellSchema>

export const NotebookSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  metadata: z.record(z.any()).optional(),
  cells: z.record(CellSchema).optional(),
})
export type Notebook = z.infer<typeof NotebookSchema>

// Update the ConnectionTypeSchema to only include currently supported types
export const ConnectionTypeSchema = z.enum(["grafana", "github"])
export type ConnectionType = z.infer<typeof ConnectionTypeSchema>

export const ConnectionSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: ConnectionTypeSchema,
  config: z.record(z.any()),
  is_default: z.boolean(),
})
export type Connection = z.infer<typeof ConnectionSchema>

export const WebSocketStatusSchema = z.enum(["connecting", "connected", "disconnected", "error"])
export type WebSocketStatus = z.infer<typeof WebSocketStatusSchema>

export const WebSocketMessageSchema = z.object({
  type: z.string(),
  data: z.any(),
})
export type WebSocketMessage = z.infer<typeof WebSocketMessageSchema>

// --- Tool Definitions ---
// Removed ToolParameter and ToolDefinition

// --- Simplified Tool Info Structure (matches backend) ---
export interface MCPToolInfo {
  name: string;
  description?: string | null; // Match Optional[str] from Pydantic
  inputSchema: Record<string, any>; // Raw JSON Schema (Record<string, any> is equivalent to Dict[str, Any])
}
// --- End Simplified Tool Info Structure ---
