import { z } from "zod"

// Define schemas with Zod for type safety
export const CellStatusSchema = z.enum(["idle", "running", "success", "error"])
export type CellStatus = z.infer<typeof CellStatusSchema>

// Update the CellTypeSchema to include github
export const CellTypeSchema = z.enum(["markdown", "sql", "code", "grafana", "log", "github"])
export type CellType = z.infer<typeof CellTypeSchema>

export const CellSchema = z.object({
  id: z.string(),
  notebook_id: z.string(),
  type: CellTypeSchema,
  content: z.string(),
  result: z.any().optional(),
  status: CellStatusSchema,
  error: z.string().optional(),
  created_at: z.string(),
  updated_at: z.string(),
  metadata: z.record(z.any()).optional(),
})
export type Cell = z.infer<typeof CellSchema>

export const NotebookSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  created_at: z.string(),
  updated_at: z.string(),
})
export type Notebook = z.infer<typeof NotebookSchema>

// Update the ConnectionTypeSchema to include python
export const ConnectionTypeSchema = z.enum(["grafana", "kubernetes", "s3", "github", "python"])
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
