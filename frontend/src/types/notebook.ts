// Notebook types

export enum CellType {
  AI_QUERY = "AI_QUERY",
  SQL = "SQL",
  PYTHON = "PYTHON",
  MARKDOWN = "MARKDOWN",
  LOG = "LOG",
  METRIC = "METRIC",
  S3 = "S3"
}

export enum CellStatus {
  IDLE = "idle",
  QUEUED = "queued",
  RUNNING = "running",
  SUCCESS = "success",
  ERROR = "error",
  STALE = "stale"
}

export interface CellResult {
  content: any;
  error?: string;
  execution_time: number;
  timestamp: string;
  stdout?: string;
  stderr?: string;
  figures?: Array<{
    format: string;
    data: string;
  }>;
  dataframe?: any;
  dataframe_html?: string;
  thinking?: string;
  plan?: any;
}

export interface Cell {
  id: string;
  type: CellType;
  content: string;
  result?: CellResult;
  status: CellStatus;
  created_at: string;
  updated_at: string;
  dependencies: Set<string>;
  dependents: Set<string>;
  metadata: {
    description?: string;
    [key: string]: any;
  };
}

export interface NotebookMetadata {
  title: string;
  description: string;
  tags: string[];
  created_by?: string;
  created_at: string;
  updated_at: string;
}

export interface Notebook {
  id: string;
  metadata: NotebookMetadata;
  cells: { [key: string]: Cell };
  cell_order: string[];
}

// WebSocket message types

export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export interface NotebookStateMessage extends WebSocketMessage {
  type: "notebook_state";
  notebook_id: string;
  data: any;
}

export interface CellUpdatedMessage extends WebSocketMessage {
  type: "cell_updated";
  cell_id: string;
  notebook_id: string;
  data: any;
}

export interface CellAddedMessage extends WebSocketMessage {
  type: "cell_added";
  cell_id: string;
  notebook_id: string;
  data: any;
}

export interface CellDeletedMessage extends WebSocketMessage {
  type: "cell_deleted";
  cell_id: string;
  notebook_id: string;
}

export interface DependencyAddedMessage extends WebSocketMessage {
  type: "dependency_added";
  dependent_id: string;
  dependency_id: string;
  notebook_id: string;
}

export interface DependencyRemovedMessage extends WebSocketMessage {
  type: "dependency_removed";
  dependent_id: string;
  dependency_id: string;
  notebook_id: string;
}

export interface ErrorMessage extends WebSocketMessage {
  type: "error";
  message: string;
}

// Connection types

export interface Connection {
  id: string;
  name: string;
  plugin_name: string;
  config: {
    [key: string]: any;
  };
}

export interface Plugin {
  name: string;
  description: string;
  version: string;
  enabled: boolean;
}