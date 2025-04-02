export enum CellType {
  MARKDOWN = 'markdown',
  PYTHON = 'python',
  SQL = 'sql',
  LOG = 'log',
  METRIC = 'metric',
  S3 = 's3',
  AI_QUERY = 'ai_query',
}

export enum CellStatus {
  IDLE = 'idle',
  RUNNING = 'running',
  COMPLETE = 'complete',
  ERROR = 'error',
}

export interface CellResult {
  content: any;
  error?: string;
  executionTime?: number;
  timestamp?: string;
}

export interface Cell {
  id: string;
  type: CellType;
  content: string;
  status: CellStatus;
  result?: CellResult;
  dependencies: string[]; // IDs of cells this cell depends on
  dependents: string[]; // IDs of cells that depend on this cell
  metadata: {
    [key: string]: any;
  };
}

export interface AIQueryCell extends Cell {
  type: CellType.AI_QUERY;
  thinking?: string;
  generatedCells: string[]; // IDs of cells generated from this query
}

export interface Notebook {
  id: string;
  name: string;
  cells: Record<string, Cell>;
  cellOrder: string[]; // IDs of cells in order
  metadata: {
    created: string;
    modified: string;
    [key: string]: any;
  };
}

export interface MCPConnection {
  id: string;
  name: string;
  type: string; // "grafana", "postgres", "prometheus", "loki", "s3", etc.
  config: Record<string, string>;
}