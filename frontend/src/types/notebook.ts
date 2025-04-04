export type CellType = 'sql' | 'python' | 'markdown' | 'log' | 'metric' | 's3' | 'ai_query';

export type CellStatus = 'idle' | 'queued' | 'running' | 'success' | 'error';

export interface Cell {
  id: string;
  type: CellType;
  content: string;
  status: CellStatus;
  result?: any;
  error?: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
  settings?: {
    use_sandbox?: boolean;
    dependencies?: string[];
    [key: string]: any;
  };
}

export interface Dependency {
  dependent_id: string;
  dependency_id: string;
}

export interface Notebook {
  id: string;
  name: string;
  description: string;
  cells: Cell[];
  dependencies: Dependency[];
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
}

export interface NotebookList {
  id: string;
  name: string;
  description: string;
  cell_count: number;
  created_at: string;
  updated_at: string;
}
