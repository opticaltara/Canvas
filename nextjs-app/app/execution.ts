import { Cell, CellType, CellStatus, CellResult, MCPConnection } from './types/notebook';
import { getExecutionManager } from './lib/execution/index';

// Execute a single cell
export async function executeCell(
  cell: Cell,
  allCells: Record<string, Cell>,
  connections: Record<string, MCPConnection>,
  variables: Record<string, any>
): Promise<CellResult & { variables?: Record<string, any> }> {
  try {
    // Get the execution manager with current connections
    const executionManager = getExecutionManager(Object.values(connections));
    
    // Execute the cell using our execution system
    const result = await executionManager.executeCell(cell, allCells);
    
    // For Python cells, capture any defined variables
    const capturedVariables: Record<string, any> = {};
    if (cell.type === CellType.PYTHON && result.content) {
      // Extract any variables from the Python execution
      if (result.content.variables) {
        Object.assign(capturedVariables, result.content.variables);
      }
    }
    
    return {
      ...result,
      variables: Object.keys(capturedVariables).length > 0 ? capturedVariables : undefined
    };
  } catch (error) {
    return {
      content: null,
      error: error instanceof Error ? error.message : String(error),
      timestamp: new Date().toISOString(),
      executionTime: 0,
    };
  }
}

// Execute all dependent cells
export async function executeDependents(
  cellId: string,
  allCells: Record<string, Cell>,
  connections: Record<string, MCPConnection>,
  variables: Record<string, any>
): Promise<void> {
  const cell = allCells[cellId];
  if (!cell) return;
  
  // Get all dependent cells
  const dependentIds = cell.dependents;
  
  // Execute each dependent
  for (const depId of dependentIds) {
    const depCell = allCells[depId];
    if (depCell) {
      // Check if all dependencies have been executed
      const allDepsExecuted = depCell.dependencies.every(id => {
        const depCell = allCells[id];
        return depCell && depCell.status === CellStatus.COMPLETE;
      });
      
      // Only execute if all dependencies are ready
      if (allDepsExecuted) {
        // Execute this cell
        const result = await executeCell(depCell, allCells, connections, variables);
        
        // Update variables if any were returned
        if (result.variables) {
          Object.assign(variables, result.variables);
        }
        
        // Mark cell as executed with its result
        depCell.status = result.error ? CellStatus.ERROR : CellStatus.COMPLETE;
        depCell.result = {
          content: result.content,
          error: result.error,
          executionTime: result.executionTime || 0,
          timestamp: result.timestamp,
        };
        
        // Recursively execute its dependents
        await executeDependents(depId, allCells, connections, variables);
      }
    }
  }
}

// Sort cells in execution order respecting dependencies
export function getExecutionOrder(
  cells: Record<string, Cell>,
  defaultOrder: string[]
): string[] {
  // Topological sort for cells
  const visited = new Set<string>();
  const result: string[] = [];
  
  // Visit function for topological sort
  const visit = (cellId: string) => {
    if (visited.has(cellId)) return;
    visited.add(cellId);
    
    // Visit all dependencies first
    const cell = cells[cellId];
    if (cell) {
      for (const depId of cell.dependencies) {
        if (cells[depId]) {
          visit(depId);
        }
      }
    }
    
    // Add this cell to result
    result.push(cellId);
  };
  
  // Visit all cells in default order
  for (const cellId of defaultOrder) {
    if (!visited.has(cellId)) {
      visit(cellId);
    }
  }
  
  return result;
}