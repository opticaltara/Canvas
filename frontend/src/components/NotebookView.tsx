import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Cell as CellType, Notebook, CellStatus, CellType as CellTypeEnum } from '../types/notebook';
import Cell from './Cell';
import CellToolbar from './CellToolbar';
import NotebookHeader from './NotebookHeader';
import useWebSocket from '../hooks/useWebSocket';
import { v4 as uuidv4 } from 'uuid';

const NotebookView: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [notebook, setNotebook] = useState<Notebook | null>(null);
  const [editingCellId, setEditingCellId] = useState<string | null>(null);
  const [selectedCellId, setSelectedCellId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const notebookRef = useRef<HTMLDivElement>(null);

  // Initialize WebSocket connection
  const { socket, connected, send } = useWebSocket(
    id ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/notebook/${id}` : null
  );

  // Placeholder for API fetch - would be replaced with an actual API call
  useEffect(() => {
    if (!id) {
      setError('Notebook ID is required');
      setIsLoading(false);
      return;
    }

    // In a real implementation, you would fetch the notebook from the API
    // For now, we will wait for the WebSocket to provide the initial state
    if (connected) {
      setIsLoading(false);
    }
  }, [id, connected]);

  // Handle WebSocket messages
  useEffect(() => {
    if (!socket) return;

    const handleMessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'notebook_state':
          // Initial notebook state
          setNotebook(deserializeNotebook(data.data));
          setIsLoading(false);
          break;
          
        case 'cell_updated':
          // Cell content or state updated
          setNotebook((prev) => {
            if (!prev) return prev;
            const updatedCells = { ...prev.cells };
            updatedCells[data.cell_id] = data.data;
            return { ...prev, cells: updatedCells };
          });
          break;
          
        case 'cell_added':
          // New cell added
          setNotebook((prev) => {
            if (!prev) return prev;
            const updatedCells = { ...prev.cells };
            updatedCells[data.cell_id] = data.data;
            const updatedCellOrder = [...prev.cell_order];
            if (!updatedCellOrder.includes(data.cell_id)) {
              updatedCellOrder.push(data.cell_id);
            }
            return { ...prev, cells: updatedCells, cell_order: updatedCellOrder };
          });
          break;
          
        case 'cell_deleted':
          // Cell deleted
          setNotebook((prev) => {
            if (!prev) return prev;
            const updatedCells = { ...prev.cells };
            delete updatedCells[data.cell_id];
            const updatedCellOrder = prev.cell_order.filter((id) => id !== data.cell_id);
            return { ...prev, cells: updatedCells, cell_order: updatedCellOrder };
          });
          break;
          
        case 'dependency_added':
          // Dependency added between cells
          setNotebook((prev) => {
            if (!prev) return prev;
            const updatedCells = { ...prev.cells };
            
            // Update dependent cell
            const dependentCell = { ...updatedCells[data.dependent_id] };
            dependentCell.dependencies = [...(dependentCell.dependencies || []), data.dependency_id];
            updatedCells[data.dependent_id] = dependentCell;
            
            // Update dependency cell
            const dependencyCell = { ...updatedCells[data.dependency_id] };
            dependencyCell.dependents = [...(dependencyCell.dependents || []), data.dependent_id];
            updatedCells[data.dependency_id] = dependencyCell;
            
            return { ...prev, cells: updatedCells };
          });
          break;
          
        case 'dependency_removed':
          // Dependency removed between cells
          setNotebook((prev) => {
            if (!prev) return prev;
            const updatedCells = { ...prev.cells };
            
            // Update dependent cell
            const dependentCell = { ...updatedCells[data.dependent_id] };
            dependentCell.dependencies = (dependentCell.dependencies || []).filter(
              (id) => id !== data.dependency_id
            );
            updatedCells[data.dependent_id] = dependentCell;
            
            // Update dependency cell
            const dependencyCell = { ...updatedCells[data.dependency_id] };
            dependencyCell.dependents = (dependencyCell.dependents || []).filter(
              (id) => id !== data.dependent_id
            );
            updatedCells[data.dependency_id] = dependencyCell;
            
            return { ...prev, cells: updatedCells };
          });
          break;
          
        case 'error':
          // Error message
          setError(data.message);
          break;
          
        default:
          console.log('Unknown message type:', data.type);
      }
    };

    socket.addEventListener('message', handleMessage);

    return () => {
      socket.removeEventListener('message', handleMessage);
    };
  }, [socket]);

  // Helper to convert serialized notebook data
  const deserializeNotebook = (data: any): Notebook => {
    return {
      id: data.id,
      metadata: data.metadata,
      cells: Object.fromEntries(
        Object.entries(data.cells).map(([id, cellData]: [string, any]) => [
          id,
          {
            ...cellData,
            dependencies: new Set(cellData.dependencies || []),
            dependents: new Set(cellData.dependents || []),
          },
        ])
      ),
      cell_order: data.cell_order,
    };
  };

  // Handle cell editing
  const handleCellEdit = useCallback(
    (cellId: string, content: string) => {
      if (!notebook || !socket) return;

      send({
        type: 'update_cell',
        notebook_id: notebook.id,
        cell_id: cellId,
        content,
      });

      setEditingCellId(null);
    },
    [notebook, socket, send]
  );

  // Handle cell execution
  const handleCellExecute = useCallback(
    (cellId: string) => {
      if (!notebook || !socket) return;

      send({
        type: 'execute_cell',
        notebook_id: notebook.id,
        cell_id: cellId,
      });
    },
    [notebook, socket, send]
  );

  // Handle cell deletion
  const handleCellDelete = useCallback(
    (cellId: string) => {
      if (!notebook || !socket) return;

      const confirmDelete = window.confirm('Are you sure you want to delete this cell?');
      if (!confirmDelete) return;

      send({
        type: 'delete_cell',
        notebook_id: notebook.id,
        cell_id: cellId,
      });

      if (editingCellId === cellId) {
        setEditingCellId(null);
      }
      if (selectedCellId === cellId) {
        setSelectedCellId(null);
      }
    },
    [notebook, socket, send, editingCellId, selectedCellId]
  );

  // Handle adding a new cell
  const handleAddCell = useCallback(
    (cellType: CellTypeEnum, position?: number) => {
      if (!notebook || !socket) return;

      // Default content based on cell type
      let content = '';
      switch (cellType) {
        case CellTypeEnum.MARKDOWN:
          content = '# New Markdown Cell';
          break;
        case CellTypeEnum.PYTHON:
          content = '# Write your Python code here\n';
          break;
        case CellTypeEnum.SQL:
          content = '-- Write your SQL query here\nSELECT * FROM table LIMIT 10;';
          break;
        case CellTypeEnum.LOG:
          content = '-- Write your log query here\n{app="example"}';
          break;
        case CellTypeEnum.METRIC:
          content = '-- Write your metric query here\nrate(http_requests_total[5m])';
          break;
        case CellTypeEnum.AI_QUERY:
          content = 'Ask AI to investigate...';
          break;
        default:
          content = '';
      }

      send({
        type: 'add_cell',
        notebook_id: notebook.id,
        cell_type: cellType,
        content,
        position,
      });
    },
    [notebook, socket, send]
  );

  // Handle adding a dependency between cells
  const handleAddDependency = useCallback(
    (dependentId: string, dependencyId: string) => {
      if (!notebook || !socket) return;

      send({
        type: 'add_dependency',
        notebook_id: notebook.id,
        dependent_id: dependentId,
        dependency_id: dependencyId,
      });
    },
    [notebook, socket, send]
  );

  // Handle removing a dependency between cells
  const handleRemoveDependency = useCallback(
    (dependentId: string, dependencyId: string) => {
      if (!notebook || !socket) return;

      send({
        type: 'remove_dependency',
        notebook_id: notebook.id,
        dependent_id: dependentId,
        dependency_id: dependencyId,
      });
    },
    [notebook, socket, send]
  );

  // Handle saving the notebook
  const handleSaveNotebook = useCallback(() => {
    if (!notebook || !socket) return;

    send({
      type: 'save_notebook',
      notebook_id: notebook.id,
    });
  }, [notebook, socket, send]);

  // Helper to get dependencies and dependents for a cell
  const getCellDependencies = (cellId: string) => {
    if (!notebook) return { dependencies: [], dependents: [] };

    const cell = notebook.cells[cellId];
    if (!cell) return { dependencies: [], dependents: [] };

    const dependencies = [...(cell.dependencies || [])].map((id) => notebook.cells[id]).filter(Boolean);
    const dependents = [...(cell.dependents || [])].map((id) => notebook.cells[id]).filter(Boolean);

    return { dependencies, dependents };
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="spinner w-12 h-12 border-4 border-blue-500 rounded-full border-t-transparent animate-spin mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-300">Loading notebook...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-red-500 text-5xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold mb-2">Error</h2>
          <p className="text-gray-600 dark:text-gray-300">{error}</p>
        </div>
      </div>
    );
  }

  if (!notebook) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">Notebook Not Found</h2>
          <p className="text-gray-600 dark:text-gray-300">The requested notebook could not be loaded.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6 max-w-5xl" ref={notebookRef}>
      <NotebookHeader
        title={notebook.metadata.title}
        description={notebook.metadata.description}
        onSave={handleSaveNotebook}
        connected={connected}
      />

      <CellToolbar onAddCell={handleAddCell} />

      <div className="mt-6 space-y-4">
        {notebook.cell_order.map((cellId) => {
          const cell = notebook.cells[cellId];
          if (!cell) return null;

          const { dependencies, dependents } = getCellDependencies(cellId);

          return (
            <Cell
              key={cellId}
              cell={cell}
              isEditing={editingCellId === cellId}
              onEdit={(content) => handleCellEdit(cellId, content)}
              onExecute={() => handleCellExecute(cellId)}
              onFocus={() => {
                setSelectedCellId(cellId);
                setEditingCellId(cellId);
              }}
              onBlur={() => setEditingCellId(null)}
              onDelete={() => handleCellDelete(cellId)}
              onAddDependency={(dependencyId) => handleAddDependency(cellId, dependencyId)}
              onRemoveDependency={(dependencyId) => handleRemoveDependency(cellId, dependencyId)}
              dependencies={dependencies}
              dependents={dependents}
              isFocused={selectedCellId === cellId}
            />
          );
        })}
      </div>

      {notebook.cell_order.length === 0 && (
        <div className="text-center py-10 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg">
          <h3 className="text-lg font-medium mb-2">This notebook is empty</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">Add a cell to get started</p>
          <div className="flex justify-center space-x-2">
            <button
              onClick={() => handleAddCell(CellTypeEnum.AI_QUERY)}
              className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
            >
              Add AI Query
            </button>
            <button
              onClick={() => handleAddCell(CellTypeEnum.MARKDOWN)}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Add Markdown
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotebookView;