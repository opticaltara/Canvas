import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Cell as CellComponent } from './Cell';
import { useWebSocket } from '../hooks/useWebSocket';
import { Cell, CellType, Notebook } from '../types/notebook';
import { cn } from '../utils/cn';
import { BACKEND_URL, WS_URL } from '../config';

export function NotebookView() {
  const { notebookId } = useParams<{ notebookId: string }>();
  const [notebook, setNotebook] = useState<Notebook | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cellTypeToAdd, setCellTypeToAdd] = useState<CellType>('markdown');

  // Create WebSocket connection for real-time updates
  const { isConnected, messages, sendMessage } = useWebSocket(
    `${WS_URL}/ws/notebook/${notebookId}`
  );

  // Fetch notebook data
  useEffect(() => {
    if (!notebookId) return;

    setLoading(true);
    fetch(`${BACKEND_URL}/api/notebooks/${notebookId}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        return response.json();
      })
      .then((data) => {
        setNotebook(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [notebookId]);

  // Update notebook state based on WebSocket messages
  useEffect(() => {
    if (!messages.length || !notebook) return;

    const latestMessage = messages[messages.length - 1];
    switch (latestMessage.type) {
      case 'notebook_state':
        setNotebook(latestMessage.data);
        break;
      case 'cell_updated':
        if (latestMessage.notebook_id === notebookId) {
          setNotebook((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              cells: prev.cells.map((cell) =>
                cell.id === latestMessage.cell_id ? latestMessage.data : cell
              ),
            };
          });
        }
        break;
      case 'cell_added':
        if (latestMessage.notebook_id === notebookId) {
          setNotebook((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              cells: [...prev.cells, latestMessage.data],
            };
          });
        }
        break;
      case 'cell_deleted':
        if (latestMessage.notebook_id === notebookId) {
          setNotebook((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              cells: prev.cells.filter(
                (cell) => cell.id !== latestMessage.cell_id
              ),
              dependencies: prev.dependencies.filter(
                (dep) =>
                  dep.dependent_id !== latestMessage.cell_id &&
                  dep.dependency_id !== latestMessage.cell_id
              ),
            };
          });
        }
        break;
      case 'dependency_added':
        if (latestMessage.notebook_id === notebookId) {
          setNotebook((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              dependencies: [
                ...prev.dependencies,
                {
                  dependent_id: latestMessage.dependent_id,
                  dependency_id: latestMessage.dependency_id,
                },
              ],
            };
          });
        }
        break;
      case 'dependency_removed':
        if (latestMessage.notebook_id === notebookId) {
          setNotebook((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              dependencies: prev.dependencies.filter(
                (dep) =>
                  !(dep.dependent_id === latestMessage.dependent_id &&
                    dep.dependency_id === latestMessage.dependency_id)
              ),
            };
          });
        }
        break;
      default:
        break;
    }
  }, [messages, notebookId, notebook]);

  // Handler for executing a cell
  const handleExecuteCell = useCallback(
    (cellId: string) => {
      if (!notebookId || !isConnected) return;

      sendMessage({
        type: 'execute_cell',
        notebook_id: notebookId,
        cell_id: cellId,
      });
    },
    [notebookId, isConnected, sendMessage]
  );

  // Handler for updating cell content and settings
  const handleUpdateCell = useCallback(
    (cellId: string, content: string, settings?: Record<string, any>) => {
      if (!notebookId || !isConnected) return;

      // Create message with basic update info
      const message: any = {
        type: 'update_cell',
        notebook_id: notebookId,
        cell_id: cellId,
        content: content,
      };
      
      // Add settings if provided
      if (settings) {
        message.settings = settings;
      }

      sendMessage(message);
    },
    [notebookId, isConnected, sendMessage]
  );

  // Handler for deleting a cell
  const handleDeleteCell = useCallback(
    (cellId: string) => {
      if (!notebookId || !isConnected) return;

      sendMessage({
        type: 'delete_cell',
        notebook_id: notebookId,
        cell_id: cellId,
      });
    },
    [notebookId, isConnected, sendMessage]
  );

  // Handler for adding a new cell
  const handleAddCell = useCallback(() => {
    if (!notebookId || !isConnected) return;

    sendMessage({
      type: 'add_cell',
      notebook_id: notebookId,
      cell_type: cellTypeToAdd,
      content: '',
    });
  }, [notebookId, isConnected, sendMessage, cellTypeToAdd]);

  // Show loading or error states
  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center p-4 max-w-lg mx-auto bg-red-50 rounded-md border border-red-200 text-red-700">
        <h2 className="text-lg font-semibold mb-2">Error Loading Notebook</h2>
        <p>{error}</p>
      </div>
    );
  }

  if (!notebook) {
    return (
      <div className="text-center p-4 max-w-lg mx-auto">
        <h2 className="text-lg font-semibold mb-2">Notebook Not Found</h2>
        <p>
          The requested notebook could not be found. Please check the URL and try
          again.
        </p>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold">{notebook.name}</h1>
          {notebook.description && (
            <p className="text-gray-600 mt-1">{notebook.description}</p>
          )}
        </div>
        <div className="flex items-center">
          <span
            className={cn(
              'h-3 w-3 rounded-full mr-2',
              isConnected ? 'bg-green-500' : 'bg-red-500'
            )}
          ></span>
          <span className="text-sm text-gray-600">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Cell list */}
      <div className="space-y-4">
        {notebook.cells.map((cell) => (
          <CellComponent
            key={cell.id}
            cell={cell}
            dependencies={notebook.dependencies}
            onExecute={handleExecuteCell}
            onUpdate={handleUpdateCell}
            onDelete={handleDeleteCell}
          />
        ))}

        {/* Add cell controls */}
        <div className="flex items-center space-x-3 p-4 border border-dashed border-gray-300 rounded-md hover:border-gray-400 transition-colors">
          <select
            value={cellTypeToAdd}
            onChange={(e) => setCellTypeToAdd(e.target.value as CellType)}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          >
            <option value="markdown">Markdown</option>
            <option value="sql">SQL</option>
            <option value="python">Python</option>
            <option value="log">Log</option>
            <option value="metric">Metric</option>
            <option value="s3">S3</option>
            <option value="ai_query">AI Query</option>
          </select>
          <button
            onClick={handleAddCell}
            className="bg-blue-500 text-white px-4 py-1 rounded text-sm hover:bg-blue-600 transition-colors"
          >
            Add Cell
          </button>
        </div>
      </div>
    </div>
  );
}
