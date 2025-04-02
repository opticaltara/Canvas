import React, { useState } from 'react';
import { Cell as CellType } from '../types/notebook';
import { CellStatusIndicator } from './CellStatusIndicator';
import { CellResult } from './CellResult';
import { DependencyIndicator } from './DependencyIndicator';
import { MarkdownRenderer } from './MarkdownRenderer';
import { cn } from '../utils/cn';
import MonacoEditor from '@monaco-editor/react';

interface CellProps {
  cell: CellType;
  dependencies: Array<{ dependent_id: string; dependency_id: string }>;
  onExecute: (cellId: string) => void;
  onUpdate: (cellId: string, content: string) => void;
  onDelete: (cellId: string) => void;
  className?: string;
}

export function Cell({ cell, dependencies, onExecute, onUpdate, onDelete, className }: CellProps) {
  const [isEditing, setIsEditing] = useState(cell.type !== 'markdown' || !cell.content);
  const [content, setContent] = useState(cell.content);

  const handleExecute = () => {
    onExecute(cell.id);
  };

  const handleSave = () => {
    onUpdate(cell.id, content);
    if (cell.type === 'markdown') {
      setIsEditing(false);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    // Save on Ctrl+Enter or Cmd+Enter
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      handleSave();
      handleExecute();
    }
  };

  const handleDelete = () => {
    if (window.confirm('Are you sure you want to delete this cell?')) {
      onDelete(cell.id);
    }
  };

  return (
    <div 
      className={cn(
        'border border-gray-200 rounded-md my-4 group',
        cell.status === 'error' ? 'border-red-300' : '',
        className
      )}
    >
      <div className="flex items-center justify-between bg-gray-50 px-3 py-1 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <div className="text-xs font-medium uppercase text-gray-500">{cell.type}</div>
          <CellStatusIndicator status={cell.status} />
          <DependencyIndicator dependencies={dependencies} cellId={cell.id} />
        </div>
        <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          {isEditing ? (
            <>
              <button 
                onClick={handleSave} 
                className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Save
              </button>
              <button 
                onClick={handleExecute} 
                className="px-2 py-1 text-xs bg-green-500 text-white rounded hover:bg-green-600"
                disabled={cell.status === 'running' || cell.status === 'queued'}
              >
                Run
              </button>
            </>
          ) : (
            <button 
              onClick={() => setIsEditing(true)} 
              className="px-2 py-1 text-xs bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              Edit
            </button>
          )}
          <button 
            onClick={handleDelete} 
            className="px-2 py-1 text-xs bg-red-500 text-white rounded hover:bg-red-600"
          >
            Delete
          </button>
        </div>
      </div>

      <div className="p-4">
        {isEditing ? (
          <div onKeyDown={handleKeyDown}>
            <MonacoEditor
              height="100px"
              language={getLanguageForCellType(cell.type)}
              theme="vs-light"
              value={content}
              onChange={(value) => setContent(value || '')}
              options={{
                minimap: { enabled: false },
                lineNumbers: 'on',
                scrollBeyondLastLine: false,
                fontSize: 14,
                tabSize: 2,
                automaticLayout: true,
              }}
            />
          </div>
        ) : cell.type === 'markdown' ? (
          <div onClick={() => setIsEditing(true)}>
            <MarkdownRenderer content={cell.content} />
          </div>
        ) : (
          <div 
            className="font-mono text-sm whitespace-pre-wrap p-2 bg-gray-50 rounded cursor-pointer"
            onClick={() => setIsEditing(true)}
          >
            {cell.content}
          </div>
        )}

        <CellResult cell={cell} />
      </div>
    </div>
  );
}

function getLanguageForCellType(cellType: string): string {
  switch (cellType) {
    case 'sql':
      return 'sql';
    case 'python':
      return 'python';
    case 'markdown':
      return 'markdown';
    case 'log':
      return 'plaintext';
    case 'metric':
      return 'plaintext'; // Or PromQL if available
    case 's3':
      return 'plaintext';
    case 'ai_query':
      return 'plaintext';
    default:
      return 'plaintext';
  }
}
