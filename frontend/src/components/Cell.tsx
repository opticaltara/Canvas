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
  onUpdate: (cellId: string, content: string, settings?: Record<string, any>) => void;
  onDelete: (cellId: string) => void;
  className?: string;
}

export function Cell({ cell, dependencies: cellDependencies, onExecute, onUpdate, onDelete, className }: CellProps) {
  const [isEditing, setIsEditing] = useState(cell.type !== 'markdown' || !cell.content);
  const [content, setContent] = useState(cell.content);
  
  // State for cell settings
  const [showSettings, setShowSettings] = useState(false);
  const [useSandbox, setUseSandbox] = useState(cell.settings?.use_sandbox || false);
  const [packageDependencies, setPackageDependencies] = useState(cell.settings?.dependencies?.join(", ") || "");

  const handleExecute = () => {
    onExecute(cell.id);
  };

  const handleSave = () => {
    // Prepare settings for Python cells
    let updatedSettings = { ...cell.settings };
    
    if (cell.type === 'python') {
      // Parse dependencies string to array
      const dependenciesArray = packageDependencies
        .split(',')
        .map(dep => dep.trim())
        .filter(dep => dep.length > 0);
      
      updatedSettings = {
        ...updatedSettings,
        use_sandbox: useSandbox,
        dependencies: dependenciesArray
      };
    }
    
    // Update cell with content and settings
    onUpdate(cell.id, content, updatedSettings);
    
    if (cell.type === 'markdown') {
      setIsEditing(false);
    }
    
    // Close settings panel
    setShowSettings(false);
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
          <DependencyIndicator dependencies={cellDependencies} cellId={cell.id} />
          {cell.type === 'python' && (
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className="px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
            >
              {showSettings ? 'Hide' : 'Settings'}
            </button>
          )}
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
        {/* Settings panel for Python cells */}
        {cell.type === 'python' && showSettings && (
          <div className="mb-4 p-3 bg-gray-50 border border-gray-200 rounded-md">
            <h3 className="text-sm font-medium mb-2">Python Cell Settings</h3>
            
            <div className="mb-2">
              <label className="flex items-center gap-2 text-sm">
                <input 
                  type="checkbox" 
                  checked={useSandbox} 
                  onChange={(e) => setUseSandbox(e.target.checked)} 
                />
                Use Sandbox Mode (WASM)
              </label>
              <p className="text-xs text-gray-500 mt-1">
                Runs Python code in an isolated WebAssembly environment for security.
              </p>
            </div>
            
            <div className="mb-2">
              <label className="block text-sm mb-1">
                Package Dependencies (comma-separated)
              </label>
              <input 
                type="text" 
                value={packageDependencies} 
                onChange={(e) => setPackageDependencies(e.target.value)}
                placeholder="e.g. pandas, numpy, matplotlib"
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded"
              />
              <p className="text-xs text-gray-500 mt-1">
                These packages will be installed in the sandbox environment.
              </p>
            </div>
          </div>
        )}
      
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

        {/* Display a sandbox indicator for Python cells */}
        {cell.type === 'python' && cell.settings?.use_sandbox && !showSettings && (
          <div className="mt-1 text-xs text-gray-500 flex items-center gap-1">
            <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" />
              <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" />
              <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" />
            </svg>
            <span>Running in Sandbox Mode</span>
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
