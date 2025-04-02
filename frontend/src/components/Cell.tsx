import React, { useState, useEffect, useRef } from 'react';
import { Editor } from '@monaco-editor/react';
import { Cell as CellType, CellStatus, CellType as CellTypeEnum } from '../types/notebook';
import MarkdownRenderer from './MarkdownRenderer';
import CellResult from './CellResult';
import CellStatusIndicator from './CellStatusIndicator';
import DependencyIndicator from './DependencyIndicator';

interface CellProps {
  cell: CellType;
  isEditing: boolean;
  onEdit: (content: string) => void;
  onExecute: () => void;
  onFocus: () => void;
  onBlur: () => void;
  onDelete: () => void;
  onAddDependency: (dependencyId: string) => void;
  onRemoveDependency: (dependencyId: string) => void;
  dependencies: CellType[];
  dependents: CellType[];
  isFocused: boolean;
}

const Cell: React.FC<CellProps> = ({
  cell,
  isEditing,
  onEdit,
  onExecute,
  onFocus,
  onBlur,
  onDelete,
  onAddDependency,
  onRemoveDependency,
  dependencies,
  dependents,
  isFocused,
}) => {
  const [localContent, setLocalContent] = useState(cell.content);
  const [showDependencies, setShowDependencies] = useState(false);
  const cellRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLocalContent(cell.content);
  }, [cell.content]);

  useEffect(() => {
    if (isFocused && cellRef.current) {
      cellRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [isFocused]);

  const handleEditorChange = (value: string | undefined) => {
    if (value !== undefined) {
      setLocalContent(value);
    }
  };

  const handleSave = () => {
    onEdit(localContent);
    onBlur();
  };

  const getLanguageForCell = (cellType: CellTypeEnum): string => {
    switch (cellType) {
      case CellTypeEnum.PYTHON:
        return 'python';
      case CellTypeEnum.SQL:
        return 'sql';
      case CellTypeEnum.MARKDOWN:
        return 'markdown';
      case CellTypeEnum.AI_QUERY:
        return 'markdown';
      case CellTypeEnum.LOG:
        return 'logql';
      case CellTypeEnum.METRIC:
        return 'promql';
      case CellTypeEnum.S3:
        return 'sql';
      default:
        return 'plaintext';
    }
  };

  const renderEditor = () => {
    return (
      <Editor
        height="auto"
        defaultLanguage={getLanguageForCell(cell.type)}
        defaultValue={localContent}
        value={localContent}
        onChange={handleEditorChange}
        options={{
          minimap: { enabled: false },
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          wordWrap: 'on',
          automaticLayout: true,
        }}
        onMount={(editor) => {
          editor.onDidFocusEditorWidget(() => onFocus());
          // Set min height based on content
          const lineCount = editor.getModel()?.getLineCount() || 1;
          const minHeight = Math.min(Math.max(lineCount * 18, 100), 400);
          editor.getDomNode()?.style.setProperty('min-height', `${minHeight}px`);
        }}
      />
    );
  };

  const renderContent = () => {
    if (isEditing) {
      return renderEditor();
    }

    switch (cell.type) {
      case CellTypeEnum.MARKDOWN:
        return <MarkdownRenderer content={cell.content} />;
      case CellTypeEnum.AI_QUERY:
        return (
          <div className="p-2 bg-indigo-50 dark:bg-indigo-900 rounded">
            <p className="font-medium mb-2">AI Query:</p>
            <p>{cell.content}</p>
          </div>
        );
      default:
        return (
          <div className="font-mono text-sm overflow-x-auto p-2">
            <pre>{cell.content}</pre>
          </div>
        );
    }
  };

  const isStale = cell.status === CellStatus.STALE;
  const isError = cell.status === CellStatus.ERROR;
  const isRunning = cell.status === CellStatus.RUNNING;
  const isQueued = cell.status === CellStatus.QUEUED;

  return (
    <div
      ref={cellRef}
      className={`border rounded mb-4 transition ${
        isFocused ? 'border-blue-500 shadow-md' : 'border-gray-300 dark:border-gray-700'
      } ${isStale ? 'bg-yellow-50 dark:bg-yellow-900/20' : ''} ${
        isError ? 'bg-red-50 dark:bg-red-900/20' : ''
      } ${isRunning || isQueued ? 'bg-blue-50 dark:bg-blue-900/20' : ''}`}
    >
      <div className="flex justify-between items-center p-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700">
        <div className="flex items-center space-x-2">
          <CellStatusIndicator status={cell.status} />
          <span className="font-medium text-sm">{cell.type}</span>
          {cell.metadata?.description && (
            <span className="text-xs text-gray-600 dark:text-gray-400">{cell.metadata.description}</span>
          )}
        </div>
        
        <div className="flex items-center space-x-1">
          <DependencyIndicator
            dependencies={dependencies}
            dependents={dependents}
            showDependencies={showDependencies}
            onToggle={() => setShowDependencies(!showDependencies)}
          />
          
          <button
            onClick={onExecute}
            className="p-1 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white"
            title="Execute cell"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
            </svg>
          </button>
          
          {isEditing ? (
            <button
              onClick={handleSave}
              className="p-1 text-green-600 hover:text-green-800 dark:text-green-400 dark:hover:text-green-300"
              title="Save changes"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
          ) : (
            <button
              onClick={onFocus}
              className="p-1 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white"
              title="Edit cell"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path d="M5.433 13.917l1.262-3.155A4 4 0 017.58 9.42l6.92-6.918a2.121 2.121 0 013 3l-6.92 6.918c-.383.383-.84.685-1.343.886l-3.154 1.262a.5.5 0 01-.65-.65z" />
                <path d="M3.5 5.75c0-.69.56-1.25 1.25-1.25H10A.75.75 0 0010 3H4.75A2.75 2.75 0 002 5.75v9.5A2.75 2.75 0 004.75 18h9.5A2.75 2.75 0 0017 15.25V10a.75.75 0 00-1.5 0v5.25c0 .69-.56 1.25-1.25 1.25h-9.5c-.69 0-1.25-.56-1.25-1.25v-9.5z" />
              </svg>
            </button>
          )}
          
          <button
            onClick={onDelete}
            className="p-1 text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
            title="Delete cell"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path
                fillRule="evenodd"
                d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.52.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      </div>
      
      <div className="p-2">{renderContent()}</div>
      
      {showDependencies && (dependencies.length > 0 || dependents.length > 0) && (
        <div className="border-t border-gray-300 dark:border-gray-700 p-2 bg-gray-50 dark:bg-gray-800/50">
          <div className="text-xs">
            {dependencies.length > 0 && (
              <div className="mb-2">
                <p className="font-semibold mb-1">Dependencies:</p>
                <ul className="space-y-1">
                  {dependencies.map((dep) => (
                    <li key={dep.id} className="flex items-center justify-between">
                      <span className="truncate">{dep.metadata?.description || dep.content.substring(0, 40)}</span>
                      <button
                        onClick={() => onRemoveDependency(dep.id)}
                        className="text-red-500 hover:text-red-700"
                        title="Remove dependency"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                          <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
                        </svg>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {dependents.length > 0 && (
              <div>
                <p className="font-semibold mb-1">Cells that depend on this:</p>
                <ul className="space-y-1">
                  {dependents.map((dep) => (
                    <li key={dep.id} className="truncate">
                      {dep.metadata?.description || dep.content.substring(0, 40)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
      
      {cell.result && (
        <div className="border-t border-gray-300 dark:border-gray-700">
          <CellResult result={cell.result} cellType={cell.type} />
        </div>
      )}
    </div>
  );
};

export default Cell;