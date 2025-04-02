'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { Cell, CellStatus, CellType } from '@/app/types/notebook';
import { Play, Trash, Edit, Check, X, ChevronRight, ChevronDown, Link } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import useNotebookStore from '@/app/lib/store';

// Dynamically import the Monaco editor with no SSR
const MonacoEditor = dynamic(
  () => import('@monaco-editor/react').then((mod) => mod.default),
  { ssr: false }
);

interface CellComponentProps {
  cell: Cell;
  executeCell: () => void;
}

export default function CellComponent({ cell, executeCell }: CellComponentProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(cell.content);
  const [showDependencies, setShowDependencies] = useState(false);
  const { updateCell, deleteCell, currentNotebook, addDependency, removeDependency } = useNotebookStore();

  const handleSave = () => {
    updateCell(cell.id, editValue);
    setIsEditing(false);
  };

  const getEditorLanguage = () => {
    switch (cell.type) {
      case CellType.PYTHON:
        return 'python';
      case CellType.SQL:
        return 'sql';
      case CellType.LOG:
        return 'text';
      case CellType.METRIC:
        return 'text';
      case CellType.S3:
        return 'text';
      case CellType.MARKDOWN:
        return 'markdown';
      default:
        return 'text';
    }
  };

  // Function to render cell result based on type
  const renderCellResult = () => {
    if (!cell.result) return null;
    
    if (cell.result.error) {
      return (
        <div className="text-red-500 whitespace-pre-wrap font-mono text-sm p-2 bg-red-50 dark:bg-red-900 dark:bg-opacity-20 rounded">
          {cell.result.error}
        </div>
      );
    }

    const content = cell.result.content;
    
    // Handle various result content types
    if (typeof content === 'string') {
      return <div className="whitespace-pre-wrap">{content}</div>;
    }
    
    if (cell.type === CellType.SQL) {
      // Render SQL results as a table
      if (content && content.rows && Array.isArray(content.rows)) {
        return (
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-700">
              <thead>
                <tr className="bg-gray-100 dark:bg-gray-800">
                  {content.columns && content.columns.map((col: string, i: number) => (
                    <th key={i} className="px-4 py-2 text-left border border-gray-300 dark:border-gray-700">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {content.rows.map((row: any, i: number) => (
                  <tr key={i} className={i % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-850'}>
                    {content.columns && content.columns.map((col: string, j: number) => (
                      <td key={j} className="px-4 py-2 border border-gray-300 dark:border-gray-700">
                        {typeof row[col] === 'object' ? JSON.stringify(row[col]) : String(row[col])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }
    }
    
    // For Python cells with text/plain output
    if (cell.type === CellType.PYTHON && content && content.type === 'text/plain') {
      return (
        <div className="python-result">
          {/* Text output */}
          {content.output && (
            <div className="font-mono text-sm whitespace-pre-wrap bg-gray-100 dark:bg-gray-800 p-2 rounded mb-4">
              {content.output}
            </div>
          )}
          
          {/* Images */}
          {content.images && content.images.length > 0 && (
            <div className="python-images mt-4">
              {content.images.map((image, idx) => (
                <div key={idx} className="mb-4">
                  <img src={image} alt={`Python output figure ${idx+1}`} className="max-w-full h-auto rounded shadow-sm" />
                </div>
              ))}
            </div>
          )}
          
          {/* HTML content (e.g. pandas dataframes) */}
          {content.html && content.html.length > 0 && (
            <div className="python-html mt-4">
              {content.html.map((html, idx) => (
                <div 
                  key={idx} 
                  className="overflow-x-auto"
                  dangerouslySetInnerHTML={{ __html: html }}
                />
              ))}
            </div>
          )}
        </div>
      );
    }
    
    // Default to JSON rendering for other types
    return (
      <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded overflow-auto">
        {JSON.stringify(content, null, 2)}
      </pre>
    );
  };

  // Component to display and manage dependencies
  const DependencyManager = () => {
    const availableCells = currentNotebook 
      ? Object.values(currentNotebook.cells).filter(c => c.id !== cell.id)
      : [];
    
    return (
      <div className="mt-2 p-2 border border-gray-200 dark:border-gray-700 rounded bg-gray-50 dark:bg-gray-800">
        <div className="font-semibold mb-2">Dependencies</div>
        
        {cell.dependencies.length > 0 ? (
          <div className="mb-4">
            <div className="text-sm font-medium mb-1">Current Dependencies:</div>
            <ul className="pl-4 list-disc">
              {cell.dependencies.map(depId => {
                const depCell = currentNotebook?.cells[depId];
                return (
                  <li key={depId} className="flex items-center justify-between">
                    <span>
                      {depCell?.type} cell
                      {depCell?.metadata?.description ? ` (${depCell.metadata.description})` : ''}
                    </span>
                    <button 
                      onClick={() => removeDependency(cell.id, depId)}
                      className="text-red-500 hover:text-red-700"
                      title="Remove dependency"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ) : (
          <div className="text-sm text-gray-500 mb-4">No dependencies</div>
        )}
        
        {availableCells.length > 0 && (
          <div>
            <div className="text-sm font-medium mb-1">Add Dependency:</div>
            <div className="flex flex-wrap gap-2">
              {availableCells.map(c => (
                <button
                  key={c.id}
                  onClick={() => addDependency(cell.id, c.id)}
                  disabled={cell.dependencies.includes(c.id)}
                  className={`
                    px-2 py-1 text-xs rounded flex items-center
                    ${
                      cell.dependencies.includes(c.id)
                        ? 'bg-gray-300 dark:bg-gray-600 cursor-not-allowed'
                        : 'bg-blue-100 dark:bg-blue-900 hover:bg-blue-200 dark:hover:bg-blue-800'
                    }
                  `}
                  title={cell.dependencies.includes(c.id) ? 'Already a dependency' : 'Add as dependency'}
                >
                  <Link className="w-3 h-3 mr-1" />
                  {c.type} cell
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div 
      className={`
        cell mb-4 p-4 rounded-lg border
        ${cell.status === CellStatus.RUNNING ? 'border-blue-500 shadow-sm bg-blue-50 dark:bg-blue-900 dark:bg-opacity-10' : 'border-gray-200 dark:border-gray-700'}
        ${cell.status === CellStatus.ERROR ? 'border-red-300 bg-red-50 dark:bg-red-900 dark:bg-opacity-10' : ''}
        ${cell.status === CellStatus.COMPLETE ? 'border-green-200 bg-gray-50 dark:bg-gray-900' : ''}
      `}
    >
      <div className="cell-header flex justify-between items-center mb-2">
        <div className="flex items-center">
          <span className="text-xs font-mono px-2 py-1 bg-gray-200 dark:bg-gray-800 rounded mr-2">
            {cell.type}
          </span>
          {cell.metadata?.description && (
            <span className="text-sm text-gray-600 dark:text-gray-300">
              {cell.metadata.description}
            </span>
          )}
        </div>
        <div className="flex items-center space-x-1">
          {isEditing ? (
            <>
              <button 
                onClick={handleSave}
                className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300"
                title="Save"
              >
                <Check className="w-4 h-4" />
              </button>
              <button 
                onClick={() => setIsEditing(false)}
                className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300"
                title="Cancel"
              >
                <X className="w-4 h-4" />
              </button>
            </>
          ) : (
            <>
              <button 
                onClick={() => {
                  setEditValue(cell.content);
                  setIsEditing(true);
                }}
                className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300"
                title="Edit cell"
              >
                <Edit className="w-4 h-4" />
              </button>
              <button 
                onClick={executeCell}
                className={`
                  p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 
                  ${cell.status === CellStatus.RUNNING ? 'opacity-50 cursor-not-allowed' : 'text-gray-600 dark:text-gray-300'}
                `}
                title="Run cell"
                disabled={cell.status === CellStatus.RUNNING}
              >
                <Play className="w-4 h-4" />
              </button>
              <button 
                onClick={() => deleteCell(cell.id)}
                className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300"
                title="Delete cell"
              >
                <Trash className="w-4 h-4" />
              </button>
            </>
          )}
        </div>
      </div>
      
      <div className="cell-content">
        {isEditing ? (
          <MonacoEditor
            height="200px"
            language={getEditorLanguage()}
            value={editValue}
            onChange={(value) => setEditValue(value || '')}
            options={{
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              fontSize: 14,
              lineNumbers: 'on',
              roundedSelection: false,
              automaticLayout: true,
              wordWrap: 'on',
            }}
          />
        ) : (
          <div>
            {cell.type === CellType.MARKDOWN ? (
              <div className="markdown-content prose dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                    code({node, inline, className, children, ...props}) {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    }
                  }}
                >
                  {cell.content}
                </ReactMarkdown>
              </div>
            ) : (
              <SyntaxHighlighter
                language={getEditorLanguage()}
                style={vscDarkPlus}
              >
                {cell.content}
              </SyntaxHighlighter>
            )}

            {/* Cell result */}
            {cell.result && (
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                {renderCellResult()}
              </div>
            )}
          </div>
        )}
      </div>
      
      <div className="cell-footer mt-4 pt-2 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center text-sm text-gray-500">
        <div className="flex items-center space-x-2">
          {cell.status === CellStatus.RUNNING && (
            <div className="flex items-center">
              <div className="w-3 h-3 mr-1 rounded-full bg-blue-500 animate-pulse"></div>
              <span>Running...</span>
            </div>
          )}
          {cell.status === CellStatus.COMPLETE && cell.result?.executionTime && (
            <div>
              Executed in {cell.result.executionTime.toFixed(2)}s
            </div>
          )}
          {cell.status === CellStatus.ERROR && (
            <div className="text-red-500">
              Error
            </div>
          )}
        </div>
        <button 
          className="flex items-center text-xs"
          onClick={() => setShowDependencies(!showDependencies)}
        >
          {showDependencies ? <ChevronDown className="w-3 h-3 mr-1" /> : <ChevronRight className="w-3 h-3 mr-1" />}
          Dependencies: {cell.dependencies.length}
        </button>
      </div>

      {showDependencies && <DependencyManager />}
    </div>
  );
}