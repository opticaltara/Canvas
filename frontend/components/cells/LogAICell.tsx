import React, { useState } from 'react';
import { Cell } from '@/store/types';

interface LogAICellProps {
  cell: Cell;
}

/**
 * Very simple renderer for Log-AI tool results.
 *   • Shows description + tool name
 *   • Renders a collapsible JSON preview of the result payload
 */
const LogAICell: React.FC<LogAICellProps> = ({ cell }) => {
  const [expanded, setExpanded] = useState(false);

  const toolName = cell.tool_name || 'log_ai_tool';
  const description = cell.content || cell.tool_arguments?.description || '';

  const error = cell.result?.error;
  const resultContent = cell.result?.content ?? cell.result ?? null;

  return (
    <div className="p-4 text-sm">
      <div className="mb-2 font-semibold text-gray-700">
        Log-AI Result &mdash; <code>{toolName}</code>
      </div>
      {description && (
        <p className="mb-2 whitespace-pre-wrap text-gray-600">{description}</p>
      )}

      {error ? (
        <div className="p-3 border rounded bg-red-50 text-red-700">
          <strong>Error:</strong> {error}
        </div>
      ) : (
        <div className="border rounded bg-gray-50">
          <button
            className="w-full text-left px-3 py-2 font-mono text-xs hover:bg-gray-100 focus:outline-none"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? '▾ Hide raw result' : '▸ Show raw result'}
          </button>
          {expanded && (
            <pre className="px-3 py-2 overflow-x-auto text-xs">
              {typeof resultContent === 'string'
                ? resultContent
                : JSON.stringify(resultContent, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
};

export default LogAICell; 