import React from 'react';
import { Cell } from '../types/notebook';
import { MarkdownRenderer } from './MarkdownRenderer';
import { cn } from '../utils/cn';

interface CellResultProps {
  cell: Cell;
  className?: string;
}

export function CellResult({ cell, className }: CellResultProps) {
  if (!cell.result && !cell.error) {
    return null;
  }

  return (
    <div 
      className={cn(
        'p-4 rounded-md mt-2 overflow-auto',
        cell.error ? 'bg-red-50 border border-red-200' : 'bg-gray-50 border border-gray-200',
        className
      )}
    >
      {cell.error ? (
        <div className="text-red-600 font-mono text-sm whitespace-pre-wrap">{cell.error}</div>
      ) : (
        <RenderCellResult cell={cell} />
      )}
    </div>
  );
}

function RenderCellResult({ cell }: { cell: Cell }) {
  const result = cell.result;
  
  // Handle null/undefined results
  if (result === null || result === undefined) {
    return <div className="text-gray-500 italic">No output</div>;
  }
  
  // Handle different result types based on cell type
  switch (cell.type) {
    case 'markdown':
      return null; // Markdown cells display their content directly
      
    case 'sql':
      return <SQLResult data={result} />;
      
    case 'log':
    case 'metric':
    case 's3':
      // Check if it's likely JSON
      if (typeof result === 'object') {
        return <JSONResult data={result} />;
      }
      return <div className="font-mono text-sm whitespace-pre-wrap">{String(result)}</div>;
      
    case 'python':
      // Python results might be markdown, plots, or data
      if (typeof result === 'string' && result.startsWith('<!-- markdown -->')) {
        return <MarkdownRenderer content={result.replace('<!-- markdown -->', '')} />;
      }
      
      if (typeof result === 'object') {
        return <JSONResult data={result} />;
      }
      
      return <div className="font-mono text-sm whitespace-pre-wrap">{String(result)}</div>;
      
    case 'ai_query':
      if (result.thinking) {
        return (
          <div>
            <details className="mb-4">
              <summary className="cursor-pointer text-sm font-medium">Show AI thinking</summary>
              <div className="mt-2 p-3 bg-gray-100 rounded text-sm">
                <MarkdownRenderer content={result.thinking} />
              </div>
            </details>
            {result.plan && <PlanResult plan={result.plan} />}
          </div>
        );
      }
      return <JSONResult data={result} />;
      
    default:
      // Fallback for unknown types
      if (typeof result === 'object') {
        return <JSONResult data={result} />;
      }
      return <div className="font-mono text-sm whitespace-pre-wrap">{String(result)}</div>;
  }
}

function SQLResult({ data }: { data: any }) {
  if (!Array.isArray(data) || data.length === 0) {
    return <div className="text-gray-500 italic">No results returned</div>;
  }
  
  const headers = Object.keys(data[0]);
  
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-100">
          <tr>
            {headers.map((header) => (
              <th 
                key={header} 
                className="px-3 py-2 text-left text-xs font-medium text-gray-700 uppercase tracking-wider"
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((row, i) => (
            <tr key={i} className="hover:bg-gray-50">
              {headers.map((header) => (
                <td 
                  key={`${i}-${header}`} 
                  className="px-3 py-2 text-sm font-mono whitespace-nowrap text-gray-800"
                >
                  {formatValue(row[header])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-xs text-gray-500 mt-2">{data.length} rows</div>
    </div>
  );
}

function JSONResult({ data }: { data: any }) {
  return (
    <pre className="text-sm overflow-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function PlanResult({ plan }: { plan: any }) {
  if (!plan || !plan.steps || !Array.isArray(plan.steps)) {
    return null;
  }
  
  return (
    <div className="mt-4">
      <h3 className="text-sm font-medium mb-2">Investigation Plan</h3>
      <ol className="list-decimal pl-5">
        {plan.steps.map((step: any, index: number) => (
          <li key={index} className="mb-1 text-sm">
            <span className="font-medium">{step.description}</span>
            <span className="ml-2 text-xs text-gray-500">({step.cell_type})</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function formatValue(value: any): string {
  if (value === null || value === undefined) {
    return 'NULL';
  }
  
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  
  return String(value);
}
