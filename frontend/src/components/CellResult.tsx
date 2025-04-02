import React, { useState } from 'react';
import { CellResult as CellResultType, CellType } from '../types/notebook';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/Tabs';
import MarkdownRenderer from './MarkdownRenderer';

interface CellResultProps {
  result: CellResultType;
  cellType: CellType;
}

const CellResult: React.FC<CellResultProps> = ({ result, cellType }) => {
  const [activeTab, setActiveTab] = useState<string>('data');
  
  if (!result) return null;
  
  const hasError = !!result.error;
  const hasDataframe = result.dataframe_html || (result.dataframe && Object.keys(result.dataframe).length > 0);
  const hasFigures = result.figures && result.figures.length > 0;
  const hasStdout = result.stdout && result.stdout.trim().length > 0;
  const hasStderr = result.stderr && result.stderr.trim().length > 0;
  
  const renderDataContent = () => {
    // Different rendering based on cell type
    switch (cellType) {
      case CellType.PYTHON:
        // For Python cells, show the execution result
        if (result.result) {
          if (typeof result.result === 'object') {
            return (
              <pre className="bg-gray-50 dark:bg-gray-800 p-3 rounded overflow-auto">
                {JSON.stringify(result.result, null, 2)}
              </pre>
            );
          } else {
            return <div className="font-mono">{String(result.result)}</div>;
          }
        } else if (hasDataframe) {
          return renderDataframe();
        } else {
          return <div className="text-gray-500 italic">No output</div>;
        }
      
      case CellType.SQL:
      case CellType.LOG:
      case CellType.METRIC:
      case CellType.S3:
        // For data query cells, show query results as tables
        return renderDataframe();
      
      case CellType.MARKDOWN:
        // Markdown cells don't have results
        return null;
      
      case CellType.AI_QUERY:
        // For AI queries, show thinking and plan
        if (result.plan) {
          return (
            <div>
              <h3 className="font-medium mb-2">Investigation Plan</h3>
              <div className="mb-4">
                {result.plan.steps && (
                  <ul className="list-disc pl-5 space-y-1">
                    {result.plan.steps.map((step: any, index: number) => (
                      <li key={index} className="text-sm">
                        <span className="font-medium">{step.description}</span>
                        {" "}
                        <span className="text-xs text-gray-500">({step.cell_type})</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {result.thinking && (
                <div>
                  <h3 className="font-medium mb-2">AI Thinking</h3>
                  <div className="text-sm bg-gray-50 dark:bg-gray-800 p-3 rounded max-h-60 overflow-auto">
                    <MarkdownRenderer content={result.thinking} />
                  </div>
                </div>
              )}
            </div>
          );
        } else {
          return <div className="text-gray-500 italic">No investigation plan generated</div>;
        }
      
      default:
        return <div className="text-gray-500 italic">No output</div>;
    }
  };
  
  const renderDataframe = () => {
    if (result.dataframe_html) {
      return (
        <div 
          className="overflow-auto max-h-96" 
          dangerouslySetInnerHTML={{ __html: result.dataframe_html }}
        />
      );
    } else if (result.data) {
      const data = Array.isArray(result.data) ? result.data : [result.data];
      
      if (data.length === 0) {
        return <div className="text-gray-500 italic">No data returned</div>;
      }
      
      // Get column headers from the first row
      const columns = Object.keys(data[0]);
      
      return (
        <div className="overflow-auto max-h-96">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase bg-gray-100 dark:bg-gray-700">
              <tr>
                {columns.map((column, idx) => (
                  <th key={idx} className="px-4 py-2">{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.slice(0, 100).map((row, rowIdx) => (
                <tr key={rowIdx} className="border-b dark:border-gray-700">
                  {columns.map((column, colIdx) => (
                    <td key={colIdx} className="px-4 py-2">
                      {formatCellValue(row[column])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {data.length > 100 && (
            <div className="text-center text-gray-500 text-xs mt-2">
              Showing 100 of {data.length} rows
            </div>
          )}
        </div>
      );
    } else {
      return <div className="text-gray-500 italic">No data returned</div>;
    }
  };
  
  const formatCellValue = (value: any): string => {
    if (value === null || value === undefined) return 'NULL';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };
  
  const renderFigures = () => {
    if (!hasFigures) {
      return <div className="text-gray-500 italic">No figures generated</div>;
    }
    
    return (
      <div className="space-y-4">
        {result.figures.map((figure, index) => (
          <div key={index} className="flex flex-col items-center">
            <img 
              src={`data:image/${figure.format};base64,${figure.data}`} 
              alt={`Figure ${index + 1}`}
              className="max-w-full"
            />
            <div className="text-sm text-gray-500 mt-1">Figure {index + 1}</div>
          </div>
        ))}
      </div>
    );
  };
  
  return (
    <div className={`p-3 ${hasError ? 'bg-red-50 dark:bg-red-900/20' : 'bg-gray-50 dark:bg-gray-800'}`}>
      {hasError ? (
        <div className="text-red-700 dark:text-red-400 whitespace-pre-wrap font-mono text-sm">
          Error: {result.error}
        </div>
      ) : (
        <Tabs 
          defaultValue="data" 
          value={activeTab} 
          onValueChange={setActiveTab} 
          className="w-full"
        >
          <TabsList className="mb-2">
            <TabsTrigger value="data">Result</TabsTrigger>
            {hasFigures && <TabsTrigger value="figures">Figures</TabsTrigger>}
            {(hasStdout || hasStderr) && <TabsTrigger value="console">Console</TabsTrigger>}
            {cellType === CellType.AI_QUERY && <TabsTrigger value="thinking">Thinking</TabsTrigger>}
          </TabsList>
          
          <TabsContent value="data" className="w-full">
            {renderDataContent()}
          </TabsContent>
          
          {hasFigures && (
            <TabsContent value="figures">
              {renderFigures()}
            </TabsContent>
          )}
          
          {(hasStdout || hasStderr) && (
            <TabsContent value="console">
              <div className="font-mono text-sm overflow-auto max-h-96">
                {hasStdout && (
                  <div className="mb-2">
                    <div className="font-semibold mb-1">Standard Output:</div>
                    <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded whitespace-pre-wrap">
                      {result.stdout}
                    </pre>
                  </div>
                )}
                
                {hasStderr && (
                  <div>
                    <div className="font-semibold mb-1 text-red-600 dark:text-red-400">Standard Error:</div>
                    <pre className="bg-red-50 dark:bg-red-900/20 p-2 rounded whitespace-pre-wrap text-red-700 dark:text-red-400">
                      {result.stderr}
                    </pre>
                  </div>
                )}
              </div>
            </TabsContent>
          )}
          
          {cellType === CellType.AI_QUERY && (
            <TabsContent value="thinking">
              <div className="text-sm bg-gray-100 dark:bg-gray-900 p-3 rounded max-h-96 overflow-auto">
                <MarkdownRenderer content={result.thinking || "No thinking process recorded"} />
              </div>
            </TabsContent>
          )}
        </Tabs>
      )}
      
      <div className="text-xs text-gray-500 mt-2">
        Execution time: {result.execution_time ? `${result.execution_time.toFixed(2)}s` : 'N/A'} · 
        {result.timestamp ? ` ${new Date(result.timestamp).toLocaleString()}` : ''}
      </div>
    </div>
  );
};

export default CellResult;