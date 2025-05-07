"use client"

import React from "react"
import { useState, useEffect } from "react"
import { PlayIcon, CopyIcon, CheckIcon, Trash2Icon, AlertCircleIcon, ServerIcon, ChevronsUpDownIcon, CodeIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
// Choose a style - you might want a different one for Python vs. GitHub
import { materialLight } from 'react-syntax-highlighter/dist/esm/styles/prism' 
import { ScrollArea } from "@/components/ui/scroll-area"
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable"
import { Separator } from "@/components/ui/separator"
import { Label } from "@/components/ui/label"

import { type Cell } from "@/store/types"

// Define the expected structure of the result content from the Python MCP server
interface PythonExecutionResult {
  status: 'success' | 'error';
  stdout?: string | null;
  stderr?: string | null;
  return_value?: any | null; // Can be any JSON-serializable type
  dependencies?: string[] | null;
  error_details?: { // If status is 'error'
    type: string;
    message: string;
    traceback?: string | null;
  } | null;
}

interface PythonCellProps {
  cell: Cell 
  onExecute: (cellId: string) => void
  // onUpdate needed if we allow editing code later
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void 
  onDelete: (cellId: string) => void
  isExecuting: boolean // Use the cell.status instead?
}

const PythonCellComponent: React.FC<PythonCellProps> = ({ cell, onExecute, onUpdate, onDelete }): React.ReactNode => {
  const [copiedCode, setCopiedCode] = useState(false);
  const [copiedStdout, setCopiedStdout] = useState(false);
  const [copiedStderr, setCopiedStderr] = useState(false);
  const [copiedReturnValue, setCopiedReturnValue] = useState(false);

  const isExecuting = cell.status === "running" || cell.status === "queued";
  const executionResult = cell.result?.content as PythonExecutionResult | null;
  const executionError = cell.result?.error; // General cell error, if any

  // Log cell rendering
  console.log(`[PythonCell ${cell.id}] Render with cell:`, JSON.stringify({ 
    id: cell.id, 
    type: cell.type, 
    status: cell.status, 
    has_result: !!cell.result,
    result_content_type: typeof cell.result?.content,
    result_content_keys: executionResult ? Object.keys(executionResult) : null,
    result_error: cell.result?.error,
    code_length: cell.content?.length
  }));

  // Copy to clipboard utility
  const copyToClipboard = (text: string | null | undefined, setter: React.Dispatch<React.SetStateAction<boolean>>) => {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setter(true);
      setTimeout(() => setter(false), 2000); // Reset after 2s
    }, (err) => {
      console.error('Could not copy text: ', err);
    });
  };

  // Parse the structured result content (which might be a string initially)
  let parsedResult: PythonExecutionResult | null = null;
  let parseError: string | null = null;
  if (executionResult) {
      if (typeof executionResult === 'string') {
          try {
              // Attempt to parse if it's a JSON string representation of PythonExecutionResult
              parsedResult = JSON.parse(executionResult) as PythonExecutionResult;
          } catch (e) {
              // If parsing fails, treat the string as likely stdout or an error message
              console.warn(`[PythonCell ${cell.id}] Failed to parse result string as JSON:`, e);
              // Heuristic: If the original cell status was error, maybe it's stderr?
              if (cell.status === "error") {
                parsedResult = { status: 'error', stderr: executionResult };
              } else {
                // Otherwise, assume it's stdout
                 parsedResult = { status: 'success', stdout: executionResult };
              }
              parseError = "Result was a string, not structured JSON.";
          }
      } else if (typeof executionResult === 'object' && executionResult !== null) {
          // It's already an object, assume it matches PythonExecutionResult structure
          parsedResult = executionResult;
      } else {
          // Handle unexpected types
          parseError = `Unexpected result content type: ${typeof executionResult}`;
          console.error(`[PythonCell ${cell.id}] ${parseError}`);
      }
  } else if (executionError) {
       // If cell.result.content is empty but cell.result.error exists
       parseError = `Cell execution failed: ${executionError}`;
       parsedResult = { status: 'error', stderr: executionError };
  }

  const pythonCode = cell.content || "# No code provided";

  return (
    <div className="border rounded-md overflow-hidden mb-3 mx-8 bg-white">
      {/* Header */}
      <div className="bg-emerald-100 border-b border-emerald-200 p-2 flex justify-between items-center">
        <div className="flex items-center">
          <CodeIcon className="h-4 w-4 mr-2 text-emerald-700" />
          <span className="font-medium text-sm text-emerald-800">Python Code</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <TooltipProvider delayDuration={100}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-gray-600 hover:bg-emerald-100 hover:text-emerald-700 h-7 w-7 px-0"
                  onClick={() => copyToClipboard(pythonCode, setCopiedCode)}
                >
                  {copiedCode ? <CheckIcon className="h-4 w-4 text-green-600" /> : <CopyIcon className="h-4 w-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent><p>Copy Code</p></TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Button
            size="sm"
            variant="outline"
            className="bg-white hover:bg-emerald-50 border-emerald-300 h-7 px-2 text-xs text-emerald-800 hover:text-emerald-900"
            onClick={() => onExecute(cell.id)}
            disabled={isExecuting}
          >
            <PlayIcon className="h-3.5 w-3.5 mr-1 text-emerald-700" />
            {isExecuting ? "Running..." : "Run Code"}
          </Button>
          <TooltipProvider delayDuration={100}>
             <Tooltip>
               <TooltipTrigger asChild>
                 <Button
                   size="sm"
                   variant="ghost"
                   className="text-gray-500 hover:bg-red-100 hover:text-red-700 h-7 w-7 px-0"
                   onClick={() => onDelete(cell.id)}
                   disabled={isExecuting}
                 >
                   <Trash2Icon className="h-4 w-4" />
                 </Button>
               </TooltipTrigger>
               <TooltipContent><p>Delete Cell</p></TooltipContent>
             </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      {/* Code and Output Area */}
      <ResizablePanelGroup direction="vertical" className="min-h-[250px]"> {/* Increased min-h */}
        {/* Code Panel */}
        <ResizablePanel defaultSize={65} minSize={30}> {/* Increased defaultSize and minSize */}
          <ScrollArea className="h-full p-1 text-xs">
            <SyntaxHighlighter
              language="python"
              style={materialLight} 
              customStyle={{ 
                margin: 0, 
                padding: '8px', 
                fontSize: '0.8rem', // Slightly smaller font
                backgroundColor: '#f8f9fa', // Light background for code
                borderRadius: '4px',
                height: '100%',
                overflow: 'auto',
                boxSizing: 'border-box'
              }}
              showLineNumbers={true}
              wrapLines={true}
              lineNumberStyle={{ color: '#adb5bd', fontSize: '0.75rem' }}
            >
              {pythonCode}
            </SyntaxHighlighter>
          </ScrollArea>
        </ResizablePanel>
        
        <ResizableHandle withHandle />

        {/* Output Panel */}
        <ResizablePanel defaultSize={50} minSize={20}>
          <div className="h-full flex flex-col">
             <div className="flex justify-between items-center p-1.5 border-b bg-gray-50">
                <span className="text-xs font-medium text-gray-700">Output</span>
                {/* Optional: Add clear output button? */} 
             </div>
             <ScrollArea className="flex-grow p-2 text-xs overflow-auto">
              {/* Status Indicator */} 
              {cell.status === "running" && (
                <div className="flex items-center text-xs text-amber-600 mb-2">
                  <div className="animate-spin h-3 w-3 border-2 border-amber-600 rounded-full border-t-transparent mr-1.5"></div>
                  Executing code...
                </div>
              )}
              {cell.status === "queued" && (
                <div className="flex items-center text-xs text-blue-600 mb-2">
                   <div className="h-3 w-3 bg-blue-600 rounded-full mr-1.5 animate-pulse"></div>
                   Queued for execution...
                </div>
              )}
              {cell.status === "stale" && (
                <div className="flex items-center text-xs text-gray-600 mb-2">
                   <div className="h-3 w-3 border border-gray-600 rounded-full mr-1.5"></div>
                   Stale (needs re-run)... 
                </div>
              )}

              {/* Display Parse Error */}
              {parseError && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded p-1.5 mb-2 text-yellow-800 text-xs">
                    <div className="flex items-center">
                       <AlertCircleIcon className="h-3.5 w-3.5 mr-1 flex-shrink-0"/>
                       <span className="font-medium">Result Parsing Issue:</span>
                    </div>
                    <div className="ml-5 mt-0.5">{parseError}</div>
                  </div>
              )}

              {/* Display General Cell Error */}
              {executionError && !parsedResult?.stderr && (
                 <div className="bg-red-50 border border-red-200 rounded p-1.5 mb-2 text-red-700 text-xs">
                    <div className="flex items-center">
                      <AlertCircleIcon className="h-3.5 w-3.5 mr-1 flex-shrink-0"/>
                      <span className="font-medium">Cell Execution Error:</span>
                    </div>
                    <pre className="whitespace-pre-wrap font-mono text-xs mt-1 ml-5">{executionError}</pre>
                  </div>
              )}

              {/* Display Python Execution Results */} 
              {parsedResult && (
                <div className="space-y-3">
                  {/* Stdout */} 
                  {parsedResult.stdout && (
                     <div>
                       <div className="flex justify-between items-center mb-0.5">
                          <Label className="text-xs text-gray-600 font-medium">stdout</Label>
                          <TooltipProvider delayDuration={100}>
                             <Tooltip>
                               <TooltipTrigger asChild>
                                 <Button variant="ghost" size="icon" className="h-5 w-5 text-gray-500 hover:bg-gray-100" onClick={() => copyToClipboard(parsedResult?.stdout, setCopiedStdout)}>
                                    {copiedStdout ? <CheckIcon className="h-3 w-3 text-green-600"/> : <CopyIcon className="h-3 w-3"/>}
                                 </Button>
                               </TooltipTrigger>
                               <TooltipContent><p>Copy stdout</p></TooltipContent>
                             </Tooltip>
                          </TooltipProvider>
                       </div>
                       <pre className="whitespace-pre-wrap font-mono text-xs bg-gray-50 p-1.5 rounded border border-gray-200 max-h-40 overflow-y-auto">{parsedResult.stdout}</pre>
                     </div>
                  )}
                  
                  {/* Stderr */} 
                  {parsedResult.stderr && (
                     <div>
                       <div className="flex justify-between items-center mb-0.5">
                          <Label className="text-xs text-red-700 font-medium">stderr</Label>
                          <TooltipProvider delayDuration={100}>
                             <Tooltip>
                               <TooltipTrigger asChild>
                                 <Button variant="ghost" size="icon" className="h-5 w-5 text-gray-500 hover:bg-gray-100" onClick={() => copyToClipboard(parsedResult?.stderr, setCopiedStderr)}>
                                    {copiedStderr ? <CheckIcon className="h-3 w-3 text-green-600"/> : <CopyIcon className="h-3 w-3"/>}
                                 </Button>
                               </TooltipTrigger>
                               <TooltipContent><p>Copy stderr</p></TooltipContent>
                             </Tooltip>
                          </TooltipProvider>
                       </div>
                       <pre className="whitespace-pre-wrap font-mono text-xs bg-red-50 p-1.5 rounded border border-red-200 text-red-800 max-h-40 overflow-y-auto">{parsedResult.stderr}</pre>
                     </div>
                  )}
                  
                  {/* Return Value */} 
                  {parsedResult.return_value !== undefined && parsedResult.return_value !== null && (
                    <div>
                       <div className="flex justify-between items-center mb-0.5">
                           <Label className="text-xs text-gray-600 font-medium">Return Value</Label>
                           <TooltipProvider delayDuration={100}>
                             <Tooltip>
                               <TooltipTrigger asChild>
                                 <Button variant="ghost" size="icon" className="h-5 w-5 text-gray-500 hover:bg-gray-100" onClick={() => copyToClipboard(JSON.stringify(parsedResult?.return_value, null, 2), setCopiedReturnValue)}>
                                     {copiedReturnValue ? <CheckIcon className="h-3 w-3 text-green-600"/> : <CopyIcon className="h-3 w-3"/>}
                                 </Button>
                               </TooltipTrigger>
                               <TooltipContent><p>Copy Return Value (JSON)</p></TooltipContent>
                             </Tooltip>
                           </TooltipProvider>
                       </div>
                       <pre className="whitespace-pre-wrap font-mono text-xs bg-gray-50 p-1.5 rounded border border-gray-200 max-h-40 overflow-y-auto">
                         {/* Pretty print if object/array, otherwise display as string */} 
                         {typeof parsedResult.return_value === 'object' 
                           ? JSON.stringify(parsedResult.return_value, null, 2)
                           : String(parsedResult.return_value)}
                       </pre>
                     </div>
                  )}

                  {/* Traceback (if error) */} 
                  {parsedResult.status === 'error' && parsedResult.error_details?.traceback && (
                     <div>
                       <Label className="text-xs text-red-700 font-medium">Traceback</Label>
                       <pre className="whitespace-pre-wrap font-mono text-xs bg-red-50 p-1.5 rounded border border-red-200 text-red-800 mt-0.5 max-h-60 overflow-y-auto">{parsedResult.error_details.traceback}</pre>
                     </div>
                  )}
                  
                  {/* Dependencies */} 
                  {parsedResult.dependencies && parsedResult.dependencies.length > 0 && (
                     <div>
                       <Label className="text-xs text-gray-600 font-medium">Dependencies Used</Label>
                       <div className="flex flex-wrap gap-1 mt-0.5">
                          {parsedResult.dependencies.map(dep => (
                            <span key={dep} className="text-xs bg-gray-100 text-gray-700 px-1.5 py-0.5 rounded border">{dep}</span>
                          ))}
                       </div>
                     </div>
                  )}
                </div>
              )}
              
              {/* No output message */} 
              {!isExecuting && !parsedResult && !parseError && !executionError && (
                  <div className="text-xs text-gray-500 italic">No output received yet.</div>
              )}
             </ScrollArea>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}

// Memoize for performance
const PythonCell = React.memo(PythonCellComponent);

export default PythonCell;
