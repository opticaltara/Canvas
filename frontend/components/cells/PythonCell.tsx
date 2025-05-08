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
import ReactMarkdown from 'react-markdown'; // Import react-markdown
import remarkGfm from 'remark-gfm'; // Import remark-gfm for table support
import { useConnectionStore } from "@/app/store/connectionStore";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { ChevronDownIcon, ChevronUpIcon } from "lucide-react"; // Might not be used but keep for parity

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

  /* ---------------------- Tool Metadata ---------------------- */
  // Determine tool name / args from either top-level or metadata so we support older cells
  const toolName: string | undefined = (cell as any).tool_name || cell.metadata?.tool_name;
  const initialToolArgs: Record<string, any> = (cell as any).tool_arguments || cell.metadata?.tool_args || {};

  /* ---------------------- State for load_csv ------------------ */
  const [toolArgs, setToolArgs] = useState<Record<string, any>>(initialToolArgs);

  // Sync local state when the cell ID changes (i.e., when a new cell instance is rendered)
  useEffect(() => {
    setToolArgs(initialToolArgs);
  }, [cell.id]);

  /* ---------------------- Connection Store -------------------- */
  // We fetch python tool definitions (load_csv, run_script, etc.) from the store so we can dynamically build a form.
  // The connection type key is assumed to be "python" (matching backend). Adjust if needed.
  const toolDefinitions = useConnectionStore((state) => state.toolDefinitions.python);
  const toolLoadingStatus = useConnectionStore((state) => state.toolLoadingStatus.python);

  // Identify the tool definition for the active tool (load_csv)
  const toolInfo = toolDefinitions?.find((def) => def.name === toolName);

  /* ---------------------- Helpers ----------------------------- */
  const handleToolArgChange = (argName: string, value: any) => {
    setToolArgs((prev) => {
      const updated = { ...prev, [argName]: value };
      // Persist the change back via onUpdate so backend sees updated args before execution
      onUpdate(cell.id, cell.content, { toolName: toolName, toolArgs: updated });
      return updated;
    });
  };

  // Render input controls for each tool argument (mirrors logic from FileSystemCell)
  const renderToolFormInputs = (): React.ReactNode => {
    // Handle loading states gracefully
    if (!toolName) {
      return <div className="text-xs text-gray-500">No tool selected.</div>;
    }
    if (toolLoadingStatus === undefined || toolLoadingStatus === "idle") {
      return <div className="text-xs text-gray-500">Initializing tool definitions...</div>;
    }
    if (toolLoadingStatus === "loading") {
      return <div className="text-xs text-gray-500">Loading tool parameters...</div>;
    }
    if (toolLoadingStatus === "error") {
      return <div className="text-xs text-red-500">Error loading tool definitions.</div>;
    }
    if (!toolInfo) {
      return <div className="text-xs text-red-500">Tool definition not found for: {toolName}</div>;
    }

    const schema = toolInfo.inputSchema;
    const properties = schema?.properties as Record<string, any> | undefined;
    const requiredFields = new Set(schema?.required || []);

    if (!properties || Object.keys(properties).length === 0) {
      return <div className="text-xs text-gray-600 mb-3">This tool requires no arguments.</div>;
    }

    const renderField = (paramName: string, paramSchema: any) => {
      const fieldId = `${paramName}-${cell.id}`;
      const label = paramSchema.title || paramName;
      const isRequired = requiredFields.has(paramName);
      const description = paramSchema.description;
      const placeholder =
        (paramSchema.examples && paramSchema.examples[0]) || description || `Enter ${label}`;
      const currentValue = toolArgs[paramName] ?? paramSchema.default ?? "";

      // Decide input type
      const schemaType = paramSchema.type;
      const schemaFormat = paramSchema.format;
      const enumValues = paramSchema.enum as string[] | undefined;

      let inputElement: React.ReactNode = null;

      if (enumValues && Array.isArray(enumValues)) {
        inputElement = (
          <Select
            value={String(currentValue)}
            onValueChange={(val) => handleToolArgChange(paramName, val)}
            required={isRequired}
          >
            <SelectTrigger id={fieldId} className="mt-1 h-8 text-xs">
              <SelectValue placeholder={placeholder} />
            </SelectTrigger>
            <SelectContent>
              {enumValues.map((option) => (
                <SelectItem key={option} value={option}>
                  {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      } else if (schemaType === "string" && (schemaFormat === "textarea" || (description && description.length > 50))) {
        inputElement = (
          <Textarea
            id={fieldId}
            value={String(currentValue)}
            onChange={(e) => handleToolArgChange(paramName, e.target.value)}
            placeholder={placeholder}
            required={isRequired}
            className="mt-1 h-16 text-xs font-mono resize-none"
          />
        );
      } else if (schemaType === "number" || schemaType === "integer") {
        inputElement = (
          <Input
            id={fieldId}
            type="number"
            value={String(currentValue)}
            onChange={(e) =>
              handleToolArgChange(paramName, e.target.value === "" ? "" : Number(e.target.value))
            }
            placeholder={placeholder}
            required={isRequired}
            min={paramSchema.minimum}
            max={paramSchema.maximum}
            step={paramSchema.multipleOf || "any"}
            className="mt-1 h-8 text-xs"
          />
        );
      } else if (schemaType === "boolean") {
        inputElement = (
          <div className="flex items-center space-x-2 mt-1 h-8">
            <input
              type="checkbox"
              id={fieldId}
              checked={!!currentValue}
              onChange={(e) => handleToolArgChange(paramName, e.target.checked)}
              className="h-4 w-4 text-emerald-600 border-gray-300 rounded focus:ring-emerald-500"
            />
          </div>
        );
      } else {
        inputElement = (
          <Input
            id={fieldId}
            type="text"
            value={String(currentValue)}
            onChange={(e) => handleToolArgChange(paramName, e.target.value)}
            placeholder={placeholder}
            required={isRequired}
            className="mt-1 h-8 text-xs font-mono"
          />
        );
      }

      return (
        <div key={fieldId} className="mb-3">
          <Label htmlFor={fieldId} className="text-xs text-gray-700">
            {label} {isRequired && <span className="text-red-500">*</span>}
          </Label>
          {inputElement}
          {description && <p className="text-xs text-gray-500 mt-1">{description}</p>}
        </div>
      );
    };

    return (
      <div className="space-y-2">
        {Object.entries(properties).map(([name, schema]) => renderField(name, schema))}
      </div>
    );
  };

  /* ============================================================
     MAIN RENDER: Two modes
     - run_script: show code panel + output (existing behavior)
     - load_csv (or other non-script): show argument form + output
  ============================================================ */

  const isRunScript = toolName === "run_script" || toolName === "run-script";

  // Helper function to render structured output from stdout
  const renderStructuredOutput = (stdout: string | null | undefined): React.ReactNode => {
    if (!stdout) return null;

    const outputParts: React.ReactNode[] = [];
    let remainingStdout = stdout;

    // Regex patterns for markers
    const dfRegex = /--- DataFrame ---\n([\s\S]*?)\n--- End DataFrame ---/g;
    const plotRegex = /<PLOT_BASE64>(.*?)<\/PLOT_BASE64>/g;
    const jsonRegex = /<JSON_OUTPUT>(.*?)<\/JSON_OUTPUT>/g;

    // Function to process matches and remaining text
    const processChunk = (text: string) => {
      if (text.trim()) {
        outputParts.push(
          <pre key={`text-${outputParts.length}`} className="whitespace-pre-wrap font-mono text-xs bg-gray-50 p-1.5 rounded border border-gray-200">
            {text.trim()}
          </pre>
        );
      }
    };

    let lastIndex = 0;
    const markers = [
      { regex: dfRegex, type: 'dataframe' },
      { regex: plotRegex, type: 'plot' },
      { regex: jsonRegex, type: 'json' },
    ];

    // Find all marker positions
    const allMatches: { index: number; length: number; type: string; content: string }[] = [];
    markers.forEach(({ regex, type }) => {
      let match;
      while ((match = regex.exec(stdout)) !== null) {
        allMatches.push({ index: match.index, length: match[0].length, type, content: match[1] });
      }
    });

    // Sort matches by index
    allMatches.sort((a, b) => a.index - b.index);

    // Process text and markers in order
    allMatches.forEach(match => {
      // Process text before the marker
      processChunk(stdout.substring(lastIndex, match.index));

      // Process the marker content
      if (match.type === 'dataframe') {
        outputParts.push(
          <div key={`df-${match.index}`} className="markdown-table-container my-1 border rounded p-1 bg-white">
             <ReactMarkdown remarkPlugins={[remarkGfm]}>{match.content}</ReactMarkdown>
          </div>
        );
      } else if (match.type === 'plot') {
        outputParts.push(
          <img 
            key={`plot-${match.index}`} 
            src={`data:image/png;base64,${match.content}`} 
            alt="Generated Plot" 
            className="my-1 max-w-full h-auto border rounded"
          />
        );
      } else if (match.type === 'json') {
        try {
          const parsedJson = JSON.parse(match.content);
          outputParts.push(
            <pre key={`json-${match.index}`} className="whitespace-pre-wrap font-mono text-xs bg-gray-50 p-1.5 rounded border border-gray-200 my-1">
              {JSON.stringify(parsedJson, null, 2)}
            </pre>
          );
        } catch (e) {
          console.error("Failed to parse JSON output:", e);
          outputParts.push(
             <div key={`json-err-${match.index}`} className="text-red-600 text-xs">Error parsing JSON output.</div>
          );
        }
      }
      lastIndex = match.index + match.length;
    });

    // Process any remaining text after the last marker
    processChunk(stdout.substring(lastIndex));

    // If no markers were found, render the whole stdout as plain text
    if (outputParts.length === 0 && stdout.trim()) {
       processChunk(stdout);
    }

    return outputParts.length > 0 ? <>{outputParts}</> : null;
  };

  return (
    <div className="border rounded-md overflow-hidden mb-3 mx-8 bg-white">
      {/* Header */}
      <div className="bg-emerald-100 border-b border-emerald-200 p-2 flex justify-between items-center">
        <div className="flex items-center">
          <CodeIcon className="h-4 w-4 mr-2 text-emerald-700" />
          <span className="font-medium text-sm text-emerald-800">Python Cell</span> 
          {/* Display Tool Name if available in metadata */}
          {toolName && (
            <span className="ml-2 text-xs text-emerald-600 bg-emerald-200 px-1.5 py-0.5 rounded">
              Tool: {toolName}
            </span>
          )}
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

      {/* Tool Arguments / Form */}
      {!isRunScript && (
        <div className="px-3 py-2 border-b bg-gray-50">
          {renderToolFormInputs()}
        </div>
      )}

      {/* For run_script we keep the code panel, for others we only show output */}
      <ResizablePanelGroup direction="vertical" className="min-h-[350px]">
        {isRunScript && (
          <>
            <ResizablePanel defaultSize={50} minSize={20}>
              <ScrollArea className="h-full p-1 text-xs">
                <SyntaxHighlighter
                  language="python"
                  style={materialLight}
                  customStyle={{
                    margin: 0,
                    padding: "8px",
                    fontSize: "0.8rem",
                    backgroundColor: "#f8f9fa",
                    borderRadius: "4px",
                    height: "100%",
                    overflow: "auto",
                    boxSizing: "border-box",
                  }}
                  showLineNumbers={true}
                  wrapLines={true}
                  lineNumberStyle={{ color: "#adb5bd", fontSize: "0.75rem" }}
                >
                  {pythonCode}
                </SyntaxHighlighter>
              </ScrollArea>
            </ResizablePanel>
            <ResizableHandle withHandle />
          </>
        )}

        {/* Output Panel (always present) */}
        <ResizablePanel defaultSize={isRunScript ? 50 : 100} minSize={20}>
          <div className="h-full flex flex-col">
             <div className="flex justify-between items-center p-1.5 border-b bg-gray-50">
                <span className="text-xs font-medium text-gray-700">Execution Result</span>
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
                  {/* Render Structured Stdout or plain stdout */}
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
                               <TooltipContent><p>Copy Raw stdout</p></TooltipContent>
                             </Tooltip>
                          </TooltipProvider>
                       </div>
                       {/* Use the new renderer */}
                       <div className="output-content-area mt-1"> 
                          {renderStructuredOutput(parsedResult.stdout)}
                       </div>
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
