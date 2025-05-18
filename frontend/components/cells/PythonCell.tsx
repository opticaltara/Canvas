"use client"

import React, { useState, useEffect, useCallback } from "react"
import { PlayIcon, CopyIcon, CheckIcon, Trash2Icon, AlertCircleIcon, ServerIcon, ChevronsUpDownIcon, CodeIcon, ChevronDownIcon, ChevronUpIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Label } from "@/components/ui/label"
import ReactMarkdown from 'react-markdown'; // Import react-markdown
import remarkGfm from 'remark-gfm'; // Import remark-gfm for table support
import { useConnectionStore } from "@/app/store/connectionStore";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"; // Added Accordion

// Import CodeMirror components
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { githubLight } from '@uiw/codemirror-theme-github'; // Or choose another theme
import type { ViewUpdate } from "@codemirror/view"; // Import type for onChange handler

import { type Cell } from "@/store/types"

// Helper for debouncing
function debounce<F extends (...args: any[]) => any>(func: F, waitFor: number) {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  const debounced = (...args: Parameters<F>) => {
    if (timeout !== null) {
      clearTimeout(timeout);
      timeout = null;
    }
    timeout = setTimeout(() => func(...args), waitFor);
  };

  return debounced as (...args: Parameters<F>) => void;
}

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
  isExecuting: boolean
}

const PythonCellComponent: React.FC<PythonCellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }): React.ReactNode => {
  const [copiedCode, setCopiedCode] = useState(false);
  const [copiedStdout, setCopiedStdout] = useState(false);
  const [copiedStderr, setCopiedStderr] = useState(false);
  const [copiedReturnValue, setCopiedReturnValue] = useState(false);
  const [isResultExpanded, setIsResultExpanded] = useState(false); // State for result card expansion

  const executionResult = cell.result?.content as PythonExecutionResult | null;
  const executionError = cell.result?.error; // General cell error, if any

  // Global expansion state from store
  const areAllCellsExpanded = useConnectionStore((state) => state.areAllCellsExpanded);

  // State for controlled accordion for Tool Arguments
  const [argsAccordionValue, setArgsAccordionValue] = useState<string>("");

  // Effect to sync Tool Arguments accordion with global state
  useEffect(() => {
    setArgsAccordionValue(areAllCellsExpanded ? "item-1" : "");
  }, [areAllCellsExpanded]);

  // Effect to sync Execution Result card expansion with global state
  useEffect(() => {
    setIsResultExpanded(areAllCellsExpanded);
  }, [areAllCellsExpanded]);

  // Log cell rendering
  console.log(`[PythonCell ${cell.id}] Render with cell:`, JSON.stringify({ 
    id: cell.id, 
    type: cell.type, 
    status: cell.status, 
    has_result: !!cell.result,
    result_content_type: typeof cell.result?.content,
    result_content_keys: executionResult ? Object.keys(executionResult) : null,
    result_error: cell.result?.error,
    // For run_script, cell.content might be a label like "Python: run_script". 
    // The actual script is in tool_arguments.script.
    // We log the length of cell.content for general info.
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

  // This pythonCode is primarily cell.content, which for run_script might just be a label.
  // The actual script for run_script is handled via toolArgs.script
  const pythonCode = cell.content || "# No code provided"; 

  /* ---------------------- Tool Metadata ---------------------- */
  const toolName: string | undefined = (cell as any).tool_name || cell.metadata?.tool_name;
  const initialToolArgs: Record<string, any> = (cell as any).tool_arguments || cell.metadata?.tool_args || {};

  /* ---------------------- State for Tool Arguments --- */
  const [toolArgs, setToolArgs] = useState<Record<string, any>>(initialToolArgs);

  useEffect(() => {
    // initialToolArgs is derived from cell.metadata.tool_args or cell.tool_arguments
    const newInitialArgsString = JSON.stringify(initialToolArgs);
    const currentLocalArgsString = JSON.stringify(toolArgs); // toolArgs is the local state

    if (currentLocalArgsString !== newInitialArgsString) {
      setToolArgs(initialToolArgs);
    }
    // The dependency array still reacts to changes in initialToolArgs stringified form
  }, [cell.id, JSON.stringify(initialToolArgs)]); 

  /* ---------------------- Connection Store -------------------- */
  const toolDefinitions = useConnectionStore((state) => state.toolDefinitions.python);
  const toolLoadingStatus = useConnectionStore((state) => state.toolLoadingStatus.python);
  const toolInfo = toolDefinitions?.find((def) => def.name === toolName);

  /* ---------------------- Helpers ----------------------------- */

  // Debounced onUpdate function
  const debouncedOnUpdate = useCallback(
    debounce((cellId: string, content: string, metadata?: Record<string, any>) => {
      onUpdate(cellId, content, metadata);
    }, 500), // 500ms debounce delay
    [onUpdate] // Dependency: onUpdate prop
  );

  const handleToolArgChange = (argName: string, value: any) => {
    setToolArgs((prev) => {
      const updated = { ...prev, [argName]: value };
      // Call the debounced function for backend update
      debouncedOnUpdate(cell.id, cell.content, { toolName: toolName, toolArgs: updated });
      return updated;
    });
  };

  const renderToolFormInputs = (): React.ReactNode => {
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
      } else if (paramName === 'script') { // Use CodeMirror for the script argument
        inputElement = (
           <div className="mt-1 border rounded-md overflow-hidden">
             <CodeMirror
               value={String(currentValue)}
               height="200px" // Adjust height as needed
               extensions={[python()]} 
               theme={githubLight} // Use the imported theme
               onChange={(value: string, viewUpdate: ViewUpdate) => handleToolArgChange(paramName, value)}
               basicSetup={{
                 foldGutter: false,
                 dropCursor: false,
                 allowMultipleSelections: false,
                 indentOnInput: false,
                 // Add other basic setup options if needed
               }}
               style={{ fontSize: '0.8rem' }} // Match font size if desired
             />
           </div>
        );
      } else if (schemaType === "string" && (schemaFormat === "textarea" || (description && description.length > 50))) {
        inputElement = (
          <Textarea
            id={fieldId}
            value={String(currentValue)}
            onChange={(e) => handleToolArgChange(paramName, e.target.value)}
            placeholder={placeholder}
            required={isRequired}
            className="mt-1 h-32 text-xs font-mono resize-y" // Increased height for script
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

  const isRunScriptTool = toolName === "run_script" || toolName === "run-script";
  const scriptArgumentContent = toolArgs?.script || initialToolArgs?.script;
  const codeForCopyButton = isRunScriptTool && typeof scriptArgumentContent === 'string'
    ? scriptArgumentContent
    : cell.content;

  const renderStructuredOutput = (stdout: string | null | undefined): React.ReactNode => {
    if (!stdout) return null;
    const outputParts: React.ReactNode[] = [];
    const dfRegex = /--- DataFrame ---\n([\s\S]*?)\n--- End DataFrame ---/g;
    const plotRegex = /<PLOT_BASE64>(.*?)<\/PLOT_BASE64>/g;
    const jsonRegex = /<JSON_OUTPUT>(.*?)<\/JSON_OUTPUT>/g;

    const processChunk = (text: string) => {
      if (text.trim()) {
        outputParts.push(
          <div key={`text-md-${outputParts.length}`} className="markdown-rendered-chunk my-1 text-xs">
             <ReactMarkdown remarkPlugins={[remarkGfm]}>{text.trim()}</ReactMarkdown>
          </div>
        );
      }
    };

    let lastIndex = 0;
    const markers = [
      { regex: dfRegex, type: 'dataframe' },
      { regex: plotRegex, type: 'plot' },
      { regex: jsonRegex, type: 'json' },
    ];
    const allMatches: { index: number; length: number; type: string; content: string }[] = [];
    markers.forEach(({ regex, type }) => {
      let match;
      while ((match = regex.exec(stdout)) !== null) {
        allMatches.push({ index: match.index, length: match[0].length, type, content: match[1] });
      }
    });
    allMatches.sort((a, b) => a.index - b.index);
    allMatches.forEach(match => {
      processChunk(stdout.substring(lastIndex, match.index));
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
    processChunk(stdout.substring(lastIndex));
    if (outputParts.length === 0 && stdout.trim()) {
       processChunk(stdout);
    }
    return outputParts.length > 0 ? <>{outputParts}</> : null;
  };

  // Keep Emerald theme for Python Cell
  const headerBgColor = "bg-emerald-100";
  const headerBorderColor = "border-emerald-200";
  const cardBorderColor = "border-emerald-300"; // Slightly darker for card borders maybe
  const resultCardBgColor = "bg-emerald-50";
  const runButtonTextColor = "text-emerald-800 hover:text-emerald-900";
  const runButtonIconColor = "text-emerald-700";

  return (
    <div className="border rounded-md overflow-hidden mb-3 mx-8 bg-white">
      {/* Header */}
      <div className={`${headerBgColor} ${headerBorderColor} border-b p-2 flex justify-between items-center`}>
        <div className="flex items-center">
          <CodeIcon className={`h-4 w-4 mr-2 ${runButtonIconColor}`} />
          <span className={`font-medium text-sm ${runButtonTextColor.split(' ')[0]}`}>Python Cell</span> 
          {toolName && (
            <span className={`ml-2 text-xs text-emerald-600 bg-emerald-200 px-1.5 py-0.5 rounded`}>
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
                  className={`text-gray-600 hover:${headerBgColor} hover:${runButtonIconColor} h-7 w-7 px-0`}
                  onClick={() => copyToClipboard(codeForCopyButton, setCopiedCode)}
                >
                  {copiedCode ? <CheckIcon className="h-4 w-4 text-green-600" /> : <CopyIcon className="h-4 w-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent><p>Copy {isRunScriptTool ? "Script" : "Code/Content"}</p></TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Button
            size="sm"
            variant="outline"
            className={`bg-white hover:bg-emerald-50 ${cardBorderColor} h-7 px-2 text-xs ${runButtonTextColor}`}
            onClick={() => onExecute(cell.id)}
            disabled={isExecuting || !toolName} // Disable if no tool is selected
          >
            <PlayIcon className={`h-3.5 w-3.5 mr-1 ${runButtonIconColor}`} />
            {isExecuting ? "Running..." : (toolName ? "Run Code" : "Select Tool")}
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

      {/* Tool Arguments in Accordion */}
      {toolName && (
        <div className="p-3">
          <Card className={cardBorderColor}>
            <CardContent className="p-0"> {/* Remove padding from CardContent if Accordion handles it */}
              <Accordion 
                type="single" 
                collapsible 
                value={argsAccordionValue} // Controlled value
                onValueChange={setArgsAccordionValue} // Update local state
                className="w-full"
              >
                <AccordionItem value="item-1" className="border-b-0">
                  <AccordionTrigger className="text-xs font-medium py-2 px-3 hover:no-underline">
                    Tool Arguments ({toolName})
                  </AccordionTrigger>
                  <AccordionContent className="pt-2 px-3 pb-3">
                    {renderToolFormInputs()}
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Execution Result Card */}
      {(cell.status !== 'idle' || cell.result || isExecuting || parseError || executionError) && (
        <div className="p-3 pt-0"> {/* Reduce top padding if arguments card exists */}
            <Card className={`border ${cardBorderColor}`}>
                <CardHeader className={`flex flex-row items-center justify-between p-2 ${resultCardBgColor} border-b ${headerBorderColor}`}>
                    <CardTitle className={`text-sm font-semibold ${runButtonTextColor.split(' ')[0]}`}>
                        Execution Result
                    </CardTitle>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setIsResultExpanded(!isResultExpanded)} // Local toggle still works
                        className={`text-xs ${runButtonTextColor}`}
                    >
                        {isResultExpanded ? <><ChevronUpIcon className="h-4 w-4 mr-1" /> Hide</> : <><ChevronDownIcon className="h-4 w-4 mr-1" /> Show</>}
                    </Button>
                </CardHeader>
                {isResultExpanded && ( // Controlled by local state, which is synced with global
                    <CardContent className="p-2">
                        <ScrollArea className="flex-grow text-xs max-h-[400px] overflow-auto pr-2">
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
                            
                            {/* No output message if not executing and no results/errors */}
                            {!isExecuting && !parsedResult && !parseError && !executionError && cell.status !== 'stale' && (
                                <div className="text-xs text-gray-500 italic ml-1 mt-1">No output received yet. Click 'Run Code' to execute.</div>
                            )}
                        </ScrollArea>
                    </CardContent>
                )}
            </Card>
        </div>
      )}
    </div>
  )
}

// Memoize for performance
const PythonCell = React.memo(PythonCellComponent);

export default PythonCell;
