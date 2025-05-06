"use client";

import React from "react";
import { useState, useEffect } from "react";
import type { Cell } from "@/store/types";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { PlayIcon, FolderIcon, Trash2Icon, ChevronDownIcon, ChevronUpIcon, GitCommitIcon, FileDiffIcon, CopyIcon, CheckIcon, Download, FileIcon, MessageSquare, GitPullRequest, CheckCircle, XCircle, Clock, AlertCircleIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import oneLight from 'react-syntax-highlighter/dist/esm/styles/prism'

import { useConnectionStore } from "@/app/store/connectionStore";

interface FileSystemCellProps {
  cell: Cell;
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void;
  onExecute: (id: string) => void;
  onDelete: (id: string) => void;
  isExecuting: boolean;
}

const FileSystemCell: React.FC<FileSystemCellProps> = ({
  cell,
  onUpdate,
  onExecute,
  onDelete,
  isExecuting,
}) => {
  const [toolName, setToolName] = useState<string>(cell.tool_name || "");
  const [toolArgs, setToolArgs] = useState<Record<string, any>>(cell.tool_arguments || {});
  const [isResultExpanded, setIsResultExpanded] = useState(true);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  const toolDefinitions = useConnectionStore((state) => state.toolDefinitions.filesystem);
  const toolLoadingStatus = useConnectionStore((state) => state.toolLoadingStatus.filesystem);
  
  const toolInfo = toolDefinitions?.find(def => def.name === toolName);
  
  useEffect(() => {
    console.log(`[FileSystemCell ${cell.id}] useEffect [id changed]: Syncing state from props.`);
    setToolName(cell.tool_name || "");
    setToolArgs(cell.tool_arguments ? { ...cell.tool_arguments } : {});
  }, [cell.id, cell.tool_name, cell.tool_arguments]);

  const handleToolArgChange = (argName: string, value: any) => {
    setToolArgs(prevArgs => {
       const newArgs = { ...prevArgs, [argName]: value };
       console.log(`[FileSystemCell ${cell.id}] Arg changed: ${argName}, New Value:`, value, "New Args State:", newArgs);
       onUpdate(cell.id, cell.content, { toolName: toolName, toolArgs: newArgs });
       return newArgs;
     });
  };

  const handleExecute = () => {
    console.log(`[FileSystemCell ${cell.id}] Executing with Tool: ${toolName}, Args:`, toolArgs);
    onUpdate(cell.id, cell.content, { toolName: toolName, toolArgs: toolArgs }); 
    onExecute(cell.id);
  };

  const handleDelete = () => {
    onDelete(cell.id);
  };

  const headerBgColor = "bg-blue-100";
  const headerBorderColor = "border-blue-200";
  const cardBorderColor = "border-blue-200";

  const resultCardBorderColor = "border-blue-200";
  const resultCardBgColor = "bg-blue-50";

  const parseCsv = (csvString: string): string[][] | null => {
    if (!csvString || typeof csvString !== 'string') return null;
    if (!csvString.includes('\n') && !csvString.includes('\r') && !csvString.includes(',')) return null;
    
    const lines = csvString.trim().split(/\r?\n/);
    if (lines.length === 0) return null;

    try {
      return lines.map(line => 
        line.split(',').map(cell => cell.trim())
      );
    } catch (e) {
      console.error("CSV parsing failed:", e);
      return null;
    }
  };

  const renderCsvTable = (data: string[][], maxRows?: number): React.ReactNode => {
    if (!data || data.length === 0) return <p className="text-xs text-gray-500">Empty CSV data.</p>;

    const headers = data[0];
    let rows = data.slice(1);
    const totalRows = rows.length;
    let isTruncated = false;

    if (maxRows !== undefined && rows.length > maxRows) {
      rows = rows.slice(0, maxRows);
      isTruncated = true;
    }

    return (
      <>
        <div className="overflow-x-auto border rounded-md max-h-[400px]">
          <Table className="min-w-full text-xs">
            <TableHeader className="bg-gray-50 sticky top-0">
              <TableRow>
                {headers.map((header, index) => (
                  <TableHead key={index} className="px-2 py-1.5 font-medium text-gray-600 whitespace-nowrap">{header}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row, rowIndex) => (
                <TableRow key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <TableCell key={cellIndex} className="px-2 py-1.5 whitespace-nowrap">{cell}</TableCell>
                  ))}
                </TableRow>
              ))}
              {rows.length === 0 && (
                <TableRow>
                   <TableCell colSpan={headers.length} className="text-center text-gray-500 py-4">No data rows found.</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
        {isTruncated && (
          <div className="text-xs text-gray-600 mt-2 flex justify-between items-center">
            <span>Showing first {maxRows} of {totalRows} data rows.</span>
            <Button
              variant="link"
              size="sm"
              className="text-xs h-auto p-0 text-blue-600 hover:text-blue-800"
              onClick={() => {
                if (typeof cell.result?.content === 'string') {
                  if (cell.tool_name === 'read_file' && cell.tool_arguments?.target_file) {
                    const filePath = cell.tool_arguments.target_file;
                    const url = `/view-csv?filePath=${encodeURIComponent(filePath)}`;
                    window.open(url, '_blank');
                  } else {
                    console.warn(`Cannot view full file: Tool is not 'read_file' (actual: ${cell.tool_name}) or 'target_file' argument is missing in cell.tool_arguments (args: ${JSON.stringify(cell.tool_arguments)}). Trying sessionStorage fallback...`);
                    // Fallback to sessionStorage (might still fail on the target page if it strictly requires filePath)
                    try {
                      sessionStorage.setItem('fullCsvData', cell.result.content);
                      window.open('/view-csv', '_blank');
                      alert("Opening file view without a direct path. This might not work as expected.");
                    } catch (e) {
                      console.error("Failed to store CSV data in sessionStorage:", e);
                      alert("Could not open full view: Data might be too large for session storage or file path is unavailable.");
                    }
                  }
                } else {
                   console.warn("Cannot view full file: Result content is not a string.");
                   alert("Could not open full view: The result content is not in a viewable format.");
                }
              }}
              title="View the full CSV file in a new tab"
            >
              View Full File
            </Button>
          </div>
        )}
      </>
    );
  };

  const renderFileSystemResult = (content: any): React.ReactNode => {
    console.log("[FileSystemCell] renderFileSystemResult received content type:", typeof content);

    if (typeof content === 'string') {
      const parsedCsv = parseCsv(content);
      if (parsedCsv) {
        return renderCsvTable(parsedCsv, 10);
      }
    }

    if (Array.isArray(content) && content.every(item => typeof item === 'string')) {
      if (content.length === 0) {
         return <p className="text-xs text-gray-500 p-1.5">Empty list.</p>;
      }
      return (
        <ul className="list-none p-1.5 text-xs font-mono bg-white border border-gray-200 rounded max-h-[400px] overflow-auto">
          {content.map((item, index) => (
            <li key={index} className="whitespace-pre-wrap break-words">{item}</li>
          ))}
        </ul>
      );
    }

    return (
      <pre className="text-xs whitespace-pre-wrap break-words font-mono bg-white p-1.5 border border-gray-200 rounded max-h-[400px] overflow-auto">
        {content && typeof content === 'object' ? JSON.stringify(content, null, 2) : String(content ?? '')}
      </pre>
    );
  };

  const renderToolFormInputs = () => {
    console.log(`[FileSystemCell ${cell.id}] Rendering inputs for: ${toolName}`);
    console.log(`[FileSystemCell ${cell.id}] Filesystem Tool loading status:`, toolLoadingStatus);
    console.log(`[FileSystemCell ${cell.id}] Available Filesystem tool definitions:`, toolDefinitions);

    if (toolLoadingStatus === undefined || toolLoadingStatus === 'idle') {
      return <div className="text-xs text-gray-500">Initializing tool definitions...</div>;
    }
    if (toolLoadingStatus === 'loading') {
      return <div className="text-xs text-gray-500">Loading tool parameters...</div>;
    }
    if (toolLoadingStatus === 'error') {
      return <div className="text-xs text-red-500">Error loading tool definitions.</div>;
    }
    if (!toolInfo) {
      if (toolName && toolLoadingStatus === 'success') {
         return <div className="text-xs text-red-500">Tool definition not found for: {toolName}</div>;
      } 
      return <div className="text-xs text-gray-500">Select a filesystem tool.</div>;
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
        const currentValue = toolArgs[paramName] ?? paramSchema.default ?? ''; 
        const placeholder = paramSchema.examples?.[0] || paramSchema.description || `Enter ${label}`;

        let inputElement: React.ReactNode = null;

        const schemaType = paramSchema.type;
        const schemaFormat = paramSchema.format;
        const enumValues = paramSchema.enum as string[] | undefined;

        if (enumValues && Array.isArray(enumValues)) {
             inputElement = (
                <Select
                  value={String(currentValue)}
                  onValueChange={(value) => handleToolArgChange(paramName, value)}
                  required={isRequired}
                >
                  <SelectTrigger id={fieldId} className="mt-1 h-8 text-xs">
                    <SelectValue placeholder={placeholder} />
                  </SelectTrigger>
                  <SelectContent>
                    {enumValues.map((option) => (
                      <SelectItem key={option} value={option}>{option}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
             );
        } else if (schemaType === 'string' && (schemaFormat === 'textarea' || (description && description.length > 50))) {
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
        } else if (schemaType === 'number' || schemaType === 'integer') {
            inputElement = (
                <Input
                    id={fieldId}
                    type="number"
                    value={String(currentValue)}
                    onChange={(e) => handleToolArgChange(paramName, e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder={placeholder}
                    required={isRequired}
                    min={paramSchema.minimum}
                    max={paramSchema.maximum}
                    step={paramSchema.multipleOf || 'any'}
                    className="mt-1 h-8 text-xs"
                 />
            );
        } else if (schemaType === 'boolean') {
             inputElement = (
                 <div className="flex items-center space-x-2 mt-1 h-8"> 
                   <input
                     type="checkbox"
                     id={fieldId}
                     checked={!!currentValue}
                     onChange={(e) => handleToolArgChange(paramName, e.target.checked)}
                     className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
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
        {Object.entries(properties).map(([paramName, paramSchema]) => 
          renderField(paramName, paramSchema)
        )}
      </div>
    );
  }

  return (
    <div className="border rounded-md overflow-hidden mb-3 mx-8">
      <div className={`p-2 flex justify-between items-center border-b ${headerBgColor} ${headerBorderColor}`}>
        <div className="flex items-center">
          <FolderIcon className="h-4 w-4 mr-2 text-blue-700 flex-shrink-0" />
          <span className="font-medium text-sm text-blue-800">
             {toolName || "Filesystem Command"}
          </span>
        </div>
        <div className="flex items-center space-x-1.5">
          <Button
            size="sm"
            variant="outline"
            className="bg-white hover:bg-blue-50 border-blue-300 h-7 px-2 text-xs"
            onClick={handleExecute}
            disabled={isExecuting || !toolName}
            title="Execute Cell"
          >
            <PlayIcon className="h-3.5 w-3.5 mr-1 text-blue-700" />
            {isExecuting ? "Running..." : "Run Command"}
          </Button>
          <TooltipProvider delayDuration={100}>
             <Tooltip>
               <TooltipTrigger asChild>
                 <Button
                   size="sm"
                   variant="ghost"
                   className="text-gray-500 hover:bg-red-100 hover:text-red-700 h-7 w-7 px-0"
                   onClick={handleDelete}
                   disabled={isExecuting}
                 >
                   <Trash2Icon className="h-4 w-4" />
                 </Button>
               </TooltipTrigger>
               <TooltipContent>
                 <p>Delete Cell</p>
               </TooltipContent>
             </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      <div className="p-3">
        <div className="mb-4">
           <Card className={cardBorderColor}>
            <CardContent className="p-3">
               <Accordion type="single" collapsible defaultValue="item-1" className="w-full">
                 <AccordionItem value="item-1" className="border-b-0">
                   <AccordionTrigger className="text-xs font-medium py-2 hover:no-underline">
                     Command Arguments ({toolName || "No tool selected"})
                   </AccordionTrigger>
                   <AccordionContent className="pt-2">
                      {toolName ? renderToolFormInputs() : <p className="text-xs text-gray-500">No tool selected or definition available.</p>}
                   </AccordionContent>
                 </AccordionItem>
               </Accordion>
            </CardContent>
           </Card>
        </div>

         {cell.status === "running" && (
           <div className="flex items-center text-xs text-amber-600 mb-3">
             <div className="animate-spin h-3 w-3 border-2 border-amber-600 rounded-full border-t-transparent mr-1.5"></div>
             Executing command...
           </div>
         )}
         {cell.status === "queued" && (
           <div className="flex items-center text-xs text-blue-600 mb-3">
              <div className="h-3 w-3 bg-blue-600 rounded-full mr-1.5 animate-pulse"></div>
              Queued for execution...
           </div>
         )}
         {cell.status === "stale" && (
           <div className="flex items-center text-xs text-gray-600 mb-3">
              <div className="h-3 w-3 border border-gray-600 rounded-full mr-1.5"></div>
              Stale (needs re-run)...
           </div>
         )}

         {cell.status === "error" && cell.result?.error && (
             <div className="bg-red-50 border border-red-200 rounded p-2 mb-3 text-red-700">
                 <div className="font-medium text-xs">Execution Error:</div>
                 <pre className="whitespace-pre-wrap text-xs mt-0.5 font-mono">{cell.result.error}</pre>
             </div>
         )}

        {cell.status !== 'queued' && cell.status !== 'running' && cell.result && (
          <Card className={`border ${resultCardBorderColor}`}>
            <CardHeader className={`flex flex-row items-center justify-between p-2 ${resultCardBgColor}`}>
              <CardTitle className="text-sm font-semibold text-blue-800">
                 Execution Result
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsResultExpanded(!isResultExpanded)}
                className="text-xs"
              >
                {isResultExpanded ? <><ChevronUpIcon className="h-4 w-4 mr-1" /> Hide</> : <><ChevronDownIcon className="h-4 w-4 mr-1" /> Show</>}
              </Button>
            </CardHeader>
             {isResultExpanded && (
                <CardContent className="p-2 border-t">
                   <div className="mt-1 rounded overflow-x-auto max-h-[450px]">
                     {renderFileSystemResult(cell.result.content)}
                     {cell.result.execution_time !== undefined && (
                       <p className="text-xs text-gray-500 mt-2">Execution Time: {cell.result.execution_time.toFixed(2)}s</p>
                     )}
                   </div>
                </CardContent>
             )}
           </Card>
        )}
        {(cell.status === 'queued' || cell.status === 'running') && !cell.result && (
           <div className="p-2 text-xs text-gray-500 flex items-center">
             <div className="animate-spin h-3 w-3 border-2 border-gray-400 rounded-full border-t-transparent mr-1.5"></div>
             Processing...
           </div>
        )}
      </div>
    </div>
  );
};

const areEqual = (prevProps: FileSystemCellProps, nextProps: FileSystemCellProps) => {
  const prevCell = prevProps.cell;
  const nextCell = nextProps.cell;

  if (prevCell === nextCell && prevProps.isExecuting === nextProps.isExecuting) {
    return true;
  }

  if (prevProps.isExecuting !== nextProps.isExecuting) {
    return false;
  }

  if (prevCell.id !== nextCell.id || 
      prevCell.status !== nextCell.status ||
      prevCell.tool_name !== nextCell.tool_name ||
      prevCell.result !== nextCell.result || 
      JSON.stringify(prevCell.tool_arguments) !== JSON.stringify(nextCell.tool_arguments)) 
  {
    return false;
  }

  return true;
};

const MemoizedFileSystemCell = React.memo(FileSystemCell, areEqual);

export default MemoizedFileSystemCell; 