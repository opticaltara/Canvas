"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Cell } from "@/store/types";
import {
  PlayIcon,
  Trash2Icon,
  ActivityIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import ResultTable from "@/components/ResultTable"; // Adjusted path

import { useConnectionStore } from "@/app/store/connectionStore";

// Helper for debouncing (can be moved to a shared utils file later)
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

// Define a stable empty object reference for default tool arguments
const DEFAULT_EMPTY_TOOL_ARGS = {};

interface LogAICellProps {
  cell: Cell;
  onExecute: (cellId: string) => void;
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void;
  onDelete: (cellId: string) => void;
  isExecuting: boolean;
}

const headerBg = "bg-rose-100"; // maroon-ish accent (tailwind rose)
const headerBorder = "border-rose-200";
const headerText = "text-rose-800";
const cardBorder = "border-rose-200";
const resultBg = "bg-rose-50";

const LogAICell: React.FC<LogAICellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }) => {
  const [isResultExpanded, setIsResultExpanded] = useState(false);
  const [argsAccordion, setArgsAccordion] = useState<string>("");

  // Global expand toggle
  const areAllExpanded = useConnectionStore((s) => s.areAllCellsExpanded);
  useEffect(() => {
    setIsResultExpanded(areAllExpanded);
    setArgsAccordion(areAllExpanded ? "item-1" : "");
  }, [areAllExpanded]);

  // --- Tool meta (Log-AI uses single tool most of the time) ---
  const [toolName, setToolName] = useState<string>(cell.tool_name || "");
  const [toolArgs, setToolArgs] = useState<Record<string, any>>(cell.tool_arguments || DEFAULT_EMPTY_TOOL_ARGS);

  // Keep local state in sync with parent updates (e.g. when cell metadata changes)
  useEffect(() => {
    const currentCellToolName = cell.tool_name || "";
    if (toolName !== currentCellToolName) {
      setToolName(currentCellToolName);
    }

    const newCellToolArguments = cell.tool_arguments || DEFAULT_EMPTY_TOOL_ARGS;
    const newCellToolArgumentsString = JSON.stringify(newCellToolArguments);
    const currentLocalToolArgsString = JSON.stringify(toolArgs);

    if (currentLocalToolArgsString !== newCellToolArgumentsString) {
      // Only update local state if the arguments have actually changed.
      // This prevents re-triggering if the parent component re-renders with
      // a new object reference for tool_arguments but the content is the same.
      setToolArgs(newCellToolArguments);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cell.id, cell.tool_name, cell.tool_arguments]); // Depend on the object itself, comparison done inside.

  // Attempt to fetch tool definitions (may be undefined if none cached)
  const toolDefinitions = useConnectionStore((s) => s.toolDefinitions.log_ai);
  const toolLoadingStatus = useConnectionStore((s) => s.toolLoadingStatus.log_ai);
  const toolInfo = toolDefinitions?.find((d) => d.name === toolName);

  /* Fetch tool definitions on mount if not loaded */
  const fetchTools = useConnectionStore((s) => s.fetchToolsForConnection);
  useEffect(() => {
    if (!toolLoadingStatus || toolLoadingStatus === "idle") {
      fetchTools("log_ai");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ---------------- Handlers ---------------- */

  // Debounced onUpdate function
  const debouncedOnUpdate = useCallback(
    debounce((cellId: string, content: string, metadata?: Record<string, any>) => {
      onUpdate(cellId, content, metadata);
    }, 500), // 500ms debounce delay
    [onUpdate] // Dependency: onUpdate prop
  );

  const handleExecute = () => {
    // Persist description (cell.content) & arguments to metadata then run
    // For execute, we want to send the latest state immediately, so no debounce here.
    // Ensure we are sending the most up-to-date local toolName and toolArgs
    const currentMetadata = {
      ...cell.metadata, // Preserve existing metadata
      toolName: toolName, // Use local state
      toolArgs: toolArgs,   // Use local state
    };
    onUpdate(cell.id, cell.content ?? "", currentMetadata);
    onExecute(cell.id);
  };

  const handleArgChange = (arg: string, value: any) => {
    setToolArgs((prev) => {
      const newArgs = { ...prev, [arg]: value };
      // Use debounced update for arg changes
      // Ensure we are sending the most up-to-date local toolName with the newArgs
      const currentMetadata = {
        ...cell.metadata, // Preserve existing metadata
        toolName: toolName, // Use local state for toolName
        toolArgs: newArgs,  // Use the newly updated args
      };
      debouncedOnUpdate(cell.id, cell.content ?? "", currentMetadata);
      return newArgs;
    });
  };

  /* --------------- Render Tool Inputs --------------- */
  const renderToolInputs = (): React.ReactNode => {
    if (!toolInfo) {
      if (toolLoadingStatus === "loading") return <p className="text-xs text-gray-500">Loading tool schema…</p>;
      if (toolLoadingStatus === "error") return <p className="text-xs text-red-500">Failed loading tool schema.</p>;
      return <p className="text-xs text-gray-500">No schema for tool <code>{toolName}</code>.</p>;
    }

    const schema = toolInfo.inputSchema;
    const properties: Record<string, any> = schema?.properties || {};
    const required = new Set(schema?.required || []);

    if (Object.keys(properties).length === 0) {
      return <p className="text-xs text-gray-600">This tool takes no arguments.</p>;
    }

    return (
      <div className="space-y-2">
        {Object.entries(properties).map(([name, prop]) => {
          const id = `${name}-${cell.id}`;
          const label = prop.title || name;
          const isReq = required.has(name);
          const placeholder = prop.examples?.[0] || prop.description || label;
          const val = toolArgs[name] ?? "";

          const type = prop.type;

          const isFileInput =
            prop.format === "file" ||
            prop.format === "filepath" ||
            (/file|path/i.test(name) && type === "string");

          const isNumberInput =
            type === "number" ||
            type === "integer" ||
            (prop.format === "int32" || prop.format === "int64");

          let field: React.ReactNode;
          if (isFileInput) {
            const fileInputId = `${id}-picker`;
            field = (
              <div className="flex items-center space-x-2 mt-1">
                {/* Hidden native file input */}
                <input
                  id={fileInputId}
                  type="file"
                  className="hidden"
                  accept=".csv,.txt,.log,text/plain,application/vnd.ms-excel"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      handleArgChange(name, (file as any).path || file.name);
                    }
                  }}
                  required={isReq}
                />

                {/* Styled trigger button */}
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-8 px-3 text-xs"
                  onClick={() => {
                    const el = document.getElementById(fileInputId) as HTMLInputElement | null;
                    el?.click();
                  }}
                >
                  Browse…
                </Button>

                {/* Chosen file path / name */}
                {val && (
                  <span className="text-[11px] font-mono text-gray-700 truncate max-w-xs" title={val}>
                    {val}
                  </span>
                )}
              </div>
            );
          } else if (type === "string" && (prop.format === "textarea" || (prop.description && prop.description.length > 50))) {
            field = (
              <Textarea
                id={id}
                className="mt-1 h-16 text-xs font-mono resize-none"
                value={val}
                placeholder={placeholder}
                onChange={(e) => handleArgChange(name, e.target.value)}
                required={isReq}
              />
            );
          } else if (isNumberInput) {
            field = (
              <Input
                id={id}
                type="number"
                className="mt-1 h-8 text-xs font-mono"
                value={val}
                placeholder={placeholder}
                onChange={(e) => handleArgChange(name, e.target.valueAsNumber ?? e.target.value)}
                required={isReq}
              />
            );
          } else {
            field = (
              <Input
                id={id}
                className="mt-1 h-8 text-xs font-mono"
                value={val}
                placeholder={placeholder}
                onChange={(e) => handleArgChange(name, e.target.value)}
                required={isReq}
              />
            );
          }

          return (
            <div key={id} className="text-xs">
              <Label htmlFor={id} className="text-gray-700">
                {label} {isReq && <span className="text-red-500">*</span>}
              </Label>
              {field}
              {prop.description && <p className="text-gray-500 mt-1">{prop.description}</p>}
            </div>
          );
        })}
      </div>
    );
  };

  /* --------------- Result Render --------------- */
  const error = cell.result?.error; // Used for displaying general errors

  const parseGenericTableString = (tableStr: string): Record<string, any>[] | null => {
    try {
      const lines = tableStr.trim().split('\n');
      if (lines.length < 1) return null; // Must have at least a header line

      // Filter out summary lines like "[6 rows x 4 columns]"
      const contentLines = lines.filter(line => !line.trim().startsWith('[') && line.trim().endsWith(']') === false && line.trim() !== '');
      if (contentLines.length < 1) return null;


      const headerLine = contentLines[0];
      // Heuristic for header splitting: split by 2 or more spaces.
      // Headers might have leading/trailing spaces from the original format.
      const headers = headerLine.split(/\s{2,}/).map(h => h.trim()).filter(h => h.length > 0);
      if (headers.length === 0) return null;

      const dataLines = contentLines.slice(1);
      const rows: Record<string, any>[] = [];

      dataLines.forEach(line => {
        // Split row by multiple spaces.
        const values = line.split(/\s{2,}/).map(v => v.trim());
        
        // Remove the initial index value if present (e.g., "0", "1 ", etc.)
        // and if the number of values after removing index matches header count.
        let finalValues = values;
        if (values.length > 0 && /^\d+$/.test(values[0].trim()) && (values.length -1 === headers.length) ) {
          finalValues = values.slice(1);
        } else if (values.length !== headers.length) {
          // If lengths don't match, this row might be malformed or not part of the table.
          // For now, we'll skip it or one could try more complex parsing.
          console.warn("Skipping row due to mismatch with header count:", line, "Headers:", headers, "Values:", values);
          return; // Skip this line
        }
        
        if (finalValues.length === headers.length) {
          const rowObject: Record<string, any> = {};
          headers.forEach((header, index) => {
            rowObject[header] = finalValues[index] !== undefined ? finalValues[index] : "";
          });
          rows.push(rowObject);
        }
      });

      if (rows.length === 0 && dataLines.length > 0) { // Parsed headers but no rows matched
        return null;
      }
      return rows.length > 0 ? rows : null; // Return null if no data rows were successfully parsed

    } catch (e) {
      console.error("Error parsing generic table string:", e);
      return null;
    }
  };

  const renderResultContent = () => {
    const resultFromCell = cell.result; // This is the object like: { content: { meta: null, content: [...], isError: false }, error: null, ... }

    if (!resultFromCell) {
      return <p className="text-xs text-gray-500">No result data.</p>;
    }

    // The actual data payload is nested under resultFromCell.content
    // resultFromCell.content = { meta: null, content: [...], isError: false }
    const dataPayloadContainer = resultFromCell.content;

    // Specific handling for get_container_logs
    if (
      toolName === "get_container_logs" &&
      dataPayloadContainer &&
      typeof dataPayloadContainer === 'object' &&
      !Array.isArray(dataPayloadContainer) &&
      dataPayloadContainer.content &&
      Array.isArray(dataPayloadContainer.content) &&
      dataPayloadContainer.content.length > 0 &&
      dataPayloadContainer.content[0] &&
      dataPayloadContainer.content[0].type === "text" &&
      typeof dataPayloadContainer.content[0].text === "string"
    ) {
      let logString = dataPayloadContainer.content[0].text;
      try {
        logString = JSON.parse(logString); // Unescape if it's a JSON-encoded string
      } catch (e) {
        // If not a JSON-encoded string, use as is
        console.warn("Log text is not a JSON-encoded string, using as is for get_container_logs:", e);
      }
      return (
        <pre className="text-xs whitespace-pre-wrap break-words font-mono bg-white p-1.5 border border-gray-200 rounded max-h-[400px] overflow-auto">
          {logString}
        </pre>
      );
    }
    // Generic table parsing logic (for list_containers and other potential table outputs)
    else if (
      dataPayloadContainer &&
      typeof dataPayloadContainer === 'object' &&
      !Array.isArray(dataPayloadContainer) && // Ensure it's the container object
      dataPayloadContainer.content &&         // Check for the inner 'content' array (dataPayloadContainer.content)
      Array.isArray(dataPayloadContainer.content) &&
      dataPayloadContainer.content.length > 0 &&
      dataPayloadContainer.content[0] &&      // Check if the first element exists
      dataPayloadContainer.content[0].type === "text" &&
      typeof dataPayloadContainer.content[0].text === "string"
    ) {
      let tableString = dataPayloadContainer.content[0].text;
      // The string itself might be quoted and contain escaped newlines.
      // Attempt to parse it as JSON to unescape.
      try {
        tableString = JSON.parse(tableString);
      } catch (e) {
        // If JSON.parse fails, it means the string was not double-quoted JSON string.
        // Proceed with tableString as is for parsing.
        console.warn("Result text is not a JSON-encoded string, attempting to parse as raw table string:", e);
      }

      const parsedTableData = parseGenericTableString(tableString);
      if (parsedTableData && parsedTableData.length > 0) {
        return (
           <div className="overflow-x-auto bg-white p-1.5 border border-gray-200 rounded max-h-[400px]">
            <ResultTable data={parsedTableData} />
           </div>
        );
      } else {
        // If parsing failed or returned no data, show the (potentially unescaped) string
         return (
          <div className="p-2 bg-yellow-50 text-yellow-700 border border-yellow-200 rounded text-xs">
            <p>Could not parse table data from the result.</p>
            <p className="mt-1">Processed text content:</p>
            <pre className="text-xs whitespace-pre-wrap break-words font-mono bg-white p-1.5 border border-gray-200 rounded max-h-[400px] overflow-auto mt-1">
              {tableString}
            </pre>
          </div>
        );
      }
    }
    
    // If the specific table structure is not matched, fall back to displaying the raw result.content or cell.result
    // This will show the JSON structure if the table parsing path wasn't hit.
    return (
      <pre className="text-xs whitespace-pre-wrap break-words font-mono bg-white p-1.5 border border-gray-200 rounded max-h-[400px] overflow-auto">
        {JSON.stringify(dataPayloadContainer ?? resultFromCell, null, 2)}
      </pre>
    );
  };

  return (
    <div className="border rounded-md overflow-hidden mb-3 mx-8">
      {/* Header */}
      <div className={`p-2 flex justify-between items-center border-b ${headerBg} ${headerBorder}`}>
        <div className="flex items-center space-x-2">
          <ActivityIcon className={`h-4 w-4 ${headerText} flex-shrink-0`} />
          {toolLoadingStatus === "loading" ? (
            <span className={`text-xs font-medium ${headerText}`}>Loading tools…</span>
          ) : toolLoadingStatus === "error" ? (
            <span className={`text-xs font-medium text-red-500`}>Failed to load.</span>
          ) : toolDefinitions && toolDefinitions.length > 0 ? (
            <div className="relative flex items-center">
              <select
                id={`tool-${cell.id}`}
                className="h-7 text-xs font-medium bg-transparent text-rose-800 pl-2 pr-7 appearance-none focus:outline-none focus:ring-0 cursor-pointer rounded-md hover:bg-rose-200/50 transition-colors"
                value={toolName}
                onChange={(e) => {
                  const newTool = e.target.value;
                  setToolName(newTool);
                  const newArgs = {}; // Reset args for new tool
                  setToolArgs(newArgs);
                  // Persist the new tool name and reset arguments
                  const currentMetadata = {
                    ...cell.metadata, // Preserve existing metadata
                    toolName: newTool,
                    toolArgs: newArgs,
                  };
                  onUpdate(cell.id, cell.content ?? "", currentMetadata);
                }}
              >
                {toolDefinitions.map((d) => (
                  <option key={d.name} value={d.name} className="text-sm text-black">
                    {d.name}
                  </option>
                ))}
              </select>
              <ChevronDownIcon className={`h-4 w-4 ${headerText} absolute right-1.5 top-1/2 -translate-y-1/2 pointer-events-none`} />
            </div>
          ) : (
            <span className={`text-xs font-medium ${headerText}`}>{toolName || "Select Tool"}</span>
          )}
        </div>
        <div className="flex items-center space-x-1.5">
          <Button
            size="sm"
            variant="outline"
            className="bg-white hover:bg-rose-50 border-rose-300 h-7 px-2 text-xs"
            onClick={handleExecute}
            disabled={isExecuting}
          >
            <PlayIcon className="h-3.5 w-3.5 mr-1 text-rose-700" />
            {isExecuting ? "Running…" : "Run"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="text-gray-500 hover:bg-red-100 hover:text-red-700 h-7 w-7 px-0"
            onClick={() => onDelete(cell.id)}
            disabled={isExecuting}
          >
            <Trash2Icon className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Body */}
      <div className="p-3 space-y-4">
        {/* Tool Args */}
        <Card className={cardBorder}>
          <CardHeader className={`p-2 ${resultBg}`}>
            <CardTitle className={`text-sm font-semibold ${headerText}`}>Tool Arguments</CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            <Accordion type="single" collapsible value={argsAccordion} onValueChange={setArgsAccordion}>
              <AccordionItem value="item-1" className="border-b-0">
                <AccordionTrigger className="text-xs font-medium py-1 hover:no-underline">
                  Configure
                </AccordionTrigger>
                <AccordionContent className="pt-2">
                  {renderToolInputs()}
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </CardContent>
        </Card>

        {/* Execution State and Result Combined */}
        {/* Show card if executing OR if there is any result/error */}
        {(isExecuting || (cell.result !== undefined && cell.result !== null)) && (
          <Card className={`border ${cardBorder}`}>
            <CardHeader className={`flex flex-row items-center justify-between p-2 ${resultBg}`}>
              <CardTitle className={`text-sm font-semibold ${headerText}`}>Execution Result</CardTitle>
              {/* Show toggle only if not executing and there's a result */}
              {!isExecuting && cell.result !== undefined && cell.result !== null && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsResultExpanded(!isResultExpanded)}
                  className="text-xs"
                >
                  {isResultExpanded ? (
                    <>
                      <ChevronUpIcon className="h-4 w-4 mr-1" /> Hide
                    </>
                  ) : (
                    <>
                      <ChevronDownIcon className="h-4 w-4 mr-1" /> Show
                    </>
                  )}
                </Button>
              )}
            </CardHeader>
            {/* Render CardContent if executing (to show progress) or if result is expanded */}
            {(isExecuting || isResultExpanded) && (
              <CardContent className="p-2">
                {isExecuting ? (
                  <div className="flex items-center text-xs text-rose-600 py-2">
                    <div className="animate-spin h-3 w-3 border-2 border-rose-600 rounded-full border-t-transparent mr-1.5"></div>
                    Executing…
                  </div>
                ) : (
                  // This part is only reached if !isExecuting AND isResultExpanded
                  error ? <div className="p-2 bg-red-50 text-red-700 border border-red-200 rounded text-xs">{error}</div> : renderResultContent()
                )}
              </CardContent>
            )}
          </Card>
        )}
      </div>
    </div>
  );
};

export default LogAICell;
