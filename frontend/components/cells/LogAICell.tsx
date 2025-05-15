"use client";

import React, { useState, useEffect } from "react";
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

import { useConnectionStore } from "@/app/store/connectionStore";

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
  const [toolArgs, setToolArgs] = useState<Record<string, any>>(cell.tool_arguments || {});

  // Keep local state in sync with parent updates (e.g. when cell metadata changes)
  useEffect(() => {
    setToolName(cell.tool_name || "");
    setToolArgs(cell.tool_arguments || {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cell.id, cell.tool_name, cell.tool_arguments]);

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
  const handleExecute = () => {
    // Persist description (cell.content) & arguments to metadata then run
    onUpdate(cell.id, cell.content ?? "", { toolName, toolArgs });
    onExecute(cell.id);
  };

  const handleArgChange = (arg: string, value: any) => {
    setToolArgs((prev) => {
      const upd = { ...prev, [arg]: value };
      onUpdate(cell.id, cell.content ?? "", { toolName, toolArgs: upd });
      return upd;
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
  const error = cell.result?.error;
  const resultContent = cell.result?.content ?? cell.result ?? null;

  const renderResultContent = () => {
    if (resultContent === null || resultContent === undefined) return <p className="text-xs text-gray-500">No result.</p>;
    return (
      <pre className="text-xs whitespace-pre-wrap break-words font-mono bg-white p-1.5 border border-gray-200 rounded max-h-[400px] overflow-auto">
        {typeof resultContent === "string" ? resultContent : JSON.stringify(resultContent, null, 2)}
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
                  setToolArgs({});
                  onUpdate(cell.id, cell.content ?? "", { toolName: newTool, toolArgs: {} });
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

        {/* Execution State */}
        {isExecuting && (
          <div className="flex items-center text-xs text-rose-600">
            <div className="animate-spin h-3 w-3 border-2 border-rose-600 rounded-full border-t-transparent mr-1.5"></div>
            Executing…
          </div>
        )}

        {/* Result */}
        {cell.status !== "running" && cell.status !== "queued" && (
          <Card className={`border ${cardBorder}`}>
            <CardHeader className={`flex flex-row items-center justify-between p-2 ${resultBg}`}>
              <CardTitle className={`text-sm font-semibold ${headerText}`}>Execution Result</CardTitle>
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
            </CardHeader>
            {isResultExpanded && <CardContent className="p-2">{error ? <div className="p-2 bg-red-50 text-red-700 border border-red-200 rounded text-xs">{error}</div> : renderResultContent()}</CardContent>}
          </Card>
        )}
      </div>
    </div>
  );
};

export default LogAICell; 