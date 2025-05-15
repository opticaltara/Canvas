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

          let field: React.ReactNode;
          if (type === "string" && (prop.format === "textarea" || (prop.description && prop.description.length > 50))) {
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
        <div className="flex items-center">
          <ActivityIcon className={`h-4 w-4 mr-2 ${headerText} flex-shrink-0`} />
          <span className={`font-medium text-sm ${headerText}`}>{toolName}</span>
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
        {/* Tool selector */}
        <div>
          <Label className="text-xs text-gray-700" htmlFor={`tool-${cell.id}`}>Select Log-AI Tool</Label>
          {toolLoadingStatus === "loading" && <p className="text-xs text-gray-500 mt-1">Loading tools…</p>}
          {toolLoadingStatus === "error" && <p className="text-xs text-red-500 mt-1">Failed to load tools.</p>}
          {toolDefinitions && toolDefinitions.length > 0 && (
            <select
              id={`tool-${cell.id}`}
              className="mt-1 h-8 text-xs border rounded px-2"
              value={toolName}
              onChange={(e) => {
                const newTool = e.target.value;
                setToolName(newTool);
                setToolArgs({});
                onUpdate(cell.id, cell.content ?? "", { toolName: newTool, toolArgs: {} });
              }}
            >
              {toolDefinitions.map((d) => (
                <option key={d.name} value={d.name}>{d.name}</option>
              ))}
            </select>
          )}
        </div>

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