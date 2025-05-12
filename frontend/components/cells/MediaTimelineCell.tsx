"use client"

import React from "react";
import type { Cell } from "../../store/types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// shadcn/ui components – keep design consistent with other cells
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionTrigger, AccordionItem, AccordionContent } from "@/components/ui/accordion";
import { Tooltip, TooltipProvider, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";

// Icons
import { ImageIcon, FileCodeIcon, Trash2Icon, ListChecksIcon, LightbulbIcon, UserCheckIcon, AlertTriangleIcon, SearchIcon, InfoIcon, FileTextIcon, EyeIcon, CalendarDaysIcon, EditIcon, XCircleIcon, CheckCircleIcon, SparklesIcon } from "lucide-react";

export interface MediaTimelineCellProps {
  cell: Cell;
  onDelete: (cellId: string) => void;
}

// Types mirroring backend structures (loosely typed for safety)
interface MediaTimelineEvent {
  image_identifier?: string;
  description?: string;
  code_references?: string[];
}

interface CorrelatorHypothesis {
  files_likely_containing_bug?: string[];
  reasoning?: string[];
  specific_code_snippets_to_check?: string[];
}

interface MediaTimelineResult {
  hypothesis?: CorrelatorHypothesis | string;
  timeline_events?: MediaTimelineEvent[];
  original_query?: string;
  error?: string;
}

const MediaTimelineCell: React.FC<MediaTimelineCellProps> = ({ cell, onDelete }) => {
  // Initial render log – mirrors other cells for consistency
  console.log(`[MediaTimelineCell ${cell.id}] Render with cell prop:`, {
    id: cell.id,
    type: cell.type,
    status: cell.status,
    hasResult: !!cell.result,
    resultContentType: typeof (cell.result as any)?.content,
    error: cell.error,
  });

  // Try to get structured data from `cell.result?.content` (may be object or JSON string)
  const structuredData: MediaTimelineResult | null = React.useMemo(() => {
    const raw = (cell.result as any)?.content;
    if (!raw) {
      console.log(`[MediaTimelineCell ${cell.id}] No result.content detected – falling back to markdown.`);
      return null;
    }

    if (typeof raw === "string") {
      try {
        const parsed = JSON.parse(raw);
        console.log(`[MediaTimelineCell ${cell.id}] Successfully parsed structured JSON string.`, parsed);
        return parsed;
      } catch (err) {
        console.warn(`[MediaTimelineCell ${cell.id}] Failed to parse result.content JSON string.`, err);
        return null;
      }
    }

    console.log(`[MediaTimelineCell ${cell.id}] Using structured data object as-is from result.content.`);
    return raw as MediaTimelineResult;
  }, [cell.result, cell.id]);

  // Log if structuredData updates (length, etc.)
  React.useEffect(() => {
    if (structuredData) {
      console.log(`[MediaTimelineCell ${cell.id}] structuredData summary ->`, {
        hasHypothesis: !!structuredData.hypothesis,
        timelineEventCount: structuredData.timeline_events?.length ?? 0,
      });
    }
  }, [structuredData, cell.id]);

  // Helper renderers
  const renderHypothesis = (hyp: MediaTimelineResult["hypothesis"]) => {
    if (!hyp) return null;
    if (typeof hyp === "string") {
      return (
        <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">{hyp}</p>
      );
    }
    const { files_likely_containing_bug, reasoning, specific_code_snippets_to_check } = hyp;
    return (
      <div className="space-y-3 text-sm">
        {files_likely_containing_bug && files_likely_containing_bug.length > 0 && (
          <div>
            <p className="font-medium mb-1 flex items-center"><FileTextIcon className="h-4 w-4 mr-1.5 text-sky-600"/>Files Likely Containing the Bug:</p>
            <ul className="list-disc list-inside ml-4 space-y-0.5">
              {files_likely_containing_bug.map((file, idx) => (
                <li key={idx} className="font-mono text-gray-800 text-xs bg-sky-50/50 px-1.5 py-0.5 rounded-sm border border-sky-200 inline-block mr-1 mb-1">{file}</li>
              ))}
            </ul>
          </div>
        )}
        {reasoning && reasoning.length > 0 && (
          <div>
            <p className="font-medium mb-1 flex items-center"><LightbulbIcon className="h-4 w-4 mr-1.5 text-amber-600"/>Reasoning:</p>
            <ul className="space-y-1">
              {reasoning.map((reason, idx) => (
                <li key={idx} className="flex items-start">
                  <CheckCircleIcon className="h-3.5 w-3.5 mr-2 mt-0.5 text-green-600 flex-shrink-0"/> 
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {specific_code_snippets_to_check && specific_code_snippets_to_check.length > 0 && (
          <div>
            <p className="font-medium mb-1 flex items-center"><SearchIcon className="h-4 w-4 mr-1.5 text-purple-600"/>Specific Code Snippets to Check:</p>
            <ul className="list-disc list-inside ml-4 space-y-0.5">
              {specific_code_snippets_to_check.map((snippet, idx) => (
                <li key={idx} className="font-mono text-xs italic text-purple-800">{snippet}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  const renderTimelineEvents = (events?: MediaTimelineEvent[]) => {
    if (!events || events.length === 0) return <p className="text-sm italic text-gray-500">No timeline events provided.</p>;
    
    // Emojis for event types (heuristic based on description)
    const getEventIcon = (description: string = "") => {
      const lowerDesc = description.toLowerCase();
      if (lowerDesc.includes("tap") || lowerDesc.includes("click")) return <EditIcon className="h-4 w-4 text-blue-500"/>;
      if (lowerDesc.includes("type") || lowerDesc.includes("enter") || lowerDesc.includes("delete")) return <EditIcon className="h-4 w-4 text-orange-500"/>;
      if (lowerDesc.includes("navigate") || lowerDesc.includes("show") || lowerDesc.includes("display")) return <EyeIcon className="h-4 w-4 text-green-500"/>;
      if (lowerDesc.includes("error") || lowerDesc.includes("fail") || lowerDesc.includes("unable")) return <XCircleIcon className="h-4 w-4 text-red-500"/>;
      if (lowerDesc.includes("video starts") || lowerDesc.includes("video ends")) return <SparklesIcon className="h-4 w-4 text-purple-500"/>;
      return <InfoIcon className="h-4 w-4 text-gray-500"/>;
    };

    return (
      <div className="relative pl-6 space-y-6 border-l-2 border-gray-200 dark:border-gray-700">
        {events.map((evt, idx) => (
          <div key={idx} className="relative">
            <div className="absolute -left-[1.6rem] top-1.5 flex items-center justify-center w-8 h-8 bg-white dark:bg-gray-800 rounded-full ring-4 ring-indigo-100 dark:ring-indigo-900">
              {getEventIcon(evt.description)}
            </div>
            <div className="ml-4 p-3 bg-gray-50/80 dark:bg-gray-800/50 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700/80">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1.5 flex items-center">
                <CalendarDaysIcon className="h-3.5 w-3.5 mr-1.5 text-indigo-500"/> 
                Event {idx + 1}
              </div>
              {/* Use ReactMarkdown for description to allow rich text */}
              <div className="prose prose-sm prose-slate dark:prose-invert max-w-none">
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]} 
                  components={{
                    p: ({node, ...props}) => <p className="text-sm text-gray-700 dark:text-gray-300" {...props} />,
                    strong: ({node, ...props}) => <strong className="font-semibold text-indigo-600 dark:text-indigo-400" {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc list-inside space-y-0.5 mt-1" {...props} />,
                    li: ({node, ...props}) => <li className="text-gray-600 dark:text-gray-400" {...props} />
                  }}
                >
                  {evt.description || "No description"}
                </ReactMarkdown>
              </div>

              {evt.image_identifier && (
                <div className="mt-2">
                  <a href={evt.image_identifier} target="_blank" rel="noopener noreferrer" className="text-xs text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 dark:hover:text-indigo-300 flex items-center group">
                    <ImageIcon className="h-3.5 w-3.5 mr-1 text-indigo-500 group-hover:text-indigo-700" />
                    View Attached Image
                    <img src={evt.image_identifier} alt={`Event ${idx+1} image`} className="hidden" /> {/* For potential future thumbnail preview */}
                  </a>
                </div>
              )}
              {evt.code_references && evt.code_references.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Code References:</p>
                  <div className="flex flex-wrap gap-1.5">
                    {evt.code_references.map((ref, ri) => (
                      <Badge key={ri} variant="outline" className="text-[0.7rem] font-mono px-2 py-0.5 flex items-center border-indigo-300 bg-indigo-50 text-indigo-700 dark:border-indigo-600 dark:bg-indigo-900/50 dark:text-indigo-300">
                        <FileCodeIcon className="h-3 w-3 mr-1" />{ref}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Fallback: render markdown content
  const renderMarkdownFallback = () => (
    <div className="prose prose-sm max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{cell.content}</ReactMarkdown>
    </div>
  );

  return (
    <Card className="mb-4 mx-8 shadow-sm border border-gray-200/80 dark:border-gray-700/60">
      <CardHeader className="py-2.5 px-4 bg-slate-50 dark:bg-slate-800/50 border-b border-gray-200 dark:border-gray-700/80 flex flex-row justify-between items-center">
        <div className="flex items-center space-x-2">
          <ListChecksIcon className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
          <CardTitle className="text-sm font-semibold text-slate-700 dark:text-slate-200">Media Timeline Analysis</CardTitle>
        </div>
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                variant="ghost"
                className="text-gray-500 hover:bg-red-100 hover:text-red-700 h-7 w-7 px-0"
                onClick={() => onDelete(cell.id)}
              >
                <Trash2Icon className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Delete Media Timeline Cell</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </CardHeader>
      <CardContent className="p-4 space-y-4 bg-white">
        {/* Error display if present */}
        {cell.error && (
          <div className="p-3 border border-red-200 bg-red-50 text-xs text-red-700 rounded">
            {cell.error}
          </div>
        )}

        {/* Structured Rendering */}
        {structuredData ? (
          <>
            {/* Hypothesis */}
            {structuredData.hypothesis && (
              <Accordion type="single" collapsible defaultValue="hypothesis" className="w-full mb-4 border rounded-lg bg-white dark:bg-slate-800 shadow-sm dark:border-slate-700">
                <AccordionItem value="hypothesis" className="border-b-0">
                  <AccordionTrigger className="text-sm font-medium px-4 py-3 hover:bg-slate-50/80 dark:hover:bg-slate-700/50 [&[data-state=open]]:bg-slate-50/80 dark:[&[data-state=open]]:bg-slate-700/50 rounded-t-lg">
                    <div className="flex items-center">
                        <UserCheckIcon className="h-4 w-4 mr-2 text-indigo-600 dark:text-indigo-400"/>
                        Hypothesis
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-4 pb-4 pt-2 text-xs bg-white dark:bg-slate-800 rounded-b-lg">
                    {renderHypothesis(structuredData.hypothesis)}
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}

            {/* Timeline Events */}
            {structuredData.timeline_events && structuredData.timeline_events.length > 0 && (
              <div>
                <h3 className="text-base font-semibold text-slate-800 dark:text-slate-100 mb-3 flex items-center">
                  <CalendarDaysIcon className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
                  Timeline of Events
                </h3>
                {renderTimelineEvents(structuredData.timeline_events)}
              </div>
            )}
            {!structuredData.timeline_events || structuredData.timeline_events.length === 0 && (
               <div className="text-center py-4">
                 <InfoIcon className="h-8 w-8 text-gray-400 dark:text-gray-500 mx-auto mb-2"/>
                 <p className="text-sm text-gray-500 dark:text-gray-400">No timeline events were found or provided for this analysis.</p>
               </div>
            )}
          </>
        ) : (
          // Fallback markdown render
          renderMarkdownFallback()
        )}
      </CardContent>
    </Card>
  );
};

export default MediaTimelineCell; 