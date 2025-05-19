"use client"

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Button } from '@/components/ui/button';
import { CodeIcon, ExternalLinkIcon, FileCodeIcon, GitCommitIcon, HelpCircleIcon, InfoIcon, ListIcon, MessageSquareIcon, PaperclipIcon, SearchIcon, ServerIcon, TagIcon, TriangleAlertIcon, CheckCircleIcon, XCircleIcon, ClockIcon, FileTextIcon, GitPullRequestIcon, Trash2Icon, BarChartIcon, QuoteIcon, PuzzleIcon, TargetIcon, WrenchIcon, LightbulbIcon, UsersIcon, ChevronRightIcon, RefreshCwIcon } from 'lucide-react'; // Added RefreshCwIcon
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Cell } from '@/store/types'; // Import base Cell type
import type { InvestigationReportData, Finding, CodeReference, RelatedItem } from '@/hooks/useInvestigationEvents'; // Import report-specific types
import { useCanvasStore } from '@/store/canvasStore';
import { v4 as uuidv4 } from 'uuid'; // Added for generating session_id

interface InvestigationReportCellProps {
  cell: Cell; // Expecting a Cell object
  onDelete: (cellId: string) => void; // ADDED onDelete prop
  // Add other props if needed, like onUpdate, etc.
}

// Helper to format date/time (or use a date library)
const formatDate = (dateString?: string | null) => {
  if (!dateString) return 'N/A';
  try {
    return new Date(dateString).toLocaleString();
  } catch (e) {
    return 'Invalid Date';
  }
};

// Helper to get language for syntax highlighting
const getLanguageFromFilename = (filename?: string | null): string | undefined => {
    if (!filename) return undefined;
    const extension = filename.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'js': case 'jsx': return 'javascript';
      case 'ts': case 'tsx': return 'typescript';
      case 'py': return 'python';
      case 'java': return 'java';
      case 'kt': case 'kts': return 'kotlin';
      case 'swift': return 'swift';
      case 'cpp': case 'cxx': case 'cc': case 'hpp': case 'hxx': case 'hh': return 'cpp';
      case 'c': case 'h': return 'c';
      case 'json': return 'json';
      case 'md': return 'markdown';
      case 'rb': return 'ruby';
      case 'go': return 'go';
      case 'php': return 'php';
      case 'html': return 'html';
      case 'css': return 'css';
      case 'sh': case 'bash': return 'bash';
      case 'diff': return 'diff';
      default: return undefined;
    }
};

// Helper to render CodeReference
const RenderCodeReference: React.FC<{ reference?: CodeReference | null; className?: string }> = ({ reference, className }) => {
  if (!reference) return null;
  const language = getLanguageFromFilename(reference.file_path);
  return (
    <div className={`border rounded p-3 text-xs bg-gray-50/80 hover:bg-gray-100/50 transition-colors ${className}`}>
      <div className="flex justify-between items-center mb-1.5">
        <span className="font-mono text-gray-700 flex items-center">
          <FileCodeIcon className="h-3 w-3 mr-1.5 flex-shrink-0" />
          {reference.file_path}{reference.line_number && `:${reference.line_number}`}
        </span>
        {reference.url && (
          <a href={reference.url} target="_blank" rel="noopener noreferrer">
            <Button variant="ghost" size="icon" className="h-5 w-5">
              <ExternalLinkIcon className="h-3 w-3" />
            </Button>
          </a>
        )}
      </div>
      {reference.component_name && <p className="italic text-gray-600 mb-1">Component: {reference.component_name}</p>}
      {reference.code_snippet && language && (
        <SyntaxHighlighter language={language} style={oneLight} customStyle={{ margin: 0, fontSize: '0.7rem', padding: '0.25rem' }} wrapLines={true}>
          {reference.code_snippet}
        </SyntaxHighlighter>
      )}
      {reference.code_snippet && !language && (
         <pre className="text-xs font-mono bg-white p-1 border rounded overflow-x-auto">{reference.code_snippet}</pre>
      )}
    </div>
  );
};

// Helper to render Finding
const RenderFinding: React.FC<{ finding?: Finding | null; title: string; confidence?: "Low" | "Medium" | "High" | null; defaultOpen?: boolean; icon?: React.ElementType }> = ({ finding, title, confidence, defaultOpen = false, icon: IconComponent }) => {
  if (!finding) return null;

  const getConfidenceColor = (level?: string | null) => {
    switch (level) {
      case 'High': return 'text-green-700 border-green-300 bg-green-50';
      case 'Medium': return 'text-yellow-700 border-yellow-300 bg-yellow-50';
      case 'Low': return 'text-red-700 border-red-300 bg-red-50';
      default: return 'text-gray-600 border-gray-300 bg-gray-50';
    }
  };

  return (
    <Accordion type="single" collapsible className="w-full mb-3 border rounded-md bg-white shadow-sm" defaultValue={defaultOpen ? 'item-1' : undefined}>
      <AccordionItem value="item-1" className="border-b-0">
        <AccordionTrigger className="text-sm font-medium px-4 py-2.5 hover:bg-gray-50 hover:no-underline [&[data-state=open]]:bg-gray-50/80 rounded-t-md">
          <div className="flex justify-between items-center w-full">
            <span className="flex items-center">
                {IconComponent && <IconComponent className="h-4 w-4 mr-2 text-indigo-600 flex-shrink-0" />}
                {title}
            </span>
            {confidence && (
              <Badge variant="outline" className={`ml-2 text-xs px-1.5 py-0.5 ${getConfidenceColor(confidence)}`}>
                Confidence: {confidence}
              </Badge>
            )}
          </div>
        </AccordionTrigger>
        <AccordionContent className="px-4 pb-4 pt-2 text-xs space-y-3">
          <p className="font-semibold text-sm text-gray-800">{finding.summary}</p>
          {finding.details && (
             <div className="prose prose-sm max-w-none text-xs text-gray-700 leading-relaxed">
                <ReactMarkdown>{finding.details}</ReactMarkdown>
             </div>
          )}
          {finding.code_reference && <RenderCodeReference reference={finding.code_reference} className="mt-2" />}
          {finding.supporting_quotes && finding.supporting_quotes.length > 0 && (
            <div className="mt-3 space-y-1.5">
              <h4 className="text-xs font-medium text-gray-600 flex items-center"><QuoteIcon className="h-3.5 w-3.5 mr-1.5 text-gray-400 flex-shrink-0"/>Supporting Quotes:</h4>
              <ul className="list-none space-y-1 pl-2">
                {finding.supporting_quotes.map((quote, i) => (
                  <li key={i} className="italic text-gray-600 bg-gray-50 p-1.5 border rounded text-[0.7rem]">&ldquo;{quote}&rdquo;</li>
                ))}
              </ul>
            </div>
          )}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
};

// Helper to render Related Items
const RenderRelatedItems: React.FC<{ items?: RelatedItem[] | null }> = ({ items }) => {
  if (!items || items.length === 0) return null;

  const getIcon = (type: RelatedItem['type']) => {
    switch (type) {
      case 'Pull Request': return <GitPullRequestIcon className="h-3.5 w-3.5 mr-1.5 text-blue-600 flex-shrink-0" />;
      case 'Issue': return <TriangleAlertIcon className="h-3.5 w-3.5 mr-1.5 text-orange-600 flex-shrink-0" />;
      case 'Commit': return <GitCommitIcon className="h-3.5 w-3.5 mr-1.5 text-gray-600 flex-shrink-0" />;
      case 'Discussion': return <MessageSquareIcon className="h-3.5 w-3.5 mr-1.5 text-purple-600 flex-shrink-0" />;
      case 'Documentation': return <FileTextIcon className="h-3.5 w-3.5 mr-1.5 text-indigo-600 flex-shrink-0" />;
      default: return <PaperclipIcon className="h-3.5 w-3.5 mr-1.5 text-gray-500 flex-shrink-0" />;
    }
  };

  const getStatusIcon = (status?: string | null) => {
     if (!status) return null;
     const lowerStatus = status.toLowerCase();
     if (lowerStatus.includes('open')) return <CheckCircleIcon className="h-3 w-3 ml-1 text-green-600" />;
     if (lowerStatus.includes('closed') || lowerStatus.includes('merged') || lowerStatus.includes('done')) return <XCircleIcon className="h-3 w-3 ml-1 text-purple-600" />;
     if (lowerStatus.includes('progress') || lowerStatus.includes('review')) return <ClockIcon className="h-3 w-3 ml-1 text-yellow-600" />;
     return null;
  };

  return (
    <Card className="mt-4 shadow-sm">
      <CardHeader className="p-3">
        <CardTitle className="text-base flex items-center"><ListIcon className="h-4 w-4 mr-2" />Related Items</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <Table className="text-xs">
          <TableHeader className="bg-gray-50">
            <TableRow>
              <TableHead className="w-[150px] px-3 py-2">Type</TableHead>
              <TableHead className="px-3 py-2">Identifier</TableHead>
              <TableHead className="px-3 py-2">Relevance</TableHead>
              <TableHead className="px-3 py-2">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((item, index) => (
              <TableRow key={index} className="hover:bg-gray-50/50">
                <TableCell className="font-medium px-3 py-2 flex items-center">{getIcon(item.type)} {item.type}</TableCell>
                <TableCell className="px-3 py-2">
                   <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline flex items-center">
                     {item.identifier} <ExternalLinkIcon className="h-3 w-3 ml-1" />
                   </a>
                </TableCell>
                <TableCell className="px-3 py-2 text-gray-600">{item.relevance || '-'}</TableCell>
                <TableCell className="px-3 py-2 flex items-center">
                    {item.status || '-'}
                    {getStatusIcon(item.status)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};

const InvestigationReportCell: React.FC<InvestigationReportCellProps> = ({ cell, onDelete }) => {
  // Log the cell prop received by the component
  console.log(`[InvestigationReportCell ${cell.id}] Render with cell prop:`, cell);

  const notebookId = cell.notebook_id; // Placed notebookId definition before its use in useCallback
  const { sendSuggestedStepToChat, isSendingChatMessage } = useCanvasStore();

  const reportData = cell.result as InvestigationReportData | null;

  if (!reportData) {
    // Handle case where result or content is missing or not the expected type
    return (
      <div className="border rounded-md overflow-hidden mb-3 mx-8 bg-yellow-50 border-yellow-200 p-3 text-xs text-yellow-800">
        <p><TriangleAlertIcon className="inline h-4 w-4 mr-1"/> Report data is missing, invalid, or not in the expected nested structure (result.content) for this cell.</p>
      </div>
    );
  }

  const getSeverityBadge = (severity?: string | null) => {
    switch (severity) {
      case 'Critical': return <Badge variant="destructive" className="bg-red-600 text-white">{severity}</Badge>;
      case 'High': return <Badge variant="destructive">{severity}</Badge>;
      case 'Medium': return <Badge className="bg-yellow-500 text-white">{severity}</Badge>;
      case 'Low': return <Badge variant="secondary">{severity}</Badge>;
      default: return <Badge variant="outline">Unknown Severity</Badge>;
    }
  };

  return (
    <div className="border rounded-md overflow-hidden mb-3 mx-8 shadow-sm">
      {/* Header - Apply Navy Blue Theme */}
      <div className="bg-indigo-100 border-b border-indigo-200 p-3 flex justify-between items-center">
        <div className="flex items-center">
           <FileTextIcon className="h-4 w-4 mr-2 text-indigo-700 flex-shrink-0" />
           <span className="font-medium text-sm text-indigo-800">{reportData.title || "Investigation Report"}</span>
        </div>
        {/* ADD Delete Button */}
        <div className="flex items-center space-x-1.5">
            <TooltipProvider delayDuration={100}>
                 <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            size="sm"
                            variant="ghost"
                            className="text-gray-500 hover:bg-indigo-100 hover:text-indigo-700 h-7 w-7 px-0"
                            onClick={() => {
                                // Implement rerun logic here
                            }}
                            disabled={cell.status === 'running' || cell.status === 'queued'} // Disable if already running/queued
                        >
                            <RefreshCwIcon className={`h-4 w-4 ${ (cell.status === 'running' || cell.status === 'queued') ? 'animate-spin' : ''}`} />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                        <p>Re-run Investigation Report</p>
                    </TooltipContent>
                </Tooltip>
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
                    <p>Delete Report Cell</p>
                  </TooltipContent>
                </Tooltip>
            </TooltipProvider>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 bg-white">
        {/* Display overall error if report generation failed */}
        {reportData.error && (
            <Alert variant="destructive" className="mb-4">
                <TriangleAlertIcon className="h-4 w-4" />
                <AlertTitle>Report Generation Error</AlertTitle>
                <AlertDescription className="text-xs">
                    {reportData.error}
                </AlertDescription>
            </Alert>
        )}

        {/* Top Section: Query, Status, Severity, Tags */}
        <Card className="mb-4 bg-white border shadow-none">
          <CardContent className="p-4 space-y-3">
            <div className="flex justify-between items-start flex-wrap gap-3">
              <p className="text-sm font-medium text-gray-700 flex items-center">
                <SearchIcon className="h-4 w-4 mr-1.5 text-gray-500"/>Original Query: <span className="ml-1 font-normal text-gray-900">{reportData.query}</span>
              </p>
              {reportData.estimated_severity && (
                <div className="flex items-center space-x-2">
                  <BarChartIcon className="h-4 w-4 text-gray-500" />
                   <span className="text-sm font-medium text-gray-700">Severity:</span>
                   {getSeverityBadge(reportData.estimated_severity)}
                </div>
              )}
            </div>
            {reportData.status && (
              <p className="text-xs text-gray-600 flex items-center">
                <InfoIcon className="h-3 w-3 mr-1.5 text-gray-400"/>Status: <span className="ml-1 font-medium text-gray-800">{reportData.status}</span>
                {reportData.status_reason && <span className="ml-1 text-gray-500">({reportData.status_reason})</span>}
              </p>
            )}
            {reportData.tags && reportData.tags.length > 0 && (
              <div className="flex flex-wrap items-center gap-1 pt-1">
                <TagIcon className="h-3.5 w-3.5 mr-1 text-gray-400"/>
                {reportData.tags.map((tag, i) => <Badge key={i} variant="secondary" className="text-xs font-normal">{tag}</Badge>)}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Core Findings: Issue Summary & Root Cause */}
        <RenderFinding finding={reportData.issue_summary} title="Issue Summary" defaultOpen={true} icon={TargetIcon} />
        <RenderFinding finding={reportData.root_cause} title="Root Cause Analysis" confidence={reportData.root_cause_confidence} defaultOpen={true} icon={WrenchIcon} />

        {/* Key Components */}
        {reportData.key_components && reportData.key_components.length > 0 && (
          <Card className="mt-4 shadow-sm">
            <CardHeader className="p-3">
              <CardTitle className="text-base flex items-center"><CodeIcon className="h-4 w-4 mr-2" />Key Code Components</CardTitle>
            </CardHeader>
            <CardContent className="p-3 grid grid-cols-1 md:grid-cols-2 gap-4">
              {reportData.key_components.map((comp, i) => <RenderCodeReference key={i} reference={comp} />)}
            </CardContent>
          </Card>
        )}

        {/* Related Items */}
        <RenderRelatedItems items={reportData.related_items} />

        {/* Proposed Fix */}
        {reportData.proposed_fix && (
          <div className="mt-4">
             <RenderFinding finding={reportData.proposed_fix} title="Proposed Fix / Solution" confidence={reportData.proposed_fix_confidence} icon={LightbulbIcon} />
          </div>
        )}

        {/* Affected Context */}
        {reportData.affected_context && (
            <div className="mt-4">
                <RenderFinding finding={reportData.affected_context} title="Affected Context" icon={UsersIcon} />
            </div>
        )}

        {/* Suggested Next Steps */}
        {reportData.suggested_next_steps && reportData.suggested_next_steps.length > 0 && (
          <Card className="mt-4 shadow-sm">
            <CardHeader className="p-3">
              <CardTitle className="text-base flex items-center"><HelpCircleIcon className="h-4 w-4 mr-2" />Suggested Next Steps</CardTitle>
            </CardHeader>
            <CardContent className="p-3">
              <ul className="list-none space-y-1.5 text-xs text-gray-700">
                {reportData.suggested_next_steps.map((step, i) => (
                  <li key={i}>
                    <button
                      onClick={() => sendSuggestedStepToChat(step)}
                      className="flex items-center w-full text-left p-1.5 rounded hover:bg-indigo-50 hover:text-indigo-800 transition-all duration-150 cursor-pointer focus:outline-none focus:ring-1 focus:ring-indigo-300"
                      // Add disabled state if a message is currently being sent via the store
                      disabled={isSendingChatMessage}
                    >
                      <ChevronRightIcon className="h-3.5 w-3.5 mr-1.5 text-indigo-400 flex-shrink-0" />
                      <span className="flex-1">{step}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

      </div>
    </div>
  );
};

export default InvestigationReportCell;
