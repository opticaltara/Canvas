"use client"

import React from "react"
import { useState, useEffect } from "react"
import { PlayIcon, ChevronDownIcon, ChevronUpIcon, ExternalLinkIcon, GitCommitIcon, FileDiffIcon, CopyIcon, CheckIcon, Trash2Icon, Download, FolderIcon, FileIcon, MessageSquare, GitPullRequest, CheckCircle, XCircle, Clock, AlertCircleIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'

// Import connection store
import { useConnectionStore } from "@/app/store/connectionStore"
// Import the updated Cell type from the store
import type { Cell } from "@/store/types"

interface GitHubCellProps {
  // Use the imported Cell type
  cell: Cell 
  onExecute: (cellId: string) => void
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void
  onDelete: (cellId: string) => void
  isExecuting: boolean
}

interface ToolForm {
  toolName: string
  toolArgs: Record<string, any>
}

const GitHubCellComponent: React.FC<GitHubCellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }): React.ReactNode => {
  // Log the cell prop received by the component
  console.log(`[GitHubCell ${cell.id}] Render with cell prop:`, JSON.stringify({ 
    id: cell.id, 
    type: cell.type, 
    status: cell.status, 
    tool_name: cell.tool_name, 
    has_result: !!cell.result,
    result_content_type: typeof cell.result?.content,
    result_error: cell.result?.error,
  }));

  if (cell.result?.error && cell.status === 'error') { // Only skip render if the cell itself is in an error state with a result error
      console.log(`[GitHubCell ${cell.id}] Backend error detected and cell status is error. Skipping render. Error:`, cell.result.error);
      // We still want to render the error *within* the cell structure if the cell isn't in a global error status
      // but the result itself indicates an error.
      // This change makes it so that renderResultData can still show the error message.
  }

  const [toolForms, setToolForms] = useState<ToolForm[]>([{ toolName: "", toolArgs: {} }])
  const [activeToolIndex, setActiveToolIndex] = useState(0) // Kept for consistency, though only one tool form
  const [isResultExpanded, setIsResultExpanded] = useState(true)
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({})

  // Access connection store for tool definitions and global expansion state
  const toolDefinitions = useConnectionStore((state) => state.toolDefinitions.github);
  const toolLoadingStatus = useConnectionStore((state) => state.toolLoadingStatus.github);
  const areAllCellsExpanded = useConnectionStore((state) => state.areAllCellsExpanded);

  // State for controlled accordion for Tool Arguments
  const [argsAccordionValue, setArgsAccordionValue] = useState<string>(areAllCellsExpanded ? "tool-args" : "");

  // Effect to sync Tool Arguments accordion with global state
  useEffect(() => {
    setArgsAccordionValue(areAllCellsExpanded ? "tool-args" : "");
  }, [areAllCellsExpanded]);

  // Effect to sync Execution Result card expansion with global state
  useEffect(() => {
    setIsResultExpanded(areAllCellsExpanded);
  }, [areAllCellsExpanded]);


  // Initialize/Reset the tool form state ONLY when the cell identity changes.
  useEffect(() => {
    const incomingToolName = cell.tool_name;
    const incomingToolArgs = cell.tool_arguments;

    if (incomingToolName) { // Removed incomingToolArgs check as it might be empty initially
      console.log(`[GitHubCell ${cell.id}] useEffect [id, tool_name, tool_arguments changed]: Initializing/Resetting form state from props.`);
      const initialForm: ToolForm = {
        toolName: incomingToolName,
        toolArgs: incomingToolArgs ? { ...incomingToolArgs } : {}, // Ensure a copy or empty object
      };
      setToolForms([initialForm]);
      setActiveToolIndex(0);
    } else {
      // Fallback if tool info is missing
      console.warn(`[GitHubCell ${cell.id}] useEffect: Tool name missing in cell prop, setting empty state.`);
      setToolForms([{ toolName: "", toolArgs: {} }]);
      setActiveToolIndex(0);
    }
  }, [cell.id, cell.tool_name, cell.tool_arguments]); // Depend on tool_name and tool_arguments as well

  // Format date to be more readable
  const formatDate = (dateString: string) => {
    if (!dateString) return ""
    try {
      const date = new Date(dateString)
      // Check if date is valid before formatting
      if (isNaN(date.getTime())) {
        return "Invalid Date";
      }
      return date.toLocaleString()
    } catch (e) {
      console.error("Error formatting date:", dateString, e);
      return "Invalid Date"; // Return a placeholder if formatting fails
    }
  }

  // Helper to get language for syntax highlighting
  const getLanguageFromFilename = (filename: string): string | undefined => {
    const extension = filename.split('.').pop()?.toLowerCase();
    // Add or ensure common languages are present
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
      case 'diff': return 'diff'; // Add diff language
      default: return undefined; // Let SyntaxHighlighter guess or handle plain text
    }
  };

  // Copy to clipboard utility
  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedStates(prev => ({ ...prev, [id]: true }));
      setTimeout(() => setCopiedStates(prev => ({ ...prev, [id]: false })), 2000); // Reset after 2s
    }, (err) => {
      console.error('Could not copy text: ', err);
    });
  };

  // Handle form input changes
  const handleToolArgChange = (toolIndex: number, argName: string, value: any) => {
    const updatedForms = [...toolForms]
    // Handle potential JSON parsing for the generic form
    if (argName === "__raw_json__") {
      updatedForms[toolIndex] = {
        ...updatedForms[toolIndex],
        toolArgs: value, // Directly set the parsed object
      }
    } else {
      updatedForms[toolIndex] = {
        ...updatedForms[toolIndex],
        toolArgs: {
          ...updatedForms[toolIndex].toolArgs,
          [argName]: value,
        },
      }
    }
    setToolForms(updatedForms)
     // Also call onUpdate to save changes to the store
    if (updatedForms[toolIndex]) {
      onUpdate(cell.id, cell.content, { toolName: updatedForms[toolIndex].toolName, toolArgs: updatedForms[toolIndex].toolArgs });
    }
  }

  // Execute the cell with the current tool forms
  const executeWithToolForms = () => {
    // ADDED: Update cell with current arguments before execution
    const currentForm = toolForms[activeToolIndex];
    if (currentForm) {
        onUpdate(cell.id, cell.content, { toolName: currentForm.toolName, toolArgs: currentForm.toolArgs });
    }
    // END ADDED
    onExecute(cell.id)
  }

  // --- Modified: Render tool form inputs dynamically based on JSON Schema --- 
  const renderToolFormInputs = (toolForm: ToolForm) => {
    const { toolName, toolArgs } = toolForm

    // Use the store hook to get the latest status and definitions
    const githubToolLoadingStatus = useConnectionStore(state => state.toolLoadingStatus.github);
    const githubToolDefinitions = useConnectionStore(state => state.toolDefinitions.github);

    // Debug Logs (keep for now)
    console.log(`[GitHubCell ${cell.id}] Rendering inputs for: ${toolName}`);
    console.log(`[GitHubCell ${cell.id}] GitHub Tool loading status from store:`, githubToolLoadingStatus);
    console.log(`[GitHubCell ${cell.id}] Available GitHub tool definitions from store:`, githubToolDefinitions);

    // Find the tool definition from the store
    const toolInfo = githubToolDefinitions?.find(def => def.name === toolName);
    console.log(`[GitHubCell ${cell.id}] Found toolInfo:`, toolInfo);

    // Handle loading state MORE ROBUSTLY
    // Case 1: Status is undefined or 'idle' (fetching hasn't started or finished yet)
    if (githubToolLoadingStatus === undefined || githubToolLoadingStatus === 'idle') {
        // Optionally trigger fetch if idle/undefined? Might be handled elsewhere.
        // For now, just show a generic loading message or potentially nothing if definitions
        // are expected to load shortly. Let's show loading.
        return <div className="text-xs text-gray-500">Initializing tool definitions...</div>;
    }
    // Case 2: Explicitly loading
    if (githubToolLoadingStatus === 'loading') {
        return <div className="text-xs text-gray-500">Loading tool parameters...</div>;
    }
    // Case 3: Explicit error state from the store
    if (githubToolLoadingStatus === 'error') {
        return <div className="text-xs text-red-500">Error loading tool definitions from store.</div>; // Corrected quote
    }
    // Case 4: Status is 'success', but the specific tool isn't found
    if (githubToolLoadingStatus === 'success' && !toolInfo) {
        return <div className="text-xs text-red-500">Tool definition not found for: {toolName}</div>;
    }
    // Case 5: Tool definition not available for other reasons (shouldn't happen if logic above is correct)
     if (!toolInfo) {
        return <div className="text-xs text-red-500">Tool definition unavailable for: {toolName}. Status: {githubToolLoadingStatus}</div>; // Fallback error
    }


    const schema = toolInfo.inputSchema;
    const properties = schema?.properties as Record<string, any> | undefined;
    const requiredFields = new Set(schema?.required || []);

    if (!properties || Object.keys(properties).length === 0) {
      return <div className="text-xs text-gray-600 mb-3">This tool requires no arguments.</div>;
    }
    
    // Helper function to render individual input fields
    const renderField = (paramName: string, paramSchema: any) => {
        const fieldId = `${paramName}-${activeToolIndex}`;
        const label = paramSchema.title || paramName; // Use title for label
        const isRequired = requiredFields.has(paramName);
        const description = paramSchema.description;
        const placeholder = paramSchema.examples?.[0] || paramSchema.description || `Enter ${label}`;
        const currentValue = toolArgs[paramName] ?? paramSchema.default ?? ''; // Use current value or default

        let inputElement: React.ReactNode = null;

        // Determine input type based on schema
        const schemaType = paramSchema.type;
        const schemaFormat = paramSchema.format;
        const enumValues = paramSchema.enum as string[] | undefined;

        if (enumValues && Array.isArray(enumValues)) {
             // Render Select for enum
             inputElement = (
                <Select
                  value={currentValue}
                  onValueChange={(value) => handleToolArgChange(activeToolIndex, paramName, value)}
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
        } else if (schemaType === 'string' && (schemaFormat === 'textarea' || (description && description.length > 50))) { // Heuristic for textarea
             // Render Textarea for long descriptions or specific format
             inputElement = (
                 <Textarea
                    id={fieldId}
                    value={currentValue}
                    onChange={(e) => handleToolArgChange(activeToolIndex, paramName, e.target.value)}
                    placeholder={placeholder}
                    required={isRequired}
                    className="mt-1 h-16 text-xs"
                 />
             );
        } else if (schemaType === 'number' || schemaType === 'integer') {
            // Render Input type number
            inputElement = (
                <Input
                    id={fieldId}
                    type="number"
                    value={currentValue}
                    onChange={(e) => handleToolArgChange(activeToolIndex, paramName, e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder={placeholder}
                    required={isRequired}
                    min={paramSchema.minimum}
                    max={paramSchema.maximum}
                    step={paramSchema.multipleOf || 'any'} // Assuming step corresponds to multipleOf
                    className="mt-1 h-8 text-xs"
                 />
            );
        } else if (schemaType === 'boolean') {
            // Render Checkbox for boolean (example - adjust UI as needed)
             inputElement = (
                 <div className="flex items-center space-x-2 mt-1 h-8"> 
                   <input
                     type="checkbox"
                     id={fieldId}
                     checked={!!currentValue} // Ensure boolean coercion
                     onChange={(e) => handleToolArgChange(activeToolIndex, paramName, e.target.checked)}
                     className="h-4 w-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
                    />
                    {/* Optional: label next to checkbox if needed */}
                 </div>
             );
        } else { // Default to string input
            inputElement = (
                <Input
                  id={fieldId}
                  type="text"
                  value={currentValue}
                  onChange={(e) => handleToolArgChange(activeToolIndex, paramName, e.target.value)}
                  placeholder={placeholder}
                  required={isRequired}
                  className="mt-1 h-8 text-xs"
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
  // --- End Modified renderToolFormInputs --- 

  // Render the result data in a formatted way
  const renderResultData = () => {
    const toolResult = cell.result
    const toolName = cell.tool_name

    // Log the data being passed to renderResultData
    console.log(`[GitHubCell ${cell.id}] renderResultData called. toolName: ${toolName}, has toolResult: ${!!toolResult}`);
    if (toolResult) {
      console.log(`[GitHubCell ${cell.id}] toolResult details:`, JSON.stringify({
        has_content: !!toolResult.content,
        content_type: typeof toolResult.content,
        has_error: !!toolResult.error,
        error_content: toolResult.error,
      }));
    }

    if (!toolResult) {
      console.log(`[GitHubCell ${cell.id}] renderResultData: No toolResult found, returning null.`);
      return null;
    }

    // Helper function to safely get the JSON text content
    const getJsonTextContent = (result: any): string | null => {
        if (result?.content?.content && Array.isArray(result.content.content) && result.content.content[0]?.text) {
            return result.content.content[0].text;
        }
        // Fallback for potentially different structure (though less likely based on API example)
        if (typeof result?.content === 'string') {
            return result.content;
        }
        if (Array.isArray(result?.content) && result.content[0]?.text) {
             // Legacy or alternative structure check
             console.warn("Accessing result content via potentially legacy path: result.content[0].text");
             return result.content[0].text;
        }
        return null;
    };

    // Helper function to get raw content if parsing fails or not needed
    const getRawContentForError = (result: any): any => {
         const jsonText = getJsonTextContent(result);
         if (jsonText) return jsonText; // Return the string if found

         // Fallback to less nested structures if the primary one fails
         if (result?.content?.content) return result.content.content;
         if (result?.content) return result.content;
         return "Unable to extract raw content."; // Default fallback
    };

    let shortSha: string | undefined;
    if (toolName === 'get_commit' && !toolResult.error && toolResult.content) {
        try {
            // Use helper to get JSON text
            const jsonText = getJsonTextContent(toolResult);
            if (jsonText) {
                const commitDataObj = JSON.parse(jsonText);
                if (commitDataObj?.sha) {
                    shortSha = commitDataObj.sha.substring(0, 7);
                }
            } else {
                 // Try parsing direct content if helper failed (unlikely based on API)
                 const commitDataObj = toolResult.content; // Assuming direct object
                 if (commitDataObj?.sha) {
                     shortSha = commitDataObj.sha.substring(0, 7);
                 }
            }
        } catch (e) {
             console.error("Error parsing SHA from get_commit result:", e);
            // Handle parsing error, shortSha remains undefined
        }
    }

    let displayContent: React.ReactNode = null

    if (toolName === 'get_commit' && !toolResult.error && toolResult.content) {
        try {
            let commitData: any;
            // Use helper to get JSON text
            const jsonText = getJsonTextContent(toolResult);
            if (jsonText) {
                 commitData = JSON.parse(jsonText);
            } else {
                 // Fallback: If getJsonTextContent failed, attempt to use content directly
                 // Check if it's already an object (post JSON serialization from backend)
                 if (typeof toolResult.content === 'object' && toolResult.content !== null) {
                     console.warn("[get_commit] Parsing direct object content.");
                     commitData = toolResult.content; 
                 } else {
                      // If it's not an object, we cannot proceed
                      throw new Error("Commit data is not in expected format (JSON string or object).");
                 }
                 // Basic validation if it's an object
                 if (typeof commitData !== 'object' || commitData === null || !commitData.sha) {
                     throw new Error("Direct content is not a valid commit object.");
                 }
            }

            const currentShortSha = commitData.sha.substring(0, 7);
            const commitId = `commit-${cell.id}`;

            displayContent = (
                <div className="space-y-3 text-xs">
                    <div className="flex justify-between items-start">
                        <div>
                            <h4 className="text-base font-semibold mb-1">{commitData.commit?.message?.split('\\n')[0] || 'No commit message'}</h4>
                            <div className="text-xs text-gray-600 flex items-center space-x-2 flex-wrap">
                                {commitData.commit?.author && (
                                    <TooltipProvider delayDuration={100}>
                                        <Tooltip>
                                            <TooltipTrigger asChild>
                                                <img src={commitData.author?.avatar_url} alt={commitData.commit.author.name || 'Author'} className="h-4 w-4 rounded-full inline-block mr-1"/>
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>{commitData.commit.author.name || 'N/A'} ({commitData.commit.author.email || 'N/A'})</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                )}
                                {commitData.commit?.author?.date && <span>authored on {formatDate(commitData.commit.author.date)}</span>}
                                {commitData.author?.login && (
                                    <a href={commitData.author.html_url} target="_blank" rel="noopener noreferrer" className="hover:underline text-blue-600">
                                        ({commitData.author.login})
                                    </a>
                                )}
                                {commitData.commit?.committer && commitData.author?.login !== commitData.committer?.login && (
                                    <>
                                        <span className="hidden md:inline">|</span>
                                        <TooltipProvider delayDuration={100}>
                                            <Tooltip>
                                                <TooltipTrigger asChild>
                                                    <img src={commitData.committer?.avatar_url} alt={commitData.commit.committer.name || 'Committer'} className="h-4 w-4 rounded-full inline-block mr-1"/>
                                                </TooltipTrigger>
                                                <TooltipContent>
                                                    <p>{commitData.commit.committer.name || 'N/A'} ({commitData.commit.committer.email || 'N/A'})</p>
                                                </TooltipContent>
                                            </Tooltip>
                                        </TooltipProvider>
                                        {commitData.commit.committer.date && <span>committed on {formatDate(commitData.commit.committer.date)}</span>}
                                        {commitData.committer?.login && (
                                            <a href={commitData.committer.html_url} target="_blank" rel="noopener noreferrer" className="hover:underline text-blue-600">
                                                ({commitData.committer.login})
                                            </a>
                                        )}
                                    </>
                                )}
                            </div>
                            {commitData.commit?.message?.includes('\\n') && (
                                <pre className="mt-1.5 text-xs whitespace-pre-wrap font-sans bg-gray-50 p-1.5 rounded border">
                                    {commitData.commit.message.substring(commitData.commit.message.indexOf('\\n') + 1)}
                                </pre>
                            )}
                        </div>
                        <div className="flex items-center space-x-1 flex-shrink-0 ml-3">
                            <TooltipProvider delayDuration={100}>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Button variant="outline" size="sm" onClick={() => copyToClipboard(commitData.sha, commitId)} className="font-mono text-xs h-auto px-1.5 py-0.5">
                                            <GitCommitIcon className="h-3 w-3 mr-1" /> {currentShortSha}
                                            {copiedStates[commitId] ? <CheckIcon className="h-3 w-3 ml-1 text-green-600" /> : <CopyIcon className="h-3 w-3 ml-1" />}
                                        </Button>
                                    </TooltipTrigger>
                                    <TooltipContent><p>Copy full SHA: {commitData.sha}</p></TooltipContent>
                                </Tooltip>
                            </TooltipProvider>
                            <a href={commitData.html_url} target="_blank" rel="noopener noreferrer">
                                <Button variant="outline" size="sm" className="text-xs h-auto px-1.5 py-0.5">
                                    <ExternalLinkIcon className="h-3 w-3 mr-1" /> GitHub
                                </Button>
                            </a>
                        </div>
                    </div>

                    <div className="flex justify-between items-center text-xs text-gray-700 flex-wrap gap-y-1">
                        <div className="flex items-center space-x-1 flex-wrap gap-y-1">
                            {commitData.stats && (
                                <>
                                    <Badge variant="outline" className="border-green-300 text-green-700 whitespace-nowrap px-1.5 py-0.5 text-xs">+{commitData.stats.additions} additions</Badge>
                                    <Badge variant="outline" className="border-red-300 text-red-700 whitespace-nowrap px-1.5 py-0.5 text-xs">-{commitData.stats.deletions} deletions</Badge>
                                    <Badge variant="secondary" className="whitespace-nowrap px-1.5 py-0.5 text-xs">{commitData.files?.length || 0} changed files</Badge>
                                </>
                            )}
                        </div>
                        {commitData.parents && commitData.parents.length > 0 && (
                            <div className="flex items-center space-x-1 flex-wrap gap-x-1">
                                <span>Parents:</span>
                                {commitData.parents.map((parent: any) => (
                                    <a key={parent.sha} href={parent.html_url} target="_blank" rel="noopener noreferrer" className="font-mono text-blue-600 hover:underline">
                                        {parent.sha.substring(0, 7)}
                                    </a>
                                ))}
                            </div>
                        )}
                    </div>

                    {commitData.files && commitData.files.length > 0 && (
                        <div>
                            <h5 className="text-xs font-medium mb-1.5">{commitData.files.length} changed files:</h5>
                            <Accordion type="single" collapsible className="w-full border rounded-md">
                                {commitData.files.map((file: any, index: number) => {
                                    const fileId = `file-${cell.id}-${index}`;
                                    if (!file) return null;
                                    return (
                                        <AccordionItem value={`item-${index}`} key={file.sha || index}>
                                            <AccordionTrigger className="text-xs px-2 py-1.5 hover:bg-gray-50">
                                                <div className="flex flex-col md:flex-row justify-between items-start md:items-center w-full gap-1">
                                                    <div className="flex items-center space-x-1.5 truncate mr-4 flex-shrink min-w-0">
                                                        <Badge
                                                            variant={file.status === 'removed' ? 'destructive' : 'outline'}
                                                            className={
                                                                `text-xs px-1 py-0 leading-tight flex-shrink-0
                                                                ${file.status === 'added' ? 'border-green-300 text-green-700 bg-green-50' : ''}
                                                                ${file.status === 'modified' ? 'border-blue-300 text-blue-700 bg-blue-50' : ''}
                                                                ${file.status === 'removed' ? 'border-red-300' : ''}
                                                                ${file.status === 'renamed' ? 'border-yellow-300 text-yellow-700 bg-yellow-50' : ''}
                                                                ${!['added', 'modified', 'removed', 'renamed'].includes(file.status) ? 'border-gray-300 text-gray-700 bg-gray-50' : ''}
                                                                `}
                                                        >
                                                            {file.status}
                                                        </Badge>
                                                        <span className="font-mono truncate text-xs" title={file.filename}>{file.filename}</span>
                                                    </div>
                                                    <div className="flex items-center space-x-1.5 flex-shrink-0 pl-6 md:pl-0">
                                                        <span className="text-green-600 text-xs">+{file.additions}</span>
                                                        <span className="text-red-600 text-xs">-{file.deletions}</span>
                                                        <TooltipProvider delayDuration={100}>
                                                            <Tooltip>
                                                                <TooltipTrigger asChild>
                                                                    <Button
                                                                        variant="ghost"
                                                                        size="icon"
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            copyToClipboard(file.patch || 'No patch available', fileId)
                                                                        }}
                                                                        className="ml-1 h-5 w-5"
                                                                    >
                                                                        {copiedStates[fileId] ? <CheckIcon className="h-3 w-3 text-green-600" /> : <CopyIcon className="h-3 w-3" />}
                                                                    </Button>
                                                                </TooltipTrigger>
                                                                <TooltipContent><p>Copy patch</p></TooltipContent>
                                                            </Tooltip>
                                                        </TooltipProvider>
                                                        {file.blob_url && (
                                                            <a href={file.blob_url} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()}>
                                                                <Button variant="ghost" size="icon" className="ml-1 h-5 w-5">
                                                                    <ExternalLinkIcon className="h-3 w-3" />
                                                                </Button>
                                                            </a>
                                                        )}
                                                    </div>
                                                </div>
                                            </AccordionTrigger>
                                            <AccordionContent className="px-0 pb-0">
                                                {file.patch ? (
                                                    <div className="text-xs bg-gray-50 border-t overflow-x-auto max-h-80 font-mono">
                                                      {file.patch.split('\\n').map((line: string, lineIndex: number) => {
                                                        let style: React.CSSProperties = { whiteSpace: 'pre-wrap', display: 'block', paddingLeft: '0.5rem', paddingRight: '0.5rem' };
                                                        let content = line;
                                                        let prefix = '';
                                                        const language = getLanguageFromFilename(file.filename) || 'diff'; // Default to diff for highlighting + / -

                                                        if (line.startsWith('+')) {
                                                          style.backgroundColor = 'rgba(217, 249, 157, 0.4)'; // Light green
                                                          content = line.substring(1);
                                                          prefix = '+ ';
                                                        } else if (line.startsWith('-')) {
                                                          style.backgroundColor = 'rgba(254, 202, 202, 0.4)'; // Light red
                                                          content = line.substring(1);
                                                          prefix = '- ';
                                                        } else if (line.startsWith('@@')) {
                                                           style.backgroundColor = '#e5e7eb'; // Gray background for headers
                                                           style.color = '#4b5563';
                                                           content = line;
                                                           prefix = '';
                                                        } else if (line.startsWith(' ')) {
                                                          // Context line, remove leading space for highlighting
                                                          content = line.substring(1);
                                                          prefix = '  '; // Keep indentation visual
                                                        } else {
                                                          // Handle empty lines or lines without standard prefixes
                                                          content = line;
                                                          prefix = '  ';
                                                        }

                                                        // Skip highlighting for header lines or empty content
                                                        const shouldHighlight = !line.startsWith('@@') && content.trim().length > 0;

                                                        return (
                                                          <div key={lineIndex} style={style} className="flex">
                                                             <span className="w-6 flex-shrink-0 text-right pr-2 text-gray-400">{prefix}</span>
                                                             <div className="flex-grow">
                                                               {shouldHighlight ? (
                                                                  <SyntaxHighlighter
                                                                    language={language}
                                                                    style={oneLight}
                                                                    customStyle={{ background: 'transparent', padding: 0, margin: 0, overflow: 'visible', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}
                                                                    wrapLines={true}
                                                                    lineProps={{style: {display: 'block'}}} // Ensure line breaks within highlighter
                                                                  >
                                                                    {content}
                                                                  </SyntaxHighlighter>
                                                                ) : (
                                                                  <span style={{color: style.color}}>{content}</span> // Render headers/empty lines plainly
                                                                )}
                                                             </div>
                                                          </div>
                                                        );
                                                      })}
                                                   </div>
                                               ) : (
                                                   <div className="text-xs p-2 text-gray-500 italic border-t">Patch not available.</div>
                                               )}
                                           </AccordionContent>
                                       </AccordionItem>
                                   )
                                })}
                            </Accordion>
                        </div>
                    )}
                </div>
            )
        } catch (e) {
            console.error("Error parsing or rendering get_commit result:", e)
            // Use helper to get raw content for error display
            const errorDisplay = getRawContentForError(toolResult);
            displayContent = (
                <div className="text-red-700 text-xs">
                    <p className="font-medium mb-1">Error rendering commit details:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                    <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                </div>
            );
        }
    } else if (toolName === 'list_commits' && !toolResult.error && toolResult.content) {
        try {
            let commitsList: any[];
            // Use helper to get JSON text
            const jsonText = getJsonTextContent(toolResult);
            if (jsonText) {
               commitsList = JSON.parse(jsonText);
            } else {
               // Fallback: Check if content is already an array
               if (Array.isArray(toolResult.content)) {
                   console.warn("[list_commits] Parsing direct array content.");
                   commitsList = toolResult.content;
               } else {
                   throw new Error("Commit list data is not in expected format (JSON string or array).");
               }
            }

            if (!Array.isArray(commitsList)) {
                throw new Error("Expected an array of commits.");
            }

            if (commitsList.length === 0) {
                displayContent = <p className="text-xs text-gray-600">No commits found for the specified criteria.</p>;
            } else {
                displayContent = (
                    <div className="space-y-3">
                        {commitsList.map((commit: any, index: number) => {
                            const commitId = `commit-${cell.id}-${index}`;
                            const shortSha = commit.sha?.substring(0, 7) || 'N/A';
                            const commitMessage = commit.commit?.message?.split('\\n')[0] || 'No commit message';
                            const author = commit.author;
                            const committer = commit.committer; // Could be different from author
                            const authorDate = commit.commit?.author?.date;

                            return (
                                <div key={commit.sha || index} className="border-b border-gray-200 pb-2.5 last:border-b-0 last:pb-0">
                                    <div className="flex justify-between items-start mb-1">
                                        <p className="text-sm font-medium truncate mr-2 flex-grow" title={commit.commit?.message || ''}>
                                            {commitMessage}
                                        </p>
                                        <div className="flex items-center space-x-1 flex-shrink-0">
                                            <TooltipProvider delayDuration={100}>
                                                <Tooltip>
                                                    <TooltipTrigger asChild>
                                                        <Button variant="outline" size="sm" onClick={() => copyToClipboard(commit.sha, commitId)} className="font-mono text-xs h-auto px-1.5 py-0.5">
                                                            <GitCommitIcon className="h-3 w-3 mr-1" /> {shortSha}
                                                            {copiedStates[commitId] ? <CheckIcon className="h-3 w-3 ml-1 text-green-600" /> : <CopyIcon className="h-3 w-3 ml-1" />}
                                                        </Button>
                                                    </TooltipTrigger>
                                                    <TooltipContent><p>Copy full SHA: {commit.sha}</p></TooltipContent>
                                                </Tooltip>
                                            </TooltipProvider>
                                            <a href={commit.html_url} target="_blank" rel="noopener noreferrer">
                                                <Button variant="outline" size="sm" className="text-xs h-auto px-1.5 py-0.5">
                                                    <ExternalLinkIcon className="h-3 w-3 mr-1" /> GitHub
                                                </Button>
                                            </a>
                                        </div>
                                    </div>
                                    <div className="text-xs text-gray-600 flex items-center space-x-1 flex-wrap">
                                        {author && (
                                            <TooltipProvider delayDuration={100}>
                                                <Tooltip>
                                                    <TooltipTrigger asChild>
                                                        <img src={author.avatar_url} alt={commit.commit?.author?.name || author.login || 'Author'} className="h-4 w-4 rounded-full inline-block"/>
                                                    </TooltipTrigger>
                                                    <TooltipContent>
                                                        <p>{commit.commit?.author?.name || 'N/A'} ({commit.commit?.author?.email || 'N/A'})</p>
                                                        {author.login && <p>GitHub: {author.login}</p>}
                                                    </TooltipContent>
                                                </Tooltip>
                                            </TooltipProvider>
                                        )}
                                        <span>{commit.commit?.author?.name || author?.login || 'Unknown author'}</span>
                                        <span>authored on {formatDate(authorDate)}</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                );
            }
        } catch (e) {
            console.error("Error parsing or rendering list_commits result:", e);
            // Use helper to get raw content for error display
            const errorDisplay = getRawContentForError(toolResult);

            displayContent = (
                <div className="text-red-700 text-xs">
                    <p className="font-medium mb-1">Error rendering commit list:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                    <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                </div>
            );
        }
    } else if (toolName === 'search_repositories' && !toolResult.error && toolResult.content) {
        try {
            let searchData: any;
            // Use helper to get JSON text
            const jsonText = getJsonTextContent(toolResult);
            if (jsonText) {
                 searchData = JSON.parse(jsonText);
            } else {
                 // Fallback: Check if content is already an object
                 if (typeof toolResult.content === 'object' && toolResult.content !== null) {
                     console.warn("[search_repositories] Parsing direct object content.");
                     searchData = toolResult.content;
                 } else {
                      throw new Error("Search data is not in expected format (JSON string or object).");
                 }
            }

            const totalCount = searchData.total_count;
            const items = searchData.items;

            if (!Array.isArray(items)) {
                throw new Error("Expected an array of repository items.");
            }

            displayContent = (
                <div className="text-xs">
                    {typeof totalCount === 'number' && (
                        <p className="mb-2 text-gray-700">
                            Found <span className="font-semibold">{totalCount}</span> repositories. {items.length < totalCount && `(Showing first ${items.length})`}
                        </p>
                    )}
                    {items.length === 0 ? (
                        <p className="text-gray-600">No repositories found matching your query.</p>
                    ) : (
                        <div className="overflow-x-auto border rounded-md">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Language</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Stars</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Forks</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Updated</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {items.map((repo: any) => (
                                        <tr key={repo.id}>
                                            <td className="px-3 py-1.5 whitespace-nowrap">
                                                <a href={repo.html_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium">
                                                    {repo.full_name}
                                                </a>
                                            </td>
                                            <td className="px-3 py-1.5 text-gray-700 max-w-xs truncate" title={repo.description}>{repo.description || '-'}</td>
                                            <td className="px-3 py-1.5 whitespace-nowrap text-gray-700">{repo.language || '-'}</td>
                                            <td className="px-3 py-1.5 whitespace-nowrap text-gray-700">{repo.stargazers_count}</td>
                                            <td className="px-3 py-1.5 whitespace-nowrap text-gray-700">{repo.forks_count}</td>
                                            <td className="px-3 py-1.5 whitespace-nowrap text-gray-700">{formatDate(repo.updated_at)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            );

        } catch (e) {
            console.error("Error parsing or rendering search_repositories result:", e);
            // Use helper to get raw content for error display
            const errorDisplay = getRawContentForError(toolResult);

            displayContent = (
                <div className="text-red-700 text-xs">
                    <p className="font-medium mb-1">Error rendering repository search results:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                    <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                </div>
            );
        }
    } else if (toolName === 'get_file_contents' && !toolResult.error && toolResult.content) {
        try {
            let parsedData: any;
             // Use helper to get JSON text
            const jsonText = getJsonTextContent(toolResult);
            if (jsonText) {
                 parsedData = JSON.parse(jsonText);
            } else {
                 // Fallback: Check if content is already an object or array
                 if (typeof toolResult.content === 'object' && toolResult.content !== null) {
                     console.warn("[get_file_contents] Parsing direct object/array content.");
                     parsedData = toolResult.content;
                 } else {
                     throw new Error("File/Directory data is not in expected format (JSON string or object/array).");
                 }
            }

            // Check if the result is an array (directory listing) or an object (single file)
            if (Array.isArray(parsedData)) {
                // --- Render Directory Listing ---
                const directoryItems = parsedData;
                if (directoryItems.length === 0) {
                    displayContent = <p className="text-xs text-gray-600">Directory is empty.</p>;
                } else {
                    displayContent = (
                        <div className="overflow-x-auto border rounded-md">
                            <table className="min-w-full divide-y divide-gray-200 text-xs">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Size</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Path</th>
                                        <th className="px-3 py-1.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {directoryItems.map((item: any) => (
                                        <tr key={item.path}>
                                            <td className="px-3 py-1.5 whitespace-nowrap font-medium text-gray-800 flex items-center">
                                                {item.type === 'dir' ? <FolderIcon className="h-3.5 w-3.5 mr-1.5 text-blue-500 flex-shrink-0"/> : <FileIcon className="h-3.5 w-3.5 mr-1.5 text-gray-500 flex-shrink-0"/>}
                                                <span className="truncate" title={item.name}>{item.name}</span>
                                            </td>
                                            <td className="px-3 py-1.5 whitespace-nowrap font-mono text-gray-600">{item.type}</td>
                                            <td className="px-3 py-1.5 whitespace-nowrap text-right text-gray-600">{item.size > 0 ? `${item.size} bytes` : '-'}</td>
                                            <td className="px-3 py-1.5 whitespace-nowrap font-mono text-gray-600 truncate" title={item.path}>{item.path}</td>
                                            <td className="px-3 py-1.5 whitespace-nowrap">
                                                <>
                                                {item.html_url && (
                                                    <a href={item.html_url} target="_blank" rel="noopener noreferrer">
                                                        <Button variant="ghost" size="sm" className="text-xs h-auto px-1 py-0.5">
                                                            <ExternalLinkIcon className="h-3 w-3" />
                                                        </Button>
                                                    </a>
                                                )}
                                                {item.download_url && (
                                                    <a href={item.download_url} target="_blank" rel="noopener noreferrer" download={item.name}>
                                                        <Button variant="ghost" size="sm" className="text-xs h-auto px-1 py-0.5">
                                                            <Download className="h-3 w-3" />
                                                        </Button>
                                                    </a>
                                                )}
                                                </>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    );
                }
            } else if (typeof parsedData === 'object' && parsedData !== null) {
                // --- Render Single File Content ---
                const fileData = parsedData;
                if (!fileData.path) { // Basic check for file object structure
                    throw new Error("Expected a file content object with a path.");
                }
                
                const decodedContent = fileData.encoding === 'base64' && fileData.content ? atob(fileData.content) : fileData.content || "(No content)";
                const contentPreview = decodedContent.substring(0, 300); // Show a preview
                const contentId = `file-content-${cell.id}`;
                const [showFullContent, setShowFullContent] = useState(false); // State for show/hide
                const language = getLanguageFromFilename(fileData.name || fileData.path || '');

                displayContent = (
                    <div className="space-y-3 text-xs">
                        {/* Header with name/path/links */}
                        <div className="flex justify-between items-start mb-1">
                            <p className="text-sm font-medium font-mono truncate mr-2 flex-grow" title={fileData.path}>
                                {fileData.name} ({fileData.path})
                            </p>
                            <div className="flex items-center space-x-1 flex-shrink-0">
                                {fileData.html_url && (
                                    <a href={fileData.html_url} target="_blank" rel="noopener noreferrer">
                                        <Button variant="outline" size="sm" className="text-xs h-auto px-1.5 py-0.5">
                                            <ExternalLinkIcon className="h-3 w-3 mr-1" /> GitHub
                                        </Button>
                                    </a>
                                )}
                                {fileData.download_url && (
                                    <a href={fileData.download_url} target="_blank" rel="noopener noreferrer" download={fileData.name}>
                                        <Button variant="outline" size="sm" className="text-xs h-auto px-1.5 py-0.5">
                                            <Download className="h-3 w-3 mr-1" /> Download
                                        </Button>
                                    </a>
                                )}
                            </div>
                        </div>
                        {/* Metadata Table */}
                        <div className="overflow-x-auto border rounded-md">
                            <table className="min-w-full divide-y divide-gray-200 text-xs">
                                <tbody className="bg-white divide-y divide-gray-200">
                                    <tr>
                                        <td className="px-3 py-1.5 font-medium text-gray-500">Path</td>
                                        <td className="px-3 py-1.5 font-mono text-gray-800">{fileData.path}</td>
                                    </tr>
                                    <tr>
                                        <td className="px-3 py-1.5 font-medium text-gray-500">Size</td>
                                        <td className="px-3 py-1.5 text-gray-800">{fileData.size} bytes</td>
                                    </tr>
                                    <tr>
                                        <td className="px-3 py-1.5 font-medium text-gray-500">Encoding</td>
                                        <td className="px-3 py-1.5 text-gray-800">{fileData.encoding || '-'}</td>
                                    </tr>
                                    <tr>
                                        <td className="px-3 py-1.5 font-medium text-gray-500">SHA</td>
                                        <td className="px-3 py-1.5 font-mono text-gray-800">{fileData.sha}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        
                        {/* Content Section */}
                        <div className="mt-3">
                            <div className="flex justify-between items-center mb-1">
                                <h5 className="text-xs font-medium">File Content {fileData.encoding === 'base64' ? '(Base64 Decoded)' : ''}</h5>
                            </div>
                            <SyntaxHighlighter
                                language={language}
                                style={oneLight} // Use the imported style
                                customStyle={{ maxHeight: showFullContent ? 'none' : '240px', overflowY: 'auto', margin: 0, fontSize: '0.75rem', border: '1px solid #e5e7eb', borderRadius: '0.25rem', padding: '0.375rem' }}
                                wrapLines={true}
                                lineProps={{style: {whiteSpace: 'pre-wrap', wordBreak: 'break-all'}}} >
                                {`${contentPreview}${decodedContent.length > 300 ? '\\n...' : ''}`}
                            </SyntaxHighlighter>
                        </div>
                    </div>
                );
            } else {
                // Handle case where parsedData is neither an array nor a valid file object
                throw new Error("Unexpected format for get_file_contents result.");
            }

        } catch (e) {
            console.error("Error parsing or rendering get_file_contents result:", e);
             // Use helper to get raw content for error display
             const errorDisplay = getRawContentForError(toolResult);

            displayContent = (
                <div className="text-red-700 text-xs">
                    <p className="font-medium mb-1">Error rendering file content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                    <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                </div>
            );
        }
    } else if (toolName === 'get_issue' && !toolResult.error && toolResult.content) {
        try {
            let issueData: any;
            const jsonText = getJsonTextContent(toolResult);
            if (jsonText) {
                 issueData = JSON.parse(jsonText);
            } else {
                 // Fallback: Check if content is already an object
                 if (typeof toolResult.content === 'object' && toolResult.content !== null) {
                     console.warn("[get_issue] Parsing direct object content.");
                     issueData = toolResult.content;
                 } else {
                     throw new Error("Issue data is not in expected format (JSON string or object).");
                 }
            }
            if (typeof issueData !== 'object' || issueData === null || !issueData.id) {
                throw new Error("Invalid issue data format.");
            }

            const issueId = `issue-${cell.id}`;

            displayContent = (
                <div className="space-y-3 text-xs">
                    {/* Header: Title, Number, State, Links */}
                    <div className="flex justify-between items-start mb-1">
                        <h4 className="text-base font-semibold mr-2 flex-grow">{issueData.title} (#{issueData.number})</h4>
                        <div className="flex items-center space-x-1 flex-shrink-0">
                            <Badge variant={issueData.state === 'open' ? 'default' : 'secondary'} className={`capitalize text-xs px-1.5 py-0.5 ${issueData.state === 'open' ? 'bg-green-100 text-green-800 border-green-200' : 'bg-purple-100 text-purple-800 border-purple-200'}`}>
                               {issueData.state}
                            </Badge>
                            {issueData.html_url && (
                                <a href={issueData.html_url} target="_blank" rel="noopener noreferrer">
                                    <Button variant="outline" size="sm" className="text-xs h-auto px-1.5 py-0.5">
                                        <ExternalLinkIcon className="h-3 w-3 mr-1" /> GitHub
                                    </Button>
                                </a>
                            )}
                         </div>
                    </div>

                    {/* Author & Dates */}
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between text-xs text-gray-600 gap-1">
                         {renderUser(issueData.user, issueData.created_at, "opened this issue")}
                         <span className="text-gray-500">Updated {formatDate(issueData.updated_at)}</span>
                    </div>

                     {/* Labels */}
                    {renderLabels(issueData.labels)}

                     {/* Body */}
                     {issueData.body && (
                         <div className="mt-3 border rounded-md p-2 bg-gray-50">
                             <h5 className="text-xs font-medium mb-1">Description</h5>
                             <div className="prose prose-sm max-w-none text-xs">
                                 <ReactMarkdown>{issueData.body}</ReactMarkdown>
                             </div>
                         </div>
                     )}

                    {/* Metadata Table */}
                     <div className="overflow-x-auto border rounded-md">
                         <table className="min-w-full divide-y divide-gray-200 text-xs">
                             <tbody className="bg-white divide-y divide-gray-200">
                                 <tr><td className="px-3 py-1.5 font-medium text-gray-500 w-1/4">ID</td><td className="px-3 py-1.5 text-gray-800">{issueData.id}</td></tr>
                                 <tr><td className="px-3 py-1.5 font-medium text-gray-500">Comments</td><td className="px-3 py-1.5 text-gray-800">{issueData.comments}</td></tr>
                                 <tr><td className="px-3 py-1.5 font-medium text-gray-500">Author Association</td><td className="px-3 py-1.5 text-gray-800">{issueData.author_association}</td></tr>
                                 <tr><td className="px-3 py-1.5 font-medium text-gray-500">Locked</td><td className="px-3 py-1.5 text-gray-800">{issueData.locked ? 'Yes' : 'No'}</td></tr>
                                 <tr><td className="px-3 py-1.5 font-medium text-gray-500">Reactions</td><td className="px-3 py-1.5 text-gray-800">{issueData.reactions?.total_count ?? 0}</td></tr>
                             </tbody>
                         </table>
                     </div>
                 </div>
            );
        } catch (e) {
            console.error("Error parsing or rendering get_issue result:", e);
            const errorDisplay = getRawContentForError(toolResult);
            displayContent = (
                <div className="text-red-700 text-xs">
                    <p className="font-medium mb-1">Error rendering issue details:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                    <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                </div>
            );
        }
    } else if (toolName === 'get_issue_comments' && !toolResult.error && toolResult.content) {
        try {
             let commentsData: any[];
             const jsonText = getJsonTextContent(toolResult);
              if (jsonText) {
                 commentsData = JSON.parse(jsonText);
             } else {
                 // Fallback: Check if content is already an array
                 if (Array.isArray(toolResult.content)) {
                     console.warn("[get_issue_comments] Parsing direct array content.");
                     commentsData = toolResult.content;
                 } else {
                     throw new Error("Comments data is not in expected format (JSON string or array).");
                 }
             }
             if (!Array.isArray(commentsData)) throw new Error("Expected an array of comments.");

             if (commentsData.length === 0) {
                 displayContent = <p className="text-xs text-gray-600">No comments found for this issue.</p>;
             } else {
                  displayContent = (
                      <div className="space-y-4">
                          {commentsData.map((comment: any, index: number) => (
                              <div key={comment.id || index} className="border rounded-md overflow-hidden">
                                  <div className="bg-gray-50 px-2 py-1.5 border-b flex justify-between items-center">
                                      {renderUser(comment.user, comment.created_at, "commented on")}
                                      <code className="text-xs bg-gray-100 px-1 py-0.5 rounded truncate ml-2" title={comment.path}>{comment.path}</code>
                                      {/* GitHub Link */}
                                  </div>
                                   {comment.diff_hunk && (
                                      <pre className="text-xs p-2 bg-gray-100 border-b overflow-x-auto max-h-40 font-mono">{comment.diff_hunk}</pre>
                                  )}
                                  <div className="p-2 text-xs prose prose-sm max-w-none">
                                      <ReactMarkdown>{comment.body}</ReactMarkdown>
                                  </div>
                                  {/* Reactions */}
                              </div>
                          ))}
                      </div>
                  );
              }
        } catch (e) {
            console.error("Error parsing or rendering get_issue_comments result:", e);
            const errorDisplay = getRawContentForError(toolResult);
            displayContent = (
                <div className="text-red-700 text-xs">
                    <p className="font-medium mb-1">Error rendering issue comments:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                    <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                </div>
            );
        }
    } else if (toolName === 'get_pull_request' && !toolResult.error && toolResult.content) {
        try {
            let prData: any;
             const jsonText = getJsonTextContent(toolResult);
            if (jsonText) {
                 prData = JSON.parse(jsonText);
            } else {
                 // Fallback: Check if content is already an object
                 if (typeof toolResult.content === 'object' && prData !== null) { // Typo: should be toolResult.content, and check for toolResult.content instead of prData
                     console.warn("[get_pull_request] Parsing direct object content.");
                     prData = toolResult.content;
                 } else {
                     throw new Error("Pull request data is not in expected format (JSON string or object).");
                 }
            }
             if (typeof prData !== 'object' || prData === null || !prData.id) {
                 throw new Error("Invalid pull request data format.");
            }

            const prId = `pr-${cell.id}`;
            let stateBadgeVariant: "default" | "secondary" | "outline" = "default";
            let stateBadgeColor = "bg-green-100 text-green-800 border-green-200"; // Open
            if (prData.state === 'closed') {
                if (prData.merged) {
                    stateBadgeVariant = "secondary";
                    stateBadgeColor = "bg-purple-100 text-purple-800 border-purple-200"; // Merged
                 } else {
                    stateBadgeVariant = "outline";
                    stateBadgeColor = "bg-red-100 text-red-800 border-red-200"; // Closed (not merged)
                 }
             }

             displayContent = (
                 <div className="space-y-3 text-xs">
                    {/* Header: Title, Number, State, Links */}
                     <div className="flex justify-between items-start mb-1">
                         <h4 className="text-base font-semibold mr-2 flex-grow">{prData.title} (#{prData.number})</h4>
                         <div className="flex items-center space-x-1 flex-shrink-0">
                             <Badge variant={stateBadgeVariant} className={`capitalize text-xs px-1.5 py-0.5 ${stateBadgeColor}`}>
                                 {prData.merged ? 'Merged' : prData.state}
                             </Badge>
                             {prData.html_url && (
                                 <a href={prData.html_url} target="_blank" rel="noopener noreferrer">
                                     <Button variant="outline" size="sm" className="text-xs h-auto px-1.5 py-0.5">
                                         <ExternalLinkIcon className="h-3 w-3 mr-1" /> GitHub
                                     </Button>
                                 </a>
                             )}
                         </div>
                     </div>
                     {/* Author & Dates */}
                     <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between text-xs text-gray-600 gap-1 flex-wrap">
                         {renderUser(prData.user, prData.created_at, `wants to merge ${prData.commits} commit(s) into`)}
                          <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">{prData.base?.label}</code> from <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">{prData.head?.label}</code>
                     </div>
                     {prData.merged && prData.merged_by && renderUser(prData.merged_by, prData.merged_at, "merged")}
                     {prData.closed_at && !prData.merged && <span className="text-xs text-gray-500">Closed on {formatDate(prData.closed_at)}</span>}

                    {/* Body */}
                     {prData.body && (
                         <div className="mt-3 border rounded-md p-2 bg-gray-50">
                             <h5 className="text-xs font-medium mb-1">Description</h5>
                             <div className="prose prose-sm max-w-none text-xs">
                                 <ReactMarkdown>{prData.body}</ReactMarkdown>
                             </div>
                         </div>
                     )}

                     {/* Stats & Metadata */}
                     <div className="flex flex-wrap gap-x-3 gap-y-1 items-center text-xs text-gray-700">
                         <Badge variant="secondary" className="whitespace-nowrap px-1.5 py-0.5 text-xs">{prData.commits || 0} Commits</Badge>
                         <Badge variant="secondary" className="whitespace-nowrap px-1.5 py-0.5 text-xs">{prData.changed_files || 0} Files Changed</Badge>
                         <Badge variant="outline" className="border-green-300 text-green-700 whitespace-nowrap px-1.5 py-0.5 text-xs">+{prData.additions} additions</Badge>
                         <Badge variant="outline" className="border-red-300 text-red-700 whitespace-nowrap px-1.5 py-0.5 text-xs">-{prData.deletions} deletions</Badge>
                         <span className="whitespace-nowrap">{prData.comments || 0} Comments</span>
                         <span className="whitespace-nowrap">{prData.review_comments || 0} Review Comments</span>
                     </div>

                     {/* Reviewers */}
                     {(prData.requested_reviewers?.length > 0 || prData.requested_teams?.length > 0) && (
                         <div className="text-xs">
                             <h5 className="font-medium mb-1">Reviewers</h5>
                             <div className="flex flex-wrap gap-2">
                                 {prData.requested_reviewers.map((reviewer: any) => renderUser(reviewer))}
                                 {prData.requested_teams.map((team: any) => (
                                     <span key={team.id} className="flex items-center space-x-1 text-gray-600">
                                         {/* Add a team icon if desired */}
                                         <span>{team.name}</span>
                                     </span>
                                 ))}
                             </div>
                         </div>
                     )}
                 </div>
             );
        } catch (e) {
            console.error("Error parsing or rendering get_pull_request result:", e);
            const errorDisplay = getRawContentForError(toolResult);
            displayContent = (
                <div className="text-red-700 text-xs">
                    <p className="font-medium mb-1">Error rendering pull request details:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                    <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                </div>
            );
        }
    } else if (toolName === 'get_pull_request_comments' && !toolResult.error && toolResult.content) {
        try {
             let commentsData: any[];
             // ... (parse JSON similar to get_issue_comments)
             const jsonText = getJsonTextContent(toolResult);
              if (jsonText) {
                 commentsData = JSON.parse(jsonText);
             } else {
                 // Fallback: Check if content is already an array
                 if (Array.isArray(toolResult.content)) {
                     console.warn("[get_pull_request_comments] Parsing direct array content.");
                     commentsData = toolResult.content;
                 } else {
                     throw new Error("Pull request comments data is not in expected format (JSON string or array).");
                 }
             }
             if (!Array.isArray(commentsData)) throw new Error("Expected an array of comments.");

             if (commentsData.length === 0) {
                 displayContent = <p className="text-xs text-gray-600">No review comments found for this pull request.</p>;
             } else {
                  displayContent = (
                      <div className="space-y-4">
                          {commentsData.map((comment: any, index: number) => (
                              <div key={comment.id || index} className="border rounded-md overflow-hidden">
                                  <div className="bg-gray-50 px-2 py-1.5 border-b flex justify-between items-center">
                                      {renderUser(comment.user, comment.created_at, "commented on")}
                                      <code className="text-xs bg-gray-100 px-1 py-0.5 rounded truncate ml-2" title={comment.path}>{comment.path}</code>
                                      {/* GitHub Link */}
                                  </div>
                                   {comment.diff_hunk && (
                                      <pre className="text-xs p-2 bg-gray-100 border-b overflow-x-auto max-h-40 font-mono">{comment.diff_hunk}</pre>
                                  )}
                                  <div className="p-2 text-xs prose prose-sm max-w-none">
                                      <ReactMarkdown>{comment.body}</ReactMarkdown>
                                  </div>
                                  {/* Reactions */}
                              </div>
                          ))}
                      </div>
                  );
              }
        } catch (e) {
            console.error("Error parsing or rendering get_pull_request_comments result:", e);
            const errorDisplay = getRawContentForError(toolResult);
            displayContent = (
                <div className="text-red-700 text-xs">
                    <p className="font-medium mb-1">Error rendering pull request comments:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                    <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                    <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                </div>
            );
        }
    } else if (toolName === 'get_pull_request_files' && !toolResult.error && toolResult.content) {
         try {
             let filesData: any[];
              // ... (parse JSON similar to get_issue_comments)
             const jsonText = getJsonTextContent(toolResult);
              if (jsonText) {
                 filesData = JSON.parse(jsonText);
             } else {
                 // Fallback: Check if content is already an array
                 if (Array.isArray(toolResult.content)) {
                     console.warn("[get_pull_request_files] Parsing direct array content.");
                     filesData = toolResult.content;
                 } else {
                     throw new Error("Pull request files data is not in expected format (JSON string or array).");
                 }
             }
             if (!Array.isArray(filesData)) throw new Error("Expected an array of files.");

              if (filesData.length === 0) {
                 displayContent = <p className="text-xs text-gray-600">No changed files found for this pull request.</p>;
             } else {
                 // Reuse the commit files rendering logic (Accordion)
                  displayContent = (
                      <div>
                          <h5 className="text-xs font-medium mb-1.5">{filesData.length} changed files:</h5>
                          <Accordion type="single" collapsible className="w-full border rounded-md">
                              {filesData.map((file: any, index: number) => {
                                  const fileId = `pr-file-${cell.id}-${index}`;
                                  if (!file) return null;
                                  // Simplified status badge logic compared to commit
                                  let badgeVariant: "default" | "destructive" | "outline" = "outline";
                                  let badgeClass = 'border-blue-300 text-blue-700 bg-blue-50'; // Modified
                                  if (file.status === 'added') { badgeVariant = 'default'; badgeClass = 'border-green-300 text-green-700 bg-green-50'; }
                                  if (file.status === 'removed') { badgeVariant = 'destructive'; badgeClass = 'border-red-300'; }
                                  if (file.status === 'renamed') { badgeVariant = 'outline'; badgeClass = 'border-yellow-300 text-yellow-700 bg-yellow-50'; }

                                  return (
                                      <AccordionItem value={`item-${index}`} key={file.sha || index}>
                                          <AccordionTrigger className="text-xs px-2 py-1.5 hover:bg-gray-50">
                                               {/* Reuse AccordionTrigger layout from get_commit */}
                                                <div className="flex flex-col md:flex-row justify-between items-start md:items-center w-full gap-1">
                                                     <div className="flex items-center space-x-1.5 truncate mr-4 flex-shrink min-w-0">
                                                         <Badge variant={badgeVariant} className={`text-xs px-1 py-0 leading-tight flex-shrink-0 ${badgeClass}`}>
                                                             {file.status}
                                                         </Badge>
                                                         <span className="font-mono truncate text-xs" title={file.filename}>{file.filename}</span>
                                                     </div>
                                                     <div className="flex items-center space-x-1.5 flex-shrink-0 pl-6 md:pl-0">
                                                         <span className="text-green-600 text-xs">+{file.additions}</span>
                                                         <span className="text-red-600 text-xs">-{file.deletions}</span>
                                                         {/* Copy Patch Button */}
                                                         {/* External Link Button */}
                                                     </div>
                                                 </div>
                                          </AccordionTrigger>
                                          <AccordionContent className="px-0 pb-0">
                                              {file.patch ? (
                                                  <div className="text-xs bg-gray-50 border-t overflow-x-auto max-h-80 font-mono">
                                                      {file.patch.split('\\n').map((line: string, lineIndex: number) => {
                                                        let style: React.CSSProperties = { whiteSpace: 'pre-wrap', display: 'block', paddingLeft: '0.5rem', paddingRight: '0.5rem' };
                                                        let content = line;
                                                        let prefix = '';
                                                        const language = getLanguageFromFilename(file.filename) || 'diff'; // Default to diff for highlighting + / -

                                                        if (line.startsWith('+')) {
                                                          style.backgroundColor = 'rgba(217, 249, 157, 0.4)'; // Light green
                                                          content = line.substring(1);
                                                          prefix = '+ ';
                                                        } else if (line.startsWith('-')) {
                                                          style.backgroundColor = 'rgba(254, 202, 202, 0.4)'; // Light red
                                                          content = line.substring(1);
                                                          prefix = '- ';
                                                        } else if (line.startsWith('@@')) {
                                                           style.backgroundColor = '#e5e7eb'; // Gray background for headers
                                                           style.color = '#4b5563';
                                                           content = line;
                                                           prefix = '';
                                                        } else if (line.startsWith(' ')) {
                                                          // Context line, remove leading space for highlighting
                                                          content = line.substring(1);
                                                          prefix = '  '; // Keep indentation visual
                                                        } else {
                                                          // Handle empty lines or lines without standard prefixes
                                                          content = line;
                                                          prefix = '  ';
                                                        }

                                                        // Skip highlighting for header lines or empty content
                                                        const shouldHighlight = !line.startsWith('@@') && content.trim().length > 0;

                                                        return (
                                                          <div key={lineIndex} style={style} className="flex">
                                                             <span className="w-6 flex-shrink-0 text-right pr-2 text-gray-400">{prefix}</span>
                                                             <div className="flex-grow">
                                                               {shouldHighlight ? (
                                                                  <SyntaxHighlighter
                                                                    language={language}
                                                                    style={oneLight}
                                                                    customStyle={{ background: 'transparent', padding: 0, margin: 0, overflow: 'visible', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}
                                                                    wrapLines={true}
                                                                    lineProps={{style: {display: 'block'}}} // Ensure line breaks within highlighter
                                                                  >
                                                                    {content}
                                                                  </SyntaxHighlighter>
                                                                ) : (
                                                                  <span style={{color: style.color}}>{content}</span> // Render headers/empty lines plainly
                                                                )}
                                                             </div>
                                                          </div>
                                                        );
                                                      })}
                                                   </div>
                                               ) : (
                                                   <div className="text-xs p-2 text-gray-500 italic border-t">Patch not available.</div>
                                               )}
                                           </AccordionContent>
                                       </AccordionItem>
                                   )
                              })}
                          </Accordion>
                      </div>
                 );
              }
         } catch (e) {
            console.error("Error parsing or rendering get_pull_request_files result:", e);
             const errorDisplay = getRawContentForError(toolResult);
             displayContent = (
                 <div className="text-red-700 text-xs">
                     <p className="font-medium mb-1">Error rendering pull request files:</p>
                     <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                     <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                     <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                 </div>
             );
         }
     } else if (toolName === 'get_me' && !toolResult.error && toolResult.content) {
         try {
             let userData: any;
              // Use helper to get JSON text
              const jsonText = getJsonTextContent(toolResult);
              if (jsonText) {
                  userData = JSON.parse(jsonText);
              } else {
                   // Fallback: Check if content is already an object
                  if (typeof toolResult.content === 'object' && toolResult.content !== null) {
                      console.warn("[get_me] Parsing direct object content.");
                      userData = toolResult.content;
                  } else {
                       throw new Error("User data is not in expected format (JSON string or object).");
                  }
              }


              if (typeof userData !== 'object' || userData === null) {
                  throw new Error("Expected a user object.");
              }

              displayContent = (
                  <div className="flex items-start space-x-3 text-xs">
                      {userData.avatar_url && (
                          <img src={userData.avatar_url} alt={`${userData.login} avatar`} className="h-12 w-12 rounded-full border" />
                      )}
                      <div className="flex-grow">
                          <div className="flex items-baseline space-x-1.5">
                              <h4 className="text-base font-semibold">{userData.name || userData.login}</h4>
                              {userData.login && (
                                  <a href={userData.html_url} target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-blue-600 hover:underline">
                                      (@{userData.login})
                                  </a>
                              )}
                          </div>
                          {userData.bio && <p className="mt-0.5 text-gray-700 text-xs">{userData.bio}</p>}
                          <div className="mt-1.5 flex items-center space-x-3 text-gray-600 text-xs">
                              <span>
                                  <span className="font-medium">{userData.public_repos ?? '-'}</span> Public Repos
                              </span>
                              <span>
                                  <span className="font-medium">{userData.followers ?? '-'}</span> Followers
                              </span>
                              <span>
                                  <span className="font-medium">{userData.following ?? '-'}</span> Following
                              </span>
                          </div>
                          <p className="mt-1 text-gray-500 text-xs">
                              GitHub member since {formatDate(userData.created_at)}
                          </p>
                      </div>
                  </div>
              );

          } catch (e) {
              console.error("Error parsing or rendering get_me result:", e);
               // Use helper to get raw content for error display
               const errorDisplay = getRawContentForError(toolResult);

              displayContent = (
                  <div className="text-red-700 text-xs">
                      <p className="font-medium mb-1">Error rendering user profile:</p>
                      <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
                      <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
                      <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
                  </div>
              );
          }
      } else {
          // Default case for tools without specific rendering or for errors
          if (toolResult.error) {
              // Error display logic
              displayContent = (
                  <div className="text-red-700 text-xs">
                      <p className="font-medium mb-1">Error:</p>
                      <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200 overflow-x-auto">{(()=>{
                          try {
                              // Try to parse if it looks like JSON, otherwise stringify
                              const errorContent = toolResult.error;
                               if (typeof errorContent === 'string' && errorContent.trim().startsWith('{')) return JSON.stringify(JSON.parse(errorContent), null, 2);
                               return String(errorContent);
                          } catch {
                              // Fallback if string conversion fails (shouldn't happen for string)
                              return String(toolResult.error); // Show raw error string
                          }
                      })()}</pre>
                  </div>
              )
          } else if (toolResult.content) {
              let finalContent: any = null;
              let isJson = false;

              // Step 1: Get the core content, trying to unwrap if necessary
              const jsonText = getJsonTextContent(toolResult);
              if (jsonText !== null) {
                  // Try to parse this text as JSON
                  try {
                      finalContent = JSON.parse(jsonText);
                      // isJson = true; // We'll check the type directly later
                  } catch {
                      // It wasn't JSON, keep it as a string
                      finalContent = jsonText;
                      // isJson = false;
                  }
              } else {
                   // Fallback: Assume toolResult.content is the data directly
                   console.warn("Parsing default/unknown tool result from potentially direct content structure.");
                   finalContent = toolResult.content;
              }


              // Step 2: Render the content (Simplified & Safer)
              if (typeof finalContent === 'object' && finalContent !== null) {
                  // Always render objects as formatted JSON
                  displayContent = (
                      <pre className="text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200 overflow-x-auto">
                          {JSON.stringify(finalContent, null, 2)}
                      </pre>
                  );
              } else {
                  // Render anything else (strings, numbers, booleans, null, undefined) as preformatted text
                  displayContent = (
                      <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200 overflow-x-auto">
                          {/* Use String() and handle null/undefined gracefully */}
                          {String(finalContent ?? '')}
                      </pre>
                  );
              }
          }
      }

      const cardBorderColor = toolResult.error ? "border-red-200" : "border-green-200";
      const cardBgColor = toolResult.error ? "bg-red-50" : "bg-green-50";

      return (
         <Card className={`mt-3 border ${cardBorderColor} ${!isResultExpanded ? cardBgColor : ''}`}>
           <CardHeader className={`flex flex-row items-center justify-between p-2 ${!isResultExpanded ? '' : cardBgColor}`}>
             <CardTitle className="text-sm font-semibold">
               Execution Result {shortSha ? `for ${shortSha}` : ''}
             </CardTitle>
             <Button
               variant="ghost"
               size="sm"
               onClick={() => setIsResultExpanded(!isResultExpanded)} // Local toggle still works
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
           {isResultExpanded && ( // Controlled by local state, which is synced with global
             <CardContent className="p-2 border-t">
               <div className="mt-1 rounded overflow-x-auto max-h-[450px]">
                 {displayContent}
               </div>
             </CardContent>
           )}
         </Card>
      )
  }

  // Render the tool form tabs (Now only ever one tab)
  const renderToolFormTabs = () => {
    // Only render if toolForms has been initialized
    if (!toolForms || toolForms.length === 0 || !toolForms[0].toolName) {
      // Use cell.tool_name if available and forms not yet ready
      const toolName = cell.tool_name;
      if (toolName) {
         return (
           <div className="flex items-center mb-2 border-b pb-1.5">
             <span className="px-3 py-1.5 text-sm font-medium text-green-700">
                Tool: {toolName}
             </span>
           </div>
         );
      }
      return <div className="h-10"></div>; // Placeholder or loading state
    }
    // Since there's only one tool per cell now, we just display its name as a non-clickable title
    return (
      <div className="flex items-center mb-2 border-b pb-1.5">
        <span className="px-3 py-1.5 text-sm font-medium text-green-700">
           Tool: {toolForms[0].toolName}
        </span>
        {/* Remove Add/Remove buttons */}
      </div>
    )
  }

  // --- START: Helper for rendering User Avatars/Links ---
  const renderUser = (user: any, date?: string, actionText?: string) => {
      if (!user) return null;
      const userId = `user-${cell.id}-${user.login}`;
      return (
           <div className="flex items-center space-x-1.5 text-xs text-gray-600">
             <TooltipProvider delayDuration={100}>
               <Tooltip>
                 <TooltipTrigger asChild>
                   <a href={user.html_url} target="_blank" rel="noopener noreferrer" className="flex items-center">
                     {user.avatar_url && <img src={user.avatar_url} alt={user.login} className="h-4 w-4 rounded-full inline-block mr-1"/>}
                     <span className="font-medium text-gray-800 hover:underline">{user.login}</span>
                   </a>
                 </TooltipTrigger>
                 <TooltipContent>
                   <p>{user.type}: {user.login}</p>
                 </TooltipContent>
               </Tooltip>
             </TooltipProvider>
             {actionText && <span>{actionText}</span>}
             {date && <span>on {formatDate(date)}</span>}
           </div>
      );
  }
  // --- END: Helper for rendering User Avatars/Links ---

  // --- START: Helper for rendering Labels ---
  const renderLabels = (labels: any[]) => {
     if (!labels || labels.length === 0) return null;
     return (
        <div className="flex flex-wrap gap-1 mt-1.5">
           {labels.map((label: any) => (
             <Badge
               key={label.id}
               variant="outline"
               className="text-xs px-1.5 py-0.5 font-normal border"
               // Basic contrast logic - might need refinement
               style={{
                 backgroundColor: `#${label.color}`,
                 color: isColorDark(`#${label.color}`) ? '#ffffff' : '#000000',
                 borderColor: `#${label.color}`
               }}
             >
               {label.name}
             </Badge>
           ))}
        </div>
     );
  }
  const isColorDark = (hexColor: string): boolean => {
    try {
      const color = hexColor.substring(1); // Remove #
      const r = parseInt(color.substring(0, 2), 16);
      const g = parseInt(color.substring(2, 4), 16);
      const b = parseInt(color.substring(4, 6), 16);
      // HSP equation from http://alienryderflex.com/hsp.html
      const hsp = Math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b));
      return hsp < 127.5; // Below 127.5 is considered dark
    } catch {
      return false; // Default to light if parsing fails
    }
  };
  // --- END: Helper for rendering Labels ---

  return (
    <div className="border rounded-md overflow-hidden mb-3 mx-8">
      <div className="bg-green-100 border-b border-green-200 p-2 flex justify-between items-center">
        <div className="flex items-center">
          {/* Use cell content as the title, which we set to "GitHub Tool: tool_name" */}
          <span className="font-medium text-sm text-green-800">{cell.content || "GitHub Tool"}</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <Button
            size="sm"
            variant="outline"
            className="bg-white hover:bg-green-50 border-green-300 h-7 px-2 text-xs"
            onClick={executeWithToolForms}
            disabled={isExecuting || toolForms.length === 0 || !toolForms[0]?.toolName} // Added check for toolName
          >
            <PlayIcon className="h-3.5 w-3.5 mr-1 text-green-700" />
            {isExecuting ? "Running..." : "Run Tool"}
          </Button>
          {/* Add Delete Button */}
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
               <TooltipContent>
                 <p>Delete Cell</p>
               </TooltipContent>
             </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      <div className="p-3">
        {/* Removed Original query content display */}

        {/* Tool form section */}
        <div className="mb-4">
          <Card className="border-green-200">
            <CardContent className="p-3">
              {renderToolFormTabs()}

              {/* Render inputs for the single active tool within a collapsible accordion */}
              {toolForms[0] && toolForms[0].toolName && ( // Ensure toolName exists before rendering accordion
                <Accordion 
                  type="single" 
                  collapsible 
                  className="w-full" 
                  value={argsAccordionValue} // Controlled value
                  onValueChange={setArgsAccordionValue} // Update local state
                >
                  <AccordionItem value="tool-args" className="border-b-0"> {/* Remove bottom border */}
                    <AccordionTrigger className="text-xs font-medium py-2 hover:no-underline">
                      Tool Arguments
                    </AccordionTrigger>
                    <AccordionContent className="pt-2">
                      {renderToolFormInputs(toolForms[0])}
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              )}
              {/* Show a message if no tool is selected yet */}
              {(!toolForms[0] || !toolForms[0].toolName) && (
                <p className="text-xs text-gray-500 py-2">No tool selected for this cell.</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Status indicator */} 
        {cell.status === "running" && (
          <div className="flex items-center text-xs text-amber-600 mb-3">
            <div className="animate-spin h-3 w-3 border-2 border-amber-600 rounded-full border-t-transparent mr-1.5"></div>
            Executing tool...
          </div>
        )}
        {/* Add queued status indicator */}
        {cell.status === "queued" && (
          <div className="flex items-center text-xs text-blue-600 mb-3">
             <div className="h-3 w-3 bg-blue-600 rounded-full mr-1.5 animate-pulse"></div>
             Queued for execution...
          </div>
        )}
        {/* Add stale status indicator */}
        {cell.status === "stale" && (
          <div className="flex items-center text-xs text-gray-600 mb-3">
             <div className="h-3 w-3 border border-gray-600 rounded-full mr-1.5"></div>
             Stale (needs re-run)...
          </div>
        )}

        {/* Update error check to use result.error */} 
        {/* This error is the cell-level error, not specific tool execution error if the tool ran but failed internally */}
        {cell.status === "error" && cell.result?.error && (
          <div className="bg-red-50 border border-red-200 rounded p-2 mb-3 text-red-700">
            <div className="font-medium text-xs">Execution Error:</div>
            {/* Displaying cell.result.error which might be a string or an object */}
            <div className="text-xs mt-0.5 whitespace-pre-wrap font-mono">
                {typeof cell.result.error === 'object' ? JSON.stringify(cell.result.error, null, 2) : cell.result.error}
            </div>
          </div>
        )}

        {/* Results section - Only render if not queued/running AND result exists */}
        {/* Also, ensure we don't try to render results if the cell itself is in a hard error state that prevented execution */}
        {cell.status !== 'queued' && cell.status !== 'running' && cell.result && cell.status !== 'error' && (
          <div>
            {renderResultData()} 
          </div>
        )}
        {/* If cell is in error status, but there's also a result (e.g. from a previous successful run), 
            renderResultData might still be useful if it can show partial/error info from the result itself. 
            The current logic in renderResultData handles toolResult.error, so we can try rendering it. */}
        {cell.status === 'error' && cell.result && (
            <div>
                {renderResultData()} 
            </div>
        )}
        {/* Optional: Show loading indicator when running/queued */}
        {(cell.status === 'queued' || cell.status === 'running') && !cell.result && (
           <div className="p-2 text-xs text-gray-500 flex items-center">
             <div className="animate-spin h-3 w-3 border-2 border-gray-400 rounded-full border-t-transparent mr-1.5"></div>
             Processing...
           </div>
        )}
      </div>
    </div>
  )
}

// Memoize with custom comparison to avoid re-render if relevant props unchanged
const areEqual = (prevProps: GitHubCellProps, nextProps: GitHubCellProps) => {
  const prevCell = prevProps.cell;
  const nextCell = nextProps.cell;

  // Quick reference equality check
  if (prevCell === nextCell && prevProps.isExecuting === nextProps.isExecuting) {
    return true;
  }

  // Shallow compare key fields that drive UI
  // Updated to reflect that isExecuting is a prop here.
  if (prevProps.isExecuting !== nextProps.isExecuting) return false;
  
  const keysToCheck: (keyof Cell)[] = [
    "id",
    "type",
    "status",
    "tool_name",
    // "content", // content rarely changes for GitHub cell once tool is set
    // "result", // Comparing result by reference, complex objects need careful thought
    // "tool_arguments" // Comparing tool_arguments by reference
  ];

  for (const key of keysToCheck) {
    if (prevCell[key] !== nextCell[key]) {
      return false;
    }
  }

  // Deep comparison for result and tool_arguments if they are objects
  if (JSON.stringify(prevCell.result) !== JSON.stringify(nextCell.result)) return false;
  if (JSON.stringify(prevCell.tool_arguments) !== JSON.stringify(nextCell.tool_arguments)) return false;


  return true;
};

const GitHubCell = React.memo(GitHubCellComponent, areEqual);

export default GitHubCell;
