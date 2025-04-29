"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { PlayIcon, ChevronDownIcon, ChevronUpIcon, ExternalLinkIcon, GitCommitIcon, FileDiffIcon, CopyIcon, CheckIcon, Trash2Icon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface GitHubCellProps {
  cell: {
    id: string
    type: string
    content: string
    result?: any
    status: "idle" | "running" | "success" | "error"
    error?: string
    metadata?: Record<string, any>
  }
  onExecute: (cellId: string, params?: any) => void
  onUpdate: (cellId: string, content: string, metadata?: Record<string, any>) => void
  onDelete: (cellId: string) => void
  isExecuting: boolean
}

interface ToolForm {
  toolName: string
  toolArgs: Record<string, any>
}

const GitHubCell: React.FC<GitHubCellProps> = ({ cell, onExecute, onUpdate, onDelete, isExecuting }) => {
  const [showToolCalls, setShowToolCalls] = useState(false)
  const [showMetadata, setShowMetadata] = useState(false)
  const [toolForms, setToolForms] = useState<ToolForm[]>([{ toolName: "", toolArgs: {} }])
  const [activeToolIndex, setActiveToolIndex] = useState(0)
  const [isResultExpanded, setIsResultExpanded] = useState(true)
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({})

  // Initialize the tool form based on cell metadata when the cell data changes
  useEffect(() => {
    const metadata = cell.metadata
    if (metadata?.toolName && metadata?.toolArgs) {
      const initialForm: ToolForm = {
        toolName: metadata.toolName,
        toolArgs: { ...metadata.toolArgs }, // Ensure a copy
      }
      setToolForms([initialForm]) // Set the state with a single form array
      setActiveToolIndex(0)
      console.log("GitHubCell initialized from metadata:", initialForm)
    } else {
      // Fallback if metadata is missing (should ideally not happen for agent-generated cells)
      console.warn("GitHubCell missing toolName/toolArgs in metadata for cell:", cell.id)
      setToolForms([{ toolName: "get_repository", toolArgs: { owner: "", repo: "" } }]) // Default fallback
      setActiveToolIndex(0)
    }
    // Dependency array ensures this runs when the cell's metadata potentially changes
  }, [cell.metadata?.toolName, cell.metadata?.toolArgs, cell.id])

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

    // Persist changes to metadata immediately
    const currentForm = updatedForms[toolIndex];
    onUpdate(cell.id, cell.content, { toolName: currentForm.toolName, toolArgs: currentForm.toolArgs });
  }

  // Execute the cell with the current tool forms
  const executeWithToolForms = () => {
    // Create execution parameters with the tool forms
    const params = {
      tools: toolForms.map((form) => ({
        name: form.toolName,
        args: form.toolArgs,
      })),
    }

    onExecute(cell.id, params)
  }

  // Render the tool form inputs based on the tool name
  const renderToolFormInputs = (toolForm: ToolForm, toolIndex: number) => {
    const { toolName, toolArgs } = toolForm

    switch (toolName) {
      case "get_repository":
        return (
          <>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div>
                <Label htmlFor={`owner-${toolIndex}`} className="text-xs text-gray-700">
                  Owner
                </Label>
                <Input
                  id={`owner-${toolIndex}`}
                  value={toolArgs.owner || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "owner", e.target.value)}
                  placeholder="e.g., octocat"
                  className="mt-1 h-8 text-xs"
                />
              </div>
              <div>
                <Label htmlFor={`repo-${toolIndex}`} className="text-xs text-gray-700">
                  Repository
                </Label>
                <Input
                  id={`repo-${toolIndex}`}
                  value={toolArgs.repo || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "repo", e.target.value)}
                  placeholder="e.g., hello-world"
                  className="mt-1 h-8 text-xs"
                />
              </div>
            </div>
          </>
        )

      case "list_repositories":
        return (
          <>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div>
                <Label htmlFor={`visibility-${toolIndex}`} className="text-xs text-gray-700">
                  Visibility
                </Label>
                <Select
                  value={toolArgs.visibility || "all"}
                  onValueChange={(value) => handleToolArgChange(toolIndex, "visibility", value)}
                >
                  <SelectTrigger id={`visibility-${toolIndex}`} className="mt-1 h-8 text-xs">
                    <SelectValue placeholder="Select visibility" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="public">Public</SelectItem>
                    <SelectItem value="private">Private</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor={`sort-${toolIndex}`} className="text-xs text-gray-700">
                  Sort By
                </Label>
                <Select
                  value={toolArgs.sort || "updated"}
                  onValueChange={(value) => handleToolArgChange(toolIndex, "sort", value)}
                >
                  <SelectTrigger id={`sort-${toolIndex}`} className="mt-1 h-8 text-xs">
                    <SelectValue placeholder="Select sort order" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="created">Created</SelectItem>
                    <SelectItem value="updated">Updated</SelectItem>
                    <SelectItem value="pushed">Pushed</SelectItem>
                    <SelectItem value="full_name">Name</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </>
        )

      case "search_repositories":
        return (
          <>
            <div className="mb-3">
              <Label htmlFor={`query-${toolIndex}`} className="text-xs text-gray-700">
                Search Query
              </Label>
              <Textarea
                id={`query-${toolIndex}`}
                value={toolArgs.query || ""}
                onChange={(e) => handleToolArgChange(toolIndex, "query", e.target.value)}
                placeholder="e.g., topic:react stars:>1000"
                className="mt-1 h-16 text-xs"
              />
            </div>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div>
                <Label htmlFor={`sort-${toolIndex}`} className="text-xs text-gray-700">
                  Sort By
                </Label>
                <Select
                  value={toolArgs.sort || "updated"}
                  onValueChange={(value) => handleToolArgChange(toolIndex, "sort", value)}
                >
                  <SelectTrigger id={`sort-${toolIndex}`} className="mt-1 h-8 text-xs">
                    <SelectValue placeholder="Select sort order" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="stars">Stars</SelectItem>
                    <SelectItem value="forks">Forks</SelectItem>
                    <SelectItem value="updated">Updated</SelectItem>
                    <SelectItem value="help-wanted-issues">Help Wanted Issues</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor={`per-page-${toolIndex}`} className="text-xs text-gray-700">
                  Results Per Page
                </Label>
                <Input
                  id={`per-page-${toolIndex}`}
                  type="number"
                  value={toolArgs.per_page || 10}
                  onChange={(e) =>
                    handleToolArgChange(toolIndex, "per_page", Number.parseInt(e.target.value) || 10)
                  }
                  min={1}
                  max={100}
                  className="mt-1 h-8 text-xs"
                />
              </div>
            </div>
          </>
        )

      case "list_commits":
        return (
          <>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div>
                <Label htmlFor={`owner-${toolIndex}`} className="text-xs text-gray-700">
                  Owner
                </Label>
                <Input
                  id={`owner-${toolIndex}`}
                  value={toolArgs.owner || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "owner", e.target.value)}
                  placeholder="e.g., octocat"
                  className="mt-1 h-8 text-xs"
                />
              </div>
              <div>
                <Label htmlFor={`repo-${toolIndex}`} className="text-xs text-gray-700">
                  Repository
                </Label>
                <Input
                  id={`repo-${toolIndex}`}
                  value={toolArgs.repo || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "repo", e.target.value)}
                  placeholder="e.g., hello-world"
                  className="mt-1 h-8 text-xs"
                />
              </div>
            </div>
            <div className="mb-3">
              <Label htmlFor={`path-${toolIndex}`} className="text-xs text-gray-700">
                Path (Optional)
              </Label>
              <Input
                id={`path-${toolIndex}`}
                value={toolArgs.path || ""}
                onChange={(e) => handleToolArgChange(toolIndex, "path", e.target.value)}
                placeholder="e.g., src/main.js"
                className="mt-1 h-8 text-xs"
              />
            </div>
          </>
        )
        
      case "get_commit":
        return (
          <>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div>
                <Label htmlFor={`owner-${toolIndex}`} className="text-xs text-gray-700">
                  Owner
                </Label>
                <Input
                  id={`owner-${toolIndex}`}
                  value={toolArgs.owner || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "owner", e.target.value)}
                  placeholder="e.g., octocat"
                  className="mt-1 h-8 text-xs"
                />
              </div>
              <div>
                <Label htmlFor={`repo-${toolIndex}`} className="text-xs text-gray-700">
                  Repository
                </Label>
                <Input
                  id={`repo-${toolIndex}`}
                  value={toolArgs.repo || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "repo", e.target.value)}
                  placeholder="e.g., hello-world"
                  className="mt-1 h-8 text-xs"
                />
              </div>
            </div>
            <div className="mb-3">
              <Label htmlFor={`ref-${toolIndex}`} className="text-xs text-gray-700">
                Commit Ref (SHA, Branch, Tag)
              </Label>
              <Input
                id={`ref-${toolIndex}`}
                value={toolArgs.ref || ""}
                onChange={(e) => handleToolArgChange(toolIndex, "ref", e.target.value)}
                placeholder="e.g., main, v1.0.0, c1dd4f2..."
                className="mt-1 h-8 text-xs"
              />
            </div>
          </>
        )

      case "get_me":
        return (
          <div className="mb-3 text-xs text-gray-600">
             This tool requires no arguments.
          </div>
        )

      default:
        // Generic form for any other tool
        return (
          <div className="mb-3">
            <Label htmlFor={`args-${toolIndex}`} className="text-xs text-gray-700">
              Arguments (JSON)
            </Label>
            <Textarea
              id={`args-${toolIndex}`}
              value={JSON.stringify(toolArgs, null, 2)}
              onChange={(e) => {
                try {
                  const parsedArgs = JSON.parse(e.target.value)
                  handleToolArgChange(toolIndex, "__raw_json__", parsedArgs) // Use a special key or handle directly
                } catch (err) {
                  // Invalid JSON, maybe provide feedback or just ignore
                  console.warn("Invalid JSON input for tool args")
                }
              }}
              className="mt-1 h-24 font-mono text-xs"
            />
             <p className="text-xs text-gray-500 mt-1">Edit the arguments directly in JSON format.</p>
          </div>
        )
    }
  }

  // Render the result data in a formatted way
  const renderResultData = () => {
    const toolResult = cell.result
    const toolName = toolForms[0]?.toolName

    if (!toolResult) return null

    let shortSha: string | undefined;
    if (toolName === 'get_commit' && !toolResult.isError && toolResult.content) {
      try {
        let commitDataObj: any;
        if (Array.isArray(toolResult.content) && toolResult.content[0]?.text) {
          commitDataObj = JSON.parse(toolResult.content[0].text);
        } else if (typeof toolResult.content === 'string') {
          commitDataObj = JSON.parse(toolResult.content);
        } else {
          commitDataObj = toolResult.content;
        }
        if (commitDataObj?.sha) {
           shortSha = commitDataObj.sha.substring(0, 7);
        }
      } catch {
        // Handle parsing error, shortSha remains undefined
      }
    }

    let displayContent: React.ReactNode = null

    if (toolName === 'get_commit' && !toolResult.isError && toolResult.content) {
      try {
        let commitData: any;
        if (Array.isArray(toolResult.content) && toolResult.content[0]?.text) {
          commitData = JSON.parse(toolResult.content[0].text);
        } else if (typeof toolResult.content === 'string') {
           commitData = JSON.parse(toolResult.content);
        } else {
           commitData = toolResult.content;
        }
        
        const currentShortSha = commitData.sha.substring(0, 7);
        const commitId = `commit-${cell.id}`;

        displayContent = (
          <div className="space-y-3 text-xs">
            <div className="flex justify-between items-start">
              <div>
                <h4 className="text-base font-semibold mb-1">{commitData.commit?.message?.split('\n')[0] || 'No commit message'}</h4>
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
                 {commitData.commit?.message?.includes('\n') && (
                   <pre className="mt-1.5 text-xs whitespace-pre-wrap font-sans bg-gray-50 p-1.5 rounded border">
                     {commitData.commit.message.substring(commitData.commit.message.indexOf('\n') + 1)}
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
                              <pre className="text-xs p-2 bg-gray-50 border-t overflow-x-auto max-h-60 font-mono">{file.patch}</pre>
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
         let errorDisplay = JSON.stringify(toolResult.content, null, 2);
         if (toolResult.content?.message) {
            errorDisplay = toolResult.content.message;
         }
         displayContent = (
            <div className="text-red-700 text-xs">
               <p className="font-medium mb-1">Error rendering commit details:</p>
               <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
               <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
               <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
            </div>
         );
      }
    } else if (toolName === 'list_commits' && !toolResult.isError && toolResult.content) {
       try {
         let commitsList: any[];
         if (Array.isArray(toolResult.content) && toolResult.content[0]?.text) {
           commitsList = JSON.parse(toolResult.content[0].text);
         } else if (typeof toolResult.content === 'string') {
           commitsList = JSON.parse(toolResult.content);
         } else {
           commitsList = toolResult.content;
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
                 const commitMessage = commit.commit?.message?.split('\n')[0] || 'No commit message';
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
         let errorDisplay = JSON.stringify(toolResult.content, null, 2);
         if (Array.isArray(toolResult.content) && toolResult.content[0]?.text) {
            errorDisplay = toolResult.content[0].text;
         } else if (typeof toolResult.content === 'string') {
            errorDisplay = toolResult.content;
         }

         displayContent = (
           <div className="text-red-700 text-xs">
             <p className="font-medium mb-1">Error rendering commit list:</p>
             <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
             <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
             <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
           </div>
         );
       }
    } else if (toolName === 'search_repositories' && !toolResult.isError && toolResult.content) {
      try {
        let searchData: any;
        if (Array.isArray(toolResult.content) && toolResult.content[0]?.text) {
          searchData = JSON.parse(toolResult.content[0].text);
        } else if (typeof toolResult.content === 'string') {
          searchData = JSON.parse(toolResult.content);
        } else {
          searchData = toolResult.content;
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
        let errorDisplay = JSON.stringify(toolResult.content, null, 2);
        if (Array.isArray(toolResult.content) && toolResult.content[0]?.text) {
           errorDisplay = toolResult.content[0].text;
         } else if (typeof toolResult.content === 'string') {
            errorDisplay = toolResult.content;
         }

        displayContent = (
           <div className="text-red-700 text-xs">
             <p className="font-medium mb-1">Error rendering repository search results:</p>
             <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{String(e)}</pre>
             <p className="font-medium mt-1.5 mb-1">Raw Result Content:</p>
             <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200">{errorDisplay}</pre>
           </div>
         );
      }
    } else if (toolName === 'get_me' && !toolResult.isError && toolResult.content) {
      try {
        let userData: any;
        if (Array.isArray(toolResult.content) && toolResult.content[0]?.text) {
          userData = JSON.parse(toolResult.content[0].text);
        } else if (typeof toolResult.content === 'string') {
          userData = JSON.parse(toolResult.content);
        } else {
          userData = toolResult.content;
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
        let errorDisplay = JSON.stringify(toolResult.content, null, 2);
         if (Array.isArray(toolResult.content) && toolResult.content[0]?.text) {
            errorDisplay = toolResult.content[0].text;
          } else if (typeof toolResult.content === 'string') {
             errorDisplay = toolResult.content;
          }

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
      // Default case for tools without specific rendering
      if (toolResult.isError) {
        // Error display logic
        displayContent = (
          <div className="text-red-700 text-xs">
            <p className="font-medium mb-1">Error:</p>
            <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200 overflow-x-auto">{(()=>{
              try {
                // Try to parse if it's a string, otherwise stringify directly
                const errorContent = typeof toolResult.content === 'string' ? JSON.parse(toolResult.content) : toolResult.content;
                return JSON.stringify(errorContent, null, 2);
              } catch {
                return String(toolResult.content); // fallback to string
              }
            })()}</pre>
          </div>
        )
      } else if (toolResult.content) {
        let finalContent: any = null;
        let isJson = false;

        // Step 1: Get the core content, trying to unwrap if necessary
        if (Array.isArray(toolResult.content) && toolResult.content.length === 1 && typeof toolResult.content[0]?.text === 'string') {
          finalContent = toolResult.content[0].text;
          // Try to parse this text as JSON
          try {
            finalContent = JSON.parse(finalContent);
            isJson = true;
          } catch {
            // It wasn't JSON, keep it as a string
            isJson = false;
          }
        } else if (typeof toolResult.content === 'string') {
           // Try to parse the string content as JSON
           try {
             finalContent = JSON.parse(toolResult.content);
             isJson = true;
           } catch {
             // It wasn't JSON, keep it as a string
             finalContent = toolResult.content;
             isJson = false;
           }
        } else {
           // Assume it's already an object/array (or some other type)
           finalContent = toolResult.content;
           // Consider it JSON-like if it's an object (for formatting purposes)
           isJson = typeof finalContent === 'object' && finalContent !== null;
        }

        // Step 2: Render the content
        if (isJson) {
          // Render as formatted JSON
          displayContent = (
            <pre className="text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200 overflow-x-auto">
              {JSON.stringify(finalContent, null, 2)}
            </pre>
          );
        } else {
           // Render as preformatted text
           displayContent = (
             <pre className="whitespace-pre-wrap text-xs font-mono bg-gray-50 p-1.5 rounded border border-gray-200 overflow-x-auto">
               {String(finalContent)}
             </pre>
           );
        }
      } else {
        displayContent = <span className="text-gray-500 italic text-xs">No result content available.</span>
      }
    }

    const cardBorderColor = toolResult.isError ? "border-red-200" : "border-green-200";
    const cardBgColor = toolResult.isError ? "bg-red-50" : "bg-green-50";

    return (
       <Card className={`mt-3 border ${cardBorderColor} ${!isResultExpanded ? cardBgColor : ''}`}>
         <CardHeader className={`flex flex-row items-center justify-between p-2 ${!isResultExpanded ? '' : cardBgColor}`}>
           <CardTitle className="text-sm font-semibold">
             Execution Result {toolName === 'get_commit' && shortSha ? `for ${shortSha}` : ''}
           </CardTitle>
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
         {isResultExpanded && (
           <CardContent className="p-2 border-t">
             <div className="mt-1 rounded overflow-x-auto max-h-[450px]">
               {displayContent}
             </div>
           </CardContent>
         )}
       </Card>
    )
  }

  // Render metadata (like allRepositories)
  const renderMetadata = () => {
    if (!cell.result?.metadata) return null

    return (
      <div className="mt-3">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowMetadata(!showMetadata)}
          className="mb-1.5 text-xs border-green-300 hover:bg-green-100"
        >
          {showMetadata ? (
            <>
              <ChevronUpIcon className="h-3 w-3 mr-1" />
              Hide Metadata
            </>
          ) : (
            <>
              <ChevronDownIcon className="h-3 w-3 mr-1" />
              Show Metadata
            </>
          )}
        </Button>

        {showMetadata && (
          <Card className="border-green-200 bg-white">
            <CardContent className="p-2">
              {cell.result.metadata.allRepositories && (
                <div>
                  <h4 className="text-xs font-medium mb-1.5">All Repositories</h4>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Name
                          </th>
                          <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Last Push
                          </th>
                          <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Status
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200 text-xs">
                        {cell.result.metadata.allRepositories.map((repo: any, index: number) => (
                          <tr key={index} className={index % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                            <td className="px-2 py-1 whitespace-nowrap">{repo.name}</td>
                            <td className="px-2 py-1 whitespace-nowrap">{formatDate(repo.pushed_at)}</td>
                            <td className="px-2 py-1 whitespace-nowrap">
                              {repo.status || (repo.lastCommitDate ? "Has commits" : "Unknown")}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {Object.entries(cell.result.metadata)
                .filter(([key]) => key !== "allRepositories")
                .map(([key, value]) => (
                  <div key={key} className="mt-2">
                    <h4 className="text-xs font-medium mb-1">{key}</h4>
                    <pre className="text-xs bg-gray-50 p-1.5 rounded overflow-x-auto">
                      {JSON.stringify(value, null, 2)}
                    </pre>
                  </div>
                ))}
            </CardContent>
          </Card>
        )}
      </div>
    )
  }

  // Render the tool form tabs (Now only ever one tab)
  const renderToolFormTabs = () => {
    // Only render if toolForms has been initialized
    if (!toolForms || toolForms.length === 0 || !toolForms[0].toolName) {
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

  return (
    <div className="border rounded-md overflow-hidden mb-3 max-w-7xl mx-auto">
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
            disabled={isExecuting || toolForms.length === 0}
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

              {/* Render inputs for the single active tool */}
              {toolForms[0] && renderToolFormInputs(toolForms[0], 0)}
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

        {cell.status === "error" && cell.error && (
          <div className="bg-red-50 border border-red-200 rounded p-2 mb-3 text-red-700">
            <div className="font-medium text-xs">Execution Error:</div>
            <div className="text-xs mt-0.5">{cell.error}</div>
          </div>
        )}

        {/* Results section - now shows the result of the single tool execution */}
        {(cell.status === "success" || cell.status === "error") && cell.result && (
          <div>
            {renderResultData()}
            {renderMetadata()}
          </div>
        )}
      </div>
    </div>
  )
}

export default GitHubCell
