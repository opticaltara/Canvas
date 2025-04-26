"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { PlayIcon, ChevronDownIcon, ChevronUpIcon, ExternalLinkIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

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
  const [toolForms, setToolForms] = useState<ToolForm[]>([])
  const [activeToolIndex, setActiveToolIndex] = useState(0)

  // Extract tool calls from cell result when it changes
  useEffect(() => {
    if (cell.result?.tool_calls && cell.result.tool_calls.length > 0) {
      const initialForms = cell.result.tool_calls.map((toolCall: any) => ({
        toolName: toolCall.tool_name,
        toolArgs: { ...toolCall.tool_args },
      }))
      setToolForms(initialForms)
    } else if (cell.metadata?.defaultTools) {
      // If no tool calls but we have default tools in metadata
      setToolForms(cell.metadata.defaultTools)
    } else {
      // Default empty tool form
      setToolForms([{ toolName: "get_repository", toolArgs: { owner: "", repo: "" } }])
    }
  }, [cell.result, cell.metadata])

  // Format date to be more readable
  const formatDate = (dateString: string) => {
    if (!dateString) return ""
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  // Handle form input changes
  const handleToolArgChange = (toolIndex: number, argName: string, value: any) => {
    const updatedForms = [...toolForms]
    updatedForms[toolIndex] = {
      ...updatedForms[toolIndex],
      toolArgs: {
        ...updatedForms[toolIndex].toolArgs,
        [argName]: value,
      },
    }
    setToolForms(updatedForms)
  }

  // Handle tool name change
  const handleToolNameChange = (toolIndex: number, toolName: string) => {
    const updatedForms = [...toolForms]

    // Define default args for different tool types
    let defaultArgs = {}
    switch (toolName) {
      case "get_repository":
        defaultArgs = { owner: "", repo: "" }
        break
      case "list_repositories":
        defaultArgs = { visibility: "all", sort: "updated" }
        break
      case "search_repositories":
        defaultArgs = { query: "", sort: "updated" }
        break
      case "list_commits":
        defaultArgs = { owner: "", repo: "", path: "" }
        break
      default:
        defaultArgs = {}
    }

    updatedForms[toolIndex] = {
      toolName,
      toolArgs: defaultArgs,
    }

    setToolForms(updatedForms)
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

  // Add a new tool form
  const addToolForm = () => {
    setToolForms([...toolForms, { toolName: "get_repository", toolArgs: { owner: "", repo: "" } }])
    setActiveToolIndex(toolForms.length)
  }

  // Remove a tool form
  const removeToolForm = (index: number) => {
    if (toolForms.length <= 1) return

    const updatedForms = toolForms.filter((_, i) => i !== index)
    setToolForms(updatedForms)

    if (activeToolIndex >= updatedForms.length) {
      setActiveToolIndex(Math.max(0, updatedForms.length - 1))
    }
  }

  // Render the tool form inputs based on the tool name
  const renderToolFormInputs = (toolForm: ToolForm, toolIndex: number) => {
    const { toolName, toolArgs } = toolForm

    switch (toolName) {
      case "get_repository":
        return (
          <>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <Label htmlFor={`owner-${toolIndex}`} className="text-sm text-gray-700">
                  Owner
                </Label>
                <Input
                  id={`owner-${toolIndex}`}
                  value={toolArgs.owner || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "owner", e.target.value)}
                  placeholder="e.g., octocat"
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor={`repo-${toolIndex}`} className="text-sm text-gray-700">
                  Repository
                </Label>
                <Input
                  id={`repo-${toolIndex}`}
                  value={toolArgs.repo || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "repo", e.target.value)}
                  placeholder="e.g., hello-world"
                  className="mt-1"
                />
              </div>
            </div>
          </>
        )

      case "list_repositories":
        return (
          <>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <Label htmlFor={`visibility-${toolIndex}`} className="text-sm text-gray-700">
                  Visibility
                </Label>
                <Select
                  value={toolArgs.visibility || "all"}
                  onValueChange={(value) => handleToolArgChange(toolIndex, "visibility", value)}
                >
                  <SelectTrigger id={`visibility-${toolIndex}`} className="mt-1">
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
                <Label htmlFor={`sort-${toolIndex}`} className="text-sm text-gray-700">
                  Sort By
                </Label>
                <Select
                  value={toolArgs.sort || "updated"}
                  onValueChange={(value) => handleToolArgChange(toolIndex, "sort", value)}
                >
                  <SelectTrigger id={`sort-${toolIndex}`} className="mt-1">
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
            <div className="mb-4">
              <Label htmlFor={`query-${toolIndex}`} className="text-sm text-gray-700">
                Search Query
              </Label>
              <Textarea
                id={`query-${toolIndex}`}
                value={toolArgs.query || ""}
                onChange={(e) => handleToolArgChange(toolIndex, "query", e.target.value)}
                placeholder="e.g., topic:react stars:>1000"
                className="mt-1 h-20"
              />
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <Label htmlFor={`sort-${toolIndex}`} className="text-sm text-gray-700">
                  Sort By
                </Label>
                <Select
                  value={toolArgs.sort || "updated"}
                  onValueChange={(value) => handleToolArgChange(toolIndex, "sort", value)}
                >
                  <SelectTrigger id={`sort-${toolIndex}`} className="mt-1">
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
                <Label htmlFor={`per-page-${toolIndex}`} className="text-sm text-gray-700">
                  Results Per Page
                </Label>
                <Input
                  id={`per-page-${toolIndex}`}
                  type="number"
                  value={toolArgs.per_page || 10}
                  onChange={(e) => handleToolArgChange(toolIndex, "per_page", Number.parseInt(e.target.value) || 10)}
                  min={1}
                  max={100}
                  className="mt-1"
                />
              </div>
            </div>
          </>
        )

      case "list_commits":
        return (
          <>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <Label htmlFor={`owner-${toolIndex}`} className="text-sm text-gray-700">
                  Owner
                </Label>
                <Input
                  id={`owner-${toolIndex}`}
                  value={toolArgs.owner || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "owner", e.target.value)}
                  placeholder="e.g., octocat"
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor={`repo-${toolIndex}`} className="text-sm text-gray-700">
                  Repository
                </Label>
                <Input
                  id={`repo-${toolIndex}`}
                  value={toolArgs.repo || ""}
                  onChange={(e) => handleToolArgChange(toolIndex, "repo", e.target.value)}
                  placeholder="e.g., hello-world"
                  className="mt-1"
                />
              </div>
            </div>
            <div className="mb-4">
              <Label htmlFor={`path-${toolIndex}`} className="text-sm text-gray-700">
                Path (Optional)
              </Label>
              <Input
                id={`path-${toolIndex}`}
                value={toolArgs.path || ""}
                onChange={(e) => handleToolArgChange(toolIndex, "path", e.target.value)}
                placeholder="e.g., src/main.js"
                className="mt-1"
              />
            </div>
          </>
        )

      default:
        // Generic form for any other tool
        return (
          <div className="mb-4">
            <Label htmlFor={`args-${toolIndex}`} className="text-sm text-gray-700">
              Arguments (JSON)
            </Label>
            <Textarea
              id={`args-${toolIndex}`}
              value={JSON.stringify(toolArgs, null, 2)}
              onChange={(e) => {
                try {
                  const parsedArgs = JSON.parse(e.target.value)
                  setToolForms((forms) => {
                    const updated = [...forms]
                    updated[toolIndex] = {
                      ...updated[toolIndex],
                      toolArgs: parsedArgs,
                    }
                    return updated
                  })
                } catch (err) {
                  // Invalid JSON, don't update
                }
              }}
              className="mt-1 h-32 font-mono text-sm"
            />
          </div>
        )
    }
  }

  // Render the result data in a formatted way
  const renderResultData = () => {
    if (!cell.result?.data) return null

    const data = cell.result.data
    return (
      <Card className="mt-4 border-green-200 bg-green-50">
        <CardContent className="p-4">
          <h3 className="text-lg font-semibold mb-2">{data.repositoryName || "Repository"}</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-4">
            {data.owner && (
              <div>
                <span className="text-sm font-medium text-gray-500">Owner:</span>
                <span className="ml-2">{data.owner}</span>
              </div>
            )}

            {data.lastCommitDate && (
              <div>
                <span className="text-sm font-medium text-gray-500">Last Commit:</span>
                <span className="ml-2">{formatDate(data.lastCommitDate)}</span>
              </div>
            )}
          </div>

          {data.description && (
            <div className="mb-2">
              <span className="text-sm font-medium text-gray-500">Description:</span>
              <p className="text-sm mt-1">{data.description}</p>
            </div>
          )}

          {data.lastCommitMessage && (
            <div className="mb-2">
              <span className="text-sm font-medium text-gray-500">Last Commit Message:</span>
              <p className="text-sm mt-1 font-mono bg-white p-2 rounded border">{data.lastCommitMessage}</p>
            </div>
          )}

          {data.url && (
            <div className="mt-3">
              <a
                href={data.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-sm text-green-700 hover:text-green-900"
              >
                View on GitHub
                <ExternalLinkIcon className="ml-1 h-3 w-3" />
              </a>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }

  // Render the tool calls history
  const renderToolCalls = () => {
    if (!cell.result?.tool_calls || cell.result.tool_calls.length === 0) return null

    return (
      <div className="mt-4">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowToolCalls(!showToolCalls)}
          className="mb-2 text-xs border-green-300 hover:bg-green-100"
        >
          {showToolCalls ? (
            <>
              <ChevronUpIcon className="h-3 w-3 mr-1" />
              Hide Tool Call History
            </>
          ) : (
            <>
              <ChevronDownIcon className="h-3 w-3 mr-1" />
              Show Tool Call History ({cell.result.tool_calls.length})
            </>
          )}
        </Button>

        {showToolCalls && (
          <div className="space-y-3">
            {cell.result.tool_calls.map((toolCall: any, index: number) => (
              <Card key={toolCall.tool_call_id || index} className="border-green-200 bg-white">
                <CardContent className="p-3">
                  <div className="flex items-center mb-2">
                    <Badge className="bg-green-600 text-white">{toolCall.tool_name}</Badge>
                  </div>

                  {Object.keys(toolCall.tool_args).length > 0 && (
                    <div className="mb-2">
                      <span className="text-xs font-medium text-gray-500">Arguments:</span>
                      <pre className="text-xs mt-1 bg-gray-50 p-2 rounded overflow-x-auto">
                        {JSON.stringify(toolCall.tool_args, null, 2)}
                      </pre>
                    </div>
                  )}

                  {toolCall.tool_result && (
                    <div>
                      <span className="text-xs font-medium text-gray-500">Result:</span>
                      <div className="text-xs mt-1 bg-gray-50 p-2 rounded overflow-x-auto max-h-40">
                        {toolCall.tool_result.isError ? (
                          <span className="text-red-500">Error: {JSON.stringify(toolCall.tool_result.content)}</span>
                        ) : (
                          <div className="overflow-auto">
                            {toolCall.tool_result.content && toolCall.tool_result.content[0]?.text ? (
                              <pre className="whitespace-pre-wrap">
                                {(() => {
                                  try {
                                    const parsed = JSON.parse(toolCall.tool_result.content[0].text)
                                    return (
                                      JSON.stringify(parsed, null, 2).substring(0, 500) +
                                      (JSON.stringify(parsed).length > 500 ? "..." : "")
                                    )
                                  } catch (e) {
                                    return (
                                      toolCall.tool_result.content[0].text.substring(0, 500) +
                                      (toolCall.tool_result.content[0].text.length > 500 ? "..." : "")
                                    )
                                  }
                                })()}
                              </pre>
                            ) : (
                              <span>No content available</span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    )
  }

  // Render metadata (like allRepositories)
  const renderMetadata = () => {
    if (!cell.result?.metadata) return null

    return (
      <div className="mt-4">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowMetadata(!showMetadata)}
          className="mb-2 text-xs border-green-300 hover:bg-green-100"
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
            <CardContent className="p-3">
              {cell.result.metadata.allRepositories && (
                <div>
                  <h4 className="text-sm font-medium mb-2">All Repositories</h4>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Name
                          </th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Last Push
                          </th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Status
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {cell.result.metadata.allRepositories.map((repo: any, index: number) => (
                          <tr key={index} className={index % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                            <td className="px-3 py-2 whitespace-nowrap text-xs">{repo.name}</td>
                            <td className="px-3 py-2 whitespace-nowrap text-xs">{formatDate(repo.pushed_at)}</td>
                            <td className="px-3 py-2 whitespace-nowrap text-xs">
                              {repo.status || (repo.lastCommitDate ? "Has commits" : "Unknown")}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Render other metadata if present */}
              {Object.entries(cell.result.metadata)
                .filter(([key]) => key !== "allRepositories")
                .map(([key, value]) => (
                  <div key={key} className="mt-3">
                    <h4 className="text-sm font-medium mb-1">{key}</h4>
                    <pre className="text-xs bg-gray-50 p-2 rounded overflow-x-auto">
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

  // Render the tool form tabs
  const renderToolFormTabs = () => {
    return (
      <div className="flex overflow-x-auto mb-2 border-b">
        {toolForms.map((form, index) => (
          <button
            key={index}
            className={`px-3 py-2 text-sm font-medium whitespace-nowrap ${
              activeToolIndex === index
                ? "text-green-700 border-b-2 border-green-500"
                : "text-gray-500 hover:text-gray-700"
            }`}
            onClick={() => setActiveToolIndex(index)}
          >
            {form.toolName}
            {toolForms.length > 1 && (
              <span
                className="ml-2 text-gray-400 hover:text-red-500"
                onClick={(e) => {
                  e.stopPropagation()
                  removeToolForm(index)
                }}
              >
                ×
              </span>
            )}
          </button>
        ))}
        <button className="px-3 py-2 text-sm font-medium text-green-600 hover:text-green-800" onClick={addToolForm}>
          + Add Tool
        </button>
      </div>
    )
  }

  return (
    <div className="border rounded-md overflow-hidden">
      <div className="bg-green-100 border-b border-green-200 p-3 flex justify-between items-center">
        <div className="flex items-center">
          <span className="font-medium text-green-800">GitHub Query</span>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            size="sm"
            variant="outline"
            className="bg-white hover:bg-green-50 border-green-300"
            onClick={executeWithToolForms}
            disabled={isExecuting}
          >
            <PlayIcon className="h-4 w-4 mr-1 text-green-700" />
            {isExecuting ? "Running..." : "Run"}
          </Button>
        </div>
      </div>

      <div className="p-4">
        {/* Original query content */}
        <div className="font-medium mb-2">Query:</div>
        <div className="bg-white border border-gray-200 rounded p-3 mb-4">{cell.content}</div>

        {/* Tool form section */}
        <div className="mb-6">
          <div className="font-medium mb-2">GitHub Tools:</div>
          <Card className="border-green-200">
            <CardContent className="p-4">
              {renderToolFormTabs()}

              <div className="mb-4">
                <Label htmlFor="tool-name" className="text-sm text-gray-700">
                  Tool
                </Label>
                <Select
                  value={toolForms[activeToolIndex]?.toolName || ""}
                  onValueChange={(value) => handleToolNameChange(activeToolIndex, value)}
                >
                  <SelectTrigger id="tool-name" className="mt-1">
                    <SelectValue placeholder="Select a tool" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="get_repository">Get Repository</SelectItem>
                    <SelectItem value="list_repositories">List Repositories</SelectItem>
                    <SelectItem value="search_repositories">Search Repositories</SelectItem>
                    <SelectItem value="list_commits">List Commits</SelectItem>
                    <SelectItem value="get_me">Get Current User</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {toolForms[activeToolIndex] && renderToolFormInputs(toolForms[activeToolIndex], activeToolIndex)}

              <Button
                onClick={executeWithToolForms}
                className="bg-green-600 hover:bg-green-700 text-white"
                disabled={isExecuting}
              >
                {isExecuting ? "Running..." : "Execute GitHub Query"}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Status indicator */}
        {cell.status === "running" && (
          <div className="flex items-center text-sm text-amber-600 mb-4">
            <div className="animate-spin h-3 w-3 border-2 border-amber-600 rounded-full border-t-transparent mr-2"></div>
            Executing GitHub query...
          </div>
        )}

        {cell.status === "error" && (
          <div className="bg-red-50 border border-red-200 rounded p-3 mb-4 text-red-700">
            <div className="font-medium">Error:</div>
            <div className="text-sm">{cell.error || "An error occurred while executing the GitHub query."}</div>
          </div>
        )}

        {/* Results section */}
        {cell.status === "success" && cell.result && (
          <div>
            {/* Main result data */}
            {renderResultData()}

            {/* Tool calls */}
            {renderToolCalls()}

            {/* Metadata */}
            {renderMetadata()}
          </div>
        )}
      </div>
    </div>
  )
}

export default GitHubCell
