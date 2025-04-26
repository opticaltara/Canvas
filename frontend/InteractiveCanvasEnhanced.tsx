"use client"

import React, { useState, useEffect, useRef } from "react"
import QuestionAnswerUI from "./QuestionAnswerUI"
import LeftSidebar from "./components/LeftSidebar"
import EmptyCanvas from "./components/EmptyCanvas"
import IntegrationsPage from "./pages/integrations"
import Cell from "./components/Cell"
import CellCreationPills from "./components/CellCreationPills"
import ChatWindow from "./components/ChatWindow"
import CanvasHeader from "./components/CanvasHeader"

interface Canvas {
  id: string
  name: string
  content: any[]
}

const InteractiveCanvasEnhanced: React.FC = () => {
  // Update the canvas name
  const [canvasName, setCanvasName] = useState("Payments Failure Investigation")

  // Update the cells array with the new content
  const [cells, setCells] = useState([
    {
      type: "markdown",
      content:
        "# Payments Failure Investigation\n\nThis notebook is for investigating the recent payment service failures. Initial alert: **Payment failures increased by 2x over the last 2 hours.**",
      status: "success",
      creator: "NM",
    },
    {
      type: "markdown",
      content: "## User Impact Analysis\n\nLet's first determine the scope of the impact on our users.",
      status: "success",
      creator: "SS",
    },
    {
      type: "datadog",
      content: "index:main-logs service:payment-processor status:failed | count | limit 10",
      status: "success",
      output: ["Total failed payments in the last 2 hours: 1500"],
      creator: "SS",
      question: "How many payment failures have we had in the last 2 hours?",
      answer:
        "I've queried the logs for failed payments in the last 2 hours. The total count is 1500 failed payments, which is significantly higher than our normal baseline.",
    },
    {
      type: "splunk",
      content:
        "index=production sourcetype=payment-service status=failed | stats dc(user_id) as affected_users | table affected_users",
      status: "success",
      output: ["affected_users", "312"],
      creator: "NM",
      question: "How many unique users were affected by the payment failures?",
      answer:
        "I've analyzed the logs to count distinct user IDs that experienced payment failures. There are 312 unique users affected by this incident.",
    },
    {
      type: "sql",
      content:
        'SELECT c.customer_name, c.tier\nFROM customers c\nJOIN payment_events p ON c.customer_id = p.customer_id\nWHERE p.status = "failed" \nAND p.timestamp >= NOW() - INTERVAL 2 HOUR\nAND c.tier = "top_20"\nORDER BY c.customer_name;',
      status: "success",
      output: [
        { customer_name: "ABC Corp", tier: "top_20" },
        { customer_name: "XYZ Industries", tier: "top_20" },
      ],
      creator: "SS",
      question: "Are any of our top customers affected by this issue?",
      answer:
        "I've identified that 2 of our Top 20 customers have been affected by the payment failures: ABC Corp and XYZ Industries. This indicates a high-severity incident that requires immediate attention.",
    },
    {
      type: "markdown",
      content: "## Error Analysis\n\nNow let's examine the error logs to identify the root cause.",
      status: "success",
      creator: "NM",
    },
    {
      type: "datadog",
      content: 'index:main-logs service:"Payment Service" status:error | sort by timestamp desc | limit 20',
      status: "success",
      output: [
        "2023-05-01 14:10:23 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:09:45 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:08:12 ERROR [Payment Service] Timeout Error: Payment gateway did not respond",
        "2023-05-01 14:07:55 ERROR [Payment Service] Database Error: Connection reset during transaction",
        "2023-05-01 14:07:30 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:06:22 ERROR [Payment Service] Timeout Error: Payment gateway did not respond",
        "2023-05-01 14:05:18 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:05:03 ERROR [Payment Service] Database Error: Connection reset during transaction",
        "2023-05-01 14:04:45 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:03:59 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:03:22 ERROR [Payment Service] Timeout Error: Payment gateway did not respond",
        "2023-05-01 14:02:47 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:02:10 ERROR [Payment Service] Database Error: Connection reset during transaction",
        "2023-05-01 14:01:38 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:01:05 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 14:00:42 ERROR [Payment Service] Timeout Error: Payment gateway did not respond",
        "2023-05-01 14:00:17 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 13:59:50 ERROR [Payment Service] Database Error: Connection reset during transaction",
        "2023-05-01 13:59:22 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
        "2023-05-01 13:58:45 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
      ],
      creator: "Sherlog",
      question: "Can you retrieve the latest error logs for the Payment Service?",
      answer:
        "I've retrieved the most recent error logs from the Payment Service. There appear to be three main types of errors occurring: config mismatches, timeout errors, and database connection resets.",
    },
    {
      type: "python",
      content: `import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Parse the logs (in a real scenario, this would come from the previous query)
logs = [
    "2023-05-01 14:10:23 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:09:45 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:08:12 ERROR [Payment Service] Timeout Error: Payment gateway did not respond",
    "2023-05-01 14:07:55 ERROR [Payment Service] Database Error: Connection reset during transaction",
    "2023-05-01 14:07:30 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:06:22 ERROR [Payment Service] Timeout Error: Payment gateway did not respond",
    "2023-05-01 14:05:18 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:05:03 ERROR [Payment Service] Database Error: Connection reset during transaction",
    "2023-05-01 14:04:45 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:03:59 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:03:22 ERROR [Payment Service] Timeout Error: Payment gateway did not respond",
    "2023-05-01 14:02:47 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:02:10 ERROR [Payment Service] Database Error: Connection reset during transaction",
    "2023-05-01 14:01:38 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:01:05 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 14:00:42 ERROR [Payment Service] Timeout Error: Payment gateway did not respond",
    "2023-05-01 14:00:17 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 13:59:50 ERROR [Payment Service] Database Error: Connection reset during transaction",
    "2023-05-01 13:59:22 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml",
    "2023-05-01 13:58:45 ERROR [Payment Service] Error 502: Config mismatch detected in payment_config.yml"
]

# Extract error types
error_types = []
for log in logs:
    if "Config mismatch" in log:
        error_types.append("Config mismatch")
    elif "Timeout Error" in log:
        error_types.append("Timeout Error")
    elif "Database Error" in log:
        error_types.append("Database Error")
    else:
        error_types.append("Other")

# Count occurrences
error_counts = Counter(error_types)

# Create a DataFrame for visualization
df = pd.DataFrame({
    'Error Type': list(error_counts.keys()),
    'Count': list(error_counts.values())
})

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['Error Type'], df['Count'], color=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Error Distribution in Payment Service')
plt.xlabel('Error Type')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(df['Count']):
    plt.text(i, v + 0.5, str(v), ha='center')

plt.tight_layout()
plt.show()

# Print anomalous log entries with analysis
print("Anomalous Log Entries:")
print("Error 502: Config mismatch detected in payment_config.yml at 2:05 PM (recorded over 1000 times in the last 2 hours)")
print("Timeout Error: Payment gateway did not respond at 2:06 PM")
print("Database Error: Connection reset during transaction at 2:07 PM")`,
      status: "success",
      output: "chart",
      creator: "SS",
      question: "Can you analyze the error logs to identify anomalous patterns?",
      answer:
        "I've analyzed the error logs and identified three main types of errors. The Config mismatch error is by far the most frequent, occurring over 1000 times in the last 2 hours. This is significantly above the baseline and indicates a potential configuration issue.",
      chartData: {
        data: [
          { "Error Type": "Config mismatch", Count: 12 },
          { "Error Type": "Timeout Error", Count: 4 },
          { "Error Type": "Database Error", Count: 4 },
        ],
        type: "bar",
        categories: ["Count"],
        index: "Error Type",
        title: "Error Distribution in Payment Service",
      },
    },
    {
      type: "markdown",
      content:
        "## Anomalous Log Entries\n\n- **Error 502: Config mismatch detected in payment_config.yml at 2:05 PM** (recorded over 1000 times in the last 2 hours)\n  - *This log deviates 50% from baseline error rates and indicates backward incompatibility in configuration.*\n\n- **Timeout Error: Payment gateway did not respond at 2:06 PM**\n  - *Spike in timeout errors observed compared to normal operation.*\n\n- **Database Error: Connection reset during transaction at 2:07 PM**\n  - *Unusual frequency of connection resets noted during peak load.*",
      status: "success",
      creator: "Sherlog",
    },
    {
      type: "markdown",
      content:
        "## Deployment Analysis\n\nLet's check if recent deployments might be related to the config mismatch errors.",
      status: "success",
      creator: "NM",
    },
    {
      type: "ai",
      content:
        "The config mismatch error typically occurs during new releases based on previous remediations. Would you like me to check recent deployments?",
      status: "success",
      output: "Yes",
      creator: "Sherlog",
    },
    {
      type: "datadog",
      content: "index:deployments service:payment-service | sort by timestamp desc | limit 5",
      status: "success",
      output: [
        "Deployment & Release Status: Release config_us_v11 was deployed 2 hours ago. Error rates began spiking within 15 minutes after deployment.",
        "Previous deployment: config_us_v10 was deployed 2 days ago with no reported issues.",
        "Deployment config_us_v11 included changes to payment gateway configuration parameters.",
        "Deployment performed by: dev-ops-team",
        "Automated tests passed before deployment.",
      ],
      creator: "Sherlog",
      question: "Can you check recent deployments to the payment service?",
      answer:
        "I've checked the deployment logs and found that Release config_us_v11 was deployed 2 hours ago, which aligns with when our error rates began spiking. This strongly suggests that the deployment is related to our current issues.",
    },
    {
      type: "python",
      content: `# Filter errors per device type for Payment Service
import pandas as pd
import matplotlib.pyplot as plt

def query_errors(service, time_range):
    # In a real scenario, this would query a database or log system
    # For this example, we'll return mock data
    if service == "Payment Service" and time_range == "last 2 hours":
        return {
            "iOS": 800,
            "Android": 200,
            "Web": 500
        }
    return {}

errors = query_errors(service="Payment Service", time_range="last 2 hours")

# Create a DataFrame for visualization
df = pd.DataFrame({
    'Platform': list(errors.keys()),
    'Error Count': list(errors.values())
})

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['Platform'], df['Error Count'], color=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Payment Service Errors by Platform')
plt.xlabel('Platform')
plt.ylabel('Error Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(df['Error Count']):
    plt.text(i, v + 20, str(v), ha='center')

plt.tight_layout()
plt.show()

# Display the results
print("iOS errors: 800; Android errors: 200; Web errors: 500")`,
      status: "success",
      output: "chart",
      creator: "NM",
      question: "Can you break down the payment errors by device type?",
      answer:
        "I've analyzed the payment errors by platform and found that iOS devices are experiencing the highest number of failures (800), followed by Web (500) and Android (200). This suggests the issue might be more severe on iOS clients.",
      chartData: {
        data: [
          { Platform: "iOS", "Error Count": 800 },
          { Platform: "Android", "Error Count": 200 },
          { Platform: "Web", "Error Count": 500 },
        ],
        type: "bar",
        categories: ["Error Count"],
        index: "Platform",
        title: "Payment Service Errors by Platform",
      },
    },
    {
      type: "markdown",
      content:
        "## Root Cause Analysis\n\nBased on our investigation, the payment failures appear to be caused by a configuration mismatch introduced in the `config_us_v11` deployment. The config change is causing errors across all platforms, with iOS clients being the most affected.\n\n### Next Steps:\n\n1. Roll back to `config_us_v10` immediately\n2. Notify affected customers, prioritizing ABC Corp and XYZ Industries\n3. Review deployment process to ensure configuration compatibility checks",
      status: "success",
      creator: "SS",
    },
  ])

  // Update the canvases array to reflect the new name
  const [canvases, setCanvases] = useState<Canvas[]>([
    { id: "1", name: "Payments Failure Investigation", content: cells },
    { id: "2", name: "New Feature Rollout", content: [] },
  ])
  const [highlightedCell, setHighlightedCell] = useState(null)
  const [currentQuestion, setCurrentQuestion] = useState(null)
  const [currentAnswer, setCurrentAnswer] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)
  const [workspace, setWorkspace] = useState("My Workspace")
  const [currentPage, setCurrentPage] = useState("canvas")
  const [isGeneratingCell, setIsGeneratingCell] = useState(false)

  const cellRefs = useRef<(HTMLDivElement | null)[]>([])

  useEffect(() => {
    if (highlightedCell !== null && cellRefs.current[highlightedCell]) {
      cellRefs.current[highlightedCell]?.scrollIntoView({ behavior: "smooth", block: "center" })
    }
  }, [highlightedCell])

  const handleAddCanvas = (name: string) => {
    const newCanvas: Canvas = { id: String(canvases.length + 1), name, content: [] }
    setCanvases([...canvases, newCanvas])
  }

  const handleSelectCanvas = (id: string) => {
    const selectedCanvas = canvases.find((canvas) => canvas.id === id)
    if (selectedCanvas) {
      setCanvasName(selectedCanvas.name)
      setCells(selectedCanvas.content)
      setCurrentPage("canvas")
    }
  }

  const handleSignOut = () => {
    console.log("Sign out")
    // Implement sign out logic here
  }

  const handleSendMessage = (message: string) => {
    setCurrentQuestion(message)
    setIsStreaming(true)
    setCurrentAnswer("")
    const answer =
      "I've queried Splunk for the most recent error logs across our production services. Here are the 5 most recent error logs:"
    let index = 0
    const interval = setInterval(() => {
      if (index < answer.length) {
        setCurrentAnswer((prev) => prev + answer[index])
        index++
      } else {
        clearInterval(interval)
        setIsStreaming(false)
        createSplunkCell()
      }
    }, 50)
  }

  const createSplunkCell = () => {
    const newCell = {
      type: "splunk",
      content: "index=production sourcetype=application error | head 5",
      status: "success",
      output: [],
      creator: "Sherlog",
    }
    setCells((prevCells) => [...prevCells, newCell])
    setTimeout(() => {
      const element = document.getElementById(`cell-${cells.length}`)
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "end" })
      }
    }, 100)
  }

  const handleCellRun = (index: number) => {
    const updatedCells = [...cells]
    const cell = updatedCells[index]
    cell.status = "running"
    cell.output = []
    setCells(updatedCells)

    // Simulate different behaviors based on cell type
    switch (cell.type) {
      case "sql":
      case "promql":
        // Simulate query execution
        setTimeout(() => {
          cell.status = "success"
          cell.output = [
            { column1: "value1", column2: "value2" },
            { column1: "value3", column2: "value4" },
          ]
          setCells([...updatedCells])
        }, 2000)
        break
      case "python":
        // Simulate Python execution
        setTimeout(() => {
          cell.status = "success"
          cell.output = "chart" // Assuming we want to show a chart for Python cells
          setCells([...updatedCells])
        }, 2000)
        break
      case "datadog":
      case "splunk":
        // Simulate log streaming
        const logs = [
          "2023-05-01 17:15:23 ERROR [order-processor] Failed to process order #3456: Payment gateway timeout",
          "2023-05-01 17:14:12 ERROR [inventory-service] Stock update failed for product ID 7890: Database lock timeout",
          "2023-05-01 17:13:01 ERROR [shipping-service] Label generation failed for order #2468: API rate limit exceeded",
          "2023-05-01 17:11:55 ERROR [user-auth] Failed login attempt for user 'jdoe': Account locked",
          "2023-05-01 17:10:30 ERROR [email-service] Failed to send notification: SMTP server not responding",
        ]

        let logIndex = 0
        const logInterval = setInterval(() => {
          if (logIndex < logs.length) {
            cell.output = [...cell.output, logs[logIndex]]
            setCells([...updatedCells])
            logIndex++
          } else {
            clearInterval(logInterval)
            cell.status = "success"
            setCells([...updatedCells])
          }
        }, 1000)
        break
      default:
        // For other cell types (like markdown), just set status to success
        cell.status = "success"
        setCells([...updatedCells])
    }
  }

  const handleAddCell = (type: string, index: number) => {
    const newCell = {
      type,
      content: `# New ${type} cell`,
      status: "success",
      creator: "User",
    }
    const updatedCells = [...cells]
    updatedCells.splice(index, 0, newCell)
    setCells(updatedCells)
  }

  const handleDeleteCell = (index: number) => {
    const updatedCells = cells.filter((_, i) => i !== index)
    setCells(updatedCells)
  }

  const handleCellContentChange = (index: number, newContent: string) => {
    const updatedCells = [...cells]
    updatedCells[index].content = newContent
    setCells(updatedCells)
  }

  const handleEditWithSherlog = (index: number) => {
    // Implement the logic to edit the cell with Sherlog
    console.log(`Editing cell ${index} with Sherlog`)
    // You can add more complex logic here, such as opening a modal or sending a request to an AI service
  }

  const handleCommentClick = (cellIndex: number) => {
    setHighlightedCell(cellIndex)

    // Remove the highlight after a short delay
    setTimeout(() => setHighlightedCell(null), 2000)
  }

  const handleAddComment = (cellIndex: number | null) => {
    //setActiveCommentBox(cellIndex)
  }

  const handleCommentSubmit = (cellIndex: number, commentText: string) => {
    //const newComment = {
    //  author: Math.random() > 0.5 ? "Navneet" : "Sidharth",
    //  text: commentText,
    //  cellIndex,
    //  textRange: { start: 0, end: 10 }, // This is a placeholder. In a real implementation, you'd determine the actual text range.
    //}
    //setComments([...comments, newComment])
    //setActiveCommentBox(null)
  }

  const handleCanvasNameChange = (newName: string) => {
    setCanvasName(newName)
  }

  const handleNavigateToIntegrations = () => {
    setCurrentPage("integrations")
  }

  const handleGenerateCell = (cellType: string, content: string) => {
    setIsGeneratingCell(true)
    const newCell = {
      type: cellType,
      content: content,
      status: "success",
      output: null,
      creator: "Sherlog",
    }
    setCells((prevCells) => [...prevCells, newCell])
    setIsGeneratingCell(false)
  }

  // Handle streaming cell completion
  const handleStreamingComplete = () => {
    console.log("Streaming completed")
  }

  return (
    <div className="h-screen flex flex-col">
      <div className="flex-1 flex overflow-hidden">
        <LeftSidebar
          workspace={workspace}
          canvases={canvases || []}
          onAddCanvas={handleAddCanvas}
          onSelectCanvas={handleSelectCanvas}
          onSignOut={handleSignOut}
          onNavigateToIntegrations={handleNavigateToIntegrations}
        />
        <div className="flex-1 flex flex-col overflow-hidden">
          {currentPage === "canvas" ? (
            <>
              <CanvasHeader name={canvasName} onNameChange={handleCanvasNameChange} />
              <div className="flex-1 overflow-y-auto p-4">
                {cells && cells.length > 0 ? (
                  <>
                    {cells.map((cell, index) => (
                      <React.Fragment key={index}>
                        {cell.question && cell.answer && (
                          <QuestionAnswerUI question={cell.question} answer={cell.answer} isStreaming={false} />
                        )}
                        <Cell
                          {...cell}
                          id={`cell-${index}`}
                          onDelete={() => handleDeleteCell(index)}
                          onContentChange={(newContent) => handleCellContentChange(index, newContent)}
                          onEditWithSherlog={() => handleEditWithSherlog(index)}
                          onRun={() => handleCellRun(index)}
                          onAddComment={() => {}} // Empty function since we're removing comments
                          onShareToSlack={() => {}}
                          hasComments={false} // Always false since we're removing comments
                          isHighlighted={index === highlightedCell}
                          cellRef={(el) => (cellRefs.current[index] = el)}
                          chartData={cell.chartData}
                        />
                        <CellCreationPills onAddCell={(type) => handleAddCell(type, index + 1)} />
                      </React.Fragment>
                    ))}
                    <CellCreationPills onAddCell={(type) => handleAddCell(type, cells.length)} />
                  </>
                ) : (
                  <EmptyCanvas />
                )}
                {isGeneratingCell && (
                  <div className="flex items-center justify-center p-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    <span className="ml-2 text-blue-500">Generating cell...</span>
                  </div>
                )}
                {currentQuestion && (
                  <QuestionAnswerUI question={currentQuestion} answer={currentAnswer} isStreaming={isStreaming} />
                )}
              </div>
              <ChatWindow onGenerateCell={handleGenerateCell} />
            </>
          ) : (
            <IntegrationsPage />
          )}
        </div>
      </div>
    </div>
  )
}

export default InteractiveCanvasEnhanced
