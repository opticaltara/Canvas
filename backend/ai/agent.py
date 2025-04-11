"""
AI Agent Module for Sherlog Canvas

This module implements the AI agent system using pydantic-ai for our notebook.
The agent is responsible for:
1. Interpreting user queries
2. Generating investigation plans
3. Creating cells with appropriate content
4. Analyzing data and providing insights

It uses Anthropic's Claude model for natural language understanding and code generation,
with tools that connect to various data sources like SQL, Prometheus, Loki, and S3.
"""

from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple
from uuid import UUID
import os

# import logfire
import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.messages import ModelMessage

from backend.config import get_settings
from backend.core.cell import AIQueryCell
from backend.core.execution import ExecutionContext
from backend.ai.planning import InvestigationPlan, InvestigationStep
from backend.services.connection_manager import get_connection_manager, BaseConnectionConfig
from backend.ai.chat_tools import NotebookCellTools, CreateCellParams

# Get logger for AI operations
ai_logger = logging.getLogger("ai")

# Get settings for API keys
settings = get_settings()

class InvestigationDependencies(BaseModel):
    """Dependencies for the investigation agent"""
    user_query: str = Field(description="The user's investigation query")
    notebook_id: Optional[UUID] = Field(description="The ID of the notebook", default=None)
    available_data_sources: List[str] = Field(
        description="Available data sources for queries",
        default_factory=list
    )


class DataQueryResult(BaseModel):
    """Result of a data source query"""
    data: Any = Field(description="The query result data")
    query: str = Field(description="The executed query")
    error: Optional[str] = Field(description="Error message if query failed", default=None)
    metadata: Dict[str, Any] = Field(description="Additional metadata", default_factory=dict)


class SQLQueryParams(BaseModel):
    """Parameters for SQL queries"""
    query: str = Field(description="SQL query to execute")
    connection_id: Optional[str] = Field(description="Database connection ID", default=None)
    parameters: Dict[str, Any] = Field(description="Query parameters", default_factory=dict)


class LogQueryParams(BaseModel):
    """Parameters for log queries"""
    query: str = Field(description="Log query to execute (e.g., Loki query)")
    source: str = Field(description="Log source (e.g., 'loki', 'grafana')", default="loki")
    time_range: Optional[Dict[str, str]] = Field(
        description="Time range for the query",
        default=None
    )


class MetricQueryParams(BaseModel):
    """Parameters for metric queries"""
    query: str = Field(description="Metric query to execute (e.g., PromQL)")
    source: str = Field(description="Metric source (e.g., 'prometheus', 'grafana')", default="prometheus")
    time_range: Optional[Dict[str, str]] = Field(
        description="Time range for the query",
        default=None
    )
    instant: bool = Field(description="Whether to perform an instant query", default=False)


class S3QueryParams(BaseModel):
    """Parameters for S3 queries"""
    query: str = Field(description="S3 query or object key")
    bucket: str = Field(description="S3 bucket name")
    prefix: Optional[str] = Field(description="Object key prefix for filtering", default=None)
    operation: str = Field(
        description="Operation to perform ('list_objects', 'get_object', or 'select_object')",
        default="list_objects"
    )


class InvestigationStepModel(BaseModel):
    """A step in the investigation plan"""
    step_type: str = Field(description="Type of step (e.g., 'sql', 'python', 'markdown')")
    description: str = Field(description="Description of what this step will do")
    dependencies: List[str] = Field(
        description="List of step IDs this step depends on",
        default_factory=list
    )
    parameters: Dict[str, Any] = Field(
        description="Parameters for the step",
        default_factory=dict
    )


class InvestigationPlanModel(BaseModel):
    """The complete investigation plan"""
    steps: List[InvestigationStepModel] = Field(
        description="Steps to execute in the investigation"
    )
    thinking: Optional[str] = Field(
        description="Reasoning behind the investigation plan",
        default=None
    )


class AIAgent:
    """
    The AI agent that handles investigation planning and execution.
    Responsible for:
    1. Creating investigation plans
    2. Executing steps and streaming results
    3. Adjusting plans based on results
    4. Creating cells directly
    """
    def __init__(self, mcp_servers: Optional[List[MCPServerHTTP]] = None):
        self.settings = get_settings()
        self.model = AnthropicModel(
            self.settings.anthropic_model,
            provider=AnthropicProvider(api_key=self.settings.anthropic_api_key)
        )
        self.mcp_servers = mcp_servers or []
        
        # Initialize investigation planner
        self.investigation_planner = Agent(
            self.model,
            deps_type=InvestigationDependencies,
            result_type=Dict,
            mcp_servers=self.mcp_servers,
            system_prompt="""
            You are an expert at software engineering investigation planning. Your task is to create detailed plans
            for investigating user queries. The plans should include:
            1. A clear understanding of what needs to be investigated
            2. A sequence of steps to gather and analyze data
            3. Dependencies between steps
            4. Parameters needed for each step
            5. Expected outputs and insights
            
            Each step should be one of these types:
            - sql: For database queries
            - python: For data analysis and processing
            - markdown: For explanations and documentation
            - log: For log analysis
            - metric: For metric analysis
            - s3: For S3 data access
            
            For each step, provide:
            - A clear description of what the step will do
            - Any dependencies on previous steps
            - Required parameters and their expected format
            - Expected output format and insights
            
            Return a JSON object with this structure:
            {
                "steps": [
                    {
                        "step_type": "type of step",
                        "description": "what this step will do",
                        "dependencies": ["step_id1", "step_id2"],
                        "parameters": {
                            "key": "value"
                        },
                        "expected_output": "description of expected output"
                    }
                ],
                "thinking": "detailed explanation of your investigation approach"
            }
            """
        )

        # Initialize content generators
        self.sql_generator = Agent(
            self.model,
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert SQL query writer. Generate a SQL query that addresses the user's request. Return ONLY the SQL query with no explanations."
        )

        self.log_generator = Agent(
            self.model,
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert in log analysis. Generate a log query that addresses the user's request. Return ONLY the log query with no explanations."
        )

        self.metric_generator = Agent(
            self.model,
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert in metric analysis. Generate a PromQL query that addresses the user's request. Return ONLY the PromQL query with no explanations."
        )

        self.python_generator = Agent(
            self.model,
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert Python programmer. Generate Python code that addresses the user's request. Return ONLY the Python code with no explanations."
        )

        self.markdown_generator = Agent(
            self.model,
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert at technical documentation. Create clear markdown to address the user's request. Return ONLY the markdown with no meta-commentary."
        )

    async def investigate(
        self, 
        query: str, 
        session_id: str, 
        notebook_id: str,
        message_history: List[ModelMessage] = [],
        cell_tools: Optional[NotebookCellTools] = None
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """
        Investigate a query by creating and executing a plan.
        Streams status updates as steps are completed.
        
        Args:
            query: The user's query
            session_id: The chat session ID
            notebook_id: The notebook ID to create cells in
            message_history: Previous messages in the session
            cell_tools: Tools for creating and managing cells
            
        Yields:
            Tuples of (step_id, status) as steps are completed
        """
        if not cell_tools:
            raise ValueError("cell_tools is required for creating cells")
            
        if not notebook_id:
            raise ValueError("notebook_id is required for creating cells")
            
        # Create initial investigation plan
        plan = await self.create_investigation_plan(query, notebook_id)
        yield "plan_created", {"status": "plan_created", "thinking": plan.thinking}
        
        # Create a markdown cell explaining the plan
        plan_cell = await cell_tools.create_cell(
            params=CreateCellParams(
                notebook_id=notebook_id,
                cell_type="markdown",
                content=f"# Investigation Plan\n\n{plan.thinking}\n\n## Steps:\n" + 
                       "\n".join(f"- {step.description}" for step in plan.steps),
                metadata={
                    "session_id": session_id,
                    "step_id": "plan",
                    "dependencies": []
                }
            )
        )
        yield "plan_cell_created", {"status": "plan_cell_created"}
        
        # Execute steps in order
        completed_steps = {}
        for step in plan.steps:
            # Wait for dependencies
            for dep_id in step.dependencies:
                if dep_id not in completed_steps:
                    raise ValueError(f"Dependency {dep_id} not found in completed steps")
            
            # Generate content for this step
            content = await self.generate_content(step.step_type, step.description)
            
            # Create cell for this step
            cell_result = await cell_tools.create_cell(
                params=CreateCellParams(
                    notebook_id=notebook_id,
                    cell_type=step.step_type,
                    content=content,
                    metadata={
                        "session_id": session_id,
                        "step_id": step.step_type,
                        "dependencies": step.dependencies
                    }
                )
            )
            
            # Execute the cell
            await cell_tools.execute_cell(cell_result["cell_id"])
            
            # Stream status update
            yield f"step_{step.step_type}_completed", {
                "status": "step_completed",
                "step_type": step.step_type
            }
            
            # Store completed step
            completed_steps[step.step_type] = content
            
            # TODO: Add logic to adjust plan based on results if needed
            
        # Create summary cell
        summary = await self.markdown_generator.run(
            f"Create a summary of the investigation results: {completed_steps}"
        )
        summary_cell = await cell_tools.create_cell(
            params=CreateCellParams(
                notebook_id=notebook_id,
                cell_type="markdown",
                content=f"# Investigation Summary\n\n{summary.data}",
                metadata={
                    "session_id": session_id,
                    "step_id": "summary",
                    "dependencies": list(completed_steps.keys())
                }
            )
        )
        yield "summary_created", {"status": "summary_created"}

    async def create_investigation_plan(self, query: str, notebook_id: str) -> InvestigationPlan:
        """Create an investigation plan for the given query"""
        # Get available data sources from connection manager
        connection_manager = get_connection_manager()
        connections = await connection_manager.get_all_connections()
        available_data_sources = [conn["type"] for conn in connections]

        # Create dependencies for the investigation planner
        deps = InvestigationDependencies(
            user_query=query,
            notebook_id=UUID(notebook_id),
            available_data_sources=available_data_sources
        )

        # Generate the plan
        result = await self.investigation_planner.run(
            user_prompt=deps.user_query,
            deps=deps
        )
        plan_data = result.data

        # Convert to InvestigationPlan
        steps = []
        for step_data in plan_data["steps"]:
            step = InvestigationStep(
                step_type=step_data["step_type"],
                description=step_data["description"],
                dependencies=step_data["dependencies"],
                parameters=step_data["parameters"]
            )
            steps.append(step)

        return InvestigationPlan(
            steps=steps,
            thinking=plan_data.get("thinking")
        )

    async def generate_content(self, step_type: str, description: str) -> str:
        """Generate content for a specific step type"""
        if step_type == "sql":
            result = await self.sql_generator.run(description)
            return result.data
        elif step_type == "log":
            result = await self.log_generator.run(description)
            return result.data
        elif step_type == "metric":
            result = await self.metric_generator.run(description)
            return result.data
        elif step_type == "python":
            result = await self.python_generator.run(description)
            return result.data
        elif step_type == "markdown":
            result = await self.markdown_generator.run(description)
            return result.data
        else:
            raise ValueError(f"Unsupported step type: {step_type}")


async def execute_ai_query_cell(cell: AIQueryCell, context: ExecutionContext) -> Dict:
    """
    Execute an AI query cell
    
    Args:
        cell: The AI query cell to execute
        context: The execution context
        
    Returns:
        The result of executing the cell
    """
    try:
        # Get connection manager
        connection_manager = get_connection_manager()
        
        # Get all connections
        connections = await connection_manager.get_all_connections()
        
        # Start MCP servers for all connections
        server_addresses = {}
        for conn in connections:
            if isinstance(conn, dict) and "id" in conn:
                connection_config = BaseConnectionConfig(**conn)
                await connection_config.start_mcp_server()
                server_addresses[conn["id"]] = connection_config.mcp_status["address"]
        
        # Create MCPServerHTTP instances for each connection
        mcp_servers = []
        for conn in connections:
            if isinstance(conn, dict) and "id" in conn and conn["id"] in server_addresses:
                server_url = server_addresses[conn["id"]]
                if server_url.startswith("http"):
                    mcp_servers.append(MCPServerHTTP(url=f"{server_url}/sse"))
        
        # Initialize the AI agent with MCP servers
        agent = AIAgent(mcp_servers=mcp_servers)
        
        # Get notebook_id from cell metadata
        notebook_id = cell.metadata.get("notebook_id")
        if not notebook_id:
            raise ValueError("Cell metadata must contain notebook_id")
        
        # Generate an investigation plan
        plan = await agent.create_investigation_plan(
            query=cell.content,
            notebook_id=notebook_id
        )
        
        # Store thinking output if available
        if plan.thinking:
            cell.thinking = plan.thinking
        
        # Return the plan for the ChatAgent to execute
        return {
            "plan": plan.model_dump(),
            "thinking": plan.thinking
        }
    
    except Exception as e:
        print(f"Error executing AI query cell: {e}")
        return {
            "error": str(e),
            "plan": None
        }