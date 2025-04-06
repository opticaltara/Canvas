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

from typing import Any, Dict, List, Optional, cast
from typing_extensions import Literal
from uuid import UUID

# import logfire
import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

from backend.config import get_settings
from backend.core.cell import AIQueryCell, CellType, create_cell
from backend.core.execution import ExecutionContext
from backend.core.notebook import Notebook
from backend.ai.planning import InvestigationPlan, InvestigationStep, PlanAdapter
from backend.services.connection_manager import ConnectionConfig

# Get logger for AI operations
ai_logger = logging.getLogger("ai")

# Get settings for API keys
settings = get_settings()
anthropic_model = settings.anthropic_model

class InvestigationDependencies(BaseModel):
    """Dependencies for the investigation agent"""
    user_query: str = Field(description="The user's investigation query")
    notebook_id: Optional[UUID] = Field(description="The ID of the notebook", default=None)
    available_data_sources: List[str] = Field(
        description="Available data sources for queries",
        default=["sql", "prometheus", "loki", "s3"]
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
    """A single step in an investigation plan"""
    step_id: int = Field(description="Unique ID for this step")
    description: str = Field(description="Description of what this step does")
    cell_type: Literal["ai_query", "sql", "python", "markdown", "log", "metric", "s3"] = Field(
        description="Type of cell to create"
    )
    content: str = Field(description="Content for the cell (query, code, text)")
    depends_on: List[int] = Field(
        description="IDs of steps this step depends on",
        default_factory=list
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata for the step",
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
    The AI agent that handles interactions with Claude 3.7
    Responsible for interpreting user queries, generating investigation plans,
    and creating/updating cells based on analysis.
    """
    def __init__(self, mcp_servers: Optional[List[MCPServerHTTP]] = None):
        self.settings = get_settings()
        self.api_key = self.settings.anthropic_api_key
        self.model = self.settings.anthropic_model
        self.mcp_servers = mcp_servers or []
        
        # Initialize agents with MCP servers
        self.investigation_planner = Agent(
            f"anthropic:{self.model}",
            deps_type=InvestigationDependencies,
            result_type=Dict,  # Changed to Dict since we'll convert to InvestigationPlan
            mcp_servers=self.mcp_servers,
            system_prompt="""
            You are an expert at software engineering investigation. Your task is to create a detailed plan
            to investigate and solve problems by breaking them down into specific steps.
            
            Create an investigation plan with ordered steps. Each step should:
            1. Have a clear purpose in the investigation
            2. Specify what type of cell to create (sql, log, metric, python, markdown)
            3. Include the actual query/code/content for that cell
            4. List any dependencies on previous steps
            
            Important considerations:
            - Start with data gathering steps (SQL queries, log analysis, metrics)
            - Add analysis steps using Python to process the gathered data
            - Include markdown cells to explain your approach and findings
            - Consider which steps depend on others and set dependencies appropriately
            - Think step by step and be thorough in your approach

            Think carefully about the logical sequence of investigation. What data do you need first?
            How will you analyze it? What conclusions can you draw?
            
            Return the plan as a dictionary with:
            {
                "query": "the original query",
                "steps": [
                    {
                        "step_id": 1,
                        "description": "step description",
                        "cell_type": "sql|log|metric|python|markdown",
                        "content": "actual query/code/content",
                        "depends_on": [list of step_ids this depends on]
                    }
                ],
                "thinking": "optional explanation of your thought process"
            }
            """
        )

        # Initialize cell content generators with MCP servers
        self.sql_generator = Agent(
            f"anthropic:{self.model}",
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert SQL query writer. Generate a SQL query that addresses the user's request. Return ONLY the SQL query with no explanations."
        )

        self.log_generator = Agent(
            f"anthropic:{self.model}",
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert in log analysis. Generate a log query that addresses the user's request. Return ONLY the log query with no explanations."
        )

        self.metric_generator = Agent(
            f"anthropic:{self.model}",
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert in metric analysis. Generate a PromQL query that addresses the user's request. Return ONLY the PromQL query with no explanations."
        )

        self.python_generator = Agent(
            f"anthropic:{self.model}",
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert Python programmer. Generate Python code that addresses the user's request. Return ONLY the Python code with no explanations."
        )

        self.markdown_generator = Agent(
            f"anthropic:{self.model}",
            result_type=str,
            mcp_servers=self.mcp_servers,
            system_prompt="You are an expert at technical documentation. Create clear markdown to address the user's request. Return ONLY the markdown with no meta-commentary."
        )
    
    async def generate_investigation_plan(
        self,
        query: str,
        available_data_sources: Optional[List[str]] = None
    ) -> InvestigationPlan:
        """
        Generate an investigation plan for a query using Pydantic AI

        Args:
            query: The user's investigation query
            available_data_sources: List of available data sources
            
        Returns:
            A structured investigation plan with steps and cell types
            
        Note: On error, returns a simple fallback plan
        """
        if available_data_sources is None:
            available_data_sources = ["sql", "prometheus", "loki", "s3"]
        
        # Set up dependencies for the agent
        dependencies = InvestigationDependencies(
            user_query=query,
            available_data_sources=available_data_sources
        )
        
        try:
            # Run the agent with MCP servers context manager
            async with self.investigation_planner.run_mcp_servers():
                result = await self.investigation_planner.run(
                    user_prompt=dependencies.user_query,
                    deps=dependencies
                )
                
                # Convert the dictionary result to an InvestigationPlan using PlanAdapter
                plan_dict = cast(Dict, result.data)
                plan_dict["query"] = query  # Ensure query is set
                return PlanAdapter.from_dict(plan_dict)
                
        except Exception as e:
            print(f"Error generating investigation plan: {e}")
            # Create a simple fallback plan
            return InvestigationPlan(
                query=query,
                steps=[
                    InvestigationStep(
                        step_id=1,
                        description="Initial analysis",
                        cell_type=CellType.MARKDOWN,
                        content=f"# Investigation for: {query}\n\nLet's break down this problem and investigate.",
                        depends_on=[]
                    )
                ]
            )
    
    async def create_cells_from_plan(
        self,
        notebook: Notebook,
        plan: InvestigationPlan,
        query_cell_id: Optional[UUID] = None
    ) -> Dict[int, UUID]:
        """
        Create cells in the notebook based on an investigation plan
        
        Args:
            notebook: The notebook to add cells to
            plan: The investigation plan
            query_cell_id: Optional ID of the query cell that initiated this plan
            
        Returns:
            Mapping from step_ids to created cell UUIDs
        """
        # Maps step_ids to cell UUIDs
        step_to_cell_map = {}
        
        # Create cells for each step
        for step in plan.steps:
            step_id = step.step_id
            description = step.description
            cell_type = step.cell_type  # Already a CellType enum
            content = step.content
            depends_on = step.depends_on
            
            # Create metadata with description and step info
            metadata = {
                "description": description,
                "step_id": step_id,
                "generated": True,
                "query_cell_id": str(query_cell_id) if query_cell_id else None
            }
            
            # Create the cell
            cell = create_cell(cell_type, content)
            cell.metadata.update(metadata)
            
            # Add to notebook
            cell = notebook.add_cell(cell)
            step_to_cell_map[step_id] = cell.id
            
            # If this is from a query cell, track the relationship
            if query_cell_id and query_cell_id in notebook.cells:
                query_cell = notebook.cells[query_cell_id]
                if isinstance(query_cell, AIQueryCell):
                    query_cell.generated_cells.append(cell.id)
        
        # Set up dependencies between cells
        for step in plan.steps:
            step_id = step.step_id
            depends_on = step.depends_on
            
            if step_id in step_to_cell_map:
                current_cell_id = step_to_cell_map[step_id]
                
                # Add dependencies for each step this depends on
                for dep_step_id in depends_on:
                    if dep_step_id in step_to_cell_map:
                        dep_cell_id = step_to_cell_map[dep_step_id]
                        notebook.add_dependency(current_cell_id, dep_cell_id)
        
        return step_to_cell_map


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
        # Get connection manager and MCP server manager
        from backend.services.connection_manager import get_connection_manager
        from backend.mcp.manager import get_mcp_server_manager
        
        connection_manager = get_connection_manager()
        mcp_manager = get_mcp_server_manager()
        
        # Get all connections
        connections = connection_manager.get_all_connections()
        
        # Start MCP servers for all connections
        connection_configs = []
        for conn in connections:
            if isinstance(conn, dict) and "id" in conn:
                connection_configs.append(ConnectionConfig(**conn))
        
        server_addresses = await mcp_manager.start_mcp_servers(connection_configs)
        
        # Create MCPServerHTTP instances for each connection
        mcp_servers = []
        for conn in connections:
            if isinstance(conn, dict) and "id" in conn and conn["id"] in server_addresses:
                server_url = server_addresses[conn["id"]]
                if server_url.startswith("http"):
                    mcp_servers.append(MCPServerHTTP(url=f"{server_url}/sse"))
        
        # Initialize the AI agent with MCP servers
        agent = AIAgent(mcp_servers=mcp_servers)
        
        # Generate an investigation plan
        plan = await agent.generate_investigation_plan(
            query=cell.content,
            available_data_sources=[conn["type"] for conn in connections if isinstance(conn, dict) and "type" in conn]
        )
        
        # Store thinking output if available
        if plan.thinking:
            cell.thinking = plan.thinking
        
        # Create cells based on the plan
        step_to_cell_map = await agent.create_cells_from_plan(
            notebook=context.notebook,
            plan=plan,
            query_cell_id=cell.id
        )
        
        # Convert plan to dict for result
        plan_dict = PlanAdapter.to_dict(plan)
        
        # Return the result
        return {
            "plan": plan_dict,
            "generated_cells": {str(step_id): str(cell_id) for step_id, cell_id in step_to_cell_map.items()},
            "thinking": plan.thinking
        }
    
    except Exception as e:
        print(f"Error executing AI query cell: {e}")
        return {
            "error": str(e),
            "plan": None,
            "generated_cells": {}
        }