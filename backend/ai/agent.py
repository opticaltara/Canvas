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

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAgent, RunContext, Tool

from backend.config import get_settings
from backend.context.engine import get_context_engine
from backend.core.cell import AIQueryCell, Cell, CellStatus, CellType, create_cell
from backend.core.execution import ExecutionContext
from backend.core.notebook import Notebook
from backend.ai.planning import InvestigationPlan, InvestigationStep, PlanAdapter


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
        description="Time range for the query (e.g., {'start': '2023-01-01T00:00:00Z', 'end': '2023-01-02T00:00:00Z'})",
        default=None
    )


class MetricQueryParams(BaseModel):
    """Parameters for metric queries"""
    query: str = Field(description="Metric query to execute (e.g., PromQL)")
    source: str = Field(description="Metric source (e.g., 'prometheus', 'grafana')", default="prometheus")
    time_range: Optional[Dict[str, str]] = Field(
        description="Time range for the query (e.g., {'start': '2023-01-01T00:00:00Z', 'end': '2023-01-02T00:00:00Z'})",
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


class DataSourceContextParams(BaseModel):
    """Parameters for retrieving data source context"""
    query: str = Field(description="Query to find relevant context for")
    source_type: Optional[str] = Field(
        description="Optional source type filter (sql, prometheus, loki, s3)",
        default=None
    )
    limit: int = Field(description="Maximum number of context items to retrieve", default=5)


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


class InvestigationPlanModel(BaseModel):
    """The complete investigation plan"""
    steps: List[InvestigationStepModel] = Field(
        description="Steps to execute in the investigation"
    )
    thinking: Optional[str] = Field(
        description="Reasoning behind the investigation plan",
        default=None
    )


# Create the pydantic-ai agent for investigation planning
investigation_planner = PydanticAgent(
    "anthropic:claude-3-7-sonnet-20250219",
    deps_type=InvestigationDependencies,
    result_type=InvestigationPlanModel,
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
    - Utilize context about data sources when provided
    - Use the get_data_source_context tool when you need information about database schemas,
      available metrics, log formats, etc.

    Think carefully about the logical sequence of investigation. What data do you need first?
    How will you analyze it? What conclusions can you draw?
    """
)


# Define tools for the agents to use

# Data source context tool for retrieving schema information
@investigation_planner.tool
async def get_data_source_context(ctx: RunContext[InvestigationDependencies], params: DataSourceContextParams) -> str:
    """
    Retrieve context about data sources such as database schemas, metric names, log formats, etc.
    
    Args:
        params: DataSourceContextParams containing the query and filters
        
    Returns:
        String containing relevant context information from data sources
    """
    # Get the context engine
    context_engine = get_context_engine()
    
    try:
        # Retrieve context items
        context_items = await context_engine.retrieve_context(
            params.query,
            params.source_type,
            params.limit
        )
        
        # Build formatted context string
        context_str = context_engine.build_context_for_ai(context_items)
        
        return context_str if context_str else "No relevant context found for the query."
    
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


# SQL query tool
@investigation_planner.tool
async def query_sql(ctx: RunContext[InvestigationDependencies], params: SQLQueryParams) -> DataQueryResult:
    """
    Execute an SQL query against a database using an MCP server.
    
    Args:
        params: SQLQueryParams containing the query and connection details
        
    Returns:
        DataQueryResult with the query results or error
    """
    try:
        # Get connection manager
        from backend.services.connection_manager import get_connection_manager
        connection_manager = get_connection_manager()
        
        connection_id = params.connection_id
        if not connection_id:
            # Get default SQL connection
            connection_config = connection_manager.get_default_connection("postgres")
            if not connection_config:
                return DataQueryResult(
                    data=[],
                    query=params.query,
                    error="No default SQL connection configured"
                )
            connection_id = connection_config.id
        
        # Get the connection
        connection_config = connection_manager.get_connection(connection_id)
        if not connection_config:
            return DataQueryResult(
                data=[],
                query=params.query,
                error=f"Connection not found: {connection_id}"
            )
        
        # Get MCP client if available
        mcp_client = ctx.state.get(f"mcp_{connection_id}", None)
        
        if mcp_client:
            # Use MCP client to execute query
            try:
                result = await mcp_client.query_db(params.query, params.parameters)
                
                return DataQueryResult(
                    data=result.get("rows", []),
                    query=params.query,
                    metadata={
                        "connection_id": connection_id,
                        "connection_name": connection_config.name,
                        "columns": result.get("columns", [])
                    }
                )
            except Exception as e:
                return DataQueryResult(
                    data=[],
                    query=params.query,
                    error=f"MCP query error: {str(e)}"
                )
        else:
            # No MCP client available
            return DataQueryResult(
                data=[],
                query=params.query,
                error=f"No MCP client available for connection: {connection_id}"
            )
    
    except Exception as e:
        return DataQueryResult(
            data=[],
            query=params.query,
            error=str(e)
        )


# Log query tool
@investigation_planner.tool
async def query_logs(ctx: RunContext[InvestigationDependencies], params: LogQueryParams) -> DataQueryResult:
    """
    Execute a log query (e.g., against Loki or Grafana) using an MCP server.
    
    Args:
        params: LogQueryParams containing the query and source details
        
    Returns:
        DataQueryResult with the query results or error
    """
    try:
        # Get connection manager
        from backend.services.connection_manager import get_connection_manager
        connection_manager = get_connection_manager()
        
        # Determine connection type based on source
        connection_type = params.source
        if connection_type not in ["loki", "grafana"]:
            connection_type = "loki"  # Default to loki
        
        # Get default connection for this type
        connection_config = connection_manager.get_default_connection(connection_type)
        if not connection_config:
            return DataQueryResult(
                data=[],
                query=params.query,
                error=f"No default connection found for {connection_type}"
            )
        
        # Get MCP client if available
        mcp_client = ctx.state.get(f"mcp_{connection_config.id}", None)
        
        if mcp_client:
            # Use MCP client to execute query
            try:
                # Set up query parameters
                query_params = {}
                if params.time_range:
                    query_params["from"] = params.time_range.get("start")
                    query_params["to"] = params.time_range.get("end")
                
                # Execute the query differently based on the source
                if connection_type == "grafana":
                    result = await mcp_client.query_loki(params.query, query_params)
                else:
                    result = await mcp_client.query_logs(params.query, query_params)
                
                return DataQueryResult(
                    data=result.get("data", []),
                    query=params.query,
                    metadata={
                        "source": params.source,
                        "time_range": params.time_range
                    }
                )
            except Exception as e:
                return DataQueryResult(
                    data=[],
                    query=params.query,
                    error=f"MCP query error: {str(e)}"
                )
        else:
            # No MCP client available
            return DataQueryResult(
                data=[],
                query=params.query,
                error=f"No MCP client available for connection: {connection_config.id}"
            )
    
    except Exception as e:
        return DataQueryResult(
            data=[],
            query=params.query,
            error=str(e)
        )


# Metrics query tool
@investigation_planner.tool
async def query_metrics(ctx: RunContext[InvestigationDependencies], params: MetricQueryParams) -> DataQueryResult:
    """
    Execute a metric query (e.g., PromQL) using an MCP server.
    
    Args:
        params: MetricQueryParams containing the query and source details
        
    Returns:
        DataQueryResult with the query results or error
    """
    try:
        # Get connection manager
        from backend.services.connection_manager import get_connection_manager
        connection_manager = get_connection_manager()
        
        # Determine connection type based on source
        connection_type = params.source
        if connection_type not in ["prometheus", "grafana"]:
            connection_type = "prometheus"  # Default to prometheus
        
        # Get default connection for this type
        connection_config = connection_manager.get_default_connection(connection_type)
        if not connection_config:
            return DataQueryResult(
                data=[],
                query=params.query,
                error=f"No default connection found for {connection_type}"
            )
        
        # Get MCP client if available
        mcp_client = ctx.state.get(f"mcp_{connection_config.id}", None)
        
        if mcp_client:
            # Use MCP client to execute query
            try:
                # Set up query parameters
                query_params = {
                    "instant": params.instant
                }
                if params.time_range:
                    query_params["from"] = params.time_range.get("start")
                    query_params["to"] = params.time_range.get("end")
                
                # Execute the query differently based on the source
                if connection_type == "grafana":
                    result = await mcp_client.query_prometheus(params.query, query_params)
                else:
                    result = await mcp_client.query_metrics(params.query, query_params)
                
                return DataQueryResult(
                    data=result.get("data", []),
                    query=params.query,
                    metadata={
                        "source": params.source,
                        "time_range": params.time_range,
                        "instant": params.instant,
                        "result_type": result.get("result_type", "unknown")
                    }
                )
            except Exception as e:
                return DataQueryResult(
                    data=[],
                    query=params.query,
                    error=f"MCP query error: {str(e)}"
                )
        else:
            # No MCP client available
            return DataQueryResult(
                data=[],
                query=params.query,
                error=f"No MCP client available for connection: {connection_config.id}"
            )
    
    except Exception as e:
        return DataQueryResult(
            data=[],
            query=params.query,
            error=str(e)
        )


# S3 query tool
@investigation_planner.tool
async def query_s3(ctx: RunContext[InvestigationDependencies], params: S3QueryParams) -> DataQueryResult:
    """
    Execute an S3 operation (list, get, or select) using an MCP server.
    
    Args:
        params: S3QueryParams containing the operation details
        
    Returns:
        DataQueryResult with the operation results or error
    """
    try:
        # Get connection manager
        from backend.services.connection_manager import get_connection_manager
        connection_manager = get_connection_manager()
        
        # Get default S3 connection
        connection_config = connection_manager.get_default_connection("s3")
        if not connection_config:
            return DataQueryResult(
                data=[],
                query=params.query,
                error="No default S3 connection found"
            )
        
        # Get MCP client if available
        mcp_client = ctx.state.get(f"mcp_{connection_config.id}", None)
        
        if mcp_client:
            # Use MCP client to execute S3 operation
            try:
                # Set up operation parameters
                operation_params = {
                    "bucket": params.bucket,
                    "operation": params.operation
                }
                
                if params.prefix:
                    operation_params["prefix"] = params.prefix
                
                if params.operation in ["get_object", "select_object"]:
                    operation_params["key"] = params.query
                
                # Execute the appropriate S3 operation
                if params.operation == "list_objects":
                    result = await mcp_client.s3_list_objects(params.bucket, params.prefix)
                elif params.operation == "get_object":
                    result = await mcp_client.s3_get_object(params.bucket, params.query)
                elif params.operation == "select_object":
                    result = await mcp_client.s3_select_object(params.bucket, params.query, params.query)
                else:
                    result = {"data": [], "error": f"Unsupported S3 operation: {params.operation}"}
                
                return DataQueryResult(
                    data=result.get("data", []),
                    query=params.query,
                    metadata={
                        "bucket": params.bucket,
                        "operation": params.operation,
                        "prefix": params.prefix
                    }
                )
            except Exception as e:
                return DataQueryResult(
                    data=[],
                    query=params.query,
                    error=f"MCP query error: {str(e)}"
                )
        else:
            # No MCP client available
            return DataQueryResult(
                data=[],
                query=params.query,
                error=f"No MCP client available for connection: {connection_config.id}"
            )
    
    except Exception as e:
        return DataQueryResult(
            data=[],
            query=params.query,
            error=str(e)
        )


# Create cell-specific content generation agents with tools
sql_generator = PydanticAgent(
    "anthropic:claude-3-7-sonnet-20250219",
    result_type=str,
    system_prompt="You are an expert SQL query writer. Generate a SQL query that addresses the user's request. Return ONLY the SQL query with no explanations.",
    tools=[query_sql, get_data_source_context]
)

log_generator = PydanticAgent(
    "anthropic:claude-3-7-sonnet-20250219",
    result_type=str,
    system_prompt="You are an expert in log analysis. Generate a log query that addresses the user's request. Return ONLY the log query with no explanations.",
    tools=[query_logs, get_data_source_context]
)

metric_generator = PydanticAgent(
    "anthropic:claude-3-7-sonnet-20250219",
    result_type=str,
    system_prompt="You are an expert in metric analysis. Generate a PromQL query that addresses the user's request. Return ONLY the PromQL query with no explanations.",
    tools=[query_metrics, get_data_source_context]
)

python_generator = PydanticAgent(
    "anthropic:claude-3-7-sonnet-20250219",
    result_type=str,
    system_prompt="You are an expert Python programmer. Generate Python code that addresses the user's request. Return ONLY the Python code with no explanations."
)

markdown_generator = PydanticAgent(
    "anthropic:claude-3-7-sonnet-20250219",
    result_type=str,
    system_prompt="You are an expert at technical documentation. Create clear markdown to address the user's request. Return ONLY the markdown with no meta-commentary."
)


class AIAgent:
    """
    The AI agent that handles interactions with Claude 3.7
    Responsible for interpreting user queries, generating investigation plans,
    and creating/updating cells based on analysis.
    """
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.anthropic_api_key
        self.model = self.settings.anthropic_model
        self.context_engine = get_context_engine()
    
    async def generate_investigation_plan(self, query: str, available_data_sources: List[str] = None, mcp_clients: Dict[str, Any] = None) -> InvestigationPlanModel:
        """
        Generate an investigation plan for a query using Pydantic AI
        
        Args:
            query: The user's investigation query
            available_data_sources: List of available data sources
            mcp_clients: Dictionary of MCP clients to use
            
        Returns:
            A structured investigation plan with steps and cell types
        """
        if available_data_sources is None:
            available_data_sources = ["sql", "prometheus", "loki", "s3"]
        
        # Set up dependencies for the agent
        dependencies = InvestigationDependencies(
            user_query=query,
            available_data_sources=available_data_sources
        )
        
        try:
            # Create a run context with the dependencies
            run_context = RunContext(dependencies)
            
            # Add MCP clients to the run context if provided
            if mcp_clients:
                run_context.state.update(mcp_clients)
            
            # Retrieve relevant context about data sources for the query
            context_params = DataSourceContextParams(query=query)
            context_str = await get_data_source_context(run_context, context_params)
            
            # Add context to the query if available
            if context_str and not context_str.startswith("Error") and not context_str.startswith("No relevant"):
                dependencies.user_query = f"{query}\n\n{context_str}"
            
            # Run the agent to generate a plan
            # Note: The standard pydantic-ai API only has `run()` not `run_with_context()`,
            # so we use the regular run method with the new context
            plan = await investigation_planner.run(dependencies, context=run_context)
            return plan
        except Exception as e:
            print(f"Error generating investigation plan: {e}")
            # Create a simple fallback plan
            return InvestigationPlanModel(
                steps=[
                    InvestigationStepModel(
                        step_id=1,
                        description="Initial analysis",
                        cell_type="markdown",
                        content=f"# Investigation for: {query}\n\nLet's break down this problem and investigate.",
                        depends_on=[]
                    )
                ]
            )
    
    async def create_cells_from_plan(
        self,
        notebook: Notebook,
        plan: InvestigationPlanModel,
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
            cell_type_str = step.cell_type.upper()
            content = step.content
            depends_on = step.depends_on
            
            # Convert cell type string to enum
            try:
                cell_type = CellType[cell_type_str]
            except KeyError:
                # Default to markdown if type is unknown
                cell_type = CellType.MARKDOWN
            
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
    # Initialize the AI agent
    agent = AIAgent()
    
    try:
        # Set up MCP clients for the AI tools if needed
        if hasattr(context, 'mcp_clients') and context.mcp_clients:
            # We already have MCP clients in the context
            pass
        else:
            # We need to initialize MCP clients
            from backend.mcp.manager import get_mcp_server_manager
            from backend.services.connection_manager import get_connection_manager
            
            # Get connection manager and MCP server manager
            connection_manager = get_connection_manager()
            mcp_manager = get_mcp_server_manager()
            
            # Get all connections
            connections = connection_manager.get_all_connections()
            
            # Start MCP servers for all connections
            server_addresses = await mcp_manager.start_mcp_servers(
                [conn for conn in connections if isinstance(conn, dict) and "id" in conn]
            )
            
            # Create basic client objects for each connection/server
            mcp_clients = {}
            for conn in connections:
                if isinstance(conn, dict) and "id" in conn and conn["id"] in server_addresses:
                    # Create a simple client object for the MCP server
                    # This is a placeholder - a real implementation would create actual client objects
                    mcp_clients[f"mcp_{conn['id']}"] = {
                        "connection_id": conn["id"],
                        "address": server_addresses[conn["id"]],
                        "query_db": lambda query, params: {"rows": [], "columns": []},
                        "query_logs": lambda query, params: {"data": []},
                        "query_metrics": lambda query, params: {"data": []},
                        "s3_list_objects": lambda bucket, prefix: {"data": []},
                        "s3_get_object": lambda bucket, key: {"data": ""},
                        "s3_select_object": lambda bucket, key, query: {"data": []},
                    }
            
            # Add MCP clients to the context
            context.mcp_clients = mcp_clients
        
        # Generate an investigation plan with MCP clients
        plan = await agent.generate_investigation_plan(
            query=cell.content,
            mcp_clients=context.mcp_clients
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
        plan_dict = plan.model_dump()
        
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