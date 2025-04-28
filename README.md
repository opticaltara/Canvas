# Sherlog Canvas

Sherlog Canvas is a reactive notebook interface for software engineering investigation tasks, focused on debugging, log analysis, metric analysis, and database queries. It integrates advanced AI capabilities to assist with complex investigations.

## Features

- **Reactive Notebook Interface**: Changes to cells automatically update dependent cells
- **Multi-Agent AI System**: Creates investigation plans and generates cells based on user queries
- **Multiple Data Source Integration**: Connect to SQL databases, Prometheus, Loki, Grafana, and S3
- **Specialized Cell Types**: SQL, log queries, metrics, Python code, and more
- **Collaborative Investigation Environment**: AI and users work together to solve problems

## Architecture

Sherlog Canvas consists of:

1. **Backend**:
   - FastAPI server with WebSocket support
   - Dependency tracking system for reactivity
   - MCP server integration for data sources
   - AI orchestration for query planning

## Quick Start

Sherlog Canvas now includes a convenient startup script that handles all components using Docker Compose:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sherlog-canvas.git
   cd sherlog-canvas
   ```

2. Create an environment file from the example:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file to add your Anthropic API key:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

4. Start all services with a single command:
   ```bash
   ./start.sh
   ```

5. Access the application at http://localhost:5173

The start script handles:
- Starting PostgreSQL MCP server in Docker
- Starting Grafana MCP server in Docker
- Starting the backend API server in Docker

All services are managed through Docker Compose for consistency and ease of use.

## MCP Servers

Sherlog Canvas uses Machine-Callable Package (MCP) servers for data source integration. These are standalone servers that expose standardized APIs for AI agents to interact with various data sources.

### Supported MCP Servers

| Data Source | MCP Implementation | Purpose                                 |
|-------------|----------------------|-----------------------------------------|
| Grafana     | mcp-grafana (Go)    | Query Grafana dashboards, metrics, logs |
| PostgreSQL  | pg-mcp (Docker)     | Query PostgreSQL databases              |
| Prometheus  | *(coming soon)*     | Query Prometheus metrics                |
| Loki        | *(coming soon)*     | Query Loki logs                         |
| S3          | *(coming soon)*     | Interact with S3 storage                |

### Setting Up Data Connections

1. Start Sherlog Canvas with `./start.sh`
2. Navigate to the "Data Connections" tab in the UI
3. Click "Add Connection" and select the connection type
4. Provide the connection details:
   - **Grafana**: URL and API key
   - **PostgreSQL**: Connection string (e.g., `postgresql://user:pass@localhost/dbname`)
   - **Prometheus**: URL and optional authentication
   - **Loki**: URL and optional authentication
   - **S3**: Endpoint, bucket, access key, and secret key

Sherlog Canvas will:
1. Store your connection details securely
2. Start the appropriate MCP server for each connection
3. Automatically connect your AI agents to these data sources

### Manual MCP Server Setup

You can also install and run MCP servers manually:

#### Grafana MCP Server (Go-based)

```bash
# Install mcp-grafana using Go
GOBIN="$HOME/go/bin" go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest

# Start Grafana MCP server
export GRAFANA_URL=https://your-grafana-url
export GRAFANA_API_KEY=your-api-key
mcp-grafana --port 9100
```

#### PostgreSQL MCP Server (Docker-based)

```bash
# Clone the repository
git clone https://github.com/stuzero/pg-mcp.git
cd pg-mcp

# Start PostgreSQL MCP server with Docker Compose
export PG_CONNECTION_STRING=postgresql://user:pass@localhost/dbname
docker-compose up -d
```
The server will be available at http://localhost:9201 by default.

## Manual Setup

If you prefer to start components individually:

### Prerequisites
- Python 3.9+
- Node.js 16+
- Go 1.18+ (for mcp-grafana)
- Git (for cloning repositories)
- Docker and Docker Compose (for pg-mcp)
- Access to data sources (PostgreSQL, Grafana, etc.)

### Backend Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

## Usage Examples

### Investigating Service Errors

1. Start with an AI query: "Investigate the increase in 500 errors in the payment service"
2. The AI will generate:
   - A PromQL query to analyze error rates
   - A Loki query to find error logs
   - An SQL query to check database state
   - Python code to correlate results

### User Session Analysis

1. Ask the AI: "Analyze user sessions that ended in abandoned carts"
2. Follow the investigation flow or modify any cell
3. Add your own cells for custom analysis

## Core Concepts

Sherlog Canvas operates on a few key concepts:

### Cells

A **Cell** is the fundamental unit of work in a Sherlog Canvas notebook. Each cell contains a specific piece of code or query, a type (e.g., `sql`, `python`, `logql`, `promql`, `markdown`), and its output. Cells can be created manually by the user or automatically by the AI agent based on an investigation query.

### Dependency Graph

Sherlog Canvas maintains a **Dependency Graph** to manage the relationships between cells. When a cell is modified, the system automatically identifies and re-executes any dependent cells downstream in the graph. This reactive nature ensures that the notebook always reflects the latest state of the investigation based on the current inputs and code. Dependencies are typically established when one cell references the output or variables defined in another cell (e.g., a Python cell using the results of an SQL query cell).

### Tool Use (AI Agents)

When you provide an investigation query (e.g., "Find the source of the 5xx errors in the payment service"), the AI agent system analyzes the request and determines which **Tools** are needed to answer it. Each tool corresponds to a specific data source or analysis capability (e.g., querying Prometheus, searching Loki logs, executing SQL against a database). The AI agent then generates the necessary cells, populating them with the appropriate queries or code to use these tools. The results are then displayed in the respective cell outputs.

### Example Investigation Flow

Let's walk through a simple example:

1.  **User Query**: "Show me the average request latency for the 'checkout' service over the last hour."
2.  **AI Agent Analysis**: The agent determines that this requires querying Prometheus metrics.
3.  **Tool Selection**: The agent selects the `prometheus_query` tool.
4.  **Cell Generation**: The agent creates a new cell of type `promql` with a query like:
    ```promql
    avg(rate(http_request_duration_seconds_sum{service="checkout"}[1h])) / avg(rate(http_request_duration_seconds_count{service="checkout"}[1h]))
    ```
5.  **Execution**: The cell is executed via the Prometheus MCP server.
6.  **Output**: The result (the average latency) is displayed in the cell's output panel.
7.  **User Interaction**: Now, suppose the user wants to see the P99 latency instead. They can manually edit the `promql` cell:
    ```promql
    histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{service="checkout"}[1h])) by (le))
    ```
8.  **Reactivity**: The cell automatically re-executes, showing the updated P99 latency. If another cell depended on the output of this cell (e.g., a Python cell plotting the latency), it would also automatically re-execute due to the dependency graph.

This reactive, AI-assisted workflow allows users to fluidly investigate issues, leveraging automated data gathering and analysis while retaining full control to manually explore and refine the investigation.

## Configuration

Configuration is managed through environment variables with the `SHERLOG_` prefix:

| Variable | Description | Default |
| --- | --- | --- |
| `SHERLOG_API_HOST` | Host to bind the API server | `0.0.0.0` |
| `SHERLOG_API_PORT` | Port for the API server | `8000` |
| `SHERLOG_DEBUG` | Enable debug mode | `false` |
| `SHERLOG_CORS_ORIGINS` | CORS origins (comma-separated) | `*` |
| `SHERLOG_AUTH_ENABLED` | Enable authentication | `false` |
| `SHERLOG_ANTHROPIC_API_KEY` | Anthropic API key for Claude | |
| `SHERLOG_ANTHROPIC_MODEL` | Claude model to use | `claude-3-7-sonnet-20250219` |
| `SHERLOG_LOGFIRE_TOKEN` | LogFire token for logging | |
| `SHERLOG_ENVIRONMENT` | Environment (development, production) | `development` |
| `SHERLOG_CONNECTION_STORAGE_TYPE` | Storage type for connections (file, env, db, none) | `file` |
| `SHERLOG_CONNECTION_FILE_PATH` | Path to connection storage file | `./data/connections.json` |
| `SHERLOG_DB_HOST` | Database host | `localhost` |
| `SHERLOG_DB_PORT` | Database port | `5432` |
| `SHERLOG_DB_NAME` | Database name | `sherlog` |
| `SHERLOG_DB_USER` | Database user | `sherlog` |
| `SHERLOG_DB_PASSWORD` | Database password | `sherlog` |
| `SHERLOG_DB_TYPE` | Database type (sqlite, postgresql) | `sqlite` |
| `SHERLOG_DB_FILE` | SQLite database file path | `./data/sherlog.db` |
| `SHERLOG_PYTHON_CELL_TIMEOUT` | Timeout for Python cell execution (seconds) | `30` |
| `SHERLOG_PYTHON_CELL_MAX_MEMORY` | Max memory for Python cell execution (MB) | `1024` |
| `SHERLOG_DEFAULT_QUERY_TIMEOUT` | Default timeout for database queries (seconds) | `30` |

## Project Structure

```
sherlog-canvas/
├── backend/                 # Backend Python code
│   ├── ai/                  # AI agent system
│   ├── core/                # Core notebook functionality
│   │   ├── executors/       # Cell executors for different cell types
│   ├── mcp/                 # MCP server integration
│   ├── services/            # Backend services
│   ├── routes/              # API routes
│   ├── config.py            # Configuration
│   └── server.py            # FastAPI server
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile.backend       # Backend Docker image
└── requirements.txt         # Python dependencies
```

## Extending Sherlog Canvas

### Adding a New MCP Server Integration

1. Find or create an MCP server for your data source
2. Add connection type to `backend/services/connection_manager.py`
3. Add MCP server startup logic in `backend/mcp/manager.py`
4. Add appropriate agent tools in `backend/ai/agent.py`

### Creating a Custom MCP Server

If you need to integrate a data source that doesn't have an existing MCP server:

1. Create a new MCP server using the MCP specification
2. Basic structure:
   ```javascript
   const express = require('express');
   const app = express();
   app.use(express.json());
   
   app.post('/mcp/query', async (req, res) => {
     const { query, parameters } = req.body;
     // Connect to your data source and execute query
     const result = await executeQuery(query, parameters);
     res.json({ data: result });
   });
   
   app.listen(process.env.PORT || 9000);
   ```

3. Register your MCP server in the MCP server manager
4. Add appropriate agent tools 

### Adding a New Cell Type

1. Add the new cell type to `backend/core/cell.py`
2. Create a cell executor in `backend/core/execution.py`
3. Update pydantic-ai tools to handle the new cell type

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.