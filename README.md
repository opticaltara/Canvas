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