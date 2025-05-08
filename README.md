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

3. Edit the `.env` file to add your OpenRouter API key:
   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key_here
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

## Detailed Setup Guide

This guide provides more comprehensive instructions for setting up Sherlog Canvas. You can choose between a Docker-based setup or a manual setup.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Git**: For cloning the repository.
*   **Docker and Docker Compose**: (Recommended for an easier setup) For running the application and its dependencies in containers.
*   **Python 3.9+**: If you choose the manual setup for the backend.
*   **Node.js 16+**: If you choose the manual setup for the frontend (not covered in this guide, assuming pre-built frontend or separate setup).
*   **Go 1.18+**: If you plan to build or run Go-based MCP servers manually (e.g., `mcp-grafana`).

### Option 1: Docker-Based Setup (Recommended)

This is the easiest way to get Sherlog Canvas up and running. The `start.sh` script utilizes Docker Compose to manage all services.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sherlog-canvas.git
    cd sherlog-canvas
    ```

2.  **Create and configure your environment file**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Open the `.env` file and add your `OPENROUTER_API_KEY`. You may also configure other variables as needed (see the [Configuration](#configuration) section).
    ```env
    OPENROUTER_API_KEY=your_openrouter_api_key_here
    SHERLOG_DB_TYPE=sqlite # Default configuration uses SQLite
    # To use PostgreSQL instead, uncomment and configure the following:
    # SHERLOG_DB_TYPE=postgresql
    # POSTGRES_USER=your_db_user
    # POSTGRES_PASSWORD=your_db_password
    # POSTGRES_DB=sherlog_canvas_db
    # DATABASE_URL=postgresql://your_db_user:your_db_password@your_db_host:5432/sherlog_canvas_db
    ```
    By default, the Docker setup uses a SQLite database stored in `./data/sherlog.db` (persisted via a Docker volume). If you prefer to use an external PostgreSQL database, set `SHERLOG_DB_TYPE=postgresql` and configure the relevant `DATABASE_URL` or other PostgreSQL variables in your `.env` file.

3.  **Run the startup script**:
    ```bash
    ./start.sh
    ```
    This script will:
    *   Build the backend Docker image (if not already built).
    *   Start the backend API server.
    *   Start common MCP servers like `mcp-grafana` as defined in `docker-compose.yml`.
    *   If `SHERLOG_DB_TYPE` is explicitly set to `postgresql` in your `.env` file (and you haven't provided an external `DATABASE_URL`), it will also start a PostgreSQL container and the `pg-mcp` server.
    *   By default (with `SHERLOG_DB_TYPE=sqlite`), a PostgreSQL container and `pg-mcp` will not be started.

4.  **Access the application**:
    Open your web browser and navigate to `http://localhost:3000` (or the configured frontend port).

5.  **Stopping the application**:
    To stop all services, run:
    ```bash
    ./stop.sh
    ```
    Or, from the `sherlog-canvas` directory:
    ```bash
    docker-compose down
    ```

### Option 2: Manual Backend Setup

If you prefer to run the backend server manually without Docker:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sherlog-canvas.git
    cd sherlog-canvas
    ```

2.  **Set up a Python virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables**:
    You can set environment variables directly in your shell or create a `.env` file and use a library like `python-dotenv` (though the application loads `.env` by default if present). Essential variables include:
    *   `OPENROUTER_API_KEY`: Your OpenRouter API key.
    *   `SHERLOG_DB_TYPE`: Set to `sqlite` (default) or `postgresql`.
    *   If `SHERLOG_DB_TYPE=sqlite`, `SHERLOG_DB_FILE` specifies the path (default: `./data/sherlog.db`).
    *   If `SHERLOG_DB_TYPE=postgresql`, ensure `DATABASE_URL` is set (e.g., `postgresql://user:pass@host:port/dbname`).
    *   Refer to the [Configuration](#configuration) section for more variables.

    Example for `.env` file (defaulting to SQLite):
    ```env
    OPENROUTER_API_KEY=your_openrouter_api_key_here
    SHERLOG_DB_TYPE=sqlite
    SHERLOG_DB_FILE=./data/sherlog.db

    # Example for PostgreSQL (if you choose to use it):
    # SHERLOG_DB_TYPE=postgresql
    # DATABASE_URL=postgresql://sherlog:sherlog@localhost:5432/sherlog
    ```

5.  **Run database migrations (if applicable)**:
    If you are using PostgreSQL and setting up the database for the first time, or if there are schema changes, you might need to run migrations. (Assuming Alembic is used, though not explicitly stated in provided context - adapt if different).
    ```bash
    # Example if Alembic is set up in backend/db
    # alembic upgrade head
    ```
    For SQLite, the database and tables are typically created automatically on first access if they don't exist, based on the SQLAlchemy models.

6.  **Start the backend server**:
    ```bash
    python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
    ```
    The backend will be accessible at `http://localhost:8000`.

7.  **Manually Start MCP Servers**:
    If you are not using the Docker setup, you will need to start any required MCP servers manually. Refer to the [MCP Servers](#mcp-servers) section for instructions on setting up servers like `mcp-grafana` or `pg-mcp`. Ensure they are configured and running so the backend can connect to them.

### Configuring Data Connections

Once Sherlog Canvas is running, you can configure connections to your data sources:

1.  **Access Sherlog Canvas**: Open the application in your browser (e.g., `http://localhost:5173`).
2.  **Navigate to Data Connections**: Find the "Data Connections" or "Settings" section in the UI.
3.  **Add a New Connection**:
    *   Click "Add Connection" or a similar button.
    *   Select the type of data source you want to connect to (e.g., Grafana, PostgreSQL, Prometheus, Loki, S3).
4.  **Provide Connection Details**:
    *   **Grafana**: URL of your Grafana instance and an API key with viewer or editor permissions.
    *   **PostgreSQL**: A standard PostgreSQL connection string (e.g., `postgresql://username:password@host:port/database_name`).
    *   **Prometheus**: URL of your Prometheus server.
    *   **Loki**: URL of your Loki server.
    *   **S3**: S3 endpoint URL, bucket name, access key ID, and secret access key.
5.  **Save Connection**:
    Sherlog Canvas will store these connection details (by default in `./data/connections.json` or as configured by `SHERLOG_CONNECTION_STORAGE_TYPE`). When a connection is added or used, Sherlog Canvas will typically ensure the relevant MCP server is running (if managed by Sherlog, e.g., via Docker) or attempt to connect to an existing one.

This setup allows the AI agents within Sherlog Canvas to query and interact with your configured data sources.

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
2. Add connection type to `