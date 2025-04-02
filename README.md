# Sherlog Canvas

Sherlog Canvas is a reactive notebook interface for software engineering investigation tasks, focused on debugging, log analysis, metric analysis, and database queries. It integrates advanced AI capabilities to assist with complex investigations.

## Features

- **Reactive Notebook Interface**: Changes to cells automatically update dependent cells
- **Multi-Agent AI System**: Creates investigation plans and generates cells based on user queries
- **Multiple Data Source Integration**: Connect to SQL databases, Prometheus, Loki, Grafana, and S3
- **Specialized Cell Types**: SQL, log queries, metrics, Python code, and more
- **Context Engine with RAG**: Uses vector embeddings to provide AI with data source schema understanding
- **Collaborative Investigation Environment**: AI and users work together to solve problems

## Architecture

Sherlog Canvas consists of:

1. **Backend**:
   - FastAPI server with WebSocket support
   - Dependency tracking system for reactivity
   - MCP server integration for data sources
   - AI orchestration for query planning
   - Context engine with RAG for data source understanding
   - Qdrant vector database for storing schema embeddings

2. **Frontend**:
   - Vite.js + React-based UI
   - Real-time updates via WebSockets
   - Monaco Editor for code/query editing
   - Dependency visualization

## Quick Start

Sherlog Canvas now includes a convenient startup script that handles all components:

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

4. Install dependencies:
   ```bash
   # Install Python dependencies
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Install Node.js dependencies
   npm install
   
   # Install frontend dependencies
   cd frontend
   npm install
   cd ..
   ```

5. Start all services with a single command:
   ```bash
   ./start.sh
   ```

6. Access the application at http://localhost:5173

The start script handles:
- Starting Qdrant vector database in Docker
- Installing and starting required MCP servers
- Starting the backend API server
- Starting the frontend development server

## MCP Servers

Sherlog Canvas uses Machine-Callable Package (MCP) servers for data source integration. These are standalone servers that expose standardized APIs for AI agents to interact with various data sources.

### Supported MCP Servers

| Data Source | MCP Package          | Purpose                                 |
|-------------|----------------------|-----------------------------------------|
| Grafana     | @anthropic-ai/mcp-grafana | Query Grafana dashboards, metrics, logs |
| PostgreSQL  | pg-mcp              | Query PostgreSQL databases              |
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
3. Index schema information for AI context
4. Automatically connect your AI agents to these data sources

### Manual MCP Server Setup

You can also install and run MCP servers manually:

```bash
# Install MCP packages globally
npm install -g @anthropic-ai/mcp-grafana pg-mcp

# Start Grafana MCP server
export GRAFANA_URL=https://your-grafana-url
export GRAFANA_API_KEY=your-api-key
npx @anthropic-ai/mcp-grafana --port 9100

# Start PostgreSQL MCP server
export PG_CONNECTION_STRING=postgresql://user:pass@localhost/dbname
npx pg-mcp --port 9200
```

## Manual Setup

If you prefer to start components individually:

### Prerequisites
- Python 3.9+
- Node.js 16+
- Docker (for Qdrant)
- Access to data sources (PostgreSQL, Grafana, etc.)

### Backend Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d --name sherlog-canvas-qdrant -p 6333:6333 qdrant/qdrant:latest

# Start the backend server
python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

Visit `http://localhost:5173` to access Sherlog Canvas.

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
|----------|-------------|---------|
| `SHERLOG_DEBUG` | Enable debug mode | `false` |
| `SHERLOG_API_HOST` | API host address | `0.0.0.0` |
| `SHERLOG_API_PORT` | API port | `8000` |
| `SHERLOG_ANTHROPIC_API_KEY` | Anthropic API key | `` |
| `SHERLOG_ANTHROPIC_MODEL` | Anthropic model | `claude-3-7-sonnet-20250219` |
| `SHERLOG_NOTEBOOK_STORAGE_TYPE` | Storage type for notebooks (file, s3, none) | `file` |
| `SHERLOG_NOTEBOOK_FILE_STORAGE_DIR` | Directory for file storage | `./data/notebooks` |
| `SHERLOG_CONNECTION_STORAGE_TYPE` | Storage type for connections (file, env, none) | `file` |

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
├── frontend/                # Frontend TypeScript/React code
│   ├── src/                 # Source code
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom React hooks
│   │   ├── styles/          # CSS styles
│   │   ├── types/           # TypeScript types
│   │   └── utils/           # Utility functions
│   ├── package.json         # NPM configuration
│   └── vite.config.ts       # Vite configuration
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile.backend       # Backend Docker image
├── Dockerfile.frontend      # Frontend Docker image
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
3. Add appropriate UI components in the frontend
4. Update pydantic-ai tools to handle the new cell type

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.