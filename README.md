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
   - Plugin system for data sources
   - AI orchestration for query planning
   - Context engine with RAG for data source understanding
   - Qdrant vector database for storing schema embeddings

2. **Frontend**:
   - Vite.js + React-based UI
   - Real-time updates via WebSockets
   - Monaco Editor for code/query editing
   - Dependency visualization

## Quick Start with Docker

The easiest way to get started is using Docker Compose:

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

4. Start the containers:
   ```bash
   docker-compose up -d
   ```

5. Access the application at http://localhost:80

## Manual Setup

### Prerequisites
- Python 3.9+
- Node.js 16+
- Access to data sources (PostgreSQL, Prometheus, Loki, etc.)

### Backend Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with uv
pip install uv
uv pip install -r requirements.txt

# Start the backend server
python -m backend.server
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
│   ├── plugins/             # Data source plugins
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

### Adding a New Data Source Plugin

1. Create a new plugin file in `backend/plugins/`
2. Implement the `PluginBase` interface
3. Register the plugin in `backend/services/connection_manager.py`

### Adding a New Cell Type

1. Add the new cell type to `backend/core/cell.py`
2. Create a cell executor in the appropriate plugin
3. Add UI components for the new cell type in the frontend

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.