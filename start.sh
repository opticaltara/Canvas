#!/bin/bash
# Sherlog Canvas Launcher Script
# This script starts all the components needed for Sherlog Canvas,
# including Docker containers, MCP servers, and application services.

# Environment setup
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Ensure .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one from .env.example."
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Check for required environment variables
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set in .env file."
    exit 1
fi

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "Error: Docker is not running or not installed."
        echo "Please start Docker and try again."
        exit 1
    fi
}

# Function to check if npm is installed
check_npm() {
    if ! command -v npm >/dev/null 2>&1; then
        echo "Error: npm is not installed."
        echo "Please install Node.js and npm, then try again."
        exit 1
    fi
}

# Function to install MCP servers
install_mcp_servers() {
    echo "Installing MCP server dependencies..."
    npm install -g @anthropic-ai/mcp-grafana pg-mcp
    echo "MCP server dependencies installed."
}

# Function to start Qdrant
start_qdrant() {
    echo "Starting Qdrant vector database..."
    
    # Check if Qdrant container is already running
    if docker ps | grep -q sherlog-canvas-qdrant; then
        echo "Qdrant is already running."
    else
        # Check if the container exists but is stopped
        if docker ps -a | grep -q sherlog-canvas-qdrant; then
            docker start sherlog-canvas-qdrant
        else
            # Create and start the container
            docker run -d --name sherlog-canvas-qdrant \
                -p 6333:6333 -p 6334:6334 \
                -v qdrant_data:/qdrant/storage \
                qdrant/qdrant:latest
        fi
        echo "Qdrant started."
    fi
}

# Function to start the backend
start_backend() {
    echo "Starting backend server..."
    
    # Create data directory if it doesn't exist
    mkdir -p data/notebooks
    
    # Start backend in a new terminal or in the background
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        osascript -e "tell app \"Terminal\" to do script \"cd '$SCRIPT_DIR' && source .env && python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000\""
    else
        # Linux and others
        if command -v gnome-terminal >/dev/null 2>&1; then
            gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && source .env && python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000; exec bash"
        else
            # Fallback to background process
            cd "$SCRIPT_DIR" && python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000 &
        fi
    fi
    
    echo "Backend server started."
}

# Function to start the frontend
start_frontend() {
    echo "Starting frontend development server..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        osascript -e "tell app \"Terminal\" to do script \"cd '$SCRIPT_DIR/frontend' && npm run dev\""
    else
        # Linux and others
        if command -v gnome-terminal >/dev/null 2>&1; then
            gnome-terminal -- bash -c "cd '$SCRIPT_DIR/frontend' && npm run dev; exec bash"
        else
            # Fallback to background process
            cd "$SCRIPT_DIR/frontend" && npm run dev &
        fi
    fi
    
    echo "Frontend development server started."
}

# Main execution
echo "Starting Sherlog Canvas..."

# Check prerequisites
check_docker
check_npm

# Start infrastructure
start_qdrant

# Install MCP servers if needed
install_mcp_servers

# Start application components
start_backend

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Start frontend
start_frontend

echo "Sherlog Canvas is now running!"
echo "Frontend: http://localhost:5173"
echo "Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the services"

# Keep script running until interrupted
trap "echo 'Shutting down Sherlog Canvas...'" INT
read -r