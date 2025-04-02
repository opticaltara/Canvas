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
    
    # Check for Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        echo "Warning: Docker Compose is not installed."
        echo "Docker Compose is required for running the PostgreSQL MCP server."
        echo "Please install Docker Compose, then try again."
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

# Function to check if Go is installed
check_go() {
    if ! command -v go >/dev/null 2>&1; then
        echo "Warning: Go is not installed."
        echo "For Grafana MCP server, Go is required or you need to manually download the binary."
        echo "Please install Go from https://golang.org/doc/install"
        echo ""
        echo "Do you want to continue anyway? (y/n)"
        read -r response
        if [[ "$response" != "y" ]]; then
            exit 1
        fi
    fi
}

# Function to install MCP servers
install_mcp_servers() {
    echo "Installing MCP server dependencies..."
    
    # Install PostgreSQL MCP server via Docker Compose
    if [ ! -d "pg-mcp" ]; then
        echo "Cloning PostgreSQL MCP server..."
        git clone https://github.com/stuzero/pg-mcp.git
    else
        echo "PostgreSQL MCP server already cloned."
    fi
    
    # Check if the Docker container exists and is running
    if docker ps -a | grep -q "pg-mcp"; then
        if docker ps | grep -q "pg-mcp"; then
            echo "PostgreSQL MCP server Docker container is already running."
            # Check if it's running on the correct port
            if ! docker port pg-mcp | grep -q "9201"; then
                echo "Stopping the container to restart it with the correct port..."
                docker stop pg-mcp
                docker rm pg-mcp
                
                # Now start it with the correct configuration
                cd pg-mcp
                docker-compose up -d
                cd ..
            fi
        else
            echo "PostgreSQL MCP server Docker container exists but is not running. Starting it..."
            # Remove the stopped container first
            docker rm pg-mcp
            
            # Start the container with the correct configuration
            cd pg-mcp
            docker-compose up -d
            cd ..
        fi
    else
        echo "Starting PostgreSQL MCP server with Docker Compose..."
        cd pg-mcp
        docker-compose up -d
        cd ..
    fi
    
    # Install Grafana MCP server via Go
    if ! command -v mcp-grafana >/dev/null 2>&1; then
        echo "Installing Grafana MCP server..."
        if command -v go >/dev/null 2>&1; then
            # If Go is installed, use it to install mcp-grafana
            echo "Installing mcp-grafana using Go..."
            GOBIN="$HOME/go/bin" go install github.com/grafana/mcp-grafana/cmd/mcp-grafana@latest
            
            # Add Go bin directory to PATH if it's not already there
            if [[ ":$PATH:" != *":$HOME/go/bin:"* ]]; then
                echo "Adding $HOME/go/bin to PATH"
                export PATH="$PATH:$HOME/go/bin"
                # Add to current session and suggest adding to shell profile
                echo "NOTE: To make this permanent, add 'export PATH=\$PATH:\$HOME/go/bin' to your shell profile"
            fi
        else
            echo "Go is not installed. Please install Go or manually download mcp-grafana from GitHub:"
            echo "https://github.com/grafana/mcp-grafana/releases"
            echo "and place it in your PATH."
            exit 1
        fi
    else
        echo "mcp-grafana is already installed."
    fi
    
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

# Function to check if git is installed
check_git() {
    if ! command -v git >/dev/null 2>&1; then
        echo "Error: git is not installed."
        echo "Please install git, then try again."
        exit 1
    fi
}

# Main execution
echo "Starting Sherlog Canvas..."

# Check prerequisites
check_docker
check_npm
check_go
check_git

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