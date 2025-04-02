#!/bin/bash
# Sherlog Canvas Launcher Script
# This script starts all the components needed for Sherlog Canvas using Docker Compose

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
        echo "Please install Docker Compose, then try again."
        exit 1
    fi
}

# Function to check if git is installed (needed for cloning pg-mcp)
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
check_git

# Create directories needed for volumes
mkdir -p data

# Get pg-mcp repository if it doesn't exist
if [ ! -d "pg-mcp" ]; then
    echo "Cloning PostgreSQL MCP server..."
    git clone https://github.com/stuzero/pg-mcp.git
else
    echo "PostgreSQL MCP server repo already present."
fi

# Print steps as they execute
set -x

# Stop any running containers
echo "Stopping running containers..."
docker-compose down

# Remove any old images to ensure clean build
echo "Removing old Docker images..."
docker-compose rm -f

# Build all services with no cache to ensure latest changes
echo "Building Docker images..."
docker-compose build --no-cache

# Start services in detached mode, force recreate
echo "Starting containers..."
docker-compose up -d --force-recreate

# Show running containers
echo "Services started. Status:"
docker-compose ps

echo "Sherlog Canvas is now running!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8080"
echo "PostgreSQL MCP: http://localhost:9211"
echo "Grafana MCP: http://localhost:9110"
echo ""
echo "To stop all services, run:"
echo "docker-compose down"
echo ""
echo "To view logs, run:"
echo "docker-compose logs -f"