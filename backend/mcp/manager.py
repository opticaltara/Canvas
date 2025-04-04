"""
MCP Server Manager

Manages the lifecycle of MCP servers for various data sources.
"""

import asyncio
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from backend.services.connection_manager import ConnectionConfig


class MCPServerManager:
    """
    Manages MCP servers for different data sources
    """
    def __init__(self):
        self.servers = {}
        self.processes = {}
        self.ready = {}
    
    async def start_mcp_servers(self, connections: List[ConnectionConfig]) -> Dict[str, str]:
        """
        Start MCP servers for the given connections
        
        Args:
            connections: List of connection configurations
            
        Returns:
            Dictionary mapping connection IDs to MCP server addresses
        """
        server_addresses = {}
        
        for connection in connections:
            try:
                address = await self.start_mcp_server(connection)
                if address:
                    server_addresses[connection.id] = address
            except Exception as e:
                print(f"Error starting MCP server for {connection.name}: {e}")
        
        return server_addresses
    
    async def start_mcp_server(self, connection: ConnectionConfig) -> Optional[str]:
        """
        Start an MCP server for a specific connection
        
        Args:
            connection: The connection configuration
            
        Returns:
            The MCP server address if successful, None otherwise
        """
        # Check if server is already running for this connection
        if connection.id in self.servers:
            return self.servers[connection.id]
        
        server_type = connection.type
        config = connection.config
        
        if server_type == "grafana":
            address = await self._start_grafana_mcp(connection)
        elif server_type == "postgres":
            address = await self._start_postgres_mcp(connection)
        else:
            print(f"Unsupported MCP server type: {server_type}")
            return None
        
        if address:
            self.servers[connection.id] = address
            return address
        
        return None
    
    async def stop_all_servers(self) -> None:
        """Stop all running MCP servers"""
        for connection_id, process in self.processes.items():
            try:
                if process is None:
                    # Skip Docker-based processes - they're managed separately
                    continue
                
                # Send SIGTERM to the process
                process.terminate()
                
                # Wait for the process to exit
                await asyncio.sleep(2)
                
                # If still running, force kill
                if process.poll() is None:
                    process.kill()
            except Exception as e:
                print(f"Error stopping MCP server for {connection_id}: {e}")
        
        self.processes = {}
        self.servers = {}
        self.ready = {}
    
    async def _start_grafana_mcp(self, connection: ConnectionConfig) -> Optional[str]:
        """
        Start a Grafana MCP server
        
        Args:
            connection: The Grafana connection configuration
            
        Returns:
            The MCP server address if successful, None otherwise
        """
        # Get configuration
        url = connection.config.get("url")
        api_key = connection.config.get("api_key")
        
        if not url or not api_key:
            print(f"Missing required configuration for Grafana MCP server: {connection.name}")
            return None
        
        # Choose a port
        port = self._find_free_port(9100, 9200)
        if not port:
            print(f"Could not find a free port for Grafana MCP server: {connection.name}")
            return None
        
        # Set environment variables
        env = os.environ.copy()
        env["GRAFANA_URL"] = url
        env["GRAFANA_API_KEY"] = api_key
        
        # Start the MCP server
        try:
            cmd = [
                "mcp-grafana", 
                "--port", str(port)
            ]
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[connection.id] = process
            
            # Wait for server to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if process.poll() is not None:
                stderr = ""
                if process.stderr:
                    stderr = process.stderr.read() or ""
                print(f"Grafana MCP server failed to start: {stderr}")
                return None
            
            # Check if we're running in Docker
            is_docker = os.path.exists('/.dockerenv')
            
            # Get the host name based on environment
            if is_docker:
                # Use the Docker Compose service name
                # The internal port is 8000, not 9100
                host = "sherlog-canvas-mcp-grafana"
                port = 8000
                address = f"{host}:{port}"
            else:
                # Use localhost when running outside Docker
                address = f"localhost:{port}"
                
            print(f"Started Grafana MCP server for {connection.name} at {address}")
            return address
        
        except Exception as e:
            print(f"Error starting Grafana MCP server: {e}")
            return None
    
    async def _start_postgres_mcp(self, connection: ConnectionConfig) -> Optional[str]:
        """
        Start a Postgres MCP server
        
        Args:
            connection: The Postgres connection configuration
            
        Returns:
            The MCP server address if successful, None otherwise
        """
        # Get configuration
        connection_string = connection.config.get("connection_string")
        
        if not connection_string:
            print(f"Missing required configuration for Postgres MCP server: {connection.name}")
            return None
        
        # Choose a port
        port = self._find_free_port(9201, 9300)
        if not port:
            print(f"Could not find a free port for Postgres MCP server: {connection.name}")
            return None
        
        # Set environment variables
        env = os.environ.copy()
        env["PG_CONNECTION_STRING"] = connection_string
        
        # Use the Docker container instead of launching a new process
        try:
            # The PostgreSQL MCP server is already running in Docker
            # We don't need to start a new process, just return the address
            # When using Docker Compose, we use the service name and internal port
            # External port: 9201, Internal port: 8000
            port = 8000
            
            # We don't need to start a process as it's running in Docker
            # Just record a placeholder in the processes dict for cleanup tracking
            self.processes[connection.id] = None
            
            # In Docker Compose, use the service name as the host
            # This works when running in Docker, but localhost is needed when running locally
            # Let's check if we're running in Docker
            is_docker = os.path.exists('/.dockerenv')
            
            # Get the host name based on environment
            if is_docker:
                # Use the Docker Compose service name
                host = "sherlog-canvas-pg-mcp"
            else:
                # Use localhost with external port when running outside Docker
                host = "localhost"
                port = 9201
            
            address = f"{host}:{port}"
            print(f"Started Postgres MCP server for {connection.name} at {address}")
            return address
        
        except Exception as e:
            print(f"Error starting Postgres MCP server: {e}")
            return None

    def _find_free_port(self, start_port: int, end_port: int) -> Optional[int]:
        """Find a free port in the given range"""
        import socket
        
        for port in range(start_port, end_port + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("localhost", port))
                sock.close()
                return port
            except OSError:
                continue
        
        return None


# Singleton instance
_mcp_server_manager_instance = None


def get_mcp_server_manager():
    """Get the singleton MCPServerManager instance"""
    global _mcp_server_manager_instance
    if _mcp_server_manager_instance is None:
        _mcp_server_manager_instance = MCPServerManager()
    return _mcp_server_manager_instance