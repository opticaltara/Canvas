"""
MCP Server Manager

Manages the lifecycle of MCP servers for various data sources.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from backend.services.connection_manager import ConnectionConfig

# Initialize logger
mcp_logger = logging.getLogger("mcp")

class MCPServerStatus:
    """Status of an MCP server"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"

class MCPServerManager:
    """
    Manages MCP servers for different data sources
    """
    def __init__(self):
        self.servers = {}  # Maps connection_id to server address
        self.processes = {}  # Maps connection_id to process
        self.status = {}  # Maps connection_id to server status
        self.last_error = {}  # Maps connection_id to last error message
        mcp_logger.info(
            "MCP Server Manager initialized",
            extra={
                'active_servers': len(self.servers)
            }
        )
    
    async def start_mcp_servers(self, connections: List[ConnectionConfig]) -> Dict[str, str]:
        """
        Start MCP servers for the given connections
        
        Args:
            connections: List of connection configurations
            
        Returns:
            Dictionary mapping connection IDs to MCP server addresses
        """
        start_time = time.time()
        server_addresses = {}
        
        for connection in connections:
            try:
                address = await self.start_mcp_server(connection)
                if address:
                    server_addresses[connection.id] = address
            except Exception as e:
                mcp_logger.error(
                    "Error starting MCP server",
                    extra={
                        'connection_id': connection.id,
                        'connection_name': connection.name,
                        'error': str(e),
                        'error_type': type(e).__name__
                    },
                    exc_info=True
                )
        
        process_time = time.time() - start_time
        mcp_logger.info(
            "MCP servers startup complete",
            extra={
                'total_connections': len(connections),
                'successful_starts': len(server_addresses),
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
        
        return server_addresses
    
    async def start_mcp_server(self, connection: ConnectionConfig) -> Optional[str]:
        """
        Start an appropriate MCP server for the given connection

        Args:
            connection: The connection configuration

        Returns:
            The MCP server address if successful, None otherwise
        """
        if not connection or not connection.id or not connection.type:
            return None

        # Check if server is already running for this connection
        status = self.get_server_status(connection.id)["status"]
        if status == MCPServerStatus.RUNNING:
            # Server is already running, return the address
            return self.servers.get(connection.id)

        # Set status to starting
        self.status[connection.id] = MCPServerStatus.STARTING
        self.last_error[connection.id] = None

        try:
            # Start the appropriate server based on connection type
            server_type = connection.type.lower()
            address = None

            if server_type == "grafana":
                address = await self._start_grafana_mcp(connection)
            elif server_type == "kubernetes":
                address = await self._start_kubernetes_mcp(connection)
            elif server_type == "python":
                address = await self._start_python_mcp(connection)

            if address:
                self.servers[connection.id] = address
                self.status[connection.id] = MCPServerStatus.RUNNING
                return address
            else:
                self.status[connection.id] = MCPServerStatus.FAILED
                if not self.last_error.get(connection.id):
                    self.last_error[connection.id] = f"Failed to start {server_type} MCP server"
                return None

        except Exception as e:
            self.status[connection.id] = MCPServerStatus.FAILED
            self.last_error[connection.id] = str(e)
            return None
    
    async def stop_all_servers(self) -> None:
        """Stop all running MCP servers"""
        start_time = time.time()
        stopped_count = 0
        error_count = 0
        
        for connection_id, process in self.processes.items():
            try:
                if process is None:
                    # Skip Docker-based processes - they're managed separately
                    continue
                
                mcp_logger.info(
                    "Stopping MCP server",
                    extra={
                        'connection_id': connection_id,
                        'server_address': self.servers.get(connection_id)
                    }
                )
                
                # Send SIGTERM to the process
                process.terminate()
                
                # Wait for the process to exit
                await asyncio.sleep(2)
                
                # If still running, force kill
                if process.poll() is None:
                    process.kill()
                    mcp_logger.warning(
                        "Force killed MCP server process",
                        extra={
                            'connection_id': connection_id
                        }
                    )
                
                # Update status
                self.status[connection_id] = MCPServerStatus.STOPPED
                stopped_count += 1
                
                mcp_logger.info(
                    "MCP server stopped successfully",
                    extra={
                        'connection_id': connection_id
                    }
                )
            except Exception as e:
                error_count += 1
                self.last_error[connection_id] = str(e)
                mcp_logger.error(
                    "Error stopping MCP server",
                    extra={
                        'connection_id': connection_id,
                        'error': str(e),
                        'error_type': type(e).__name__
                    },
                    exc_info=True
                )
        
        # Clear all server data
        self.processes = {}
        self.servers = {}
        
        process_time = time.time() - start_time
        mcp_logger.info(
            "All MCP servers stopped",
            extra={
                'total_servers': stopped_count + error_count,
                'successful_stops': stopped_count,
                'failed_stops': error_count,
                'processing_time_ms': round(process_time * 1000, 2)
            }
        )
    
    async def stop_server(self, connection_id: str) -> bool:
        """
        Stop a specific MCP server
        
        Args:
            connection_id: The ID of the connection to stop
            
        Returns:
            True if stopped successfully, False otherwise
        """
        start_time = time.time()
        
        if connection_id not in self.processes:
            self.status[connection_id] = MCPServerStatus.STOPPED
            mcp_logger.info(
                "MCP server already stopped",
                extra={
                    'connection_id': connection_id
                }
            )
            return True
            
        process = self.processes[connection_id]
        
        try:
            if process is None:
                # For Docker-based processes, we just update the status
                self.status[connection_id] = MCPServerStatus.STOPPED
                del self.processes[connection_id]
                if connection_id in self.servers:
                    del self.servers[connection_id]
                mcp_logger.info(
                    "Docker-based MCP server marked as stopped",
                    extra={
                        'connection_id': connection_id
                    }
                )
                return True
            
            mcp_logger.info(
                "Stopping MCP server",
                extra={
                    'connection_id': connection_id,
                    'server_address': self.servers.get(connection_id)
                }
            )
            
            # Send SIGTERM to the process
            process.terminate()
            
            # Wait for the process to exit
            await asyncio.sleep(2)
            
            # If still running, force kill
            if process.poll() is None:
                process.kill()
                mcp_logger.warning(
                    "Force killed MCP server process",
                    extra={
                        'connection_id': connection_id
                    }
                )
            
            # Update status
            self.status[connection_id] = MCPServerStatus.STOPPED
            
            # Remove from active processes and servers
            del self.processes[connection_id]
            if connection_id in self.servers:
                del self.servers[connection_id]
            
            process_time = time.time() - start_time
            mcp_logger.info(
                "MCP server stopped successfully",
                extra={
                    'connection_id': connection_id,
                    'processing_time_ms': round(process_time * 1000, 2)
                }
            )
                
            return True
        except Exception as e:
            process_time = time.time() - start_time
            self.last_error[connection_id] = str(e)
            mcp_logger.error(
                "Error stopping MCP server",
                extra={
                    'connection_id': connection_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'processing_time_ms': round(process_time * 1000, 2)
                },
                exc_info=True
            )
            return False
            
    def get_server_status(self, connection_id: str) -> Dict:
        """
        Get the status of an MCP server
        
        Args:
            connection_id: The ID of the connection
            
        Returns:
            Dictionary with status information
        """
        status_info = {
            "status": self.status.get(connection_id, MCPServerStatus.STOPPED),
            "address": self.servers.get(connection_id),
            "error": self.last_error.get(connection_id)
        }
        
        mcp_logger.debug(
            "Retrieved server status",
            extra={
                'connection_id': connection_id,
                'status': status_info["status"],
                'has_error': status_info["error"] is not None
            }
        )
        
        return status_info
        
    def get_all_server_statuses(self) -> Dict[str, Dict]:
        """
        Get the status of all MCP servers
        
        Returns:
            Dictionary mapping connection IDs to status information
        """
        statuses = {}
        for connection_id in set(list(self.status.keys()) + list(self.servers.keys())):
            statuses[connection_id] = self.get_server_status(connection_id)
        return statuses
    
    async def _start_grafana_mcp(self, connection: ConnectionConfig) -> Optional[str]:
        """
        Start a Grafana MCP server
        
        Args:
            connection: The Grafana connection configuration
            
        Returns:
            The MCP server address if successful, None otherwise
        """
        # Get configuration
        grafana_url = connection.config.get("url")
        api_key = connection.config.get("api_key")
        
        if not grafana_url or not api_key:
            self.last_error[connection.id] = "Missing required configuration for Grafana MCP server"
            print(f"Missing required configuration for Grafana MCP server: {connection.name}")
            return None
            
        # Choose a port for local development (when not using Docker)
        port = self._find_free_port(9101, 9200)
        if not port:
            self.last_error[connection.id] = f"Could not find a free port for Grafana MCP server"
            print(f"Could not find a free port for Grafana MCP server: {connection.name}")
            return None
            
        # Set environment variables
        env = os.environ.copy()
        env["GRAFANA_URL"] = grafana_url
        env["GRAFANA_API_KEY"] = api_key
            
        try:
            # When running in Docker, no need to start a process
            # Check if we're running in Docker
            is_docker = os.path.exists('/.dockerenv')
            
            # For Docker, we don't need to start a process
            # Just record a placeholder in the processes dict for cleanup tracking
            self.processes[connection.id] = None
            
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
    
    async def _start_kubernetes_mcp(self, connection: ConnectionConfig) -> Optional[str]:
        """
        Start a Kubernetes MCP server
        
        Args:
            connection: The Kubernetes connection configuration
            
        Returns:
            The MCP server address if successful, None otherwise
        """
        # Get configuration
        kubeconfig = connection.config.get("kubeconfig")
        context = connection.config.get("context")
        
        if not kubeconfig:
            self.last_error[connection.id] = "Missing required kubeconfig for Kubernetes MCP server"
            print(f"Missing required kubeconfig for Kubernetes MCP server: {connection.name}")
            return None
        
        # Choose a port
        port = self._find_free_port(9301, 9400)
        if not port:
            self.last_error[connection.id] = f"Could not find a free port for Kubernetes MCP server"
            print(f"Could not find a free port for Kubernetes MCP server: {connection.name}")
            return None
        
        # Set environment variables
        env = os.environ.copy()
        
        # Write the kubeconfig to a temporary file if it's provided as content
        # Otherwise use the path directly
        kubeconfig_path = None
        if kubeconfig.startswith("{") or kubeconfig.startswith("apiVersion:"):
            # This is a kubeconfig content rather than a path
            import tempfile
            kubeconfig_fd, kubeconfig_path = tempfile.mkstemp(suffix=".yaml")
            with os.fdopen(kubeconfig_fd, 'w') as f:
                f.write(kubeconfig)
            env["KUBECONFIG"] = kubeconfig_path
        else:
            # This is a path to the kubeconfig file
            env["KUBECONFIG"] = os.path.expanduser(kubeconfig)
            
        # Set context if provided
        if context:
            env["KUBE_CONTEXT"] = context
            
        # Start the MCP server using the kubernetes MCP package
        try:
            # For local development, use node to run our custom server
            cmd = [
                "node",
                "mcp-kubernetes/server.js"
            ]
            # Add port as an environment variable
            env["PORT"] = str(port)
            
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
                self.last_error[connection.id] = f"Kubernetes MCP server failed to start: {stderr}"
                print(f"Kubernetes MCP server failed to start: {stderr}")
                
                # Clean up the temp file if we created one
                if kubeconfig_path and kubeconfig.startswith(("{", "apiVersion:")):
                    try:
                        os.unlink(kubeconfig_path)
                    except:
                        pass
                        
                return None
            
            # Check if we're running in Docker
            is_docker = os.path.exists('/.dockerenv')
            
            # Get the host name based on environment
            if is_docker:
                # Use the Docker Compose service name
                host = "sherlog-canvas-mcp-kubernetes"
                # The internal port is 8000, not our dynamic port
                port = 8000
            else:
                # Use localhost when running outside Docker
                host = "localhost" 
                # Use the dynamically assigned port
                
            address = f"{host}:{port}"
            print(f"Started Kubernetes MCP server for {connection.name} at {address}")
            return address
        
        except Exception as e:
            self.last_error[connection.id] = f"Error starting Kubernetes MCP server: {str(e)}"
            print(f"Error starting Kubernetes MCP server: {e}")
            
            # Clean up the temp file if we created one
            if kubeconfig_path and kubeconfig.startswith(("{", "apiVersion:")):
                try:
                    os.unlink(kubeconfig_path)
                except:
                    pass
                    
            return None
            
    async def _start_python_mcp(self, connection: ConnectionConfig) -> Optional[str]:
        """
        Start a Python MCP server for sandboxed code execution using Pydantic MCP
        
        Args:
            connection: The Python connection configuration
            
        Returns:
            The MCP server address if successful, None otherwise
        """
        # For Python MCP, there's no specific configuration needed since
        # it's a utility server rather than a data connection
        
        # Choose a port for local development (when not using Docker)
        port = self._find_free_port(9401, 9500)
        if not port:
            self.last_error[connection.id] = f"Could not find a free port for Python MCP server"
            print(f"Could not find a free port for Python MCP server: {connection.name}")
            return None
        
        # Set environment variables
        env = os.environ.copy()
            
        # Start the MCP server for Python execution
        try:
            # Check if we're running in Docker
            is_docker = os.path.exists('/.dockerenv')
            
            if not is_docker:
                # For local development, we'll use Deno to start the server
                cmd = [
                    "deno", "run", "-A", "--unstable-worker-options",
                    "jsr:@pydantic/mcp-run-python", "sse"
                ]
                
                # Add port as an environment variable
                env["PORT"] = str(port)
                
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
                    self.last_error[connection.id] = f"Python MCP server failed to start: {stderr}"
                    print(f"Python MCP server failed to start: {stderr}")
                    return None
                
                host = "localhost"
            else:
                # In Docker, we don't need to start a process as it's managed by Docker Compose
                # Just record a placeholder in the processes dict for cleanup tracking
                self.processes[connection.id] = None
                
                # Use the Docker Compose service name
                host = "sherlog-canvas-mcp-python"
                # The internal port is 8000, not our dynamic port
                port = 8000
            
            address = f"{host}:{port}"
            print(f"Started Python MCP server for {connection.name} at {address}")
            return address
        
        except Exception as e:
            self.last_error[connection.id] = f"Error starting Python MCP server: {str(e)}"
            print(f"Error starting Python MCP server: {e}")
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