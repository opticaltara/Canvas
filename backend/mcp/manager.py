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
from uuid import uuid4

from backend.services.connection_manager import BaseConnectionConfig
from backend.core.logging import get_logger

# Initialize logger
mcp_logger = get_logger('mcp')

# Make sure correlation_id is set for all logs
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

# Add the filter to our logger if not already added by server.py
mcp_logger.addFilter(CorrelationIdFilter())

# Ensure at least a basic console handler is attached if none exists
# This is a safeguard in case this module is imported before logging is configured
if not mcp_logger.handlers:
    # Check if logger has a parent with handlers
    has_parent_handlers = mcp_logger.parent and hasattr(mcp_logger.parent, 'handlers') and mcp_logger.parent.handlers
    
    if not has_parent_handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        mcp_logger.addHandler(console_handler)
        mcp_logger.setLevel(logging.INFO)

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
    
    async def start_mcp_servers(self, connections: List[BaseConnectionConfig]) -> Dict[str, str]:
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
    
    async def start_mcp_server(self, connection: BaseConnectionConfig) -> Optional[str]:
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

        correlation_id = str(uuid4())
        mcp_logger.info(
            f"Starting MCP server for connection",
            extra={
                'correlation_id': correlation_id,
                'connection_id': connection.id,
                'connection_name': connection.name,
                'connection_type': connection.type
            }
        )

        try:
            # Start the appropriate server based on connection type
            server_type = connection.type.lower()
            address = None

            mcp_logger.info(
                f"Using MCP server type",
                extra={
                    'correlation_id': correlation_id,
                    'connection_id': connection.id,
                    'server_type': server_type
                }
            )

            if server_type == "grafana":
                address = await self._start_grafana_mcp(connection)
            elif server_type == "python":
                address = await self._start_python_mcp(connection)
            elif server_type == "github":
                address = await self._start_github_mcp(connection)
            else:
                error_msg = f"Unsupported MCP server type: {server_type}"
                mcp_logger.error(
                    error_msg,
                    extra={
                        'correlation_id': correlation_id,
                        'connection_id': connection.id
                    }
                )
                self.last_error[connection.id] = error_msg

            if address:
                self.servers[connection.id] = address
                self.status[connection.id] = MCPServerStatus.RUNNING
                mcp_logger.info(
                    "MCP server started successfully",
                    extra={
                        'correlation_id': correlation_id,
                        'connection_id': connection.id,
                        'server_address': address
                    }
                )
                return address
            else:
                self.status[connection.id] = MCPServerStatus.FAILED
                if not self.last_error.get(connection.id):
                    self.last_error[connection.id] = f"Failed to start {server_type} MCP server"
                
                mcp_logger.error(
                    "Failed to start MCP server",
                    extra={
                        'correlation_id': correlation_id,
                        'connection_id': connection.id,
                        'error': self.last_error.get(connection.id)
                    }
                )
                return None

        except Exception as e:
            self.status[connection.id] = MCPServerStatus.FAILED
            self.last_error[connection.id] = str(e)
            
            mcp_logger.error(
                "Exception starting MCP server",
                extra={
                    'correlation_id': correlation_id,
                    'connection_id': connection.id,
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
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
                "MCP server process not found (already stopped or never started?)",
                extra={
                    'connection_id': connection_id
                }
            )
            # Clear server address if present
            self.servers.pop(connection_id, None)
            return True
            
        process = self.processes[connection_id]
        
        try:
            mcp_logger.info(
                "Stopping MCP server process/container",
                extra={
                    'connection_id': connection_id,
                    'server_address': self.servers.get(connection_id),
                    'pid': process.pid if process else 'N/A'
                }
            )
            
            # Send SIGTERM to the process (docker run)
            process.terminate()
            
            # Wait for the process to exit (give docker run time to stop/rm container)
            await asyncio.sleep(2) # Adjust if needed
            
            # If still running, force kill
            if process.poll() is None:
                process.kill()
                mcp_logger.warning(
                    "Force killed MCP server process (docker run)",
                    extra={
                        'connection_id': connection_id
                    }
                )
            
            # Update status
            self.status[connection_id] = MCPServerStatus.STOPPED
            
            # Remove from active processes and servers
            self.processes.pop(connection_id, None) # Use pop with default
            self.servers.pop(connection_id, None) # Use pop with default
            
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
            # Attempt cleanup even on error
            self.processes.pop(connection_id, None)
            self.servers.pop(connection_id, None)
            self.status[connection_id] = MCPServerStatus.FAILED # Mark as failed if stop errored
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
        
        mcp_logger.info(
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
    
    async def _start_grafana_mcp(self, connection: BaseConnectionConfig) -> Optional[str]:
        """
        Start a Grafana MCP server by building and running the local Dockerfile.
        
        Args:
            connection: The Grafana connection configuration
            
        Returns:
            The MCP server address if successful, None otherwise
        """
        correlation_id = str(uuid4())
        mcp_logger.info(
            "Attempting to start Grafana MCP server from Dockerfile",
            extra={
                'correlation_id': correlation_id,
                'connection_id': connection.id,
                'connection_name': connection.name
            }
        )
        
        # Get configuration from the config field
        grafana_url = connection.config.get("url")
        api_key = connection.config.get("api_key")
        
        if not grafana_url or not api_key:
            error_msg = "Missing required configuration (url, api_key) for Grafana MCP server"
            self.last_error[connection.id] = error_msg
            mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id})
            return None

        dockerfile_path = "./Dockerfile.mcp-grafana" # Corrected filename
        image_tag = "sherlog/grafana-mcp-server:latest"
        build_context = "." # Assuming Dockerfile is at root or build context is root

        # --- Build Image Step (Consider optimizing this) --- 
        # TODO: Optimize image building - check existence first or build outside this flow.
        build_cmd = [
            "docker", "build", "-t", image_tag, "-f", dockerfile_path, build_context
        ]
        try:
            mcp_logger.info(f"Building Grafana MCP image: {' '.join(build_cmd)}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
            build_process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await build_process.communicate()
            if build_process.returncode != 0:
                error_msg = f"Failed to build Grafana MCP image. Exit code: {build_process.returncode}. Stderr: {stderr.decode(errors='ignore')}"
                mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                self.last_error[connection.id] = error_msg
                return None
            mcp_logger.info(f"Successfully built Grafana MCP image: {image_tag}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
        except Exception as build_e:
            error_msg = f"Error during Grafana MCP image build: {build_e}"
            mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id}, exc_info=True)
            self.last_error[connection.id] = error_msg
            return None
        # --- End Build Image Step --- 
            
        # Find a free port for the host machine
        container_port = 8000 # Assuming Dockerfile exposes 8000
        host_port = self._find_free_port(start_port=9101, end_port=9200) # Example range
        if not host_port:
            error_msg = "Could not find a free port for Grafana MCP server"
            self.last_error[connection.id] = error_msg
            mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id})
            return None
            
        # Prepare docker run command
        run_cmd = [
            "docker", "run", "-i", "--rm",
            "-p", f"{host_port}:{container_port}",
            "-e", f"GRAFANA_URL={grafana_url}",
            "-e", f"GRAFANA_API_KEY={api_key}",
            image_tag
        ]
            
        try:
            mcp_logger.info(f"Running Docker command: {' '.join(run_cmd)}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
            process = await asyncio.create_subprocess_exec(
                *run_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.processes[connection.id] = process
            mcp_logger.info(
                f"Grafana MCP Docker process started (PID: {process.pid}) for connection {connection.id}",
                 extra={'correlation_id': correlation_id}
            )

            await asyncio.sleep(3) # Give server time to start

            if process.returncode is not None:
                 stderr_output = await process.stderr.read() if process.stderr else b''
                 error_msg = f"Grafana MCP Docker process failed to start or exited prematurely. Exit code: {process.returncode}. Stderr: {stderr_output.decode(errors='ignore')}"
                 mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                 self.last_error[connection.id] = error_msg
                 self.processes.pop(connection.id, None)
                 return None

            server_address = f"http://localhost:{host_port}/sse"
            mcp_logger.info(f"Grafana MCP server assumed running at {server_address}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
            return server_address
        
        except Exception as e:
            error_msg = f"Failed to start Grafana MCP Docker container: {str(e)}"
            mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id}, exc_info=True)
            self.last_error[connection.id] = error_msg
            if connection.id in self.processes and self.processes[connection.id]:
                 try:
                     self.processes[connection.id].terminate()
                 except ProcessLookupError:
                      pass 
                 self.processes.pop(connection.id, None)
            return None
    
    async def _start_python_mcp(self, connection: BaseConnectionConfig) -> Optional[str]:
        """
        Start a Python MCP server for sandboxed code execution using Pydantic MCP
        
        Args:
            connection: The Python connection configuration
            
        Returns:
            The MCP server address if successful, None otherwise
        """
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

    async def _start_github_mcp(self, connection: BaseConnectionConfig) -> Optional[str]:
        """Starts the GitHub MCP server wrapped by Supergateway over SSE using npx."""
        correlation_id = str(uuid4())
        mcp_logger.info(
            "Attempting to start GitHub MCP server via Supergateway (npx)",
            extra={
                'correlation_id': correlation_id,
                'connection_id': connection.id
            }
        )
        
        token = connection.config.get("github_personal_access_token")
        if not token:
            error_msg = "GitHub Personal Access Token not found in connection config."
            mcp_logger.error(
                error_msg,
                extra={
                    'correlation_id': correlation_id,
                    'connection_id': connection.id
                }
            )
            self.last_error[connection.id] = error_msg
            return None

        # Find a free port on the host/container for Supergateway to listen on
        supergateway_host_port = self._find_free_port(start_port=8100, end_port=8200)
        if supergateway_host_port is None:
            error_msg = "Could not find a free port for GitHub MCP Supergateway (npx)."
            mcp_logger.error(
                error_msg,
                extra={
                    'correlation_id': correlation_id,
                    'connection_id': connection.id
                }
            )
            self.last_error[connection.id] = error_msg
            return None

        github_mcp_stdio_command = (
            f"docker run -i --rm "
            f"-e GITHUB_PERSONAL_ACCESS_TOKEN='{token}' "
            f"ghcr.io/github/github-mcp-server"
        )
        
        # Define the command to run Supergateway via npx
        sse_path = "/sse" # Default SSE path for Supergateway itself

        cmd = [
            "npx", "-y", "supergateway", # Use npx to fetch and run supergateway
            "--stdio", github_mcp_stdio_command, # The command for supergateway to run via stdio
            "--port", str(supergateway_host_port), # Port for Supergateway to listen on
            "--ssePath", sse_path, # SSE path for Supergateway
        ]

        try:
            mcp_logger.info(f"Running Supergateway npx command: {' '.join(cmd)}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
            
            # Start the Supergateway npx process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.processes[connection.id] = process # Store the supergateway npx process
            mcp_logger.info(
                f"GitHub MCP Supergateway npx process started (PID: {process.pid}) for connection {connection.id}",
                 extra={'correlation_id': correlation_id}
            )

            # --- Check Supergateway Output (Refactored) ---
            output_lines = []
            flags = {"confirmed": False, "error": False}
            start_time = asyncio.get_event_loop().time()

            async def _read_stream_and_check(stream, stream_name):
                if not stream:
                    return
                try:
                    async for line_bytes in stream:
                        if flags["confirmed"] or flags["error"]:
                            break # Stop reading if outcome is decided
                        line = line_bytes.decode(errors='ignore').strip()
                        if line:
                            output_lines.append(f"[{stream_name}] {line}")
                            mcp_logger.info(f"Supergateway Output ({connection.id}, {stream_name}): {line}", extra={'correlation_id': correlation_id})
                            # Check for success
                            if "GitHub MCP Server running on stdio" in line:
                                mcp_logger.info("GitHub MCP Server confirmed running via Supergateway output.", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                                flags["confirmed"] = True
                                break
                            # Check for known error patterns
                            if "[supergateway] stdio: docker" in line or "Usage:  docker [OPTIONS] COMMAND" in line:
                                mcp_logger.error("Supergateway output indicates stdio command failed (Docker help shown).", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                                flags["error"] = True
                                break
                        # Check timeout explicitly within the loop as well
                        if asyncio.get_event_loop().time() - start_time > 30.0:
                            break
                except Exception as e:
                    # Log errors reading from the stream but don't necessarily stop the check
                    mcp_logger.warning(f"Error reading from Supergateway {stream_name}: {e}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})

            reader_tasks = []
            if process.stdout:
                reader_tasks.append(_read_stream_and_check(process.stdout, "stdout"))
            if process.stderr:
                reader_tasks.append(_read_stream_and_check(process.stderr, "stderr"))

            if not reader_tasks:
                mcp_logger.error("Supergateway process has no stdout/stderr streams.", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                flags["error"] = True
            else:
                try:
                    # Run readers concurrently with a 30s timeout
                    await asyncio.wait_for(asyncio.gather(*reader_tasks), timeout=30.0)
                except asyncio.TimeoutError:
                    mcp_logger.warning("Timeout waiting for Supergateway startup confirmation.", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                    # Check flags even on timeout, confirmation might have just happened
                except Exception as gather_exc:
                    mcp_logger.error(f"Error during Supergateway stream reading: {gather_exc}", extra={'correlation_id': correlation_id, 'connection_id': connection.id}, exc_info=True)
                    flags["error"] = True # Treat gather errors as failure

            # Final check based on flags and process status
            if process.returncode is not None:
                 mcp_logger.error(f"Supergateway process exited during/after startup check. Code: {process.returncode}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                 flags["error"] = True # Process exiting is an error if not confirmed

            if not flags["confirmed"] or flags["error"]:
                error_msg = f"Failed to confirm GitHub MCP server startup via Supergateway. Final status - Confirmed: {flags['confirmed']}, Error: {flags['error']}. Output: {chr(10).join(output_lines)}" # Use newline for readability
                mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                self.last_error[connection.id] = error_msg
                # Attempt to terminate the potentially problematic process
                if process.returncode is None:
                    try:
                        process.terminate()
                        # Optionally wait briefly for termination
                        # await asyncio.wait_for(process.wait(), timeout=2.0)
                    except ProcessLookupError:
                         pass # Already gone
                    except Exception as term_exc:
                         mcp_logger.warning(f"Error terminating Supergateway process during cleanup: {term_exc}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                self.processes.pop(connection.id, None)
                return None
            # --- End Check Supergateway Output ---


            # Construct the SSE address exposed by Supergateway (running via npx)
            # Assuming access via localhost within the backend container's context
            server_address = f"http://localhost:{supergateway_host_port}{sse_path}"
            mcp_logger.info(f"GitHub MCP server via Supergateway (npx) confirmed running at {server_address}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
            return server_address

        except FileNotFoundError:
             error_msg = "Failed to start GitHub MCP Supergateway: 'npx' command not found. Is Node.js/npx installed in the environment?"
             mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id}, exc_info=True)
             self.last_error[connection.id] = error_msg
             self.processes.pop(connection.id, None) # Clean up potential partial process entry
             return None
        except Exception as e:
            error_msg = f"Failed to start GitHub MCP Supergateway via npx: {str(e)}"
            mcp_logger.error(error_msg, extra={'correlation_id': correlation_id, 'connection_id': connection.id}, exc_info=True)
            self.last_error[connection.id] = error_msg
            # Ensure process is cleaned up if creation failed mid-way or during checks
            if connection.id in self.processes and self.processes[connection.id]:
                 process_to_term = self.processes[connection.id]
                 if process_to_term and process_to_term.returncode is None:
                     try:
                         process_to_term.terminate()
                         # Allow some time for termination
                         await asyncio.wait_for(process_to_term.wait(), timeout=2.0)
                     except asyncio.TimeoutError:
                          mcp_logger.warning(f"Supergateway process {process_to_term.pid} did not terminate gracefully, killing.", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                          process_to_term.kill()
                     except ProcessLookupError:
                         pass # Process already gone
                     except Exception as term_exc:
                         mcp_logger.warning(f"Error terminating Supergateway npx process during cleanup: {term_exc}", extra={'correlation_id': correlation_id, 'connection_id': connection.id})
                 self.processes.pop(connection.id, None) # Remove after attempting termination
            return None


_mcp_server_manager_instance = None
_manager_lock = asyncio.Lock()

async def get_mcp_server_manager_async():
    """Get the singleton MCPServerManager instance asynchronously with locking"""
    global _mcp_server_manager_instance, _manager_lock
    
    async with _manager_lock:
        if _mcp_server_manager_instance is None:
            _mcp_server_manager_instance = MCPServerManager()
            mcp_logger.info("Created new MCPServerManager instance")
        return _mcp_server_manager_instance

def get_mcp_server_manager():
    """Get the singleton MCPServerManager instance"""
    global _mcp_server_manager_instance
    
    if _mcp_server_manager_instance is None:
        # In a synchronous context, just create it directly
        _mcp_server_manager_instance = MCPServerManager()
        mcp_logger.info("Created new MCPServerManager instance (sync)")
    
    return _mcp_server_manager_instance