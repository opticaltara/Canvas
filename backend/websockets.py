import asyncio
import json
from typing import Dict, Set
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect

from backend.core.logging import get_logger

websocket_logger = get_logger('websocket')

# Enhanced WebSocket manager with detailed logging
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[UUID, Set[WebSocket]] = {}
        self.logger = websocket_logger
    
    async def connect(self, websocket: WebSocket, notebook_id: UUID):
        await websocket.accept()
        if notebook_id not in self.active_connections:
            self.active_connections[notebook_id] = set()
        self.active_connections[notebook_id].add(websocket)
        self.logger.info(
            f"WebSocket connected", 
            extra={'notebook_id': str(notebook_id), 'client': websocket.client}
        )

    def disconnect(self, websocket: WebSocket, notebook_id: UUID):
        if notebook_id in self.active_connections:
            self.active_connections[notebook_id].remove(websocket)
            if not self.active_connections[notebook_id]:
                del self.active_connections[notebook_id]
            self.logger.info(
                f"WebSocket disconnected", 
                extra={'notebook_id': str(notebook_id), 'client': websocket.client}
            )
        else:
             self.logger.warning(
                f"Attempted to disconnect WebSocket for non-existent notebook ID",
                extra={'notebook_id': str(notebook_id)}
            )


    async def broadcast(self, notebook_id: UUID, message: Dict):
        """Broadcast a message to all connected clients for a specific notebook."""
        if notebook_id not in self.active_connections:
             self.logger.warning(
                f"Attempted broadcast to notebook with no active connections",
                extra={'notebook_id': str(notebook_id)}
             )
             return

        disconnected_clients = set()
        message_json = json.dumps(message)
        # Create a list of tasks for sending messages
        tasks = []
        
        # Iterate over a copy of the set to allow modification during iteration
        connections = list(self.active_connections.get(notebook_id, set()))
        self.logger.debug(f"Broadcasting to {len(connections)} clients for notebook {notebook_id}")
        
        for connection in connections:
            tasks.append(self._send_message(connection, message_json, notebook_id, disconnected_clients))

        # Wait for all send tasks to complete
        if tasks:
            await asyncio.gather(*tasks)

        # Remove disconnected clients
        if disconnected_clients:
            self.logger.info(f"Removing {len(disconnected_clients)} disconnected clients for notebook {notebook_id}")
            if notebook_id in self.active_connections:
                 self.active_connections[notebook_id].difference_update(disconnected_clients)
                 # Clean up the entry if no connections remain
                 if not self.active_connections[notebook_id]:
                     del self.active_connections[notebook_id]


    async def _send_message(self, websocket: WebSocket, message_json: str, notebook_id: UUID, disconnected_clients: Set[WebSocket]):
        """Helper function to send a message and handle potential disconnections."""
        try:
            await websocket.send_text(message_json)
            self.logger.debug(
                f"Sent message to WebSocket", 
                extra={'notebook_id': str(notebook_id), 'client': websocket.client}
            )
        except WebSocketDisconnect:
            self.logger.info(
                f"WebSocket disconnected during broadcast", 
                extra={'notebook_id': str(notebook_id), 'client': websocket.client}
            )
            disconnected_clients.add(websocket)
        except Exception as e:
            # Catch other potential exceptions during send
            self.logger.error(
                f"Error sending message to WebSocket: {str(e)}",
                extra={'notebook_id': str(notebook_id), 'client': websocket.client},
                exc_info=True
            )
            disconnected_clients.add(websocket) # Assume client is lost if send fails


# Singleton instance of the WebSocket manager
ws_manager = WebSocketManager()

def get_ws_manager() -> WebSocketManager:
    """Dependency function to get the WebSocket manager instance."""
    return ws_manager 