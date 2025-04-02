import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export function useWebSocket(url: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Create WebSocket connection
    const socket = new WebSocket(url);
    socketRef.current = socket;

    // Connection opened
    socket.addEventListener('open', () => {
      setIsConnected(true);
    });

    // Listen for messages
    socket.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prev) => [...prev, data]);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });

    // Connection closed or error
    socket.addEventListener('close', () => {
      setIsConnected(false);
    });

    socket.addEventListener('error', (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    });

    // Cleanup on unmount
    return () => {
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close();
      }
    };
  }, [url]);

  // Send message function
  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
    } else {
      console.error('WebSocket is not connected');
    }
  }, []);

  // Clear messages
  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { isConnected, messages, sendMessage, clearMessages };
}
