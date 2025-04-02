import { useState, useEffect, useRef, useCallback } from 'react';

type WebSocketStatus = 'connecting' | 'open' | 'closing' | 'closed';

interface UseWebSocketReturn {
  socket: WebSocket | null;
  status: WebSocketStatus;
  connected: boolean;
  send: (data: any) => void;
  error: Error | null;
  reconnect: () => void;
}

/**
 * React hook for WebSocket connections
 * 
 * @param url WebSocket URL to connect to (null or empty for no connection)
 * @param options Connection options
 * @returns WebSocket state and methods
 */
function useWebSocket(
  url: string | null,
  options: {
    reconnectInterval?: number;
    reconnectAttempts?: number;
    onOpen?: (event: Event) => void;
    onClose?: (event: CloseEvent) => void;
    onMessage?: (event: MessageEvent) => void;
    onError?: (event: Event) => void;
    autoReconnect?: boolean;
  } = {}
): UseWebSocketReturn {
  const {
    reconnectInterval = 3000,
    reconnectAttempts = 5,
    onOpen,
    onClose,
    onMessage,
    onError,
    autoReconnect = true,
  } = options;

  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [status, setStatus] = useState<WebSocketStatus>('closed');
  const [error, setError] = useState<Error | null>(null);
  
  const reconnectCount = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  
  // Memoize the URL to prevent unnecessary reconnections
  const urlRef = useRef(url);
  useEffect(() => {
    urlRef.current = url;
  }, [url]);
  
  // Cleanup function to close socket and clear timers
  const cleanup = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }
    
    if (socket && status !== 'closed') {
      socket.close();
    }
  }, [socket, status]);
  
  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!urlRef.current) {
      console.log('No WebSocket URL provided');
      return;
    }
    
    cleanup();
    
    try {
      const ws = new WebSocket(urlRef.current);
      setSocket(ws);
      setStatus('connecting');
      setError(null);
      
      ws.onopen = (event) => {
        console.log('WebSocket connected');
        setStatus('open');
        reconnectCount.current = 0;
        
        if (onOpen) {
          onOpen(event);
        }
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket closed');
        setStatus('closed');
        
        if (onClose) {
          onClose(event);
        }
        
        // Attempt to reconnect if not a clean close and auto-reconnect is enabled
        if (autoReconnect && !event.wasClean) {
          attemptReconnect();
        }
      };
      
      ws.onmessage = (event) => {
        if (onMessage) {
          onMessage(event);
        }
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(new Error('WebSocket error'));
        
        if (onError) {
          onError(event);
        }
      };
    } catch (err) {
      console.error('WebSocket connection error:', err);
      setError(err instanceof Error ? err : new Error(String(err)));
      
      // Attempt to reconnect on connection error
      if (autoReconnect) {
        attemptReconnect();
      }
    }
  }, [cleanup, onOpen, onClose, onMessage, onError, autoReconnect]);
  
  // Attempt to reconnect
  const attemptReconnect = useCallback(() => {
    if (reconnectCount.current >= reconnectAttempts) {
      console.log(`Max reconnect attempts (${reconnectAttempts}) reached`);
      return;
    }
    
    reconnectCount.current += 1;
    
    const timeoutMs = reconnectInterval * Math.pow(1.5, reconnectCount.current - 1);
    console.log(`Reconnecting in ${timeoutMs}ms (attempt ${reconnectCount.current}/${reconnectAttempts})`);
    
    reconnectTimerRef.current = setTimeout(() => {
      connect();
    }, timeoutMs);
  }, [connect, reconnectAttempts, reconnectInterval]);
  
  // Manual reconnect function
  const reconnect = useCallback(() => {
    reconnectCount.current = 0;
    connect();
  }, [connect]);
  
  // Send data through the WebSocket
  const send = useCallback(
    (data: any) => {
      if (socket && status === 'open') {
        socket.send(typeof data === 'string' ? data : JSON.stringify(data));
      } else {
        console.warn('Cannot send message, WebSocket is not open');
      }
    },
    [socket, status]
  );
  
  // Initialize connection when URL changes
  useEffect(() => {
    if (urlRef.current) {
      connect();
    }
    
    return cleanup;
  }, [connect, cleanup]);
  
  return {
    socket,
    status,
    connected: status === 'open',
    send,
    error,
    reconnect,
  };
}

export default useWebSocket;