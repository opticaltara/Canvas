import { useEffect, useState } from 'react';
import { BACKEND_URL } from '../config';

type ServerStatus = 'starting' | 'running' | 'stopped' | 'failed';

interface MCPServerStatusProps {
  connectionId: string;
  showControls?: boolean;
}

export function MCPServerStatus({ connectionId, showControls = false }: MCPServerStatusProps) {
  const [status, setStatus] = useState<ServerStatus>('stopped');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch status on mount and periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/connections/${connectionId}/mcp/status`);
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        setStatus(data.status as ServerStatus);
        setError(data.error);
      } catch (err: any) {
        console.error('Error fetching MCP server status:', err);
        // Don't update status on error to avoid flashing
      }
    };

    // Fetch immediately
    fetchStatus();

    // Then set up interval to fetch every 5 seconds
    const intervalId = setInterval(fetchStatus, 5000);

    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, [connectionId]);

  // Handle starting the server
  const handleStart = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/connections/${connectionId}/mcp/start`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setStatus(data.status as ServerStatus);
      setError(data.error);
    } catch (err: any) {
      console.error('Error starting MCP server:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle stopping the server
  const handleStop = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/connections/${connectionId}/mcp/stop`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setStatus(data.status as ServerStatus);
      setError(data.error);
    } catch (err: any) {
      console.error('Error stopping MCP server:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Status indicator colors
  const getStatusColor = (status: ServerStatus) => {
    switch (status) {
      case 'running':
        return 'text-green-600';
      case 'starting':
        return 'text-yellow-600';
      case 'failed':
        return 'text-red-600';
      case 'stopped':
      default:
        return 'text-gray-500';
    }
  };

  // Status display text
  const getStatusText = (status: ServerStatus) => {
    switch (status) {
      case 'running':
        return 'Running';
      case 'starting':
        return 'Starting...';
      case 'failed':
        return 'Failed';
      case 'stopped':
      default:
        return 'Stopped';
    }
  };

  return (
    <div className="flex items-center">
      <div className="flex items-center">
        <div className={`h-2 w-2 rounded-full mr-2 ${getStatusColor(status)}`} />
        <span className={`font-medium ${getStatusColor(status)}`}>
          {getStatusText(status)}
        </span>
      </div>
      
      {error && (
        <div className="text-xs text-red-500 ml-2" title={error}>
          ⚠️
        </div>
      )}
      
      {showControls && (
        <div className="ml-3 flex space-x-2">
          {status !== 'running' && status !== 'starting' && (
            <button 
              onClick={handleStart}
              disabled={loading}
              className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded hover:bg-green-200 disabled:opacity-50"
            >
              Start
            </button>
          )}
          
          {status === 'running' && (
            <button 
              onClick={handleStop}
              disabled={loading}
              className="text-xs px-2 py-1 bg-red-100 text-red-800 rounded hover:bg-red-200 disabled:opacity-50"
            >
              Stop
            </button>
          )}
        </div>
      )}
    </div>
  );
}