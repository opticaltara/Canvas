'use client';

import { useState } from 'react';
import { MCPConnection } from '../types/notebook';
import useNotebookStore from '../lib/store';
import { testPostgresConnection, testGrafanaConnection } from '../lib/connection-manager';
import { nanoid } from 'nanoid';

// Connection type definitions
const CONNECTION_TYPES = [
  { id: 'postgres', name: 'PostgreSQL' },
  { id: 'grafana', name: 'Grafana (Prometheus/Loki)' },
  { id: 's3', name: 'S3' },
];

export default function ConnectionManager() {
  const { connections, addConnection, updateConnection, deleteConnection } = useNotebookStore();
  const [isAddingConnection, setIsAddingConnection] = useState(false);
  const [editingConnection, setEditingConnection] = useState<MCPConnection | null>(null);
  const [showConnectionDetails, setShowConnectionDetails] = useState<string | null>(null);
  
  // Form state
  const [formData, setFormData] = useState<{
    name: string;
    type: string;
    config: Record<string, string>;
  }>({
    name: '',
    type: 'postgres',
    config: {},
  });
  
  // Test connection state
  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);
  
  // Reset form
  const resetForm = () => {
    setFormData({
      name: '',
      type: 'postgres',
      config: {},
    });
    setTestResult(null);
  };
  
  // Handle form change
  const handleFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    if (name === 'type') {
      // Reset config when type changes
      setFormData({
        ...formData,
        type: value,
        config: {},
      });
    } else if (name.startsWith('config.')) {
      // Update config field
      const configField = name.replace('config.', '');
      setFormData({
        ...formData,
        config: {
          ...formData.config,
          [configField]: value,
        },
      });
    } else {
      // Update other fields
      setFormData({
        ...formData,
        [name]: value,
      });
    }
  };
  
  // Test connection
  const handleTestConnection = async () => {
    try {
      let success = false;
      let message = '';
      
      switch (formData.type) {
        case 'postgres':
          const pgResult = await testPostgresConnection(formData.config);
          success = pgResult.isValid;
          message = pgResult.message || '';
          break;
        case 'grafana':
          const grafanaResult = await testGrafanaConnection(formData.config);
          success = grafanaResult.isValid;
          message = grafanaResult.message || '';
          break;
        default:
          success = false;
          message = 'Testing not implemented for this connection type';
      }
      
      setTestResult({
        success,
        message: message || (success ? 'Connection successful!' : 'Connection failed'),
      });
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : 'An unknown error occurred',
      });
    }
  };
  
  // Save connection
  const handleSaveConnection = () => {
    if (!formData.name.trim()) {
      alert('Please provide a name for the connection');
      return;
    }
    
    if (editingConnection) {
      // Update existing connection
      updateConnection(
        editingConnection.id,
        formData.name,
        formData.config
      );
    } else {
      // Add new connection
      addConnection(
        formData.name,
        formData.type,
        formData.config
      );
    }
    
    // Reset state
    setIsAddingConnection(false);
    setEditingConnection(null);
    resetForm();
  };
  
  // Start editing connection
  const handleEditConnection = (connection: MCPConnection) => {
    setEditingConnection(connection);
    setFormData({
      name: connection.name,
      type: connection.type,
      config: { ...connection.config },
    });
    setIsAddingConnection(true);
  };
  
  // Cancel editing/adding
  const handleCancel = () => {
    setIsAddingConnection(false);
    setEditingConnection(null);
    resetForm();
  };
  
  // Render config fields based on connection type
  const renderConfigFields = () => {
    switch (formData.type) {
      case 'postgres':
        return (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1">Connection string</label>
              <input
                type="text"
                name="config.connection_string"
                value={formData.config.connection_string || ''}
                onChange={handleFormChange}
                placeholder="postgresql://username:password@host:port/database"
                className="w-full px-3 py-2 border rounded"
              />
              <p className="text-xs text-gray-500 mt-1">
                Format: postgresql://username:password@host:port/database
              </p>
            </div>
          </div>
        );
      case 'grafana':
        return (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1">Grafana URL</label>
              <input
                type="text"
                name="config.url"
                value={formData.config.url || ''}
                onChange={handleFormChange}
                placeholder="https://grafana.example.com"
                className="w-full px-3 py-2 border rounded"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">API Key</label>
              <input
                type="password"
                name="config.api_key"
                value={formData.config.api_key || ''}
                onChange={handleFormChange}
                placeholder="Grafana API key"
                className="w-full px-3 py-2 border rounded"
              />
            </div>
          </div>
        );
      case 's3':
        return (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1">S3 Bucket</label>
              <input
                type="text"
                name="config.bucket"
                value={formData.config.bucket || ''}
                onChange={handleFormChange}
                placeholder="my-bucket"
                className="w-full px-3 py-2 border rounded"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">AWS Region</label>
              <input
                type="text"
                name="config.region"
                value={formData.config.region || ''}
                onChange={handleFormChange}
                placeholder="us-east-1"
                className="w-full px-3 py-2 border rounded"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Access Key</label>
              <input
                type="text"
                name="config.access_key"
                value={formData.config.access_key || ''}
                onChange={handleFormChange}
                placeholder="AWS Access Key"
                className="w-full px-3 py-2 border rounded"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Secret Key</label>
              <input
                type="password"
                name="config.secret_key"
                value={formData.config.secret_key || ''}
                onChange={handleFormChange}
                placeholder="AWS Secret Key"
                className="w-full px-3 py-2 border rounded"
              />
            </div>
          </div>
        );
      default:
        return null;
    }
  };
  
  // Render connection details
  const renderConnectionDetails = (connection: MCPConnection) => {
    switch (connection.type) {
      case 'postgres':
        return (
          <div className="text-sm mt-2">
            <div><span className="font-semibold">Connection:</span> {maskConnectionString(connection.config.connection_string || '')}</div>
          </div>
        );
      case 'grafana':
        return (
          <div className="text-sm mt-2">
            <div><span className="font-semibold">URL:</span> {connection.config.url}</div>
            <div><span className="font-semibold">API Key:</span> {maskString(connection.config.api_key || '')}</div>
          </div>
        );
      case 's3':
        return (
          <div className="text-sm mt-2">
            <div><span className="font-semibold">Bucket:</span> {connection.config.bucket}</div>
            <div><span className="font-semibold">Region:</span> {connection.config.region}</div>
            <div><span className="font-semibold">Access Key:</span> {maskString(connection.config.access_key || '')}</div>
          </div>
        );
      default:
        return null;
    }
  };
  
  // Helper to mask sensitive strings
  const maskString = (str: string) => {
    if (!str) return '';
    if (str.length <= 8) return '********';
    return str.substring(0, 4) + '********' + str.substring(str.length - 4);
  };
  
  // Helper to mask connection strings
  const maskConnectionString = (connStr: string) => {
    try {
      // Try to parse as URL
      const url = new URL(connStr);
      let masked = `${url.protocol}//`;
      
      // Mask username/password if present
      if (url.username || url.password) {
        masked += '********:********@';
      }
      
      // Add host, port, path
      masked += url.host + url.pathname;
      
      return masked;
    } catch (e) {
      // Not a valid URL, just mask it
      return maskString(connStr);
    }
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Data Connections</h2>
        {!isAddingConnection && (
          <button
            onClick={() => setIsAddingConnection(true)}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
          >
            Add Connection
          </button>
        )}
      </div>
      
      {/* Connection list */}
      {connections.length === 0 && !isAddingConnection ? (
        <div className="text-center py-6 text-gray-500">
          No connections configured. Add a connection to get started.
        </div>
      ) : (
        <div className="space-y-3 mb-4">
          {connections.map((connection) => (
            <div
              key={connection.id}
              className="border rounded p-3 bg-gray-50 dark:bg-gray-700"
            >
              <div className="flex justify-between items-center">
                <div>
                  <div className="font-medium">{connection.name}</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {CONNECTION_TYPES.find((t) => t.id === connection.type)?.name || connection.type}
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setShowConnectionDetails(
                      showConnectionDetails === connection.id ? null : connection.id
                    )}
                    className="text-blue-500 hover:text-blue-700 text-sm"
                  >
                    {showConnectionDetails === connection.id ? 'Hide' : 'Details'}
                  </button>
                  <button
                    onClick={() => handleEditConnection(connection)}
                    className="text-blue-500 hover:text-blue-700 text-sm"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => {
                      if (confirm(`Delete connection "${connection.name}"?`)) {
                        deleteConnection(connection.id);
                      }
                    }}
                    className="text-red-500 hover:text-red-700 text-sm"
                  >
                    Delete
                  </button>
                </div>
              </div>
              
              {showConnectionDetails === connection.id && (
                renderConnectionDetails(connection)
              )}
            </div>
          ))}
        </div>
      )}
      
      {/* Add/Edit connection form */}
      {isAddingConnection && (
        <div className="border p-4 rounded bg-gray-50 dark:bg-gray-700">
          <h3 className="text-lg font-medium mb-4">
            {editingConnection ? 'Edit Connection' : 'Add New Connection'}
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Connection Name</label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleFormChange}
                placeholder="My Database"
                className="w-full px-3 py-2 border rounded"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Connection Type</label>
              <select
                name="type"
                value={formData.type}
                onChange={handleFormChange}
                className="w-full px-3 py-2 border rounded"
                disabled={!!editingConnection} // Can't change type of existing connection
              >
                {CONNECTION_TYPES.map((type) => (
                  <option key={type.id} value={type.id}>
                    {type.name}
                  </option>
                ))}
              </select>
            </div>
            
            {/* Type-specific config fields */}
            {renderConfigFields()}
            
            {/* Test connection result */}
            {testResult && (
              <div
                className={`p-3 rounded text-sm ${
                  testResult.success
                    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:bg-opacity-30 dark:text-green-300'
                    : 'bg-red-100 text-red-800 dark:bg-red-900 dark:bg-opacity-30 dark:text-red-300'
                }`}
              >
                {testResult.message}
              </div>
            )}
            
            {/* Actions */}
            <div className="flex justify-end space-x-3 pt-2">
              <button
                onClick={handleTestConnection}
                className="px-3 py-1 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500"
              >
                Test Connection
              </button>
              <button
                onClick={handleCancel}
                className="px-3 py-1 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveConnection}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                {editingConnection ? 'Update' : 'Add'} Connection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}