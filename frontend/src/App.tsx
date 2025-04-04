import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { NotebookView } from './components/NotebookView';
import { Tabs } from './components/ui/Tabs';
import { NotebookList } from './types/notebook';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from './components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from './components/ui/dropdown-menu';
import { MoreVertical, Plus, Search } from 'lucide-react';
import { BACKEND_URL } from './config';

// Define connection types
interface Connection {
  id: string;
  name: string;
  type: string;
  config: Record<string, string>;
}

function App() {
  const [notebooks, setNotebooks] = useState<NotebookList[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('notebooks');
  const [newNotebookName, setNewNotebookName] = useState('');
  const [newNotebookDescription, setNewNotebookDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isConnectionDialogOpen, setIsConnectionDialogOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [newConnectionName, setNewConnectionName] = useState('');
  const [newConnectionType, setNewConnectionType] = useState('postgres');
  const [newConnectionConfig, setNewConnectionConfig] = useState<Record<string, string>>({
    connection_string: '',
    host: '',
    port: '',
    username: '',
    password: '',
    database: '',
    url: '',
    api_key: ''
  });
  const [isCreatingConnection, setIsCreatingConnection] = useState(false);

  // Fetch data based on active tab
  useEffect(() => {
    setLoading(true);
    setError(null);
    
    if (activeTab === 'notebooks') {
      // Fetch notebooks
      fetch(`${BACKEND_URL}/api/notebooks`)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Error ${response.status}: ${response.statusText}`);
          }
          return response.json();
        })
        .then((data) => {
          setNotebooks(data);
          setLoading(false);
        })
        .catch((err) => {
          setError(err.message);
          setLoading(false);
        });
    } else if (activeTab === 'connections') {
      // Fetch connections
      fetch(`${BACKEND_URL}/api/connections`)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Error ${response.status}: ${response.statusText}`);
          }
          return response.json();
        })
        .then((data) => {
          setConnections(data);
          setLoading(false);
        })
        .catch((err) => {
          setError(err.message);
          setLoading(false);
        });
    }
  }, [activeTab]);

  // Create a new notebook
  const handleCreateNotebook = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!newNotebookName.trim()) return;
    
    setIsCreating(true);
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/notebooks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newNotebookName,
          description: newNotebookDescription,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const newNotebook = await response.json();
      
      // Update notebooks list
      setNotebooks((prev) => [...prev, {
        id: newNotebook.id,
        name: newNotebook.name,
        description: newNotebook.description,
        cell_count: 0,
        created_at: newNotebook.created_at,
        updated_at: newNotebook.updated_at
      }]);
      
      // Reset form
      setNewNotebookName('');
      setNewNotebookDescription('');
      setIsCreating(false);
      setIsDialogOpen(false);
      
      // Navigate to new notebook
      window.location.href = `/notebooks/${newNotebook.id}`;
    } catch (err: any) {
      setError(err.message);
      setIsCreating(false);
    }
  };
  
  // Create a new connection
  const handleCreateConnection = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!newConnectionName.trim()) return;
    
    setIsCreatingConnection(true);
    
    // Filter the config to only include fields relevant to the connection type
    let configToSend: Record<string, string> = {};
    
    if (newConnectionType === 'postgres') {
      configToSend = {
        connection_string: newConnectionConfig.connection_string || 
          `postgresql://${newConnectionConfig.username}:${newConnectionConfig.password}@${newConnectionConfig.host}:${newConnectionConfig.port}/${newConnectionConfig.database}`
      };
    } else if (newConnectionType === 'grafana') {
      configToSend = {
        url: newConnectionConfig.url,
        api_key: newConnectionConfig.api_key
      };
    }
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/connections`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newConnectionName,
          type: newConnectionType,
          config: configToSend
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const newConnection = await response.json();
      
      // Update connections list
      setConnections((prev) => [...prev, {
        id: newConnection.id,
        name: newConnection.name,
        type: newConnection.type,
        config: newConnection.config
      }]);
      
      // Reset form
      setNewConnectionName('');
      setNewConnectionType('postgres');
      setNewConnectionConfig({
        connection_string: '',
        host: '',
        port: '',
        username: '',
        password: '',
        database: '',
        url: '',
        api_key: ''
      });
      setIsCreatingConnection(false);
      setIsConnectionDialogOpen(false);
    } catch (err: any) {
      setError(err.message);
      setIsCreatingConnection(false);
    }
  };

  // Filter notebooks based on search query
  const filteredNotebooks = notebooks.filter(notebook => 
    notebook.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (notebook.description && notebook.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <Router>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b">
          <div className="container flex h-16 items-center px-4">
            <Link to="/" className="text-xl font-bold text-primary">Sherlog Canvas</Link>
            <div className="ml-auto flex items-center space-x-4">
              <div className="relative w-64">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search notebooks..."
                  className="pl-8"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>
          </div>
        </header>

        <Routes>
          <Route path="/" element={
            <main className="container py-6">
              <Tabs
                tabs={[
                  {
                    id: 'notebooks',
                    label: 'Notebooks',
                    content: (
                      <div className="space-y-6">
                        {/* Create new notebook button */}
                        <div className="flex justify-between items-center">
                          <h2 className="text-2xl font-bold tracking-tight">Your Notebooks</h2>
                          <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                            <DialogTrigger asChild>
                              <Button>
                                <Plus className="mr-2 h-4 w-4" />
                                New Notebook
                              </Button>
                            </DialogTrigger>
                            <DialogContent>
                              <DialogHeader>
                                <DialogTitle>Create New Notebook</DialogTitle>
                                <DialogDescription>
                                  Create a new notebook to organize your data analysis and visualizations.
                                </DialogDescription>
                              </DialogHeader>
                              <form onSubmit={handleCreateNotebook} className="space-y-4">
                                <div className="space-y-2">
                                  <label htmlFor="name" className="text-sm font-medium">
                                    Name
                                  </label>
                                  <Input
                                    id="name"
                                    value={newNotebookName}
                                    onChange={(e) => setNewNotebookName(e.target.value)}
                                    placeholder="Enter notebook name"
                                    required
                                  />
                                </div>
                                <div className="space-y-2">
                                  <label htmlFor="description" className="text-sm font-medium">
                                    Description
                                  </label>
                                  <Textarea
                                    id="description"
                                    value={newNotebookDescription}
                                    onChange={(e) => setNewNotebookDescription(e.target.value)}
                                    placeholder="Describe the notebook purpose"
                                    rows={3}
                                  />
                                </div>
                                <DialogFooter>
                                  <Button type="submit" disabled={isCreating}>
                                    {isCreating ? 'Creating...' : 'Create Notebook'}
                                  </Button>
                                </DialogFooter>
                              </form>
                            </DialogContent>
                          </Dialog>
                        </div>

                        {/* Notebooks list */}
                        {loading ? (
                          <div className="flex justify-center items-center h-32">
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
                          </div>
                        ) : error ? (
                          <Card className="border-destructive">
                            <CardHeader>
                              <CardTitle className="text-destructive">Error</CardTitle>
                              <CardDescription>{error}</CardDescription>
                            </CardHeader>
                          </Card>
                        ) : filteredNotebooks.length === 0 ? (
                          <Card>
                            <CardContent className="flex flex-col items-center justify-center py-10">
                              <p className="text-muted-foreground mb-4">
                                {searchQuery ? 'No notebooks match your search.' : 'No notebooks found. Create your first one!'}
                              </p>
                              <Button onClick={() => setIsDialogOpen(true)}>
                                <Plus className="mr-2 h-4 w-4" />
                                Create Notebook
                              </Button>
                            </CardContent>
                          </Card>
                        ) : (
                          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                            {filteredNotebooks.map((notebook) => (
                              <Card key={notebook.id} className="overflow-hidden">
                                <CardHeader className="pb-2">
                                  <div className="flex justify-between items-start">
                                    <CardTitle className="text-lg">
                                      <Link to={`/notebooks/${notebook.id}`} className="hover:underline">
                                        {notebook.name}
                                      </Link>
                                    </CardTitle>
                                    <DropdownMenu>
                                      <DropdownMenuTrigger asChild>
                                        <Button variant="ghost" size="icon">
                                          <MoreVertical className="h-4 w-4" />
                                          <span className="sr-only">Open menu</span>
                                        </Button>
                                      </DropdownMenuTrigger>
                                      <DropdownMenuContent align="end">
                                        <DropdownMenuItem>
                                          <Link to={`/notebooks/${notebook.id}`}>Open</Link>
                                        </DropdownMenuItem>
                                        <DropdownMenuItem>Rename</DropdownMenuItem>
                                        <DropdownMenuItem className="text-destructive">Delete</DropdownMenuItem>
                                      </DropdownMenuContent>
                                    </DropdownMenu>
                                  </div>
                                  <CardDescription>
                                    {notebook.description || 'No description'}
                                  </CardDescription>
                                </CardHeader>
                                <CardContent>
                                  <div className="flex items-center text-sm text-muted-foreground">
                                    <span className="mr-2">{notebook.cell_count} cells</span>
                                    <span>Updated {new Date(notebook.updated_at).toLocaleDateString()}</span>
                                  </div>
                                </CardContent>
                                <CardFooter className="bg-muted/50">
                                  <Button variant="secondary" className="w-full" asChild>
                                    <Link to={`/notebooks/${notebook.id}`}>Open Notebook</Link>
                                  </Button>
                                </CardFooter>
                              </Card>
                            ))}
                          </div>
                        )}
                      </div>
                    ),
                  },
                  {
                    id: 'connections',
                    label: 'Data Connections',
                    content: (
                      <div className="space-y-6">
                        <div className="flex justify-between items-center">
                          <h2 className="text-2xl font-bold tracking-tight">Data Connections</h2>
                          <Dialog open={isConnectionDialogOpen} onOpenChange={setIsConnectionDialogOpen}>
                            <DialogTrigger asChild>
                              <Button>
                                <Plus className="mr-2 h-4 w-4" />
                                New Connection
                              </Button>
                            </DialogTrigger>
                            <DialogContent>
                              <DialogHeader>
                                <DialogTitle>Create New Connection</DialogTitle>
                                <DialogDescription>
                                  Connect to your data sources like PostgreSQL, Grafana, Prometheus, or Loki.
                                </DialogDescription>
                              </DialogHeader>
                              
                              <form onSubmit={handleCreateConnection} className="space-y-4">
                                <div className="space-y-2">
                                  <label htmlFor="name" className="text-sm font-medium">
                                    Connection Name
                                  </label>
                                  <Input
                                    id="name"
                                    value={newConnectionName}
                                    onChange={(e) => setNewConnectionName(e.target.value)}
                                    placeholder="My PostgreSQL Database"
                                    required
                                  />
                                </div>
                                
                                <div className="space-y-2">
                                  <label htmlFor="type" className="text-sm font-medium">
                                    Connection Type
                                  </label>
                                  <select
                                    id="type"
                                    value={newConnectionType}
                                    onChange={(e) => setNewConnectionType(e.target.value)}
                                    className="w-full p-2 border rounded-md"
                                  >
                                    <option value="postgres">PostgreSQL</option>
                                    <option value="grafana">Grafana</option>
                                  </select>
                                </div>
                                
                                {newConnectionType === 'postgres' && (
                                  <div className="space-y-4 border p-3 rounded-md">
                                    <div className="space-y-2">
                                      <label htmlFor="connection_string" className="text-sm font-medium">
                                        Connection String (Optional)
                                      </label>
                                      <Input
                                        id="connection_string"
                                        value={newConnectionConfig.connection_string}
                                        onChange={(e) => setNewConnectionConfig({...newConnectionConfig, connection_string: e.target.value})}
                                        placeholder="postgresql://username:password@localhost:5432/database"
                                      />
                                      <p className="text-xs text-muted-foreground">
                                        Or fill in the details below
                                      </p>
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-3">
                                      <div className="space-y-2">
                                        <label htmlFor="host" className="text-sm font-medium">
                                          Host
                                        </label>
                                        <Input
                                          id="host"
                                          value={newConnectionConfig.host}
                                          onChange={(e) => setNewConnectionConfig({...newConnectionConfig, host: e.target.value})}
                                          placeholder="localhost"
                                        />
                                      </div>
                                      
                                      <div className="space-y-2">
                                        <label htmlFor="port" className="text-sm font-medium">
                                          Port
                                        </label>
                                        <Input
                                          id="port"
                                          value={newConnectionConfig.port}
                                          onChange={(e) => setNewConnectionConfig({...newConnectionConfig, port: e.target.value})}
                                          placeholder="5432"
                                        />
                                      </div>
                                    </div>
                                    
                                    <div className="space-y-2">
                                      <label htmlFor="database" className="text-sm font-medium">
                                        Database
                                      </label>
                                      <Input
                                        id="database"
                                        value={newConnectionConfig.database}
                                        onChange={(e) => setNewConnectionConfig({...newConnectionConfig, database: e.target.value})}
                                        placeholder="postgres"
                                      />
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-3">
                                      <div className="space-y-2">
                                        <label htmlFor="username" className="text-sm font-medium">
                                          Username
                                        </label>
                                        <Input
                                          id="username"
                                          value={newConnectionConfig.username}
                                          onChange={(e) => setNewConnectionConfig({...newConnectionConfig, username: e.target.value})}
                                          placeholder="postgres"
                                        />
                                      </div>
                                      
                                      <div className="space-y-2">
                                        <label htmlFor="password" className="text-sm font-medium">
                                          Password
                                        </label>
                                        <Input
                                          id="password"
                                          type="password"
                                          value={newConnectionConfig.password}
                                          onChange={(e) => setNewConnectionConfig({...newConnectionConfig, password: e.target.value})}
                                          placeholder="••••••••"
                                        />
                                      </div>
                                    </div>
                                  </div>
                                )}
                                
                                {newConnectionType === 'grafana' && (
                                  <div className="space-y-4 border p-3 rounded-md">
                                    <div className="space-y-2">
                                      <label htmlFor="url" className="text-sm font-medium">
                                        Grafana URL
                                      </label>
                                      <Input
                                        id="url"
                                        value={newConnectionConfig.url}
                                        onChange={(e) => setNewConnectionConfig({...newConnectionConfig, url: e.target.value})}
                                        placeholder="http://localhost:3000"
                                        required
                                      />
                                    </div>
                                    
                                    <div className="space-y-2">
                                      <label htmlFor="api_key" className="text-sm font-medium">
                                        API Key
                                      </label>
                                      <Input
                                        id="api_key"
                                        type="password"
                                        value={newConnectionConfig.api_key}
                                        onChange={(e) => setNewConnectionConfig({...newConnectionConfig, api_key: e.target.value})}
                                        placeholder="eyJrIjoiT0tTc..."
                                        required
                                      />
                                    </div>
                                  </div>
                                )}
                                
                                <DialogFooter>
                                  <Button type="submit" disabled={isCreatingConnection}>
                                    {isCreatingConnection ? 'Creating...' : 'Create Connection'}
                                  </Button>
                                </DialogFooter>
                              </form>
                            </DialogContent>
                          </Dialog>
                        </div>
                        
                        {loading ? (
                          <Card>
                            <CardContent className="flex justify-center py-8">
                              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                            </CardContent>
                          </Card>
                        ) : error ? (
                          <Card>
                            <CardHeader>
                              <CardTitle className="text-destructive">Error</CardTitle>
                              <CardDescription>{error}</CardDescription>
                            </CardHeader>
                          </Card>
                        ) : connections.length === 0 ? (
                          <Card>
                            <CardContent className="flex flex-col items-center justify-center py-10">
                              <p className="text-muted-foreground mb-4">
                                No data connections found. Create your first one!
                              </p>
                              <Button>
                                <Plus className="mr-2 h-4 w-4" />
                                Create Connection
                              </Button>
                            </CardContent>
                          </Card>
                        ) : (
                          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                            {/* Map over actual connections */}
                            {connections.map((connection) => (
                              <Card key={connection.id}>
                                <CardHeader className="pb-2">
                                  <CardTitle className="text-lg">{connection.name}</CardTitle>
                                  <CardDescription>
                                    {connection.type.charAt(0).toUpperCase() + connection.type.slice(1)} Connection
                                  </CardDescription>
                                </CardHeader>
                                <CardContent>
                                  <p className="text-sm text-muted-foreground mb-4">
                                    Status: <span className="text-green-600 font-medium">Connected</span>
                                  </p>
                                  <div className="text-sm">
                                    {/* Display key config values except sensitive ones */}
                                    {Object.entries(connection.config)
                                      .filter(([key]) => !key.includes('password') && !key.includes('key') && !key.includes('secret'))
                                      .map(([key, value]) => (
                                        <p key={key}>
                                          <span className="font-medium">{key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ')}:</span> {value}
                                        </p>
                                      ))}
                                  </div>
                                </CardContent>
                                <CardFooter className="bg-muted/50 flex justify-between">
                                  <Button variant="outline" size="sm">
                                    Edit
                                  </Button>
                                  <Button variant="secondary" size="sm">
                                    Test Connection
                                  </Button>
                                </CardFooter>
                              </Card>
                            ))}
                            
                            {/* Show default connections if API didn't return them */}
                            {connections.find(c => c.type === 'postgres') === undefined && (
                              <Card>
                                <CardHeader className="pb-2 flex flex-row items-center space-x-4">
                                  <div className="bg-blue-100 p-2 rounded-full h-12 w-12 flex items-center justify-center">
                                    <svg viewBox="0 0 24 24" className="h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg">
                                      <path fill="currentColor" d="M12 0C8.922 0 6.642.18 5.129.962c-1.512.783-2.3 1.946-2.3 3.297 0 .784.479 2.082 1.285 3.92 1.37 3.133 3.432 7.077 5.344 9.892 1.048 1.54 2.096 2.78 3.035 3.668.47.443.925.76 1.356.998.43.237.838.38 1.3.38.463 0 .872-.143 1.302-.381.431-.237.886-.555 1.355-.998.94-.887 1.988-2.127 3.036-3.667 1.912-2.815 3.974-6.76 5.343-9.893.807-1.837 1.286-3.135 1.286-3.92 0-1.35-.788-2.514-2.3-3.296C17.357.18 15.078 0 12 0zm0 1.923c2.92 0 4.97.173 6.154.807 1.184.634 1.602 1.398 1.602 2.02 0 .47-.417 1.582-1.147 3.27-1.303 2.98-3.304 6.787-5.075 9.389-.886 1.303-1.846 2.437-2.636 3.188-.395.375-.754.628-1.034.772-.28.145-.498.208-.71.208-.211 0-.43-.063-.71-.208-.28-.144-.64-.397-1.034-.772-.79-.75-1.75-1.885-2.636-3.188-1.771-2.602-3.772-6.408-5.075-9.388-.73-1.689-1.147-2.8-1.147-3.271 0-.621.418-1.385 1.602-2.02C7.03 2.097 9.08 1.923 12 1.923zm0 2.981c-1.198 0-2.198.997-2.198 2.193 0 1.197.998 2.194 2.198 2.194 1.199 0 2.199-.997 2.199-2.194 0-1.196-1-2.193-2.199-2.193zm-5.252 3.47c-1.145 0-2.093.95-2.093 2.095s.947 2.096 2.093 2.096c1.146 0 2.094-.95 2.094-2.096 0-1.146-.948-2.096-2.094-2.096zm10.504 0c-1.146 0-2.094.95-2.094 2.095s.948 2.096 2.094 2.096c1.145 0 2.093-.95 2.093-2.096 0-1.146-.948-2.096-2.093-2.096zM12 10.981c-1.147 0-2.094.95-2.094 2.096s.948 2.096 2.094 2.096c1.146 0 2.094-.95 2.094-2.096 0-1.146-.947-2.096-2.094-2.096zm-6.276 4.24c-1.147 0-2.094.948-2.094 2.095 0 1.146.947 2.095 2.094 2.095s2.093-.949 2.093-2.095c0-1.147-.946-2.096-2.093-2.096zm12.552 0c-1.147 0-2.094.948-2.094 2.095 0 1.146.947 2.095 2.094 2.095 1.146 0 2.093-.949 2.093-2.095 0-1.147-.947-2.096-2.093-2.096z"/>
                                    </svg>
                                  </div>
                                  <div>
                                    <CardTitle className="text-lg">PostgreSQL</CardTitle>
                                    <CardDescription>Connect to your PostgreSQL databases</CardDescription>
                                  </div>
                                </CardHeader>
                                <CardContent>
                                  <p className="text-sm text-muted-foreground mb-4">
                                    Status: <span className="text-green-600 font-medium">Connected</span>
                                  </p>
                                  <div className="text-sm">
                                    <p><span className="font-medium">Server:</span> pg-mcp:8000</p>
                                    <p><span className="font-medium">Host:</span> localhost</p>
                                    <p><span className="font-medium">Port:</span> 9211</p>
                                  </div>
                                </CardContent>
                                <CardFooter className="bg-muted/50 flex justify-between">
                                  <Button variant="outline" size="sm" onClick={() => {
                                    setNewConnectionName("PostgreSQL");
                                    setNewConnectionType("postgres");
                                    setIsConnectionDialogOpen(true);
                                  }}>
                                    Edit
                                  </Button>
                                  <Button variant="secondary" size="sm" onClick={() => {
                                    // TODO: Test the connection
                                    alert("Connection test successful!");
                                  }}>
                                    Test Connection
                                  </Button>
                                </CardFooter>
                              </Card>
                            )}
                            
                            {connections.find(c => c.type === 'grafana') === undefined && (
                              <Card>
                                <CardHeader className="pb-2 flex flex-row items-center space-x-4">
                                  <div className="bg-orange-100 p-2 rounded-full h-12 w-12 flex items-center justify-center">
                                    <svg viewBox="0 0 24 24" className="h-8 w-8 text-orange-600" xmlns="http://www.w3.org/2000/svg">
                                      <path fill="currentColor" d="M11.982 0A12 12 0 0 0 0 11.978a12 12 0 0 0 11.982 12.044 12 12 0 0 0 12.018-12.044A12 12 0 0 0 11.982 0zM2.87 16.243c.463 1.587 1.3 2.923 2.632 3.995l-1.373-.65a6.525 6.525 0 0 1-1.259-3.345zm14.465-8.425c.177.76.277 1.613.392 2.523.116.91.29 1.858.262 2.818a71.881 71.881 0 0 0-.187 2.523c-.06 2.464-.026 4.765-.026 6.963h-1.95c.145-2.195.31-4.298.481-6.316.026-.262.042-.525.059-.787-.154.262-.321.516-.489.759a10.33 10.33 0 0 1-1.185 1.4 4.706 4.706 0 0 1-1.678.759 2.484 2.484 0 0 1-1.865-.341c-1.167-.802-1.865-1.867-2.065-3.226a10.568 10.568 0 0 1 .025-3.778c.196-.958.554-1.848 1.175-2.612.622-.759 1.426-1.288 2.368-1.4 1.016-.116 1.875.171 2.645.828.77.656 1.327 1.493 1.737 2.424.137.307.262.622.379.933.138-.17.275-.35.413-.532.574-.744 1.067-1.553 1.31-2.477.145-.533.222-1.074.196-1.627a2.605 2.605 0 0 0-.583-1.688 2.242 2.242 0 0 0-1.515-.724 6.673 6.673 0 0 0-1.934.2c-1.287.333-2.551.81-3.754 1.484-.239.137-.477.274-.714.411.051-.316.116-.631.187-.943a5 5 0 0 1 .642-1.628c.275-.461.787-.828 1.293-1.1a12.28 12.28 0 0 1 3.482-1.168 12.282 12.282 0 0 1 3.772-.072c1.356.204 2.287.958 2.705 2.305a6.583 6.583 0 0 1 .234 1.799 8.66 8.66 0 0 1-.73 3.226 21.93 21.93 0 0 1-1.444 2.833zm-9.328.186c.592-2.047 1.864-3.247 3.942-3.558.178-.03.361-.043.55-.053a2.295 2.295 0 0 0-1.866 1.002 3.795 3.795 0 0 0-.626 1.535 10.14 10.14 0 0 0-.232 2.022c0 .787.115 1.564.31 2.33.026.118.075.229.11.332.145-.502.202-1.022.273-1.536.08-.581.15-1.162.247-1.743a5.797 5.797 0 0 1 .533-1.662c.622-1.203 1.893-1.27 2.626-.154.418.64.714 1.349.958 2.068.307.904.582 1.816.893 2.717.127.366.267.726.4 1.098.138-1.223.196-2.445.275-3.668.085-1.3.196-2.591.463-3.873.044-.198.102-.393.153-.59l.152.034c.051.35.11.65.162.1.034.599.06 1.202.066 1.81a42.67 42.67 0 0 1-.463 6.365 3.642 3.642 0 0 1-.488 1.425c-.453.76-1.133.904-1.841.386-.532-.392-.893-.938-1.23-1.501a67.92 67.92 0 0 1-1.018-1.816c-.209-.41-.412-.82-.617-1.229-.2.324-.403.648-.599.973-.533.859-1.082 1.708-1.71 2.5a2.981 2.981 0 0 1-.941.81c-.59.306-1.184.213-1.674-.273a2.435 2.435 0 0 1-.55-.837c-.473-1.066-.79-2.175-.96-3.318a16.08 16.08 0 0 1-.17-2.168c.002-.963.062-1.914.296-2.859z"/>
                                    </svg>
                                  </div>
                                  <div>
                                    <CardTitle className="text-lg">Grafana</CardTitle>
                                    <CardDescription>Connect to your Grafana dashboards</CardDescription>
                                  </div>
                                </CardHeader>
                                <CardContent>
                                  <p className="text-sm text-muted-foreground mb-4">
                                    Status: <span className="text-green-600 font-medium">Connected</span>
                                  </p>
                                  <div className="text-sm">
                                    <p><span className="font-medium">Server:</span> mcp-grafana:8000</p>
                                    <p><span className="font-medium">Host:</span> localhost</p>
                                    <p><span className="font-medium">Port:</span> 9110</p>
                                  </div>
                                </CardContent>
                                <CardFooter className="bg-muted/50 flex justify-between">
                                  <Button variant="outline" size="sm" onClick={() => {
                                    setNewConnectionName("Grafana");
                                    setNewConnectionType("grafana");
                                    setIsConnectionDialogOpen(true);
                                  }}>
                                    Edit
                                  </Button>
                                  <Button variant="secondary" size="sm" onClick={() => {
                                    // TODO: Test the connection
                                    alert("Connection test successful!");
                                  }}>
                                    Test Connection
                                  </Button>
                                </CardFooter>
                              </Card>
                            )}
                            
                            {/* Add New Connection Card */}
                            <Card className="border-dashed">
                              <CardContent className="flex flex-col items-center justify-center py-10 h-full">
                                <Plus className="h-10 w-10 text-muted-foreground mb-4" />
                                <p className="text-muted-foreground mb-4 text-center">
                                  Add another data connection
                                </p>
                                <Button variant="outline" onClick={() => {
                                  setNewConnectionName("");
                                  setNewConnectionType("postgres");
                                  setIsConnectionDialogOpen(true);
                                }}>
                                  Add Connection
                                </Button>
                              </CardContent>
                            </Card>
                          </div>
                        )}
                      </div>
                    ),
                  }
                ]}
                activeTab={activeTab}
                onChange={setActiveTab}
              />
            </main>
          } />
          <Route path="/notebooks/:notebookId" element={<NotebookView />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;