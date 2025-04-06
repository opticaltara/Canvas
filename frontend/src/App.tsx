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
import { MCPServerStatus } from './components/MCPServerStatus';

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
  const [newConnectionType, setNewConnectionType] = useState('grafana');
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
    
    // Process the connection configuration
    let connectionConfig: Record<string, string> = {};
    
    if (newConnectionType === 'grafana') {
      connectionConfig = {
        url: newConnectionConfig.url || '',
        api_key: newConnectionConfig.api_key || ''
      };
    } else if (newConnectionType === 'kubernetes') {
      connectionConfig = {
        kubeconfig: newConnectionConfig.kubeconfig || '',
        context: newConnectionConfig.context || ''
      };
    } else if (newConnectionType === 's3') {
      connectionConfig = {
        bucket: newConnectionConfig.bucket || '',
        prefix: newConnectionConfig.prefix || '',
        region: newConnectionConfig.region || ''
      };
    }
    
    // Create the connection object
    const connection = {
      name: newConnectionName,
      type: newConnectionType,
      config: connectionConfig
    };
    
    // Send the connection to the API
    try {
      const response = await fetch(`${BACKEND_URL}/api/connections`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(connection)
      });
      
      if (!response.ok) {
        throw new Error('Failed to create connection');
      }
      
      // Refresh the connections list
      const newConnection = await response.json();
      setConnections((prev) => [...prev, newConnection]);
      
      // Reset the form
      setNewConnectionName('');
      setNewConnectionConfig({});
      setNewConnectionType('grafana');
      setIsConnectionDialogOpen(false);
    } catch (error) {
      console.error('Error creating connection:', error);
      // TODO: Show error message to user
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
                                    <option value="grafana">Grafana</option>
                                    <option value="kubernetes">Kubernetes</option>
                                    <option value="s3">S3</option>
                                  </select>
                                </div>
                                
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
                                
                                {newConnectionType === 'kubernetes' && (
                                  <div className="space-y-4 border p-3 rounded-md">
                                    <div className="space-y-2">
                                      <label htmlFor="kubeconfig_type" className="text-sm font-medium">
                                        Kubeconfig Source
                                      </label>
                                      <select 
                                        id="kubeconfig_type"
                                        className="w-full p-2 border rounded-md"
                                        value={newConnectionConfig.kubeconfig_type || 'file'}
                                        onChange={(e) => setNewConnectionConfig({
                                          ...newConnectionConfig, 
                                          kubeconfig_type: e.target.value
                                        })}
                                      >
                                        <option value="file">Use kubeconfig file path</option>
                                        <option value="content">Paste kubeconfig content</option>
                                      </select>
                                    </div>
                                    
                                    {(newConnectionConfig.kubeconfig_type === 'file' || !newConnectionConfig.kubeconfig_type) && (
                                      <div className="space-y-2">
                                        <label htmlFor="kubeconfig" className="text-sm font-medium">
                                          Kubeconfig File Path
                                        </label>
                                        <Input
                                          id="kubeconfig"
                                          value={newConnectionConfig.kubeconfig || ''}
                                          onChange={(e) => setNewConnectionConfig({...newConnectionConfig, kubeconfig: e.target.value})}
                                          placeholder="~/.kube/config"
                                          required
                                        />
                                        <p className="text-xs text-muted-foreground">
                                          Path to your kubeconfig file (e.g., ~/.kube/config)
                                        </p>
                                      </div>
                                    )}
                                    
                                    {newConnectionConfig.kubeconfig_type === 'content' && (
                                      <div className="space-y-2">
                                        <label htmlFor="kubeconfig_content" className="text-sm font-medium">
                                          Kubeconfig Content
                                        </label>
                                        <Textarea
                                          id="kubeconfig_content"
                                          value={newConnectionConfig.kubeconfig || ''}
                                          onChange={(e) => setNewConnectionConfig({...newConnectionConfig, kubeconfig: e.target.value})}
                                          placeholder="apiVersion: v1\nkind: Config\nclusters:\n..."
                                          className="min-h-[150px]"
                                          required
                                        />
                                        <p className="text-xs text-muted-foreground">
                                          Paste the full content of your kubeconfig file
                                        </p>
                                      </div>
                                    )}
                                    
                                    <div className="space-y-2">
                                      <label htmlFor="context" className="text-sm font-medium">
                                        Kubernetes Context (Optional)
                                      </label>
                                      <Input
                                        id="context"
                                        value={newConnectionConfig.context || ''}
                                        onChange={(e) => setNewConnectionConfig({...newConnectionConfig, context: e.target.value})}
                                        placeholder="my-cluster-context"
                                      />
                                      <p className="text-xs text-muted-foreground">
                                        Specify which context to use from your kubeconfig
                                      </p>
                                    </div>
                                    
                                    <div className="space-y-2">
                                      <label htmlFor="namespace" className="text-sm font-medium">
                                        Default Namespace (Optional)
                                      </label>
                                      <Input
                                        id="namespace"
                                        value={newConnectionConfig.namespace || ''}
                                        onChange={(e) => setNewConnectionConfig({...newConnectionConfig, namespace: e.target.value})}
                                        placeholder="default"
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
                                <CardHeader className="pb-2 flex flex-row items-center space-x-4">
                                  <div className={`bg-${connection.type === 'grafana' ? 'orange' : connection.type === 'kubernetes' ? 'green' : 'gray'}-100 p-2 rounded-full h-12 w-12 flex items-center justify-center`}>
                                    {connection.type === 'grafana' && (
                                      <svg viewBox="0 0 24 24" className="h-8 w-8 text-orange-600" xmlns="http://www.w3.org/2000/svg">
                                        <path fill="currentColor" d="M11.982 0A12 12 0 0 0 0 11.978a12 12 0 0 0 11.982 12.044 12 12 0 0 0 12.018-12.044A12 12 0 0 0 11.982 0zM2.87 16.243c.463 1.587 1.3 2.923 2.632 3.995l-1.373-.65a6.525 6.525 0 0 1-1.259-3.345zm14.465-8.425c.177.76.277 1.613.392 2.523.116.91.29 1.858.262 2.818a71.881 71.881 0 0 0-.187 2.523c-.06 2.464-.026 4.765-.026 6.963h-1.95c.145-2.195.31-4.298.481-6.316.026-.262.042-.525.059-.787-.154.262-.321.516-.489.759a10.33 10.33 0 0 1-1.185 1.4 4.706 4.706 0 0 1-1.678.759 2.484 2.484 0 0 1-1.865-.341c-1.167-.802-1.865-1.867-2.065-3.226a10.568 10.568 0 0 1 .025-3.778c.196-.958.554-1.848 1.175-2.612.622-.759 1.426-1.288 2.368-1.4 1.016-.116 1.875.171 2.645.828.77.656 1.327 1.493 1.737 2.424.137.307.262.622.379.933.138-.17.275-.35.413-.532.574-.744 1.067-1.553 1.31-2.477.145-.533.222-1.074.196-1.627a2.605 2.605 0 0 0-.583-1.688 2.242 2.242 0 0 0-1.515-.724 6.673 6.673 0 0 0-1.934.2c-1.287.333-2.551.81-3.754 1.484-.239.137-.477.274-.714.411.051-.316.116-.631.187-.943a5 5 0 0 1 .642-1.628c.275-.461.787-.828 1.293-1.1a12.28 12.28 0 0 1 3.482-1.168 12.282 12.282 0 0 1 3.772-.072c1.356.204 2.287.958 2.705 2.305a6.583 6.583 0 0 1 .234 1.799 8.66 8.66 0 0 1-.73 3.226 21.93 21.93 0 0 1-1.444 2.833zm-9.328.186c.592-2.047 1.864-3.247 3.942-3.558.178-.03.361-.043.55-.053a2.295 2.295 0 0 0-1.866 1.002 3.795 3.795 0 0 0-.626 1.535 10.14 10.14 0 0 0-.232 2.022c0 .787.115 1.564.31 2.33.026.118.075.229.11.332.145-.502.202-1.022.273-1.536.08-.581.15-1.162.247-1.743a5.797 5.797 0 0 1 .533-1.662c.622-1.203 1.893-1.27 2.626-.154.418.64.714 1.349.958 2.068.307.904.582 1.816.893 2.717.127.366.267.726.4 1.098.138-1.223.196-2.445.275-3.668.085-1.3.196-2.591.463-3.873.044-.198.102-.393.153-.59l.152.034c.051.35.11.65.162.1.034.599.06 1.202.066 1.81a42.67 42.67 0 0 1-.463 6.365 3.642 3.642 0 0 1-.488 1.425c-.453.76-1.133.904-1.841.386-.532-.392-.893-.938-1.23-1.501a67.92 67.92 0 0 1-1.018-1.816c-.209-.41-.412-.82-.617-1.229-.2.324-.403.648-.599.973-.533.859-1.082 1.708-1.71 2.5a2.981 2.981 0 0 1-.941.81c-.59.306-1.184.213-1.674-.273a2.435 2.435 0 0 1-.55-.837c-.473-1.066-.79-2.175-.96-3.318a16.08 16.08 0 0 1-.17-2.168c.002-.963.062-1.914.296-2.859z"/>
                                      </svg>
                                    )}
                                    {connection.type === 'kubernetes' && (
                                      <svg viewBox="0 0 24 24" className="h-8 w-8 text-green-500" xmlns="http://www.w3.org/2000/svg">
                                        <path fill="currentColor" d="M10.204 14.35l.007.01-.999 2.413a5.171 5.171 0 0 1-2.075-2.597l2.578-.437.013.06a.525.525 0 0 0 .476.55zm-.198-1.153a.525.525 0 0 0-.578-.32l-2.825.48a5.203 5.203 0 0 1-.096-1.307l2.708-1.25a.528.528 0 0 0 .109.205c.38.475.568.974.479.815a.524.524 0 0 0 .204-.623zm.811-1.924a.524.524 0 0 0-.775.152l-1.648 2.647a5.2 5.2 0 0 1-.99-.847l1.907-2.27a.525.525 0 0 0 .072-.11.525.525 0 0 0 .20.152c.777.439.915.561 1.057.637a.397.397 0 0 1 .122.103c.018.026.044.076.055.119M12 8.1c.18 0 .355.026.523.047l.963-2.842a.525.525 0 0 0-.128-.504.526.526 0 0 0-.476-.153C11.386 4.93 9.64 5.46 9.64 5.46a5.18 5.18 0 0 1 2.36-1.36zM8.277 8.8a5.18 5.18 0 0 1 1.74-1.654l.658 2.946a.525.525 0 0 0-.076.172.525.525 0 0 0-.131-.213c-.164-.163-1.003-1.145-1.48-1.628a.526.526 0 0 0-.715.378zm5.484.929c.051-.21.093-.424.122-.642l3.01-.573a.525.525 0 0 0 .126-.035c.002.023.006.046.008.069a5.181 5.181 0 0 1-.616 3.06c-.964-1.922-1.845-1.195-2.65-1.879zm.106 5.101a5.2 5.2 0 0 1-1.609.926l-.387-3.013a.526.526 0 0 0 .483-.574.526.526 0 0 0 .245.508c.323.214 1.935.271 1.268 2.153zm.446-1.747a.525.525 0 0 0 .266-.447.526.526 0 0 0-.351-.487c-.544-.225-1.407-1.581-.697-2.615a5.185 5.185 0 0 1 1.526 2.446l-.744 1.103zm4.241-8.723c.053.273-.165.32-.232.773-.031.213-.285 2.363-.32 2.598a.525.525 0 0 0 .329.542.525.525 0 0 0-.585-.144c-.483.219-1.464 1.254-3.17-.33a5.17 5.17 0 0 1 3.978-3.439zM12 3.53a8.47 8.47 0 1 0 0 16.94 8.47 8.47 0 0 0 0-16.94zm7.045 9.67a7.28 7.28 0 0 1-.43 1.569.525.525 0 0 0-.519-.05.525.525 0 0 0-.27.43l-.336 3.307a7.384 7.384 0 0 1-11.077-1.305l1.123-2.702a.525.525 0 0 0 .273-.363.526.526 0 0 0-.607-.038 7.28 7.28 0 0 1-.932-1.316l2.86-.485a.525.525 0 0 0 .396-.318.526.526 0 0 0-.19-.615L7.18 9.391a7.367 7.367 0 0 1 .773-1.334l1.972 2.34a.526.526 0 0 0 .561.15.526.526 0 0 0 .353-.454l.44-2.958c.397-.154.81-.268 1.23-.346l.51 3.143a.525.525 0 0 0 .3.394.525.525 0 0 0 .501-.03l2.46-1.474c.2.373.367.766.495 1.17l-2.593 1.268a.526.526 0 0 0-.184.743c.507.778.507.778.614.943l.276.433 1.257-1.86a7.284 7.284 0 0 1 .276 1.63l-2.523.48a.526.526 0 0 0-.18.924l1.887 1.664a7.284 7.284 0 0 1-.638 1.498z"/>
                                      </svg>
                                    )}
                                    {!['grafana', 'kubernetes'].includes(connection.type) && (
                                      <svg viewBox="0 0 24 24" className="h-8 w-8 text-gray-600" xmlns="http://www.w3.org/2000/svg">
                                        <path fill="currentColor" d="M18.88 14.88c.08-.79.12-1.79.12-3 0-1.33-.04-2.33-.12-3-.08-.79-.31-1.52-.69-2.2-.37-.69-.9-1.23-1.56-1.62s-1.4-.59-2.19-.59c-.48 0-.96.09-1.46.26s-.96.43-1.4.77c-.4.32-.72.61-.97.88-.24.28-.52.58-.82.97l-1.39 1.65 1.39 1.65c.3.39.58.69.82.97.25.27.57.56.97.88.44.34.9.6 1.4.77s.98.26 1.46.26c.79 0 1.52-.2 2.19-.59s1.19-.93 1.56-1.61c.38-.68.61-1.41.69-2.19zm-7.29.79L9.5 17.76c-.05.06-.09.12-.15.17s-.2.09-.38.09H6.03l3.95-9.76h2.92l.16.43 1.97 4.8-1.93 2.28c-.05.17.01-.05-.41-.8z"/>
                                      </svg>
                                    )}
                                  </div>
                                  <div>
                                    <CardTitle className="text-lg">{connection.name}</CardTitle>
                                    <CardDescription>
                                      {connection.type.charAt(0).toUpperCase() + connection.type.slice(1)} Connection
                                    </CardDescription>
                                  </div>
                                </CardHeader>
                                <CardContent>
                                  <div className="text-sm mb-4">
                                    <div className="flex items-center mb-2">
                                      <p className="mr-2">Connection Status:</p>
                                      <span className="text-green-600 font-medium">Available</span>
                                    </div>
                                    <div className="flex items-center">
                                      <p className="mr-2">MCP Server:</p>
                                      <MCPServerStatus connectionId={connection.id} showControls={true} />
                                    </div>
                                  </div>
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
                                  <Button variant="outline" size="sm" onClick={() => {
                                    setNewConnectionName(connection.name);
                                    setNewConnectionType(connection.type);
                                    setIsConnectionDialogOpen(true);
                                  }}>
                                    Edit
                                  </Button>
                                  <Button variant="secondary" size="sm" onClick={() => {
                                    fetch(`${BACKEND_URL}/api/connections/${connection.id}/mcp/start`, {
                                      method: 'POST',
                                    }).then(response => {
                                      if (response.ok) {
                                        alert("MCP server started successfully");
                                      } else {
                                        alert("Failed to start MCP server");
                                      }
                                    });
                                  }}>
                                    Start MCP
                                  </Button>
                                </CardFooter>
                              </Card>
                            ))}
                            
                            {/* Show default connections if API didn't return them */}
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
                                  <div className="text-sm mb-4">
                                    <div className="flex items-center mb-2">
                                      <p className="mr-2">Connection Status:</p>
                                      <span className="text-green-600 font-medium">Available</span>
                                    </div>
                                    <div className="flex items-center">
                                      <p className="mr-2">MCP Server:</p>
                                      <MCPServerStatus connectionId="grafana-default" showControls={true} />
                                    </div>
                                  </div>
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
                                  setNewConnectionType("grafana");
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