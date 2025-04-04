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

function App() {
  const [notebooks, setNotebooks] = useState<NotebookList[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('notebooks');
  const [newNotebookName, setNewNotebookName] = useState('');
  const [newNotebookDescription, setNewNotebookDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch notebooks
  useEffect(() => {
    setLoading(true);
    fetch('/api/notebooks')
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
  }, []);

  // Create a new notebook
  const handleCreateNotebook = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!newNotebookName.trim()) return;
    
    setIsCreating(true);
    
    try {
      const response = await fetch('/api/notebooks', {
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
                      <Card>
                        <CardHeader>
                          <CardTitle>Data Connections</CardTitle>
                          <CardDescription>
                            Configure your data sources here. Connect to SQL databases, Prometheus, Loki, or S3.
                          </CardDescription>
                        </CardHeader>
                        <CardContent>
                          <div className="rounded-md bg-muted p-4 text-muted-foreground">
                            <p>Coming soon! This feature is currently under development.</p>
                          </div>
                        </CardContent>
                      </Card>
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