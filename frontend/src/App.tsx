import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { NotebookView } from './components/NotebookView';
import { Tabs } from './components/ui/Tabs';
import { NotebookList } from './types/notebook';

function App() {
  const [notebooks, setNotebooks] = useState<NotebookList[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('notebooks');
  const [newNotebookName, setNewNotebookName] = useState('');
  const [newNotebookDescription, setNewNotebookDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);

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
      
      // Navigate to new notebook
      window.location.href = `/notebooks/${newNotebook.id}`;
    } catch (err: any) {
      setError(err.message);
      setIsCreating(false);
    }
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 text-gray-900">
        <header className="bg-white shadow">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <Link to="/" className="text-xl font-bold text-blue-600">Sherlog Canvas</Link>
          </div>
        </header>

        <Routes>
          <Route path="/" element={
            <main className="max-w-7xl mx-auto px-4 py-6">
              <Tabs
                tabs={[
                  {
                    id: 'notebooks',
                    label: 'Notebooks',
                    content: (
                      <div>
                        {/* Create new notebook form */}
                        <div className="mb-6 bg-white p-4 rounded-md shadow">
                          <h2 className="text-lg font-semibold mb-4">Create New Notebook</h2>
                          <form onSubmit={handleCreateNotebook} className="space-y-4">
                            <div>
                              <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                                Name
                              </label>
                              <input
                                type="text"
                                id="name"
                                value={newNotebookName}
                                onChange={(e) => setNewNotebookName(e.target.value)}
                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                                placeholder="Enter notebook name"
                                required
                              />
                            </div>
                            <div>
                              <label htmlFor="description" className="block text-sm font-medium text-gray-700">
                                Description
                              </label>
                              <textarea
                                id="description"
                                value={newNotebookDescription}
                                onChange={(e) => setNewNotebookDescription(e.target.value)}
                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                                placeholder="Describe the notebook purpose"
                                rows={3}
                              />
                            </div>
                            <button
                              type="submit"
                              className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                              disabled={isCreating}
                            >
                              {isCreating ? 'Creating...' : 'Create Notebook'}
                            </button>
                          </form>
                        </div>

                        {/* Notebooks list */}
                        <div className="bg-white rounded-md shadow overflow-hidden">
                          <div className="px-4 py-5 sm:px-6">
                            <h3 className="text-lg font-medium leading-6 text-gray-900">Your Notebooks</h3>
                          </div>
                          
                          {loading ? (
                            <div className="flex justify-center items-center h-32">
                              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                            </div>
                          ) : error ? (
                            <div className="px-4 py-5 text-center text-red-500">
                              {error}
                            </div>
                          ) : notebooks.length === 0 ? (
                            <div className="px-4 py-5 text-center text-gray-500">
                              No notebooks found. Create your first one!
                            </div>
                          ) : (
                            <ul className="divide-y divide-gray-200">
                              {notebooks.map((notebook) => (
                                <li key={notebook.id} className="hover:bg-gray-50">
                                  <Link
                                    to={`/notebooks/${notebook.id}`}
                                    className="block px-4 py-4 sm:px-6"
                                  >
                                    <div className="flex items-center justify-between">
                                      <div className="flex-1 min-w-0">
                                        <p className="text-sm font-medium text-blue-600 truncate">
                                          {notebook.name}
                                        </p>
                                        <p className="text-sm text-gray-500 truncate">
                                          {notebook.description || 'No description'}
                                        </p>
                                      </div>
                                      <div className="ml-4 flex flex-shrink-0">
                                        <p className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                          {notebook.cell_count} cells
                                        </p>
                                        <p className="ml-2 text-xs text-gray-500">
                                          Updated {new Date(notebook.updated_at).toLocaleDateString()}
                                        </p>
                                      </div>
                                    </div>
                                  </Link>
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                      </div>
                    ),
                  },
                  {
                    id: 'connections',
                    label: 'Data Connections',
                    content: (
                      <div className="bg-white rounded-md shadow p-6">
                        <h2 className="text-lg font-semibold mb-4">Data Connections</h2>
                        <p className="text-gray-500">
                          Configure your data sources here. Connect to SQL databases, Prometheus, Loki, or S3.
                        </p>
                        <div className="mt-4 p-4 bg-blue-50 rounded-md text-blue-700">
                          <p>Coming soon! This feature is currently under development.</p>
                        </div>
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