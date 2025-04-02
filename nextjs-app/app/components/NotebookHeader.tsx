'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Notebook } from '@/app/types/notebook';
import { Save, Play, Settings, Home } from 'lucide-react';

interface NotebookHeaderProps {
  notebook: Notebook;
  onExecuteAll: () => void;
}

export default function NotebookHeader({ notebook, onExecuteAll }: NotebookHeaderProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [notebookName, setNotebookName] = useState(notebook.name);

  const handleSave = () => {
    // Save notebook name logic would go here
    setIsEditing(false);
  };

  return (
    <header className="bg-white dark:bg-gray-900 shadow-sm sticky top-0 z-10">
      <div className="container mx-auto px-4 py-3 flex justify-between items-center max-w-5xl">
        <div className="flex items-center">
          <Link href="/" className="mr-4">
            <Home className="w-5 h-5 text-gray-500 hover:text-primary-500" />
          </Link>
          
          {isEditing ? (
            <div className="flex items-center">
              <input
                type="text"
                value={notebookName}
                onChange={(e) => setNotebookName(e.target.value)}
                className="border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-lg font-medium"
                autoFocus
                onBlur={handleSave}
                onKeyDown={(e) => e.key === 'Enter' && handleSave()}
              />
            </div>
          ) : (
            <h1 
              className="text-lg font-medium cursor-pointer" 
              onClick={() => setIsEditing(true)}
            >
              {notebook.name}
            </h1>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          <button 
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500"
            title="Save notebook"
          >
            <Save className="w-5 h-5" />
          </button>
          
          <button 
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500"
            title="Run all cells"
            onClick={onExecuteAll}
          >
            <Play className="w-5 h-5" />
          </button>
          
          <button 
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500"
            title="Notebook settings"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
}