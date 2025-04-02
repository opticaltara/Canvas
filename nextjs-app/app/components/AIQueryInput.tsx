'use client';

import { useState, useRef } from 'react';
import { Search, Sparkles, Loader2 } from 'lucide-react';
import { useUIState } from 'ai/react';

interface AIQueryInputProps {
  onSubmit: (query: string) => void;
  isGenerating?: boolean;
  generatingStatus?: string;
}

export default function AIQueryInput({ onSubmit, isGenerating = false, generatingStatus = '' }: AIQueryInputProps) {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isGenerating) return;
    
    onSubmit(query);
    setQuery('');
  };

  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm p-4">
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Sparkles className="text-primary-500 w-5 h-5" />
          <h2 className="text-lg font-medium">AI Investigation</h2>
        </div>
        
        <form onSubmit={handleSubmit} className="relative">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe the issue you want to investigate..."
            className="w-full p-3 pr-12 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-primary-400"
            disabled={isGenerating}
          />
          <button
            type="submit"
            className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-gray-400 hover:text-primary-500 focus:outline-none disabled:opacity-50"
            disabled={!query.trim() || isGenerating}
          >
            {isGenerating ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Search className="w-5 h-5" />
            )}
          </button>
        </form>
        
        {isGenerating && (
          <div className="flex items-center justify-center mt-2 space-x-2">
            <Loader2 className="w-4 h-4 animate-spin text-primary-500" />
            <span className="text-sm text-gray-500 dark:text-gray-400">{generatingStatus || 'Generating cells...'}</span>
          </div>
        )}
        
        <div className="text-xs text-gray-500">
          <p>Example queries:</p>
          <div className="flex flex-wrap gap-2 mt-1">
            <QueryPill 
              text="Why are we seeing a spike in error rates?" 
              onClick={() => {
                setQuery("Why are we seeing a spike in error rates?");
                inputRef.current?.focus();
              }}
            />
            <QueryPill 
              text="Analyze database performance issues" 
              onClick={() => {
                setQuery("Analyze database performance issues");
                inputRef.current?.focus();
              }}
            />
            <QueryPill 
              text="Investigate slow API response times" 
              onClick={() => {
                setQuery("Investigate slow API response times");
                inputRef.current?.focus();
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

interface QueryPillProps {
  text: string;
  onClick: () => void;
}

function QueryPill({ text, onClick }: QueryPillProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="px-2 py-1 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-full text-xs transition-colors"
    >
      {text}
    </button>
  );
}