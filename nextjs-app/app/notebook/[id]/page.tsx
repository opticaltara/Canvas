'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useMessage } from 'ai/react';
import useNotebookStore from '@/app/lib/store';
import { CellType, CellStatus } from '@/app/types/notebook';
import { generateStructuredPlan, createCellsFromPlan, InvestigationPlan } from '@/app/lib/ai-agent';

// Import components
import NotebookHeader from '@/app/components/NotebookHeader';
import CellComponent from '@/app/components/CellComponent';
import AddCellButton from '@/app/components/AddCellButton';
import AIQueryInput from '@/app/components/AIQueryInput';

export default function NotebookPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const { id } = params;
  const { currentNotebook, loadNotebook, addCell, executeCell, executeNotebook } = useNotebookStore();
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatingStatus, setGeneratingStatus] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  // Load the notebook
  useEffect(() => {
    if (id === 'new') {
      router.push('/notebook/new');
      return;
    }

    loadNotebook(id);
  }, [id, router, loadNotebook]);

  // Handle AI generation
  const handleAIQuery = async (query: string) => {
    if (!query.trim() || !currentNotebook) return;

    try {
      setIsGenerating(true);
      setGeneratingStatus('Generating investigation plan...');

      // Generate an investigation plan
      const plan = await generateStructuredPlan(query);
      
      setGeneratingStatus('Creating cells from plan...');
      
      // Create cells from the plan
      const cellMap = createCellsFromPlan(plan);
      
      // Add cells to the notebook
      const cellsArray = Array.from(cellMap.values());
      for (const cell of cellsArray) {
        addCell(cell.type, cell.content);
      }
      
      // Scroll to the bottom
      setTimeout(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
      
    } catch (error) {
      console.error('Error generating investigation:', error);
    } finally {
      setIsGenerating(false);
      setGeneratingStatus('');
    }
  };

  // Return loading state if notebook is not loaded
  if (!currentNotebook) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Loading notebook...</h1>
          <div className="w-16 h-16 border-t-4 border-b-4 border-primary-500 rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <NotebookHeader 
        notebook={currentNotebook} 
        onExecuteAll={() => executeNotebook()}
      />
      
      <main className="flex-1 container mx-auto py-4 px-4 lg:px-8 max-w-5xl">
        <div className="notebook">
          {/* AI query input at the top */}
          <div className="mb-8">
            <AIQueryInput 
              onSubmit={handleAIQuery} 
              isGenerating={isGenerating}
              generatingStatus={generatingStatus}
            />
          </div>
          
          {/* Cells */}
          {currentNotebook.cellOrder.length === 0 ? (
            <div className="text-center py-16 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <h3 className="text-xl mb-4">This notebook is empty</h3>
              <p className="mb-4">Start by adding a cell or asking the AI to generate an investigation</p>
              <div className="flex justify-center space-x-4">
                <button 
                  onClick={() => addCell(CellType.MARKDOWN)}
                  className="bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded"
                >
                  Add Markdown Cell
                </button>
                <button 
                  onClick={() => addCell(CellType.PYTHON)}
                  className="bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded"
                >
                  Add Python Cell
                </button>
              </div>
            </div>
          ) : (
            <>
              {currentNotebook.cellOrder.map((cellId, index) => (
                <div key={cellId}>
                  <CellComponent 
                    cell={currentNotebook.cells[cellId]} 
                    executeCell={() => executeCell(cellId)}
                  />
                  <AddCellButton 
                    onAddCell={(type) => addCell(type, '', cellId)} 
                  />
                </div>
              ))}
            </>
          )}
          
          {/* Bottom reference for scrolling */}
          <div ref={bottomRef} />
        </div>
      </main>
    </div>
  );
}