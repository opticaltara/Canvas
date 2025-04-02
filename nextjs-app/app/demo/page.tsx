'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import useNotebookStore from '../lib/store';
import { CellType } from '../types/notebook';
import EnhancedCellComponent from '../components/EnhancedCellComponent';
import ConnectionManager from '../components/ConnectionManager';
import NotebookSettings from '../components/NotebookSettings';
import DependencyGraph from '../components/DependencyGraph';

export default function DemoPage() {
  const { 
    currentNotebook, 
    createNotebook, 
    addCell, 
    executeCell,
    executeNotebook
  } = useNotebookStore();
  
  const [activeTab, setActiveTab] = useState<'notebook' | 'connections' | 'settings' | 'graph'>('notebook');
  const [autoExecuteEnabled, setAutoExecuteEnabled] = useState(false);
  
  // Create demo notebook on load if none exists
  useEffect(() => {
    if (!currentNotebook) {
      createNotebook('Demo Notebook');
      
      // Setup will be handled after notebook is created
      setTimeout(() => {
        setupDemoNotebook();
      }, 100);
    }
  }, [currentNotebook, createNotebook]);
  
  // Function to setup demo notebook with example cells
  const setupDemoNotebook = () => {
    // Add a markdown cell with instructions
    const markdownId = addCell(CellType.MARKDOWN, `# Reactive Notebook Demo
    
This notebook demonstrates reactivity features in Sherlog Canvas. Try modifying the Python cells and see how dependent cells automatically update.

## Instructions

1. Edit the cells below
2. Add dependencies between cells
3. See how changes propagate automatically
4. Explore the dependency graph
`);
    
    // Add a Python cell for data generation
    const pythonDataId = addCell(CellType.PYTHON, `# Generate sample data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a random dataset
np.random.seed(42)
data = np.random.randn(100)
df = pd.DataFrame({
    'values': data,
    'squared': data ** 2,
    'abs': np.abs(data)
})

print("Data statistics:")
print(df.describe())

# Display the dataframe
df.head(10)
`);
    
    // Add a Python cell for visualization
    const pythonVizId = addCell(CellType.PYTHON, `# Visualize the data from the previous cell
import matplotlib.pyplot as plt

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(df['values'], label='Values')
plt.plot(df['squared'], label='Squared')
plt.plot(df['abs'], label='Absolute')
plt.title('Data Visualization')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
`);
    
    // Add a cell for statistics
    const pythonStatsId = addCell(CellType.PYTHON, `# Calculate statistics on the data
# This cell depends on the data generation cell

# Calculate correlation matrix
corr_matrix = df.corr()
print("Correlation matrix:")
print(corr_matrix)

# Calculate additional statistics
stats = {
    'mean': df.mean().to_dict(),
    'median': df.median().to_dict(),
    'std': df.std().to_dict(),
    'skew': df.skew().to_dict()
}

print("\\nDetailed statistics:")
for stat_name, values in stats.items():
    print(f"{stat_name}:")
    for col, val in values.items():
        print(f"  {col}: {val:.4f}")

# Return the stats for use in other cells
stats
`);
    
    // Set up dependencies
    setTimeout(() => {
      if (addDependency) {
        addDependency(pythonVizId, pythonDataId);
        addDependency(pythonStatsId, pythonDataId);
        
        // Execute cells
        executeNotebook();
      }
    }, 100);
  };
  
  if (!currentNotebook) {
    return <div className="p-8 text-center">Loading notebook...</div>;
  }
  
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">{currentNotebook.name}</h1>
      
      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
        <div className="flex">
          <button
            className={`px-4 py-2 ${activeTab === 'notebook' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('notebook')}
          >
            Notebook
          </button>
          <button
            className={`px-4 py-2 ${activeTab === 'connections' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('connections')}
          >
            Connections
          </button>
          <button
            className={`px-4 py-2 ${activeTab === 'settings' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('settings')}
          >
            Settings
          </button>
          <button
            className={`px-4 py-2 ${activeTab === 'graph' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('graph')}
          >
            Dependency Graph
          </button>
        </div>
      </div>
      
      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'notebook' && (
          <div>
            <div className="mb-4 flex justify-between items-center">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => executeNotebook()}
                  className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Run All Cells
                </button>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="autoExecute"
                    checked={autoExecuteEnabled}
                    onChange={(e) => setAutoExecuteEnabled(e.target.checked)}
                    className="mr-2"
                  />
                  <label htmlFor="autoExecute">Auto-execute cells</label>
                </div>
              </div>
              <div className="space-x-2">
                <button
                  onClick={() => addCell(CellType.MARKDOWN)}
                  className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300"
                >
                  + Markdown
                </button>
                <button
                  onClick={() => addCell(CellType.PYTHON)}
                  className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300"
                >
                  + Python
                </button>
                <button
                  onClick={() => addCell(CellType.SQL)}
                  className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300"
                >
                  + SQL
                </button>
              </div>
            </div>
            
            {/* Cells */}
            <div className="cells-container space-y-4">
              {currentNotebook.cellOrder.map((cellId) => {
                const cell = currentNotebook.cells[cellId];
                if (!cell) return null;
                
                return (
                  <EnhancedCellComponent
                    key={cellId}
                    cell={cell}
                    executeCell={() => executeCell(cellId)}
                    autoExecuteEnabled={autoExecuteEnabled}
                  />
                );
              })}
            </div>
          </div>
        )}
        
        {activeTab === 'connections' && (
          <ConnectionManager />
        )}
        
        {activeTab === 'settings' && (
          <NotebookSettings />
        )}
        
        {activeTab === 'graph' && (
          <DependencyGraph
            cells={currentNotebook.cells}
            cellOrder={currentNotebook.cellOrder}
            onCellClick={(cellId) => {
              // Scroll to cell when clicked in graph
              const cellElement = document.getElementById(`cell-${cellId}`);
              if (cellElement) {
                cellElement.scrollIntoView({ behavior: 'smooth' });
              }
              
              // Switch to notebook tab
              setActiveTab('notebook');
            }}
          />
        )}
      </div>
    </div>
  );
}