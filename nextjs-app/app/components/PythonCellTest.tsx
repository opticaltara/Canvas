/**
 * This is a test component to demonstrate Python cell execution with Pyodide
 */
'use client';

import { useState, useEffect } from 'react';
import { CellType, CellStatus } from '../types/notebook';
import { getExecutionManager } from '../lib/execution';

export default function PythonCellTest() {
  const [code, setCode] = useState(`
# A simple Python example that uses matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

# Display the plot
plt.show()

# Return a dictionary of values
{'x': x.tolist(), 'y': y.tolist()}
  `);

  const [output, setOutput] = useState<{ content: any; error?: string }>({ content: null });
  const [status, setStatus] = useState<'idle' | 'running' | 'complete' | 'error'>('idle');

  async function runCode() {
    setStatus('running');
    
    try {
      // Create a simple cell object
      const cell = {
        id: 'test-cell',
        type: CellType.PYTHON,
        content: code,
        status: CellStatus.RUNNING,
        dependencies: [],
        dependents: [],
        metadata: {},
      };
      
      // Get the execution manager
      const executionManager = getExecutionManager();
      
      // Execute the cell
      const result = await executionManager.executeCell(cell, { 'test-cell': cell });
      
      // Update the output
      setOutput({
        content: result.content,
        error: result.error,
      });
      
      setStatus(result.error ? 'error' : 'complete');
    } catch (error) {
      setOutput({
        content: null,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      setStatus('error');
    }
  }
  
  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">Python Cell Test</h2>
      
      <div className="mb-4">
        <label htmlFor="code-editor" className="block mb-2">Python Code:</label>
        <textarea
          id="code-editor"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className="w-full h-64 p-2 font-mono text-sm bg-gray-100 border rounded"
        />
      </div>
      
      <button
        onClick={runCode}
        disabled={status === 'running'}
        className={`px-4 py-2 rounded ${
          status === 'running'
            ? 'bg-gray-400'
            : 'bg-blue-500 hover:bg-blue-600 text-white'
        }`}
      >
        {status === 'running' ? 'Running...' : 'Run Code'}
      </button>
      
      <div className="mt-6">
        <h3 className="text-xl font-semibold mb-2">Output:</h3>
        <div className={`p-4 rounded ${status === 'error' ? 'bg-red-100' : 'bg-gray-100'}`}>
          {status === 'idle' ? (
            <p className="text-gray-500">Code not executed yet</p>
          ) : status === 'running' ? (
            <p className="text-blue-500">Running...</p>
          ) : status === 'error' ? (
            <pre className="text-red-500 overflow-x-auto whitespace-pre-wrap">
              {output.error || 'Unknown error'}
            </pre>
          ) : (
            <div>
              {/* Display text output */}
              {output.content?.output && (
                <pre className="mb-4 overflow-x-auto whitespace-pre-wrap">
                  {output.content.output}
                </pre>
              )}
              
              {/* Display plot if present */}
              {output.content?.result?.plot && (
                <div className="my-4">
                  <img 
                    src={output.content.result.plot} 
                    alt="Python plot" 
                    className="max-w-full h-auto border"
                  />
                </div>
              )}
              
              {/* Display result object */}
              {output.content?.result && typeof output.content.result === 'object' && !output.content.result.plot && (
                <pre className="overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(output.content.result, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}