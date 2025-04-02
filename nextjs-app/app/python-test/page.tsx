'use client';

import PythonCellTest from '../components/PythonCellTest';

export default function PythonTestPage() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Python Execution Test</h1>
      <p className="mb-4">
        This page demonstrates the execution of Python code in the browser using Pyodide.
        You can edit the code and run it to see the results.
      </p>
      <div className="my-6">
        <PythonCellTest />
      </div>
    </div>
  );
}