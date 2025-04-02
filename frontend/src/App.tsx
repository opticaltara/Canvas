import React from 'react';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 text-gray-900">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold">Sherlog Canvas</h1>
          <div className="flex space-x-2">
            <button className="px-3 py-1 rounded bg-blue-600 text-white">New Notebook</button>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="bg-white p-6 rounded shadow">
          <h2 className="text-xl font-semibold mb-4">Welcome to Sherlog Canvas</h2>
          <p className="mb-4">
            A reactive notebook interface for software engineering investigation tasks.
          </p>
          <div className="p-4 bg-gray-100 rounded">
            <p className="text-sm text-gray-700">
              Backend connection status: <span className="text-red-500">Disconnected</span>
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;