import { useEffect } from 'react';
import Link from 'next/link';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 md:p-24">
      <div className="max-w-4xl w-full text-center">
        <h1 className="text-4xl md:text-6xl font-bold mb-6">
          Sherlog Canvas
        </h1>
        <p className="text-xl mb-8">
          A reactive notebook for software engineering investigations
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link 
            href="/notebook/new" 
            className="bg-primary-500 hover:bg-primary-600 text-white font-bold py-3 px-6 rounded-lg transition-colors"
          >
            Create New Notebook
          </Link>
          
          <Link 
            href="/notebooks" 
            className="bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 font-bold py-3 px-6 rounded-lg transition-colors"
          >
            Open Existing Notebook
          </Link>
        </div>
        
        {/* Demo Links */}
        <div className="mt-8 mb-4">
          <h2 className="text-2xl font-bold mb-4">Demos & Examples</h2>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/demo"
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg transition-colors"
            >
              Reactive Notebook Demo
            </Link>
            <Link
              href="/python-test"
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors"
            >
              Python Execution Test
            </Link>
          </div>
        </div>
        
        <div className="mt-16 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          <FeatureCard 
            title="Reactive Cells" 
            description="Changes to cells automatically update dependent cells"
          />
          <FeatureCard 
            title="AI Assistance" 
            description="Intelligent investigation planning and cell generation"
          />
          <FeatureCard 
            title="Multi-Source" 
            description="Connect to SQL, logs, metrics, and S3 data sources"
          />
          <FeatureCard 
            title="Python Execution" 
            description="Run Python code directly in the browser with full visualization support"
          />
          <FeatureCard 
            title="Auto Dependencies" 
            description="Smart detection of dependencies between cells"
          />
          <FeatureCard 
            title="Dependency Visualization" 
            description="Interactive graph of cell dependencies"
          />
        </div>
      </div>
    </main>
  );
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow">
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="text-gray-600 dark:text-gray-300">{description}</p>
    </div>
  );
}