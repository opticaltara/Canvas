import React from 'react';
import { Cell } from '../types/notebook';

interface DependencyIndicatorProps {
  dependencies: Cell[];
  dependents: Cell[];
  showDependencies: boolean;
  onToggle: () => void;
}

const DependencyIndicator: React.FC<DependencyIndicatorProps> = ({
  dependencies,
  dependents,
  showDependencies,
  onToggle,
}) => {
  const hasDependencies = dependencies.length > 0;
  const hasDependents = dependents.length > 0;
  const hasAny = hasDependencies || hasDependents;
  
  if (!hasAny) {
    return null;
  }
  
  return (
    <button
      onClick={onToggle}
      className={`flex items-center space-x-1 px-2 py-1 rounded text-xs ${
        showDependencies
          ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
      }`}
      title="Toggle dependencies view"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="currentColor"
        className="w-4 h-4"
      >
        <path
          fillRule="evenodd"
          d="M12.97 3.97a.75.75 0 011.06 0l7.5 7.5a.75.75 0 010 1.06l-7.5 7.5a.75.75 0 11-1.06-1.06l6.22-6.22H3a.75.75 0 010-1.5h16.19l-6.22-6.22a.75.75 0 010-1.06z"
          clipRule="evenodd"
        />
      </svg>
      
      <span className="font-medium">
        {hasDependencies && hasDependents
          ? `${dependencies.length}↑ ${dependents.length}↓`
          : hasDependencies
          ? `${dependencies.length}↑`
          : `${dependents.length}↓`}
      </span>
    </button>
  );
};

export default DependencyIndicator;