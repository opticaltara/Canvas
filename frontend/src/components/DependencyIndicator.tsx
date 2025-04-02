import React from 'react';
import { Dependency } from '../types/notebook';
import { cn } from '../utils/cn';

interface DependencyIndicatorProps {
  dependencies: Dependency[];
  cellId: string;
  className?: string;
}

export function DependencyIndicator({ dependencies, cellId, className }: DependencyIndicatorProps) {
  // Count dependencies this cell depends on
  const dependsOn = dependencies.filter(d => d.dependent_id === cellId).length;
  
  // Count dependencies that depend on this cell
  const dependedBy = dependencies.filter(d => d.dependency_id === cellId).length;
  
  if (dependsOn === 0 && dependedBy === 0) {
    return null;
  }
  
  return (
    <div className={cn('flex items-center gap-3 text-xs text-gray-500', className)}>
      {dependsOn > 0 && (
        <div title="This cell depends on other cells">
          <span>Inputs: {dependsOn}</span>
        </div>
      )}
      {dependedBy > 0 && (
        <div title="Other cells depend on this cell">
          <span>Outputs: {dependedBy}</span>
        </div>
      )}
    </div>
  );
}
