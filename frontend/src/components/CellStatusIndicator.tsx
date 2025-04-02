import React from 'react';
import { CellStatus } from '../types/notebook';
import { cn } from '../utils/cn';

interface CellStatusIndicatorProps {
  status: CellStatus;
  className?: string;
}

export function CellStatusIndicator({ status, className }: CellStatusIndicatorProps) {
  let color = '';
  let label = '';

  switch (status) {
    case 'idle':
      color = 'bg-gray-300';
      label = 'Idle';
      break;
    case 'queued':
      color = 'bg-yellow-300';
      label = 'Queued';
      break;
    case 'running':
      color = 'bg-blue-400 animate-pulse';
      label = 'Running';
      break;
    case 'success':
      color = 'bg-green-400';
      label = 'Success';
      break;
    case 'error':
      color = 'bg-red-400';
      label = 'Error';
      break;
  }

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div 
        className={cn(
          'h-3 w-3 rounded-full', 
          color
        )} 
        title={label}
      />
      <span className="text-xs text-gray-500">{label}</span>
    </div>
  );
}
