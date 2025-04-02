import React from 'react';
import { CellStatus } from '../types/notebook';

interface CellStatusIndicatorProps {
  status: CellStatus;
}

const CellStatusIndicator: React.FC<CellStatusIndicatorProps> = ({ status }) => {
  // Define status colors
  const statusStyles: Record<CellStatus, { color: string; icon: JSX.Element; text: string }> = {
    [CellStatus.IDLE]: {
      color: 'text-gray-400',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
          <path fillRule="evenodd" d="M2 10a8 8 0 1116 0 8 8 0 01-16 0zm8 1a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
      ),
      text: 'Idle',
    },
    [CellStatus.QUEUED]: {
      color: 'text-blue-400',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 animate-pulse">
          <path d="M4 5h12v7H4z" />
          <path fillRule="evenodd" d="M1 3.5A1.5 1.5 0 012.5 2h15A1.5 1.5 0 0119 3.5v10a1.5 1.5 0 01-1.5 1.5h-5.248a1.5 1.5 0 00-1.275.713l-1.58 2.515a.75.75 0 01-1.27.001l-1.582-2.516A1.5 1.5 0 006.27 15H1.5A1.5 1.5 0 010 13.5v-10A1.5 1.5 0 011.5 2h15A1.5 1.5 0 0119 3.5" clipRule="evenodd" />
        </svg>
      ),
      text: 'Queued',
    },
    [CellStatus.RUNNING]: {
      color: 'text-blue-500',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 animate-spin">
          <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z" clipRule="evenodd" />
        </svg>
      ),
      text: 'Running',
    },
    [CellStatus.SUCCESS]: {
      color: 'text-green-500',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
        </svg>
      ),
      text: 'Success',
    },
    [CellStatus.ERROR]: {
      color: 'text-red-500',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
      ),
      text: 'Error',
    },
    [CellStatus.STALE]: {
      color: 'text-yellow-500',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
          <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
      ),
      text: 'Stale',
    },
  };
  
  const { color, icon, text } = statusStyles[status];
  
  return (
    <div className="flex items-center" title={text}>
      <span className={color}>{icon}</span>
    </div>
  );
};

export default CellStatusIndicator;