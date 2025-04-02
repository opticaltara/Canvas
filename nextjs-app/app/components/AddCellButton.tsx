'use client';

import { useState } from 'react';
import { Plus, Code, FileText, Database, BarChart2, FileBox, Lock } from 'lucide-react';
import { CellType } from '@/app/types/notebook';

interface AddCellButtonProps {
  onAddCell: (type: CellType) => void;
}

export default function AddCellButton({ onAddCell }: AddCellButtonProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const handleAddCell = (type: CellType) => {
    onAddCell(type);
    setIsMenuOpen(false);
  };

  return (
    <div className="relative my-2 group">
      <div className="absolute left-1/2 transform -translate-x-1/2 z-10">
        <div 
          className={`flex items-center justify-center w-8 h-8 rounded-full 
            ${isMenuOpen ? 'bg-primary-500 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300 opacity-0 group-hover:opacity-100'} 
            cursor-pointer transition-all duration-150`}
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          <Plus className="w-5 h-5" />
        </div>
        
        {isMenuOpen && (
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-1 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-2 min-w-max">
            <div className="grid grid-cols-2 gap-1">
              <CellButton 
                icon={<FileText className="w-4 h-4" />}
                text="Markdown"
                onClick={() => handleAddCell(CellType.MARKDOWN)}
              />
              <CellButton 
                icon={<Code className="w-4 h-4" />}
                text="Python"
                onClick={() => handleAddCell(CellType.PYTHON)}
              />
              <CellButton 
                icon={<Database className="w-4 h-4" />}
                text="SQL"
                onClick={() => handleAddCell(CellType.SQL)}
              />
              <CellButton 
                icon={<FileText className="w-4 h-4" />}
                text="Log"
                onClick={() => handleAddCell(CellType.LOG)}
              />
              <CellButton 
                icon={<BarChart2 className="w-4 h-4" />}
                text="Metric"
                onClick={() => handleAddCell(CellType.METRIC)}
              />
              <CellButton 
                icon={<FileBox className="w-4 h-4" />}
                text="S3"
                onClick={() => handleAddCell(CellType.S3)}
              />
              <CellButton 
                icon={<Lock className="w-4 h-4" />}
                text="AI Query"
                onClick={() => handleAddCell(CellType.AI_QUERY)}
              />
            </div>
          </div>
        )}
      </div>
      
      {/* Divider line */}
      <div className="border-t border-gray-200 dark:border-gray-800 my-1"></div>
    </div>
  );
}

interface CellButtonProps {
  icon: React.ReactNode;
  text: string;
  onClick: () => void;
}

function CellButton({ icon, text, onClick }: CellButtonProps) {
  return (
    <button 
      className="flex items-center space-x-2 p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded text-left text-sm"
      onClick={onClick}
    >
      <span>{icon}</span>
      <span>{text}</span>
    </button>
  );
}