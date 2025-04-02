import React from 'react';
import { cn } from '../../utils/cn';

interface TabsProps {
  tabs: Array<{
    id: string;
    label: React.ReactNode;
    content: React.ReactNode;
  }>;
  activeTab: string;
  onChange: (tabId: string) => void;
  className?: string;
}

export function Tabs({ tabs, activeTab, onChange, className }: TabsProps) {
  return (
    <div className={cn('w-full', className)}>
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-4" aria-label="Tabs">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onChange(tab.id)}
                className={cn(
                  'whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm',
                  isActive
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                )}
                aria-current={isActive ? 'page' : undefined}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>
      <div className="py-4">
        {tabs.find((tab) => tab.id === activeTab)?.content}
      </div>
    </div>
  );
}
