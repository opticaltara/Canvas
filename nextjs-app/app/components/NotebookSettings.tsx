'use client';

import { useState } from 'react';
import useNotebookStore from '../lib/store';

interface NotebookSettings {
  autoExecuteEnabled: boolean;
  autoSaveEnabled: boolean;
  autoSaveInterval: number; // in seconds
  theme: 'light' | 'dark' | 'system';
  editorFontSize: number;
  executionTimeout: number; // in seconds
}

export default function NotebookSettings() {
  const { currentNotebook } = useNotebookStore();
  
  // Get settings from notebook metadata or use defaults
  const defaultSettings: NotebookSettings = {
    autoExecuteEnabled: false,
    autoSaveEnabled: true,
    autoSaveInterval: 60,
    theme: 'system',
    editorFontSize: 14,
    executionTimeout: 30,
  };
  
  // Get current settings from notebook metadata
  const initialSettings = currentNotebook?.metadata?.settings 
    ? { ...defaultSettings, ...currentNotebook.metadata.settings }
    : defaultSettings;
  
  const [settings, setSettings] = useState<NotebookSettings>(initialSettings);
  
  // Update a single setting
  const updateSetting = <K extends keyof NotebookSettings>(key: K, value: NotebookSettings[K]) => {
    setSettings({
      ...settings,
      [key]: value,
    });
  };
  
  // Save settings to notebook metadata
  const saveSettings = () => {
    if (!currentNotebook) return;
    
    const updatedNotebook = {
      ...currentNotebook,
      metadata: {
        ...currentNotebook.metadata,
        settings,
      },
    };
    
    // In a real implementation, this would save to the backend
    console.log('Saving notebook settings:', settings);
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm">
      <h2 className="text-xl font-semibold mb-4">Notebook Settings</h2>
      
      <div className="space-y-4">
        {/* Execution settings */}
        <div>
          <h3 className="text-lg font-medium mb-2">Execution</h3>
          
          <div className="flex items-center mb-2">
            <input
              type="checkbox"
              id="autoExecute"
              checked={settings.autoExecuteEnabled}
              onChange={(e) => updateSetting('autoExecuteEnabled', e.target.checked)}
              className="mr-2"
            />
            <label htmlFor="autoExecute">Auto-execute cells when content changes</label>
          </div>
          
          <div className="flex items-center mb-4">
            <label htmlFor="executionTimeout" className="mr-2">Execution timeout (seconds):</label>
            <input
              type="number"
              id="executionTimeout"
              value={settings.executionTimeout}
              onChange={(e) => updateSetting('executionTimeout', Number(e.target.value))}
              min={1}
              max={300}
              className="w-20 px-2 py-1 border rounded"
            />
          </div>
        </div>
        
        {/* Auto-save settings */}
        <div>
          <h3 className="text-lg font-medium mb-2">Auto-save</h3>
          
          <div className="flex items-center mb-2">
            <input
              type="checkbox"
              id="autoSave"
              checked={settings.autoSaveEnabled}
              onChange={(e) => updateSetting('autoSaveEnabled', e.target.checked)}
              className="mr-2"
            />
            <label htmlFor="autoSave">Auto-save notebook</label>
          </div>
          
          <div className="flex items-center mb-4">
            <label htmlFor="autoSaveInterval" className="mr-2">Auto-save interval (seconds):</label>
            <input
              type="number"
              id="autoSaveInterval"
              value={settings.autoSaveInterval}
              onChange={(e) => updateSetting('autoSaveInterval', Number(e.target.value))}
              min={10}
              max={3600}
              disabled={!settings.autoSaveEnabled}
              className={`w-20 px-2 py-1 border rounded ${!settings.autoSaveEnabled ? 'opacity-50' : ''}`}
            />
          </div>
        </div>
        
        {/* Editor settings */}
        <div>
          <h3 className="text-lg font-medium mb-2">Editor</h3>
          
          <div className="flex items-center mb-2">
            <label htmlFor="theme" className="mr-2">Theme:</label>
            <select
              id="theme"
              value={settings.theme}
              onChange={(e) => updateSetting('theme', e.target.value as NotebookSettings['theme'])}
              className="px-2 py-1 border rounded"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </div>
          
          <div className="flex items-center mb-4">
            <label htmlFor="editorFontSize" className="mr-2">Editor font size:</label>
            <input
              type="number"
              id="editorFontSize"
              value={settings.editorFontSize}
              onChange={(e) => updateSetting('editorFontSize', Number(e.target.value))}
              min={10}
              max={24}
              className="w-20 px-2 py-1 border rounded"
            />
          </div>
        </div>
        
        {/* Save button */}
        <div className="flex justify-end">
          <button
            onClick={saveSettings}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}