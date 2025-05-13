// Defines shared types and interfaces for connection forms

export interface ConnectionConfig {
  [key: string]: any; // Allows for arbitrary configuration keys
}

export interface ConnectionComponentProps {
  config: ConnectionConfig;
  updateConfig: (newConfig: ConnectionConfig) => void;
  isTesting: boolean;
  // Add any other common props needed by connection forms
}
