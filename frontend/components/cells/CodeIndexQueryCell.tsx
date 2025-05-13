import React from 'react';
import { Cell } from '@/store/types'; // Corrected import to use Cell

interface CodeIndexQueryResultItem {
  id: string; // Qdrant point ID
  score: number;
  payload: any; // The actual content of the document stored
  // Depending on qdrant-mcp-server's find tool, 'payload' might contain 'metadata', 'document', etc.
  // For this example, we'll assume payload is the main content or has a 'page_content' and 'metadata'.
  metadata?: {
    file_path?: string;
    repo_url?: string;
    type?: 'summary' | 'tree' | 'file';
    [key: string]: any; // Allow other metadata fields
  };
  document?: { // If payload is structured like LangChain's Document
    page_content?: string;
    metadata?: {
        file_path?: string;
        repo_url?: string;
        type?: 'summary' | 'tree' | 'file';
        [key: string]: any;
    };
  };
}

interface CodeIndexQueryCellProps {
  cell: Cell; // Use the corrected frontend type
}

const CodeIndexQueryCell: React.FC<CodeIndexQueryCellProps> = ({ cell }) => {
  const results: CodeIndexQueryResultItem[] = cell.result?.content || [];
  const query = cell.tool_arguments?.search_query || cell.content; // Fallback to content if specific arg not found

  if (!cell.result) {
    return (
      <div className="p-4 text-sm text-gray-500">
        Code Index Query cell has not been executed yet or has no result.
      </div>
    );
  }

  if (cell.result.error) {
    return (
      <div className="p-4 text-sm text-red-500">
        <p><strong>Error executing Code Index Query:</strong></p>
        <pre className="whitespace-pre-wrap">{cell.result.error}</pre>
      </div>
    );
  }

  return (
    <div className="p-4 text-sm">
      <div className="mb-2 font-semibold text-gray-700">
        Code Search Results for: <code>{query}</code>
        {cell.tool_arguments?.collection_name && (
          <span className="ml-2 text-xs text-gray-500">
            (Collection: {cell.tool_arguments.collection_name})
          </span>
        )}
      </div>
      {results.length === 0 ? (
        <p className="text-gray-500">No results found.</p>
      ) : (
        <ul className="space-y-3">
          {results.map((item, index) => {
            // Try to extract relevant info, adapting to possible structures
            const metadata = item.metadata || item.payload?.metadata || item.document?.metadata;
            const pageContent = item.document?.page_content || item.payload?.page_content || item.payload?.content || JSON.stringify(item.payload);
            
            const filePath = metadata?.file_path || 'N/A';
            const repoUrl = metadata?.repo_url;
            const docType = metadata?.type;

            return (
              <li key={item.id || index} className="p-3 border rounded-md shadow-sm bg-gray-50">
                <div className="flex justify-between items-center mb-1">
                  <strong className="text-gray-800">
                    {docType ? `${docType.charAt(0).toUpperCase() + docType.slice(1)}: ` : ''}
                    {filePath !== 'N/A' ? filePath : (repoUrl || 'Repository Level Result')}
                  </strong>
                  <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
                    Score: {item.score.toFixed(4)}
                  </span>
                </div>
                {repoUrl && filePath === 'N/A' && ( // Show repo URL if it's a repo-level item like summary/tree
                   <p className="text-xs text-gray-500 mb-1">Repository: {repoUrl}</p>
                )}
                <pre className="mt-1 p-2 bg-white text-xs text-gray-700 border rounded whitespace-pre-wrap max-h-60 overflow-y-auto">
                  {typeof pageContent === 'string' ? pageContent : JSON.stringify(pageContent, null, 2)}
                </pre>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default CodeIndexQueryCell;
