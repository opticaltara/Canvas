'use client';

import { useEffect, useRef, useState } from 'react';
import { Cell, CellType } from '../types/notebook';

interface DependencyGraphProps {
  cells: Record<string, Cell>;
  cellOrder: string[];
  activeCellId?: string;
  onCellClick?: (cellId: string) => void;
}

// Colors for different cell types
const CELL_COLORS = {
  [CellType.MARKDOWN]: '#60a5fa', // blue
  [CellType.PYTHON]: '#34d399', // green
  [CellType.SQL]: '#f97316', // orange
  [CellType.LOG]: '#8b5cf6', // purple
  [CellType.METRIC]: '#ec4899', // pink
  [CellType.S3]: '#fbbf24', // yellow
  [CellType.AI_QUERY]: '#6b7280', // gray
};

// Minimap version of the graph
function GraphMinimap({ cells, cellOrder, activeCellId, onCellClick }: DependencyGraphProps) {
  return (
    <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded mb-4 overflow-x-auto">
      <div className="flex flex-wrap gap-1">
        {cellOrder.map((cellId) => {
          const cell = cells[cellId];
          if (!cell) return null;
          
          const hasIncoming = Object.values(cells).some(c => 
            c.dependencies.includes(cellId)
          );
          
          return (
            <div
              key={cellId}
              onClick={() => onCellClick?.(cellId)}
              className={`
                w-6 h-6 rounded-sm cursor-pointer flex items-center justify-center text-xs
                ${activeCellId === cellId ? 'ring-2 ring-blue-500' : ''}
                ${hasIncoming || cell.dependencies.length > 0 ? 'opacity-100' : 'opacity-60'}
              `}
              style={{ backgroundColor: CELL_COLORS[cell.type] || '#6b7280' }}
              title={`${cell.type} cell ${cell.metadata?.description || ''}`}
            >
              {cell.dependencies.length > 0 && (
                <span className="text-white font-bold">{cell.dependencies.length}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function DependencyGraph({ cells, cellOrder, activeCellId, onCellClick }: DependencyGraphProps) {
  const [selectedCellId, setSelectedCellId] = useState<string | undefined>(activeCellId);
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Update selected cell when activeCellId changes
  useEffect(() => {
    setSelectedCellId(activeCellId);
  }, [activeCellId]);
  
  // Handle cell selection
  const handleCellSelect = (cellId: string) => {
    setSelectedCellId(cellId);
    onCellClick?.(cellId);
  };
  
  // Render a detailed dependency graph
  const renderDetailedGraph = () => {
    // Display only cells connected to the selected cell
    const relevantCells = new Set<string>();
    
    if (selectedCellId) {
      // Add the selected cell
      relevantCells.add(selectedCellId);
      
      // Add direct dependencies
      const selectedCell = cells[selectedCellId];
      if (selectedCell) {
        selectedCell.dependencies.forEach(id => relevantCells.add(id));
        selectedCell.dependents.forEach(id => relevantCells.add(id));
      }
      
      // Add cells that may be indirectly connected
      const checkConnections = (cellId: string, visited = new Set<string>(), depth = 0) => {
        if (visited.has(cellId) || depth > 2) return; // Limit depth to prevent too many cells
        visited.add(cellId);
        
        const cell = cells[cellId];
        if (!cell) return;
        
        // Add dependencies and check their connections
        cell.dependencies.forEach(id => {
          relevantCells.add(id);
          checkConnections(id, visited, depth + 1);
        });
        
        // Add dependents and check their connections
        cell.dependents.forEach(id => {
          relevantCells.add(id);
          checkConnections(id, visited, depth + 1);
        });
      };
      
      checkConnections(selectedCellId);
    } else {
      // No cell selected, show cells with dependencies
      Object.entries(cells).forEach(([cellId, cell]) => {
        if (cell.dependencies.length > 0 || cell.dependents.length > 0) {
          relevantCells.add(cellId);
          cell.dependencies.forEach(id => relevantCells.add(id));
          cell.dependents.forEach(id => relevantCells.add(id));
        }
      });
    }
    
    // Position cells in a force-directed-like layout (simplified)
    const positions: Record<string, { x: number; y: number }> = {};
    const relevantCellsArray = Array.from(relevantCells);
    
    // Simple positioning for cells
    const cols = Math.ceil(Math.sqrt(relevantCellsArray.length));
    const cellSize = 100;
    const padding = 50;
    
    relevantCellsArray.forEach((cellId, index) => {
      const col = index % cols;
      const row = Math.floor(index / cols);
      
      positions[cellId] = {
        x: col * (cellSize + padding) + cellSize / 2,
        y: row * (cellSize + padding) + cellSize / 2,
      };
    });
    
    // Draw the graph
    const width = cols * (cellSize + padding);
    const height = Math.ceil(relevantCellsArray.length / cols) * (cellSize + padding);
    
    return (
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="border rounded bg-white dark:bg-gray-900"
      >
        {/* Draw edges first so they appear behind nodes */}
        {relevantCellsArray.map(cellId => {
          const cell = cells[cellId];
          if (!cell) return null;
          
          return cell.dependencies.map(depId => {
            if (!positions[depId] || !positions[cellId]) return null;
            
            const start = positions[depId];
            const end = positions[cellId];
            
            // Draw a path from dependency to cell
            return (
              <path
                key={`${depId}-${cellId}`}
                d={`M ${start.x} ${start.y} L ${end.x} ${end.y}`}
                stroke="#9ca3af"
                strokeWidth="2"
                strokeDasharray={cellId === selectedCellId || depId === selectedCellId ? "none" : "4"}
                markerEnd="url(#arrowhead)"
              />
            );
          });
        })}
        
        {/* Draw nodes */}
        {relevantCellsArray.map(cellId => {
          const cell = cells[cellId];
          if (!cell || !positions[cellId]) return null;
          
          const pos = positions[cellId];
          const isSelected = cellId === selectedCellId;
          
          return (
            <g
              key={cellId}
              transform={`translate(${pos.x - 40}, ${pos.y - 25})`}
              onClick={() => handleCellSelect(cellId)}
              style={{ cursor: 'pointer' }}
            >
              {/* Cell background */}
              <rect
                width="80"
                height="50"
                rx="4"
                fill={CELL_COLORS[cell.type] || '#6b7280'}
                stroke={isSelected ? '#3b82f6' : 'none'}
                strokeWidth={isSelected ? 2 : 0}
              />
              
              {/* Cell type */}
              <text
                x="40"
                y="20"
                textAnchor="middle"
                fontSize="14"
                fontWeight="bold"
                fill="white"
              >
                {cell.type}
              </text>
              
              {/* Dependency count */}
              <text
                x="40"
                y="35"
                textAnchor="middle"
                fontSize="10"
                fill="white"
              >
                {cell.dependencies.length} in / {cell.dependents.length} out
              </text>
            </g>
          );
        })}
        
        {/* Arrow marker definition */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" />
          </marker>
        </defs>
      </svg>
    );
  };
  
  return (
    <div className="dependency-graph">
      <h3 className="text-lg font-medium mb-2">Dependency Graph</h3>
      
      {/* Show minimap */}
      <GraphMinimap
        cells={cells}
        cellOrder={cellOrder}
        activeCellId={selectedCellId}
        onCellClick={handleCellSelect}
      />
      
      {/* Detailed graph */}
      <div className="overflow-auto">
        {renderDetailedGraph()}
      </div>
      
      {/* Explanation */}
      <div className="mt-2 text-xs text-gray-500">
        <p>Click on a cell to highlight its dependencies and dependents.</p>
      </div>
    </div>
  );
}