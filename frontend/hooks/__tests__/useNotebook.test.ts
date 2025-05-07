import { renderHook, act } from '@testing-library/react'
import { useNotebook } from '../useNotebook'
import { api, Cell, Notebook } from '../../api/client'
import { useWebSocket } from '../useWebSocket'

// Mock API client
jest.mock('../../api/client', () => ({
  api: {
    notebooks: {
      get: jest.fn(),
    },
    cells: {
      list: jest.fn(),
      execute: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
      create: jest.fn(),
    },
  },
}))

// Mock useWebSocket hook
jest.mock('../useWebSocket')

const mockUseWebSocket = useWebSocket as jest.Mock

describe('useNotebook', () => {
  const notebookId = 'test-notebook-id'
  const mockNotebook: Notebook = { id: notebookId, name: 'Test Notebook', created_at: '2023-01-01T00:00:00Z', updated_at: '2023-01-01T00:00:00Z' }
  const mockCells: Cell[] = [
    { id: 'cell-1', notebook_id: notebookId, type: 'markdown', content: 'Hello', status: 'idle', created_at: '2023-01-01T00:00:00Z', updated_at: '2023-01-01T00:00:00Z' },
    { id: 'cell-2', notebook_id: notebookId, type: 'code', content: 'print("world")', status: 'idle', created_at: '2023-01-01T00:00:00Z', updated_at: '2023-01-01T00:00:00Z' },
  ]
  let mockHandleWebSocketMessage: (message: any) => void

  beforeEach(() => {
    jest.clearAllMocks()
    // Setup mock for useWebSocket to capture the message handler
    mockUseWebSocket.mockImplementation((id, handler) => {
      mockHandleWebSocketMessage = handler // Capture the handler
      return { status: 'connected', sendMessage: jest.fn() }
    })
  })

  it('should initialize with loading true and then load notebook and cells', async () => {
    (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
    (api.cells.list as jest.Mock).mockResolvedValue(mockCells)

    const { result, rerender } = renderHook(() => useNotebook(notebookId))

    expect(result.current.loading).toBe(true)
    expect(result.current.notebook).toBeNull()
    expect(result.current.cells).toEqual([])

    // Wait for useEffect to run and API calls to resolve
    await act(async () => {
      // This is a bit of a hack to wait for promises in useEffect to settle
      // A more robust way might involve waitFor from @testing-library/react
      await new Promise(resolve => setTimeout(resolve, 0));
      rerender(); // Rerender to reflect state changes from async operations
    });


    expect(result.current.loading).toBe(false)
    expect(result.current.notebook).toEqual(mockNotebook)
    expect(result.current.cells).toEqual(mockCells)
    expect(api.notebooks.get).toHaveBeenCalledWith(notebookId)
    expect(api.cells.list).toHaveBeenCalledWith(notebookId)
  })

  it('should set error state if loading fails', async () => {
    (api.notebooks.get as jest.Mock).mockRejectedValue(new Error('Failed to load'))

    const { result } = renderHook(() => useNotebook(notebookId))
    
    await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBe('Failed to load notebook. Please try again.')
    expect(result.current.notebook).toBeNull()
    expect(result.current.cells).toEqual([])
  })

  // Test WebSocket message handling
  describe('WebSocket message handling', () => {
    beforeEach(async () => {
      (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
      (api.cells.list as jest.Mock).mockResolvedValue([mockCells[0]]); // Start with one cell
      
      // Initial load
      const { rerender } = renderHook(() => useNotebook(notebookId));
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0));
        rerender();
      });
    });

    it('should handle cell_update message for existing cell', async () => {
      const { result, rerender } = renderHook(() => useNotebook(notebookId))
       await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
      });

      const updatedCellData = { ...mockCells[0], content: 'Updated Content' }
      act(() => {
        mockHandleWebSocketMessage({ type: 'cell_update', data: updatedCellData })
      })
      
      expect(result.current.cells.find(c => c.id === mockCells[0].id)?.content).toBe('Updated Content')
    })

    it('should handle cell_update message for new cell', async () => {
      const { result, rerender } = renderHook(() => useNotebook(notebookId))
       await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
      });
      const newCellData = mockCells[1] // Use the second cell from mockCells as new
      act(() => {
        mockHandleWebSocketMessage({ type: 'cell_update', data: newCellData })
      })
      expect(result.current.cells).toContainEqual(newCellData)
      expect(result.current.cells.length).toBe(2) // Assuming initial load had 1 cell
    })

    it('should handle cell_execution_started message', async () => {
      const { result, rerender } = renderHook(() => useNotebook(notebookId))
       await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
      });
      act(() => {
        mockHandleWebSocketMessage({ type: 'cell_execution_started', data: { cell_id: 'cell-1' } })
      })
      expect(result.current.executingCells.has('cell-1')).toBe(true)
    })

    it('should handle cell_execution_completed message', async () => {
      const { result, rerender } = renderHook(() => useNotebook(notebookId))
       await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        result.current.executeCell('cell-1'); // Start execution
        rerender();
      });

      act(() => {
        mockHandleWebSocketMessage({ type: 'cell_execution_completed', data: { cell_id: 'cell-1' } })
      })
      expect(result.current.executingCells.has('cell-1')).toBe(false)
    })

    it('should handle notebook_update message', async () => {
      const { result, rerender } = renderHook(() => useNotebook(notebookId))
       await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
      });
      const updatedNotebookData = { ...mockNotebook, name: 'Updated Notebook Name' }
      act(() => {
        mockHandleWebSocketMessage({ type: 'notebook_update', data: updatedNotebookData })
      })
      expect(result.current.notebook?.name).toBe('Updated Notebook Name')
    })
  })

  it('executeCell should call api.cells.execute and update executingCells', async () => {
    (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
    (api.cells.list as jest.Mock).mockResolvedValue(mockCells);
    (api.cells.execute as jest.Mock).mockResolvedValue({})

    const { result, rerender } = renderHook(() => useNotebook(notebookId))
    await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
    });

    await act(async () => {
      result.current.executeCell('cell-1')
    })

    expect(api.cells.execute).toHaveBeenCalledWith(notebookId, 'cell-1')
    expect(result.current.executingCells.has('cell-1')).toBe(true)
  })

   it('executeCell should remove cell from executingCells on API failure', async () => {
    (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
    (api.cells.list as jest.Mock).mockResolvedValue(mockCells);
    (api.cells.execute as jest.Mock).mockRejectedValue(new Error("Execute failed"));

    const { result, rerender } = renderHook(() => useNotebook(notebookId))
     await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
    });
    
    // Add to executing set first to simulate optimistic update
    act(() => {
        result.current.executeCell('cell-1');
    });
    expect(result.current.executingCells.has('cell-1')).toBe(true); // Optimistically added

    await act(async () => {
      // Wait for the rejection to be processed
      try {
        await result.current.executeCell('cell-1');
      } catch (e) {
        // error expected
      }
    });
    
    expect(result.current.executingCells.has('cell-1')).toBe(false); // Should be removed on failure
  });


  it('updateCell should call api.cells.update', async () => {
    (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
    (api.cells.list as jest.Mock).mockResolvedValue(mockCells);
    (api.cells.update as jest.Mock).mockResolvedValue({})
    const { result, rerender } = renderHook(() => useNotebook(notebookId))
     await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
    });

    await act(async () => {
      result.current.updateCell('cell-1', 'new content', { meta: 'data' })
    })

    expect(api.cells.update).toHaveBeenCalledWith(notebookId, 'cell-1', { content: 'new content', metadata: { meta: 'data' } })
  })

  it('deleteCell should call api.cells.delete and remove cell from state', async () => {
    (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
    (api.cells.list as jest.Mock).mockResolvedValue(mockCells); // Start with 2 cells
    (api.cells.delete as jest.Mock).mockResolvedValue({})
    const { result, rerender } = renderHook(() => useNotebook(notebookId))
    
    await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
    });
    expect(result.current.cells.length).toBe(2);


    await act(async () => {
      result.current.deleteCell('cell-1')
    })

    expect(api.cells.delete).toHaveBeenCalledWith(notebookId, 'cell-1')
    expect(result.current.cells.length).toBe(1)
    expect(result.current.cells.find(c => c.id === 'cell-1')).toBeUndefined()
  })

  it('createCell should call api.cells.create and add cell to state', async () => {
    (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
    (api.cells.list as jest.Mock).mockResolvedValue([]); // Start with 0 cells
    const newCellData: Cell = { id: 'new-cell', notebook_id: notebookId, type: 'markdown', content: '# New Markdown Cell...', status: 'idle', created_at: '', updated_at: '' }
    ;(api.cells.create as jest.Mock).mockResolvedValue(newCellData)
    
    const { result, rerender } = renderHook(() => useNotebook(notebookId))
    await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
    });
    expect(result.current.cells.length).toBe(0);

    let createdCell
    await act(async () => {
      createdCell = await result.current.createCell('markdown')
    })

    expect(api.cells.create).toHaveBeenCalledWith(notebookId, {
      type: 'markdown',
      content: '# New Markdown Cell\n\nEnter your markdown content here.',
      status: 'idle',
    })
    expect(createdCell).toEqual(newCellData)
    expect(result.current.cells.length).toBe(1)
    expect(result.current.cells[0]).toEqual(newCellData)
  })

  it('isExecuting should correctly report execution status', async () => {
    (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
    (api.cells.list as jest.Mock).mockResolvedValue(mockCells);
    const { result, rerender } = renderHook(() => useNotebook(notebookId))
    await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 0)); // initial load
        rerender();
    });

    expect(result.current.isExecuting('cell-1')).toBe(false)
    act(() => {
      // Simulate starting execution (e.g., via executeCell or WebSocket message)
      mockHandleWebSocketMessage({ type: 'cell_execution_started', data: { cell_id: 'cell-1' } })
    })
    expect(result.current.isExecuting('cell-1')).toBe(true)
    act(() => {
      mockHandleWebSocketMessage({ type: 'cell_execution_completed', data: { cell_id: 'cell-1' } })
    })
    expect(result.current.isExecuting('cell-1')).toBe(false)
  })
})
