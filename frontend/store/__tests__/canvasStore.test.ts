import { useCanvasStore, NotebookState } from '../canvasStore'
import { api } from '../../api/client'
import { Cell, Notebook, WebSocketStatus } from '../types'

// Mock the API client
jest.mock('../../api/client', () => ({
  api: {
    notebooks: {
      get: jest.fn(),
      list: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
    },
    cells: {
      list: jest.fn(),
      get: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
      execute: jest.fn(),
    },
  },
}))

// Original initial state from the store definition (data properties only)
const originalInitialState = {
  notebook: null,
  cells: [],
  activeNotebookId: null,
  loading: false,
  error: null,
  executingCells: new Set<string>(),
  wsStatus: 'disconnected' as WebSocketStatus,
  wsMessages: [],
  sendChatMessageFunction: null,
  isSendingChatMessage: false,
};

describe('useCanvasStore', () => {
  beforeEach(() => {
    // Reset the store to its defined initial state for data properties
    // This preserves the actual action implementations.
    // Default for the second arg (replace) is false, which means it merges.
    useCanvasStore.setState(originalInitialState); 
    jest.clearAllMocks(); // Clear all mocks on the api client etc.
  })

  it('should have correct initial state', () => {
    const state = useCanvasStore.getState()
    expect(state.notebook).toBeNull()
    expect(state.cells).toEqual([])
    expect(state.activeNotebookId).toBeNull()
    expect(state.loading).toBe(false)
    expect(state.error).toBeNull()
    expect(state.executingCells.size).toBe(0)
    expect(state.wsStatus).toBe('disconnected')
    expect(state.wsMessages).toEqual([])
    expect(state.sendChatMessageFunction).toBeNull()
    expect(state.isSendingChatMessage).toBe(false)
  })

  it('setActiveNotebook should update activeNotebookId', () => {
    const notebookId = 'test-notebook-id'
    useCanvasStore.getState().setActiveNotebook(notebookId)
    expect(useCanvasStore.getState().activeNotebookId).toBe(notebookId)
  })

  it('updateWsStatus should update wsStatus', () => {
    const newStatus: WebSocketStatus = 'connected'
    useCanvasStore.getState().updateWsStatus(newStatus)
    expect(useCanvasStore.getState().wsStatus).toBe(newStatus)
  })

  it('isExecuting should return true if cellId is in executingCells', () => {
    const cellId = 'cell-1'
    useCanvasStore.setState({ executingCells: new Set([cellId]) })
    expect(useCanvasStore.getState().isExecuting(cellId)).toBe(true)
  })

  it('isExecuting should return false if cellId is not in executingCells', () => {
    const cellId = 'cell-1'
    expect(useCanvasStore.getState().isExecuting(cellId)).toBe(false)
  })

  describe('loadNotebook', () => {
    const notebookId = 'notebook-1'
    const mockNotebook: Notebook = { id: notebookId, name: 'Test Notebook', created_at: '', updated_at: '' }
    const mockCells: Cell[] = [{ id: 'cell-1', notebook_id: notebookId, type: 'markdown', content: 'Hello', status: 'idle', created_at: '', updated_at: '' }]

    it('should load notebook and cells successfully', async () => {
      (api.notebooks.get as jest.Mock).mockResolvedValue(mockNotebook);
      (api.cells.list as jest.Mock).mockResolvedValue(mockCells)

      await useCanvasStore.getState().loadNotebook(notebookId)

      const state = useCanvasStore.getState()
      expect(state.loading).toBe(false)
      expect(state.notebook).toEqual(mockNotebook)
      expect(state.cells).toEqual(mockCells)
      expect(state.activeNotebookId).toBe(notebookId)
      expect(state.error).toBeNull()
      expect(api.notebooks.get).toHaveBeenCalledWith(notebookId)
      expect(api.cells.list).toHaveBeenCalledWith(notebookId)
    })

    it('should set error state if loading notebook fails', async () => {
      const errorMessage = 'Failed to load notebook'
      ;(api.notebooks.get as jest.Mock).mockRejectedValue(new Error(errorMessage))
      ;(api.cells.list as jest.Mock).mockResolvedValue(mockCells) // Assume cells list might still be called or not, depending on Promise.all behavior

      await useCanvasStore.getState().loadNotebook(notebookId)

      const state = useCanvasStore.getState()
      expect(state.loading).toBe(false)
      expect(state.error).toBe('Failed to load notebook. Please try again.')
      expect(state.notebook).toBeNull() // Or initial state, depending on reset logic on error
      expect(state.cells).toEqual([]) // Or initial state
    })

     it('should not attempt to load if notebookId is null or undefined', async () => {
      await useCanvasStore.getState().loadNotebook(null as any)
      expect(api.notebooks.get).not.toHaveBeenCalled()
      expect(api.cells.list).not.toHaveBeenCalled()
      // Ensure state remains unchanged or in initial loading: false state
      const state = useCanvasStore.getState()
      expect(state.loading).toBe(false)
    })
  })

  describe('createCell', () => {
    const notebookId = 'notebook-1'
    const newCellData: Cell = { id: 'new-cell', notebook_id: notebookId, type: 'markdown', content: '# New Markdown Cell...', status: 'idle', created_at: '', updated_at: '' }

    beforeEach(() => {
      useCanvasStore.setState({ activeNotebookId: notebookId, cells: [] })
    })

    it('should create a markdown cell and add it to the store', async () => {
      (api.cells.create as jest.Mock).mockResolvedValue(newCellData)

      const createdCell = await useCanvasStore.getState().createCell('markdown')

      expect(api.cells.create).toHaveBeenCalledWith(notebookId, {
        type: 'markdown',
        content: '# New Markdown Cell\n\nEnter your markdown content here.',
      })
      expect(createdCell).toEqual(newCellData)
      const state = useCanvasStore.getState()
      expect(state.cells).toContainEqual(newCellData)
      expect(state.cells.length).toBe(1)
    })

    it('should create a github cell with default content', async () => {
      const githubCellData = { ...newCellData, type: 'github', content: 'GitHub Tool Cell - Configure in UI' }
      ;(api.cells.create as jest.Mock).mockResolvedValue(githubCellData)

      await useCanvasStore.getState().createCell('github')
      expect(api.cells.create).toHaveBeenCalledWith(notebookId, {
        type: 'github',
        content: 'GitHub Tool Cell - Configure in UI',
      })
    })

    it('should create a summarization cell with default content', async () => {
        const summarizationCellData = { ...newCellData, type: 'summarization', content: 'Summarization results will appear here.' }
        ;(api.cells.create as jest.Mock).mockResolvedValue(summarizationCellData)

        await useCanvasStore.getState().createCell('summarization')
        expect(api.cells.create).toHaveBeenCalledWith(notebookId, {
            type: 'summarization',
            content: 'Summarization results will appear here.',
        })
    })


    it('should use initialContent if provided', async () => {
      const initialContent = 'Custom content'
      const customCellData = { ...newCellData, content: initialContent }
      ;(api.cells.create as jest.Mock).mockResolvedValue(customCellData)

      await useCanvasStore.getState().createCell('markdown', initialContent)
      expect(api.cells.create).toHaveBeenCalledWith(notebookId, {
        type: 'markdown',
        content: initialContent,
      })
    })

    it('should return null and not update store if API call fails', async () => {
      (api.cells.create as jest.Mock).mockRejectedValue(new Error('API Error'))
      const result = await useCanvasStore.getState().createCell('markdown')
      expect(result).toBeNull()
      const state = useCanvasStore.getState()
      expect(state.cells.length).toBe(0) // Assuming initial cells was empty
    })

    it('should return null if activeNotebookId is null', async () => {
      useCanvasStore.setState({ activeNotebookId: null })
      const result = await useCanvasStore.getState().createCell('markdown')
      expect(result).toBeNull()
      expect(api.cells.create).not.toHaveBeenCalled()
    })
  })

  describe('deleteCell', () => {
    const notebookId = 'notebook-1'
    const cellIdToDelete = 'cell-to-delete'
    const initialCells: Cell[] = [
      { id: 'cell-1', notebook_id: notebookId, type: 'markdown', content: 'Hello', status: 'idle', created_at: '', updated_at: '' },
      { id: cellIdToDelete, notebook_id: notebookId, type: 'markdown', content: 'Delete me', status: 'idle', created_at: '', updated_at: '' },
    ]

    beforeEach(() => {
      useCanvasStore.setState({ activeNotebookId: notebookId, cells: [...initialCells] })
    })

    it('should delete a cell and remove it from the store', async () => {
      (api.cells.delete as jest.Mock).mockResolvedValue({}) // Assume API returns success

      await useCanvasStore.getState().deleteCell(cellIdToDelete)

      expect(api.cells.delete).toHaveBeenCalledWith(notebookId, cellIdToDelete)
      const state = useCanvasStore.getState()
      expect(state.cells.find(c => c.id === cellIdToDelete)).toBeUndefined()
      expect(state.cells.length).toBe(initialCells.length - 1)
    })

    it('should not change cells if API call fails', async () => {
      (api.cells.delete as jest.Mock).mockRejectedValue(new Error('API Error'))
      await useCanvasStore.getState().deleteCell(cellIdToDelete)
      const state = useCanvasStore.getState()
      expect(state.cells.length).toBe(initialCells.length)
      expect(state.cells.find(c => c.id === cellIdToDelete)).toBeDefined()
    })

    it('should not call API if activeNotebookId is null', async () => {
      useCanvasStore.setState({ activeNotebookId: null })
      await useCanvasStore.getState().deleteCell(cellIdToDelete)
      expect(api.cells.delete).not.toHaveBeenCalled()
    })
  })

  describe('updateCell', () => {
    const notebookId = 'notebook-1'
    const cellIdToUpdate = 'cell-to-update'
    const initialContent = 'Old content'
    const newContent = 'New content'
    const newMetadata = { some: 'data' }
    const initialCells: Cell[] = [
      { id: cellIdToUpdate, notebook_id: notebookId, type: 'markdown', content: initialContent, status: 'idle', created_at: '', updated_at: '' },
    ]

     beforeEach(() => {
      useCanvasStore.setState({ activeNotebookId: notebookId, cells: [...initialCells] })
    })

    it('should call api.cells.update with correct parameters', async () => {
      (api.cells.update as jest.Mock).mockResolvedValue({})

      await useCanvasStore.getState().updateCell(cellIdToUpdate, newContent, newMetadata)

      expect(api.cells.update).toHaveBeenCalledWith(notebookId, cellIdToUpdate, { content: newContent, metadata: newMetadata })
      // Note: This action doesn't update the local store directly, it relies on WebSocket for updates.
    })

    it('should not throw if API call fails (error is logged)', async () => {
       (api.cells.update as jest.Mock).mockRejectedValue(new Error('API Error'))
       await expect(useCanvasStore.getState().updateCell(cellIdToUpdate, newContent, newMetadata)).resolves.not.toThrow()
    })

    it('should not call API if activeNotebookId is null', async () => {
      useCanvasStore.setState({ activeNotebookId: null })
      await useCanvasStore.getState().updateCell(cellIdToUpdate, newContent, newMetadata)
      expect(api.cells.update).not.toHaveBeenCalled()
    })
  })

  describe('executeCell', () => {
    const notebookId = 'notebook-1'
    const cellIdToExecute = 'cell-to-execute'

    beforeEach(() => {
      useCanvasStore.setState({ activeNotebookId: notebookId, executingCells: new Set() })
    })

    it('should call api.cells.execute and add cellId to executingCells', async () => {
      (api.cells.execute as jest.Mock).mockResolvedValue({})

      await useCanvasStore.getState().executeCell(cellIdToExecute)

      expect(api.cells.execute).toHaveBeenCalledWith(notebookId, cellIdToExecute)
      const state = useCanvasStore.getState()
      expect(state.executingCells.has(cellIdToExecute)).toBe(true)
      // Completion of execution is handled by WebSocket, so executingCells is not cleared here.
    })

    it('should remove cellId from executingCells if API call fails', async () => {
      (api.cells.execute as jest.Mock).mockRejectedValue(new Error('API Error'))
      await useCanvasStore.getState().executeCell(cellIdToExecute)
      const state = useCanvasStore.getState()
      expect(state.executingCells.has(cellIdToExecute)).toBe(false)
    })

    it('should not call API if activeNotebookId is null', async () => {
      useCanvasStore.setState({ activeNotebookId: null })
      await useCanvasStore.getState().executeCell(cellIdToExecute)
      expect(api.cells.execute).not.toHaveBeenCalled()
    })
  })

  // WebSocket related handlers
  describe('WebSocket Handlers', () => {
    const cellId = 'cell-1'
    const notebookId = 'notebook-1'
    const initialCell: Cell = { id: cellId, notebook_id: notebookId, type: 'markdown', content: 'Initial', status: 'idle', created_at: '', updated_at: '' }

    it('handleCellExecutionStarted should add cellId to executingCells', () => {
      useCanvasStore.getState().handleCellExecutionStarted(cellId)
      expect(useCanvasStore.getState().executingCells.has(cellId)).toBe(true)
    })

    it('handleCellExecutionCompleted should remove cellId from executingCells', () => {
      useCanvasStore.setState({ executingCells: new Set([cellId]) })
      useCanvasStore.getState().handleCellExecutionCompleted(cellId)
      expect(useCanvasStore.getState().executingCells.has(cellId)).toBe(false)
    })

    it('handleNotebookUpdate should update the notebook data', () => {
      const notebookData: Notebook = { id: notebookId, name: 'Updated Notebook', created_at: '', updated_at: '' }
      useCanvasStore.getState().handleNotebookUpdate(notebookData)
      expect(useCanvasStore.getState().notebook).toEqual(notebookData)
    })

    describe('handleCellUpdate', () => {
        const cellDataUpdate: Cell = { ...initialCell, content: 'Updated Content', status: 'success' }
        const newCellData: Cell = { id: 'new-cell-ws', notebook_id: notebookId, type: 'python', content: 'print("hi")', status: 'idle', created_at: '', updated_at: '' }

        it('should update an existing cell if found', () => {
            useCanvasStore.setState({ cells: [initialCell], activeNotebookId: notebookId })
            useCanvasStore.getState().handleCellUpdate(cellDataUpdate)
            const updatedCell = useCanvasStore.getState().cells.find(c => c.id === cellId)
            expect(updatedCell).toBeDefined()
            expect(updatedCell?.content).toBe('Updated Content')
            expect(updatedCell?.status).toBe('success')
        })

        it('should add a new cell if not found', () => {
            useCanvasStore.setState({ cells: [initialCell], activeNotebookId: notebookId })
            useCanvasStore.getState().handleCellUpdate(newCellData)
            const addedCell = useCanvasStore.getState().cells.find(c => c.id === newCellData.id)
            expect(addedCell).toBeDefined()
            expect(addedCell).toEqual(newCellData)
            expect(useCanvasStore.getState().cells.length).toBe(2)
        })

        it('should not update state if cell data is identical (shallow comparison)', () => {
            useCanvasStore.setState({ cells: [initialCell], activeNotebookId: notebookId })
            const consoleSpy = jest.spyOn(console, 'log')
            useCanvasStore.getState().handleCellUpdate({ ...initialCell }) // Pass a copy with same values
            expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('No changes detected for cell'), initialCell.id, expect.stringContaining('skipping state update'))
            const cell = useCanvasStore.getState().cells.find(c => c.id === cellId)
            expect(cell).toEqual(initialCell) // Ensure it's the original reference or deeply equal
            consoleSpy.mockRestore()
        })
    })


    it('addWsMessage should add message and trigger handlers', () => {
      const cellUpdateMessage = { type: 'cell_update', data: { id: cellId, content: 'WS Update' } }
      const execStartMessage = { type: 'cell_execution_started', data: { cell_id: cellId } }
      const execEndMessage = { type: 'cell_execution_completed', data: { cell_id: cellId } }
      const notebookUpdateMessage = { type: 'notebook_update', data: { id: notebookId, name: 'WS Notebook Update' } }

      const handleCellUpdateMock = jest.spyOn(useCanvasStore.getState(), 'handleCellUpdate')
      const handleCellExecutionStartedMock = jest.spyOn(useCanvasStore.getState(), 'handleCellExecutionStarted')
      const handleCellExecutionCompletedMock = jest.spyOn(useCanvasStore.getState(), 'handleCellExecutionCompleted')
      const handleNotebookUpdateMock = jest.spyOn(useCanvasStore.getState(), 'handleNotebookUpdate')

      useCanvasStore.getState().addWsMessage(cellUpdateMessage)
      expect(useCanvasStore.getState().wsMessages).toContainEqual(cellUpdateMessage)
      expect(handleCellUpdateMock).toHaveBeenCalledWith(cellUpdateMessage.data)

      useCanvasStore.getState().addWsMessage(execStartMessage)
      expect(handleCellExecutionStartedMock).toHaveBeenCalledWith(execStartMessage.data.cell_id)

      useCanvasStore.getState().addWsMessage(execEndMessage)
      expect(handleCellExecutionCompletedMock).toHaveBeenCalledWith(execEndMessage.data.cell_id)

      useCanvasStore.getState().addWsMessage(notebookUpdateMessage)
      expect(handleNotebookUpdateMock).toHaveBeenCalledWith(notebookUpdateMessage.data)

      handleCellUpdateMock.mockRestore()
      handleCellExecutionStartedMock.mockRestore()
      handleCellExecutionCompletedMock.mockRestore()
      handleNotebookUpdateMock.mockRestore()
    })
  })

  // Chat Panel Interaction
  describe('Chat Panel Interaction', () => {
    const mockSendChatMessage = jest.fn()

    beforeEach(() => {
        useCanvasStore.setState({ sendChatMessageFunction: null, isSendingChatMessage: false, error: null })
        mockSendChatMessage.mockClear()
    })

    it('registerSendChatMessageFunction should update the function', () => {
        useCanvasStore.getState().registerSendChatMessageFunction(mockSendChatMessage)
        expect(useCanvasStore.getState().sendChatMessageFunction).toBe(mockSendChatMessage)
    })

    it('sendSuggestedStepToChat should call registered function if available and not already sending', async () => {
        useCanvasStore.setState({ sendChatMessageFunction: mockSendChatMessage })
        const message = "Test chat message"
        mockSendChatMessage.mockResolvedValue(undefined) // Simulate successful send

        await useCanvasStore.getState().sendSuggestedStepToChat(message)

        expect(mockSendChatMessage).toHaveBeenCalledWith(message)
        expect(useCanvasStore.getState().isSendingChatMessage).toBe(false) // Should reset after completion
        expect(useCanvasStore.getState().error).toBeNull()
    })

    it('sendSuggestedStepToChat should not call function if none registered', async () => {
        const consoleWarnSpy = jest.spyOn(console, 'warn')
        const message = "Test chat message"
        await useCanvasStore.getState().sendSuggestedStepToChat(message)
        expect(mockSendChatMessage).not.toHaveBeenCalled()
        expect(consoleWarnSpy).toHaveBeenCalledWith(
            expect.stringContaining("No chat message function registered. Cannot send suggested step:"),
            message
        )
        consoleWarnSpy.mockRestore()
    })

    it('sendSuggestedStepToChat should not call function if already sending', async () => {
        useCanvasStore.setState({ sendChatMessageFunction: mockSendChatMessage, isSendingChatMessage: true })
        const consoleWarnSpy = jest.spyOn(console, 'warn')
        const message = "Test chat message"

        await useCanvasStore.getState().sendSuggestedStepToChat(message)

        expect(mockSendChatMessage).not.toHaveBeenCalled()
        expect(consoleWarnSpy).toHaveBeenCalledWith(
            expect.stringContaining("Chat message sending already in progress. Ignoring suggested step:"),
            message
        )
        consoleWarnSpy.mockRestore()
    })

    it('sendSuggestedStepToChat should set error and reset sending flag if registered function throws', async () => {
        const errorMessage = "Chat send failed"
        useCanvasStore.setState({ sendChatMessageFunction: mockSendChatMessage })
        mockSendChatMessage.mockRejectedValue(new Error(errorMessage))
        const consoleErrorSpy = jest.spyOn(console, 'error')

        const message = "Test chat message"
        await useCanvasStore.getState().sendSuggestedStepToChat(message)

        expect(mockSendChatMessage).toHaveBeenCalledWith(message)
        expect(useCanvasStore.getState().isSendingChatMessage).toBe(false)
        expect(useCanvasStore.getState().error).toBe("Failed to send suggested step to chat.")
        
        // Let's check the exact call arguments
        // console.log('consoleErrorSpy calls:', consoleErrorSpy.mock.calls); 
        // The above line can be added temporarily if run locally to debug calls

        expect(consoleErrorSpy).toHaveBeenCalledWith(
            "[CanvasStore] Error sending suggested step to chat:", // Exact first argument
            expect.any(Error)
        )
        consoleErrorSpy.mockRestore() // Restored after assertion
    })
  }) // Closes 'Chat Panel Interaction' describe
}) // Closes 'useCanvasStore' describe
