import axios from 'axios'
import { notebooks } from '../notebooks' // Assuming this is the correct export from your notebooks.ts
import { BACKEND_URL } from '@/config/api-config'
import type { Notebook, Cell } from '@/store/types'

jest.mock('axios')
const mockedAxios = axios as jest.Mocked<typeof axios>

const API_BASE_URL = `${BACKEND_URL}/api`

describe('Notebooks API', () => {
  afterEach(() => {
    jest.clearAllMocks()
  })

  describe('list', () => {
    it('should fetch a list of notebooks', async () => {
      const mockNotebooks: Notebook[] = [
        { id: '1', name: 'Notebook 1', created_at: '2023-01-01T00:00:00Z', updated_at: '2023-01-01T00:00:00Z' },
        { id: '2', name: 'Notebook 2', created_at: '2023-01-01T00:00:00Z', updated_at: '2023-01-01T00:00:00Z' },
      ]
      mockedAxios.get.mockResolvedValue({ data: mockNotebooks })

      const result = await notebooks.list()

      expect(mockedAxios.get).toHaveBeenCalledWith(`${API_BASE_URL}/notebooks`)
      expect(result).toEqual(mockNotebooks)
    })
  })

  describe('get', () => {
    it('should fetch a single notebook by ID', async () => {
      const notebookId = '1'
      const mockNotebook: Notebook = { id: notebookId, name: 'Notebook 1', created_at: '2023-01-01T00:00:00Z', updated_at: '2023-01-01T00:00:00Z' }
      mockedAxios.get.mockResolvedValue({ data: mockNotebook })

      const result = await notebooks.get(notebookId)

      expect(mockedAxios.get).toHaveBeenCalledWith(`${API_BASE_URL}/notebooks/${notebookId}`)
      expect(result).toEqual(mockNotebook)
    })
  })

  describe('create', () => {
    it('should create a new notebook', async () => {
      const newNotebookData = { title: 'New Notebook', description: 'A description' }
      const mockCreatedNotebook: Notebook = { id: '3', name: newNotebookData.title, description: newNotebookData.description, created_at: '2023-01-01T00:00:00Z', updated_at: '2023-01-01T00:00:00Z' }
      mockedAxios.post.mockResolvedValue({ data: mockCreatedNotebook })

      const result = await notebooks.create(newNotebookData)

      expect(mockedAxios.post).toHaveBeenCalledWith(`${API_BASE_URL}/notebooks`, newNotebookData)
      expect(result).toEqual(mockCreatedNotebook)
    })
  })

  describe('update', () => {
    it('should update an existing notebook', async () => {
      const notebookId = '1'
      const updateData: Partial<Notebook> = { name: 'Updated Notebook Name' }
      const mockUpdatedNotebook: Notebook = { id: notebookId, name: 'Updated Notebook Name', created_at: '2023-01-01T00:00:00Z', updated_at: '2023-01-01T00:00:00Z' }
      mockedAxios.put.mockResolvedValue({ data: mockUpdatedNotebook })

      const result = await notebooks.update(notebookId, updateData)

      expect(mockedAxios.put).toHaveBeenCalledWith(`${API_BASE_URL}/notebooks/${notebookId}`, updateData)
      expect(result).toEqual(mockUpdatedNotebook)
    })
  })

  describe('delete', () => {
    it('should delete a notebook', async () => {
      const notebookId = '1'
      mockedAxios.delete.mockResolvedValue({})

      await notebooks.delete(notebookId)

      expect(mockedAxios.delete).toHaveBeenCalledWith(`${API_BASE_URL}/notebooks/${notebookId}`)
    })
  })

  describe('sendMessage', () => {
    it('should send a message to a cell within a notebook', async () => {
      const notebookId = '1'
      const cellId = 'cell-abc'
      const message = 'Hello Cell'
      const mockCellResponse: Cell = {
        id: cellId,
        notebook_id: notebookId,
        type: 'markdown', // Example type
        content: 'Response content',
        status: 'success', // Example status
        created_at: '2023-01-01T00:00:00Z',
        updated_at: '2023-01-01T00:00:00Z',
      }
      mockedAxios.post.mockResolvedValue({ data: mockCellResponse })

      const result = await notebooks.sendMessage(notebookId, cellId, message)

      expect(mockedAxios.post).toHaveBeenCalledWith(`${API_BASE_URL}/notebooks/${notebookId}/cells/${cellId}/message`, { message })
      expect(result).toEqual(mockCellResponse)
    })
  })
})
