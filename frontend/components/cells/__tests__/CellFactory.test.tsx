import React from 'react'
import { render, screen } from '@testing-library/react'
import CellFactory from '../CellFactory'
import type { Cell, CellType } from '../../../store/types' // Adjusted path

// Mock next/dynamic
jest.mock('next/dynamic', () => ({
  __esModule: true,
  default: (loader: () => Promise<any>) => {
    // In a test environment, we can resolve the dynamic import immediately
    // or return a mock. For simplicity, let's try to load it.
    // This might still require the loader to be a function that returns a module with a default export.
    let LoadedComponent: React.ComponentType<any> | null = null;
    try {
      // This is a bit of a hack and might not always work depending on how Jest handles the import()
      // A more robust solution might involve specific mocks for each dynamic component path.
      const dynamicImport = loader();
      dynamicImport.then((mod) => {
        LoadedComponent = mod.default;
      }).catch(() => {
        // Fallback or error
      });
    } catch (e) {
      // Fallback
    }
    // Return a component that tries to render the loaded one, or a placeholder
    return (props: any) => {
      if (LoadedComponent) {
        return <LoadedComponent {...props} />;
      }
      // Fallback if direct loading in mock fails, rely on individual mocks below.
      // This part might be tricky. The individual mocks below should take precedence if this fails.
      return <div data-testid="dynamic-loading-fallback">Loading...</div>;
    };
  },
}));


// Mock individual cell components
// It's better to import the mocked components to access their .mock property
import MarkdownCellMock from '../MarkdownCell';
import GitHubCellMock from '../GitHubCell';
import SummarizationCellMock from '../SummarizationCell';
import InvestigationReportCellMock from '../InvestigationReportCell';
import FileSystemCellMock from '../FileSystemCell';
import PythonCellMock from '../PythonCell';

jest.mock('../MarkdownCell', () => ({
  __esModule: true,
  default: jest.fn((props) => <div data-testid="markdown-cell" data-props={JSON.stringify(props)} />),
}));
jest.mock('../GitHubCell', () => ({
  __esModule: true,
  default: jest.fn((props) => <div data-testid="github-cell" data-props={JSON.stringify(props)} />),
}));
jest.mock('../SummarizationCell', () => ({
  __esModule: true,
  default: jest.fn((props) => <div data-testid="summarization-cell" data-props={JSON.stringify(props)} />),
}));
jest.mock('../InvestigationReportCell', () => ({
  __esModule: true,
  default: jest.fn((props) => <div data-testid="investigation-report-cell" data-props={JSON.stringify(props)} />),
}));
jest.mock('../FileSystemCell', () => ({
  __esModule: true,
  default: jest.fn((props) => <div data-testid="filesystem-cell" data-props={JSON.stringify(props)} />),
}));
jest.mock('../PythonCell', () => ({
  __esModule: true,
  default: jest.fn((props) => <div data-testid="python-cell" data-props={JSON.stringify(props)} />),
}));


describe('CellFactory', () => {
  const mockOnExecute = jest.fn()
  const mockOnUpdate = jest.fn()
  const mockOnDelete = jest.fn()
  const baseCellProps = {
    onExecute: mockOnExecute,
    onUpdate: mockOnUpdate,
    onDelete: mockOnDelete,
    isExecuting: false,
  }

  const createCell = (type: CellType, overrides = {}): Cell => ({
    id: `cell-${type}`,
    notebook_id: 'notebook-1',
    type,
    content: `Content for ${type}`,
    status: 'idle',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  })

  afterEach(() => {
    jest.clearAllMocks()
  })

  it('renders MarkdownCell for "markdown" type', () => {
    const cell = createCell('markdown')
    render(<CellFactory cell={cell} {...baseCellProps} />)
    expect(screen.getByTestId('markdown-cell')).toBeInTheDocument();
    const receivedProps = (MarkdownCellMock as unknown as jest.Mock).mock.calls[0][0];
    expect(receivedProps.cell).toEqual(cell);
    expect(receivedProps.onUpdate).toBe(mockOnUpdate);
    expect(receivedProps.onDelete).toBe(mockOnDelete);
  })

  it('renders GitHubCell for "github" type', () => {
    const cell = createCell('github')
    render(<CellFactory cell={cell} {...baseCellProps} />)
    expect(screen.getByTestId('github-cell')).toBeInTheDocument();
    const receivedProps = (GitHubCellMock as unknown as jest.Mock).mock.calls[0][0];
    expect(receivedProps.cell).toEqual(cell);
    expect(receivedProps.onExecute).toBe(mockOnExecute);
    expect(receivedProps.onUpdate).toBe(mockOnUpdate);
    expect(receivedProps.onDelete).toBe(mockOnDelete);
    expect(receivedProps.isExecuting).toBe(false);
  })

  it('renders SummarizationCell for "summarization" type', () => {
    const cell = createCell('summarization')
    render(<CellFactory cell={cell} {...baseCellProps} />)
    expect(screen.getByTestId('summarization-cell')).toBeInTheDocument();
    const receivedProps = (SummarizationCellMock as unknown as jest.Mock).mock.calls[0][0];
    expect(receivedProps.cell).toEqual(cell);
    expect(receivedProps.onDelete).toBe(mockOnDelete);
    expect(receivedProps.onExecute).toBeUndefined();
    expect(receivedProps.onUpdate).toBeUndefined();
  })

  it('renders InvestigationReportCell for "investigation_report" type', () => {
    const cell = createCell('investigation_report')
    render(<CellFactory cell={cell} {...baseCellProps} />)
    expect(screen.getByTestId('investigation-report-cell')).toBeInTheDocument();
    const receivedProps = (InvestigationReportCellMock as unknown as jest.Mock).mock.calls[0][0];
    expect(receivedProps.cell).toEqual(cell);
    expect(receivedProps.onDelete).toBe(mockOnDelete);
    expect(receivedProps.onExecute).toBeUndefined();
    expect(receivedProps.onUpdate).toBeUndefined();
  })

  it('renders FileSystemCell for "filesystem" type', () => {
    const cell = createCell('filesystem')
    render(<CellFactory cell={cell} {...baseCellProps} />)
    expect(screen.getByTestId('filesystem-cell')).toBeInTheDocument();
    const receivedProps = (FileSystemCellMock as unknown as jest.Mock).mock.calls[0][0];
    expect(receivedProps.cell).toEqual(cell);
    expect(receivedProps.onUpdate).toBe(mockOnUpdate);
    expect(receivedProps.onExecute).toBe(mockOnExecute);
    expect(receivedProps.onDelete).toBe(mockOnDelete);
    expect(receivedProps.isExecuting).toBe(false);
  })

  it('renders PythonCell for "python" type', () => {
    const cell = createCell('python')
    render(<CellFactory cell={cell} {...baseCellProps} />)
    expect(screen.getByTestId('python-cell')).toBeInTheDocument();
    const receivedProps = (PythonCellMock as unknown as jest.Mock).mock.calls[0][0];
    expect(receivedProps.cell).toEqual(cell);
    expect(receivedProps.onExecute).toBe(mockOnExecute);
    expect(receivedProps.onUpdate).toBe(mockOnUpdate);
    expect(receivedProps.onDelete).toBe(mockOnDelete);
    expect(receivedProps.isExecuting).toBe(false);
  })

  it('renders unknown cell type message for an unsupported type', () => {
    // Cast to any to bypass type checking for this specific test case
    const cell = createCell('unknown_type' as any)
    render(<CellFactory cell={cell} {...baseCellProps} />)
    expect(screen.getByText(/Unknown or unsupported cell type: unknown_type/i)).toBeInTheDocument()
    // Check if the cell data is stringified in the output
    expect(screen.getByText(/"id": "cell-unknown_type"/)).toBeInTheDocument();
  })
})
