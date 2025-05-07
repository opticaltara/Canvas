import { render, screen } from '@testing-library/react'
import Header from '../Header'
import '@testing-library/jest-dom'
import { usePathname } from 'next/navigation'

// Mock next/navigation
jest.mock('next/navigation', () => ({
  usePathname: jest.fn(),
}))

describe('Header', () => {
  beforeEach(() => {
    // Reset mocks before each test
    (usePathname as jest.Mock).mockReturnValue('/')
  })

  it('renders the brand name', () => {
    render(<Header />)
    const brandName = screen.getByText(/Sherlog Canvas/i)
    expect(brandName).toBeInTheDocument()
  })

  it('renders navigation links', () => {
    render(<Header />)
    const canvasesLink = screen.getByRole('link', { name: /Canvases/i })
    const dataConnectionsLink = screen.getByRole('link', { name: /Data Connections/i })

    expect(canvasesLink).toBeInTheDocument()
    expect(dataConnectionsLink).toBeInTheDocument()
  })

  it('applies active styles to the Canvases link when on the homepage', () => {
    (usePathname as jest.Mock).mockReturnValue('/')
    render(<Header />)
    const canvasesLink = screen.getByRole('link', { name: /Canvases/i })
    expect(canvasesLink).toHaveClass('border-primary text-primary font-medium')
  })

  it('applies active styles to the Data Connections link when on the connections page', () => {
    (usePathname as jest.Mock).mockReturnValue('/connections')
    render(<Header />)
    const dataConnectionsLink = screen.getByRole('link', { name: /Data Connections/i })
    expect(dataConnectionsLink).toHaveClass('border-primary text-primary font-medium')
  })
})
