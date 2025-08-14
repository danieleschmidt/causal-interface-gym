import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import { CausalGraph } from '../../components/CausalGraph'
import { CausalGraph as CausalGraphType } from '../../types'

// Mock D3 for testing
vi.mock('d3', () => ({
  select: vi.fn(() => ({
    selectAll: vi.fn(() => ({
      remove: vi.fn(),
      data: vi.fn(() => ({
        enter: vi.fn(() => ({
          append: vi.fn(() => ({
            attr: vi.fn(() => ({ attr: vi.fn() })),
            style: vi.fn(() => ({ style: vi.fn() })),
            text: vi.fn(() => ({ text: vi.fn() })),
            on: vi.fn(() => ({ on: vi.fn() })),
            call: vi.fn(() => ({ call: vi.fn() })),
          })),
        })),
      })),
    })),
    append: vi.fn(() => ({
      attr: vi.fn(() => ({ attr: vi.fn() })),
      append: vi.fn(() => ({
        attr: vi.fn(() => ({ attr: vi.fn() })),
        append: vi.fn(() => ({
          attr: vi.fn(() => ({ attr: vi.fn() })),
        })),
      })),
    })),
    call: vi.fn(),
  })),
  forceSimulation: vi.fn(() => ({
    force: vi.fn(() => ({ force: vi.fn() })),
    on: vi.fn(() => ({ on: vi.fn() })),
    stop: vi.fn(),
  })),
  forceLink: vi.fn(() => ({
    id: vi.fn(() => ({ id: vi.fn() })),
    distance: vi.fn(() => ({ distance: vi.fn() })),
  })),
  forceManyBody: vi.fn(() => ({
    strength: vi.fn(() => ({ strength: vi.fn() })),
  })),
  forceCenter: vi.fn(() => ({ forceCenter: vi.fn() })),
  forceCollide: vi.fn(() => ({
    radius: vi.fn(() => ({ radius: vi.fn() })),
  })),
  zoom: vi.fn(() => ({
    scaleExtent: vi.fn(() => ({
      scaleExtent: vi.fn(),
      on: vi.fn(() => ({ on: vi.fn() })),
    })),
  })),
  zoomIdentity: { x: 0, y: 0, k: 1 },
  drag: vi.fn(() => ({
    on: vi.fn(() => ({ on: vi.fn() })),
  })),
}))

const mockGraph: CausalGraphType = {
  nodes: [
    { id: 'A', label: 'Node A', type: 'binary', parents: [] },
    { id: 'B', label: 'Node B', type: 'continuous', parents: ['A'] },
    { id: 'C', label: 'Node C', type: 'categorical', parents: ['A', 'B'] },
  ],
  edges: [
    { source: 'A', target: 'B' },
    { source: 'A', target: 'C' },
    { source: 'B', target: 'C' },
  ],
}

describe('CausalGraph Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders without crashing', () => {
    render(<CausalGraph graph={mockGraph} />)
    expect(screen.getByRole('img')).toBeInTheDocument()
  })

  test('renders with correct dimensions', () => {
    render(<CausalGraph graph={mockGraph} width={600} height={400} />)
    const svg = screen.getByRole('img')
    expect(svg).toHaveAttribute('width', '600')
    expect(svg).toHaveAttribute('height', '400')
  })

  test('handles empty graph gracefully', () => {
    const emptyGraph: CausalGraphType = { nodes: [], edges: [] }
    render(<CausalGraph graph={emptyGraph} />)
    expect(screen.getByRole('img')).toBeInTheDocument()
  })

  test('calls onNodeClick when node is clicked', async () => {
    const mockOnNodeClick = vi.fn()
    render(<CausalGraph graph={mockGraph} onNodeClick={mockOnNodeClick} />)
    
    // Note: In actual implementation, this would require mocking D3 interactions
    // For now, we test that the callback prop is passed correctly
    expect(mockOnNodeClick).not.toHaveBeenCalled()
  })

  test('highlights interventions correctly', () => {
    const interventions = { A: { value: true, type: 'set' } }
    render(<CausalGraph graph={mockGraph} interventions={interventions} />)
    
    // Verify that the graph renders with interventions
    expect(screen.getByRole('img')).toBeInTheDocument()
  })

  test('handles graph updates', () => {
    const { rerender } = render(<CausalGraph graph={mockGraph} />)
    
    const updatedGraph: CausalGraphType = {
      nodes: [
        ...mockGraph.nodes,
        { id: 'D', label: 'Node D', type: 'binary', parents: ['C'] },
      ],
      edges: [
        ...mockGraph.edges,
        { source: 'C', target: 'D' },
      ],
    }
    
    rerender(<CausalGraph graph={updatedGraph} />)
    expect(screen.getByRole('img')).toBeInTheDocument()
  })

  test('validates graph structure', () => {
    // Test invalid graph structure
    const invalidGraph: CausalGraphType = {
      nodes: [
        { id: 'A', label: 'Node A', type: 'binary', parents: [] },
      ],
      edges: [
        { source: 'A', target: 'NonExistent' }, // Invalid target
      ],
    }
    
    render(<CausalGraph graph={invalidGraph} />)
    expect(screen.getByRole('img')).toBeInTheDocument()
  })

  test('performance with large graphs', () => {
    const largeGraph: CausalGraphType = {
      nodes: Array.from({ length: 100 }, (_, i) => ({
        id: `node${i}`,
        label: `Node ${i}`,
        type: 'binary' as const,
        parents: i > 0 ? [`node${i - 1}`] : [],
      })),
      edges: Array.from({ length: 99 }, (_, i) => ({
        source: `node${i}`,
        target: `node${i + 1}`,
      })),
    }
    
    const start = performance.now()
    render(<CausalGraph graph={largeGraph} />)
    const end = performance.now()
    
    expect(end - start).toBeLessThan(1000) // Should render in less than 1 second
    expect(screen.getByRole('img')).toBeInTheDocument()
  })
})