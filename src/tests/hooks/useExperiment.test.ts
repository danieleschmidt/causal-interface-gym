import { renderHook, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from 'react-query'
import { useExperiment } from '../../hooks/useExperiment'
import { apiClient } from '../../api/client'
import { CausalEnvironment, Intervention } from '../../types'

// Mock the API client
jest.mock('../../api/client', () => ({
  apiClient: {
    runExperiment: jest.fn(),
    getExperimentHistory: jest.fn(),
    getEnvironments: jest.fn(),
    validateEnvironment: jest.fn(),
  },
}))

const mockApiClient = apiClient as jest.Mocked<typeof apiClient>

describe('useExperiment Hook', () => {
  let queryClient: QueryClient

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    })
    jest.clearAllMocks()
  })

  const mockEnvironment: CausalEnvironment = {
    name: 'Test Environment',
    description: 'Test description',
    graph: {
      nodes: [
        { id: 'A', label: 'Node A', type: 'binary', parents: [] },
        { id: 'B', label: 'Node B', type: 'continuous', parents: ['A'] },
      ],
      edges: [
        { source: 'A', target: 'B' },
      ],
    },
    variables: {},
    type: 'custom',
  }

  const mockExperimentConfig = {
    environment: mockEnvironment,
    interventions: { A: { value: true, type: 'set' } },
    agent_config: {
      provider: 'openai',
      model: 'gpt-4',
    },
  }

  test('successfully runs experiment', async () => {
    const mockResult = {
      experiment_id: 'exp_123',
      agent_type: 'gpt-4',
      interventions: [{ variable: 'A', value: true, type: 'set' }],
      initial_beliefs: {},
      intervention_results: [],
      causal_analysis: {
        causal_score: 0.85,
        intervention_vs_observation: {},
      },
    }

    mockApiClient.runExperiment.mockResolvedValueOnce(mockResult)

    const { result } = renderHook(() => useExperiment(), { wrapper })

    act(() => {
      result.current.runExperiment(mockExperimentConfig)
    })

    expect(result.current.isRunning).toBe(true)

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(result.current.isRunning).toBe(false)
    expect(result.current.result).toEqual(mockResult)
    expect(result.current.error).toBeNull()
  })

  test('handles validation errors', async () => {
    const { result } = renderHook(() => useExperiment(), { wrapper })

    const invalidConfig = {
      ...mockExperimentConfig,
      interventions: {}, // Empty interventions should fail validation
    }

    act(() => {
      result.current.runExperiment(invalidConfig)
    })

    expect(result.current.validationErrors).not.toBeNull()
    expect(result.current.validationErrors?.isValid).toBe(false)
  })

  test('validates experiment config', () => {
    const { result } = renderHook(() => useExperiment(), { wrapper })

    const validation = result.current.validateConfig(mockExperimentConfig)
    expect(validation.isValid).toBe(true)
    expect(validation.errors).toHaveLength(0)
  })

  test('handles API errors', async () => {
    const mockError = new Error('API Error')
    mockApiClient.runExperiment.mockRejectedValueOnce(mockError)

    const { result } = renderHook(() => useExperiment(), { wrapper })

    act(() => {
      result.current.runExperiment(mockExperimentConfig)
    })

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(result.current.error).toBeTruthy()
    expect(result.current.isRunning).toBe(false)
  })

  test('calls success callback', async () => {
    const mockResult = {
      experiment_id: 'exp_123',
      agent_type: 'gpt-4',
      interventions: [],
      initial_beliefs: {},
      intervention_results: [],
      causal_analysis: {
        causal_score: 0.85,
        intervention_vs_observation: {},
      },
    }

    mockApiClient.runExperiment.mockResolvedValueOnce(mockResult)

    const onSuccess = jest.fn()
    const { result } = renderHook(() => useExperiment({ onSuccess }), { wrapper })

    act(() => {
      result.current.runExperiment(mockExperimentConfig)
    })

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(onSuccess).toHaveBeenCalledWith(mockResult)
  })

  test('resets state correctly', () => {
    const { result } = renderHook(() => useExperiment(), { wrapper })

    act(() => {
      result.current.runExperiment(mockExperimentConfig)
    })

    act(() => {
      result.current.reset()
    })

    expect(result.current.result).toBeUndefined()
    expect(result.current.error).toBeNull()
    expect(result.current.validationErrors).toBeNull()
  })
})