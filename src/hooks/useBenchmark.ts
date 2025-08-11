import { useState, useEffect } from 'react'
import { useQuery, useMutation } from 'react-query'
import { apiClient, APIError } from '../api/client'
import { BenchmarkResult } from '../types'

interface BenchmarkConfig {
  models: string[]
  test_categories: string[]
  options?: Record<string, any>
}

interface BenchmarkStatus {
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  results?: BenchmarkResult[]
  error?: string
}

interface UseBenchmarkOptions {
  onComplete?: (results: BenchmarkResult[]) => void
  onError?: (error: APIError) => void
  pollInterval?: number
}

export const useBenchmark = (options: UseBenchmarkOptions = {}) => {
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<BenchmarkStatus>({
    status: 'pending',
    progress: 0,
  })

  const { pollInterval = 2000 } = options

  // Query for existing benchmark results
  const {
    data: existingResults,
    error: existingResultsError,
    refetch: refetchResults,
  } = useQuery<BenchmarkResult[], APIError>(
    'benchmarkResults',
    () => apiClient.getBenchmarkResults(),
    {
      staleTime: 5 * 60 * 1000,
      cacheTime: 10 * 60 * 1000,
      refetchOnWindowFocus: false,
    }
  )

  // Query for benchmark status (only when job is running)
  const { data: jobStatus } = useQuery<BenchmarkStatus, APIError>(
    ['benchmarkStatus', jobId],
    () => jobId ? apiClient.getBenchmarkStatus(jobId) : Promise.reject('No job ID'),
    {
      enabled: !!jobId,
      refetchInterval: pollInterval,
      refetchOnWindowFocus: false,
      retry: false,
    }
  )

  // Mutation to start new benchmark
  const {
    mutate: startBenchmark,
    isLoading: isStarting,
    error: startError,
  } = useMutation<{ job_id: string }, APIError, BenchmarkConfig>(
    'startBenchmark',
    (config) => apiClient.runBenchmark(config),
    {
      onSuccess: (data) => {
        setJobId(data.job_id)
        setStatus({ status: 'running', progress: 0 })
      },
      onError: (error) => {
        setStatus({ status: 'failed', progress: 0, error: error.message })
        options.onError?.(error)
      },
    }
  )

  // Update status when job status changes
  useEffect(() => {
    if (jobStatus) {
      setStatus(jobStatus)
      
      if (jobStatus.status === 'completed' && jobStatus.results) {
        options.onComplete?.(jobStatus.results)
        setJobId(null) // Clear job ID to stop polling
        refetchResults() // Refresh existing results
      } else if (jobStatus.status === 'failed') {
        setJobId(null) // Clear job ID to stop polling
      }
    }
  }, [jobStatus, options, refetchResults])

  const cancelBenchmark = () => {
    setJobId(null)
    setStatus({ status: 'pending', progress: 0 })
  }

  const isRunning = status.status === 'running' || isStarting

  return {
    // Current benchmark
    startBenchmark,
    cancelBenchmark,
    isRunning,
    isStarting,
    status,
    error: startError,
    
    // Existing results
    existingResults,
    existingResultsError,
    refetchResults,
  }
}

export const useBenchmarkComparison = (results: BenchmarkResult[]) => {
  const [sortBy, setSortBy] = useState<keyof BenchmarkResult['categories'] | 'overall_score'>('overall_score')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')

  const sortedResults = [...results].sort((a, b) => {
    let aValue: number
    let bValue: number

    if (sortBy === 'overall_score') {
      aValue = a.overall_score
      bValue = b.overall_score
    } else {
      aValue = a.categories[sortBy]
      bValue = b.categories[sortBy]
    }

    return sortOrder === 'desc' ? bValue - aValue : aValue - bValue
  })

  const toggleSort = (column: keyof BenchmarkResult['categories'] | 'overall_score') => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')
    } else {
      setSortBy(column)
      setSortOrder('desc')
    }
  }

  const getTopPerformer = (category?: keyof BenchmarkResult['categories']) => {
    if (!results.length) return null

    if (category) {
      return results.reduce((best, current) => 
        current.categories[category] > best.categories[category] ? current : best
      )
    }

    return results.reduce((best, current) => 
      current.overall_score > best.overall_score ? current : best
    )
  }

  const getCategoryStats = (category: keyof BenchmarkResult['categories']) => {
    if (!results.length) return { min: 0, max: 0, avg: 0 }

    const scores = results.map(r => r.categories[category])
    return {
      min: Math.min(...scores),
      max: Math.max(...scores),
      avg: scores.reduce((sum, score) => sum + score, 0) / scores.length,
    }
  }

  return {
    sortedResults,
    sortBy,
    sortOrder,
    toggleSort,
    getTopPerformer,
    getCategoryStats,
  }
}