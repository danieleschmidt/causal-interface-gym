import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios'
import { CausalEnvironment, ExperimentResult, BenchmarkResult } from '../types'

class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string,
    public details?: any
  ) {
    super(message)
    this.name = 'APIError'
  }
}

class APIClient {
  private client: AxiosInstance

  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token')
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        
        // Add request ID for tracing
        config.headers['X-Request-ID'] = crypto.randomUUID()
        
        return config
      },
      (error) => Promise.reject(error)
    )

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          // Server responded with error status
          const { status, data } = error.response
          throw new APIError(
            data?.message || 'An error occurred',
            status,
            data?.error_code,
            data?.details
          )
        } else if (error.request) {
          // Network error
          throw new APIError('Network error - please check your connection')
        } else {
          // Request setup error
          throw new APIError('Request configuration error')
        }
      }
    )
  }

  async runExperiment(config: {
    environment: CausalEnvironment
    interventions: Record<string, any>
    agent_config: any
    options?: any
  }): Promise<ExperimentResult> {
    try {
      const response = await this.client.post('/experiments/run', config)
      return response.data
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      throw new APIError('Failed to run experiment')
    }
  }

  async getBenchmarkResults(): Promise<BenchmarkResult[]> {
    try {
      const response = await this.client.get('/benchmarks/results')
      return response.data
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      throw new APIError('Failed to fetch benchmark results')
    }
  }

  async runBenchmark(config: {
    models: string[]
    test_categories: string[]
    options?: any
  }): Promise<{ job_id: string }> {
    try {
      const response = await this.client.post('/benchmarks/run', config)
      return response.data
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      throw new APIError('Failed to start benchmark')
    }
  }

  async getBenchmarkStatus(jobId: string): Promise<{ status: string; progress: number; results?: BenchmarkResult[] }> {
    try {
      const response = await this.client.get(`/benchmarks/status/${jobId}`)
      return response.data
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      throw new APIError('Failed to fetch benchmark status')
    }
  }

  async getEnvironments(): Promise<CausalEnvironment[]> {
    try {
      const response = await this.client.get('/environments')
      return response.data
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      throw new APIError('Failed to fetch environments')
    }
  }

  async validateEnvironment(environment: CausalEnvironment): Promise<{ valid: boolean; errors?: string[] }> {
    try {
      const response = await this.client.post('/environments/validate', environment)
      return response.data
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      throw new APIError('Failed to validate environment')
    }
  }

  async getExperimentHistory(limit = 50): Promise<ExperimentResult[]> {
    try {
      const response = await this.client.get(`/experiments/history?limit=${limit}`)
      return response.data
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      throw new APIError('Failed to fetch experiment history')
    }
  }

  async exportResults(experimentId: string, format = 'json'): Promise<string> {
    try {
      const response = await this.client.get(`/experiments/${experimentId}/export?format=${format}`)
      return response.data
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      throw new APIError('Failed to export results')
    }
  }
}

// Create singleton instance
export const apiClient = new APIClient()
export { APIError }