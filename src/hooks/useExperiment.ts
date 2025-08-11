import { useState, useCallback } from 'react'
import { useMutation, useQuery } from 'react-query'
import { apiClient, APIError } from '../api/client'
import { ExperimentResult, CausalEnvironment, Intervention } from '../types'
import { validateExperimentConfig, ValidationResult } from '../utils/validation'

interface UseExperimentOptions {
  onSuccess?: (result: ExperimentResult) => void
  onError?: (error: APIError) => void
}

interface ExperimentConfig {
  environment: CausalEnvironment
  interventions: Record<string, any>
  agent_config: {
    provider: string
    model: string
    config?: Record<string, any>
  }
  options?: Record<string, any>
}

export const useExperiment = (options: UseExperimentOptions = {}) => {
  const [validationErrors, setValidationErrors] = useState<ValidationResult | null>(null)

  const {
    mutate: runExperiment,
    isLoading: isRunning,
    error,
    data: result,
    reset,
  } = useMutation<ExperimentResult, APIError, ExperimentConfig>(
    'runExperiment',
    async (config) => {
      // Client-side validation
      const interventionArray: Intervention[] = Object.entries(config.interventions).map(
        ([variable, interventionConfig]) => ({
          variable,
          value: interventionConfig.value,
          type: interventionConfig.type,
        })
      )

      const validation = validateExperimentConfig({
        environment: config.environment,
        interventions: interventionArray,
        agent_config: config.agent_config,
      })

      if (!validation.isValid) {
        setValidationErrors(validation)
        throw new APIError('Validation failed', 400, 'VALIDATION_ERROR', validation.errors)
      }

      setValidationErrors(null)
      return apiClient.runExperiment({
        ...config,
        interventions: config.interventions,
      })
    },
    {
      onSuccess: (data) => {
        options.onSuccess?.(data)
      },
      onError: (error) => {
        options.onError?.(error)
      },
    }
  )

  const validateConfig = useCallback((config: Partial<ExperimentConfig>): ValidationResult => {
    const interventionArray: Intervention[] = config.interventions
      ? Object.entries(config.interventions).map(([variable, interventionConfig]) => ({
          variable,
          value: interventionConfig.value,
          type: interventionConfig.type,
        }))
      : []

    return validateExperimentConfig({
      environment: config.environment,
      interventions: interventionArray,
      agent_config: config.agent_config,
    })
  }, [])

  return {
    runExperiment,
    isRunning,
    error: error as APIError | null,
    result,
    validationErrors,
    validateConfig,
    reset,
  }
}

export const useExperimentHistory = () => {
  return useQuery<ExperimentResult[], APIError>(
    'experimentHistory',
    () => apiClient.getExperimentHistory(),
    {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        if (error.status === 401 || error.status === 403) {
          return false
        }
        return failureCount < 3
      },
    }
  )
}

export const useEnvironments = () => {
  return useQuery<CausalEnvironment[], APIError>(
    'environments',
    () => apiClient.getEnvironments(),
    {
      staleTime: 30 * 60 * 1000, // 30 minutes
      cacheTime: 60 * 60 * 1000, // 1 hour
      refetchOnWindowFocus: false,
    }
  )
}

export const useEnvironmentValidation = () => {
  const { mutate: validateEnvironment, isLoading, data, error } = useMutation<
    { valid: boolean; errors?: string[] },
    APIError,
    CausalEnvironment
  >('validateEnvironment', (environment) => apiClient.validateEnvironment(environment))

  return {
    validateEnvironment,
    isValidating: isLoading,
    validationResult: data,
    validationError: error,
  }
}