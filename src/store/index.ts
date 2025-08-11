import { create } from 'zustand'
import { subscribeWithSelector, devtools, persist } from 'zustand/middleware'
import { CausalEnvironment, ExperimentResult, BenchmarkResult } from '../types'

interface AppState {
  // UI State
  theme: 'light' | 'dark'
  sidebarOpen: boolean
  
  // Experiment State
  currentEnvironment: CausalEnvironment | null
  interventions: Record<string, any>
  experimentHistory: ExperimentResult[]
  isExperimentRunning: boolean
  
  // Benchmark State
  benchmarkResults: BenchmarkResult[]
  isBenchmarkRunning: boolean
  benchmarkProgress: number
  
  // Cache State
  cachedEnvironments: CausalEnvironment[]
  lastFetchTime: number
  
  // Actions
  setTheme: (theme: 'light' | 'dark') => void
  toggleSidebar: () => void
  setCurrentEnvironment: (env: CausalEnvironment | null) => void
  setInterventions: (interventions: Record<string, any>) => void
  addExperimentResult: (result: ExperimentResult) => void
  setExperimentRunning: (running: boolean) => void
  setBenchmarkResults: (results: BenchmarkResult[]) => void
  setBenchmarkRunning: (running: boolean) => void
  setBenchmarkProgress: (progress: number) => void
  setCachedEnvironments: (environments: CausalEnvironment[]) => void
  clearCache: () => void
  reset: () => void
}

const initialState = {
  theme: 'light' as const,
  sidebarOpen: true,
  currentEnvironment: null,
  interventions: {},
  experimentHistory: [],
  isExperimentRunning: false,
  benchmarkResults: [],
  isBenchmarkRunning: false,
  benchmarkProgress: 0,
  cachedEnvironments: [],
  lastFetchTime: 0,
}

export const useAppStore = create<AppState>()(
  devtools(
    subscribeWithSelector(
      persist(
        (set, get) => ({
          ...initialState,

          setTheme: (theme) => {
            set({ theme }, false, 'setTheme')
          },

          toggleSidebar: () => {
            set((state) => ({ sidebarOpen: !state.sidebarOpen }), false, 'toggleSidebar')
          },

          setCurrentEnvironment: (environment) => {
            set({ currentEnvironment: environment }, false, 'setCurrentEnvironment')
          },

          setInterventions: (interventions) => {
            set({ interventions }, false, 'setInterventions')
          },

          addExperimentResult: (result) => {
            set((state) => ({
              experimentHistory: [result, ...state.experimentHistory].slice(0, 100) // Keep last 100
            }), false, 'addExperimentResult')
          },

          setExperimentRunning: (running) => {
            set({ isExperimentRunning: running }, false, 'setExperimentRunning')
          },

          setBenchmarkResults: (results) => {
            set({ benchmarkResults: results }, false, 'setBenchmarkResults')
          },

          setBenchmarkRunning: (running) => {
            set({ isBenchmarkRunning: running }, false, 'setBenchmarkRunning')
          },

          setBenchmarkProgress: (progress) => {
            set({ benchmarkProgress: progress }, false, 'setBenchmarkProgress')
          },

          setCachedEnvironments: (environments) => {
            set({ 
              cachedEnvironments: environments,
              lastFetchTime: Date.now()
            }, false, 'setCachedEnvironments')
          },

          clearCache: () => {
            set({ 
              cachedEnvironments: [],
              lastFetchTime: 0
            }, false, 'clearCache')
          },

          reset: () => {
            set(initialState, false, 'reset')
          },
        }),
        {
          name: 'causal-interface-gym-storage',
          partialize: (state) => ({
            theme: state.theme,
            sidebarOpen: state.sidebarOpen,
            experimentHistory: state.experimentHistory.slice(0, 10), // Persist only last 10 experiments
            benchmarkResults: state.benchmarkResults,
          }),
        }
      )
    ),
    {
      name: 'causal-interface-gym',
    }
  )
)

// Selectors for optimized re-renders
export const useTheme = () => useAppStore((state) => state.theme)
export const useExperimentState = () => useAppStore((state) => ({
  currentEnvironment: state.currentEnvironment,
  interventions: state.interventions,
  isRunning: state.isExperimentRunning,
}))
export const useBenchmarkState = () => useAppStore((state) => ({
  results: state.benchmarkResults,
  isRunning: state.isBenchmarkRunning,
  progress: state.benchmarkProgress,
}))

// Performance-optimized hooks with shallow comparison
export const useOptimizedExperimentHistory = () => 
  useAppStore((state) => state.experimentHistory, (a, b) => a.length === b.length)

export const useOptimizedBenchmarkResults = () =>
  useAppStore((state) => state.benchmarkResults, (a, b) => a.length === b.length)