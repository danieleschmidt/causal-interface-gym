export interface CausalNode {
  id: string
  label: string
  type: 'binary' | 'continuous' | 'categorical'
  parents: string[]
  position?: { x: number; y: number }
}

export interface CausalEdge {
  source: string
  target: string
  strength?: number
  mechanism?: string
}

export interface CausalGraph {
  nodes: CausalNode[]
  edges: CausalEdge[]
}

export interface Intervention {
  variable: string
  value: any
  type: 'set' | 'randomize' | 'prevent'
}

export interface BeliefMeasurement {
  belief_statement: string
  condition: string
  probability: number
  timestamp: number
}

export interface ExperimentResult {
  experiment_id: string
  agent_type: string
  interventions: Intervention[]
  initial_beliefs: Record<string, number>
  intervention_results: any[]
  causal_analysis: {
    causal_score: number
    intervention_vs_observation: Record<string, any>
  }
}

export interface LLMProvider {
  name: 'openai' | 'anthropic' | 'local'
  model: string
  config: Record<string, any>
}

export interface CausalEnvironment {
  name: string
  description: string
  graph: CausalGraph
  variables: Record<string, any>
  type: 'classic' | 'economic' | 'medical' | 'custom'
}

export interface BenchmarkResult {
  model: string
  overall_score: number
  categories: {
    intervention_understanding: number
    backdoor_identification: number
    frontdoor_adjustment: number
    counterfactual_reasoning: number
  }
  detailed_results: Record<string, any>
}