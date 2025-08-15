import { CausalGraph, CausalNode, CausalEdge, Intervention } from '../types'

export interface ValidationError {
  field: string
  message: string
  code: string
}

export class ValidationResult {
  constructor(
    public isValid: boolean,
    public errors: ValidationError[] = []
  ) {}

  static success(): ValidationResult {
    return new ValidationResult(true, [])
  }

  static failure(errors: ValidationError[]): ValidationResult {
    return new ValidationResult(false, errors)
  }

  addError(field: string, message: string, code: string): void {
    this.errors.push({ field, message, code })
    this.isValid = false
  }
}

export const validateCausalGraph = (graph: CausalGraph): ValidationResult => {
  const result = new ValidationResult(true)

  // Validate nodes
  if (!graph.nodes || graph.nodes.length === 0) {
    result.addError('nodes', 'Graph must have at least one node', 'EMPTY_NODES')
    return result
  }

  // Check for unique node IDs
  const nodeIds = new Set<string>()
  for (const node of graph.nodes) {
    if (!node.id || typeof node.id !== 'string' || node.id.trim() === '') {
      result.addError('nodes', 'All nodes must have valid non-empty string IDs', 'INVALID_NODE_ID')
      continue
    }

    if (nodeIds.has(node.id)) {
      result.addError('nodes', `Duplicate node ID: ${node.id}`, 'DUPLICATE_NODE_ID')
    }
    nodeIds.add(node.id)

    // Validate node label
    if (!node.label || typeof node.label !== 'string' || node.label.trim() === '') {
      result.addError('nodes', `Node ${node.id} must have a valid label`, 'INVALID_NODE_LABEL')
    }

    // Validate node type
    if (!['binary', 'continuous', 'categorical'].includes(node.type)) {
      result.addError('nodes', `Node ${node.id} has invalid type: ${node.type}`, 'INVALID_NODE_TYPE')
    }
  }

  // Validate edges
  if (!graph.edges) {
    result.addError('edges', 'Graph must have edges array (can be empty)', 'MISSING_EDGES')
    return result
  }

  for (const edge of graph.edges) {
    // Check source and target exist
    if (!nodeIds.has(edge.source)) {
      result.addError('edges', `Edge source node not found: ${edge.source}`, 'INVALID_EDGE_SOURCE')
    }
    if (!nodeIds.has(edge.target)) {
      result.addError('edges', `Edge target node not found: ${edge.target}`, 'INVALID_EDGE_TARGET')
    }

    // Check for self-loops
    if (edge.source === edge.target) {
      result.addError('edges', `Self-loop not allowed: ${edge.source}`, 'SELF_LOOP')
    }
  }

  // Check for cycles (DAG validation)
  const cycleResult = detectCycles(graph)
  if (!cycleResult.isAcyclic) {
    result.addError('graph', `Graph contains cycles: ${cycleResult.cycles.join(', ')}`, 'GRAPH_CYCLES')
  }

  return result
}

export const validateIntervention = (intervention: Intervention, nodes: CausalNode[]): ValidationResult => {
  const result = new ValidationResult(true)

  // Check variable exists
  const node = nodes.find(n => n.id === intervention.variable)
  if (!node) {
    result.addError('variable', `Variable not found: ${intervention.variable}`, 'VARIABLE_NOT_FOUND')
    return result
  }

  // Validate intervention type
  if (!['set', 'randomize', 'prevent'].includes(intervention.type)) {
    result.addError('type', `Invalid intervention type: ${intervention.type}`, 'INVALID_INTERVENTION_TYPE')
  }

  // Validate value based on variable type and intervention type
  if (intervention.type === 'set') {
    switch (node.type) {
      case 'binary':
        if (typeof intervention.value !== 'boolean' && intervention.value !== 0 && intervention.value !== 1) {
          result.addError('value', 'Binary variable requires boolean or 0/1 value', 'INVALID_BINARY_VALUE')
        }
        break
      case 'continuous':
        if (typeof intervention.value !== 'number' || isNaN(intervention.value)) {
          result.addError('value', 'Continuous variable requires numeric value', 'INVALID_CONTINUOUS_VALUE')
        }
        break
      case 'categorical':
        if (typeof intervention.value !== 'string' || intervention.value.trim() === '') {
          result.addError('value', 'Categorical variable requires non-empty string value', 'INVALID_CATEGORICAL_VALUE')
        }
        break
    }
  }

  return result
}

export const validateExperimentConfig = (config: {
  environment?: any
  interventions?: Intervention[]
  agent_config?: any
}): ValidationResult => {
  const result = new ValidationResult(true)

  // Validate environment
  if (!config.environment) {
    result.addError('environment', 'Environment is required', 'MISSING_ENVIRONMENT')
  } else if (config.environment.graph) {
    const graphValidation = validateCausalGraph(config.environment.graph)
    if (!graphValidation.isValid) {
      graphValidation.errors.forEach(error => {
        result.addError(`environment.${error.field}`, error.message, error.code)
      })
    }
  }

  // Validate interventions
  if (!config.interventions || !Array.isArray(config.interventions)) {
    result.addError('interventions', 'Interventions must be an array', 'INVALID_INTERVENTIONS')
  } else if (config.interventions.length === 0) {
    result.addError('interventions', 'At least one intervention is required', 'NO_INTERVENTIONS')
  } else if (config.environment?.graph?.nodes) {
    config.interventions.forEach((intervention, index) => {
      const interventionValidation = validateIntervention(intervention, config.environment.graph.nodes)
      if (!interventionValidation.isValid) {
        interventionValidation.errors.forEach(error => {
          result.addError(`interventions[${index}].${error.field}`, error.message, error.code)
        })
      }
    })
  }

  // Validate agent config
  if (!config.agent_config) {
    result.addError('agent_config', 'Agent configuration is required', 'MISSING_AGENT_CONFIG')
  } else {
    if (!config.agent_config.provider) {
      result.addError('agent_config.provider', 'Agent provider is required', 'MISSING_AGENT_PROVIDER')
    }
    if (!config.agent_config.model) {
      result.addError('agent_config.model', 'Agent model is required', 'MISSING_AGENT_MODEL')
    }
  }

  return result
}

interface CycleDetectionResult {
  isAcyclic: boolean
  cycles: string[]
}

const detectCycles = (graph: CausalGraph): CycleDetectionResult => {
  const visited = new Set<string>()
  const recursionStack = new Set<string>()
  const cycles: string[] = []

  const dfs = (nodeId: string, path: string[]): boolean => {
    if (recursionStack.has(nodeId)) {
      // Found a cycle
      const cycleStart = path.indexOf(nodeId)
      const cycle = [...path.slice(cycleStart), nodeId].join(' → ')
      cycles.push(cycle)
      return false
    }

    if (visited.has(nodeId)) {
      return true
    }

    visited.add(nodeId)
    recursionStack.add(nodeId)

    // Get all outgoing edges from this node
    const outgoingEdges = graph.edges.filter(edge => edge.source === nodeId)
    
    for (const edge of outgoingEdges) {
      if (!dfs(edge.target, [...path, nodeId])) {
        return false
      }
    }

    recursionStack.delete(nodeId)
    return true
  }

  // Check all nodes
  for (const node of graph.nodes) {
    if (!visited.has(node.id)) {
      if (!dfs(node.id, [])) {
        return { isAcyclic: false, cycles }
      }
    }
  }

  return { isAcyclic: true, cycles: [] }
}

export const sanitizeInput = (input: string): string => {
  if (typeof input !== 'string') return ''
  
  return input
    .trim()
    .replace(/[<>\"'&]/g, (char) => {
      const map: Record<string, string> = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '&': '&amp;',
      }
      return map[char] || char
    })
    .slice(0, 1000) // Limit length
}

export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

export const validateURL = (url: string): boolean => {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}