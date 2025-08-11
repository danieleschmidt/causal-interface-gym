import {
  validateCausalGraph,
  validateIntervention,
  validateExperimentConfig,
  sanitizeInput,
  ValidationResult,
} from '../../utils/validation'
import { CausalGraph, CausalNode, Intervention } from '../../types'

describe('Validation Utilities', () => {
  describe('validateCausalGraph', () => {
    test('validates correct graph', () => {
      const validGraph: CausalGraph = {
        nodes: [
          { id: 'A', label: 'Node A', type: 'binary', parents: [] },
          { id: 'B', label: 'Node B', type: 'continuous', parents: ['A'] },
        ],
        edges: [
          { source: 'A', target: 'B' },
        ],
      }

      const result = validateCausalGraph(validGraph)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    test('rejects empty nodes', () => {
      const invalidGraph: CausalGraph = {
        nodes: [],
        edges: [],
      }

      const result = validateCausalGraph(invalidGraph)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'nodes',
          code: 'EMPTY_NODES',
        })
      )
    })

    test('rejects duplicate node IDs', () => {
      const invalidGraph: CausalGraph = {
        nodes: [
          { id: 'A', label: 'Node A', type: 'binary', parents: [] },
          { id: 'A', label: 'Node A Duplicate', type: 'binary', parents: [] },
        ],
        edges: [],
      }

      const result = validateCausalGraph(invalidGraph)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'nodes',
          code: 'DUPLICATE_NODE_ID',
        })
      )
    })

    test('rejects invalid node types', () => {
      const invalidGraph: CausalGraph = {
        nodes: [
          { id: 'A', label: 'Node A', type: 'invalid' as any, parents: [] },
        ],
        edges: [],
      }

      const result = validateCausalGraph(invalidGraph)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'nodes',
          code: 'INVALID_NODE_TYPE',
        })
      )
    })

    test('rejects edges with non-existent nodes', () => {
      const invalidGraph: CausalGraph = {
        nodes: [
          { id: 'A', label: 'Node A', type: 'binary', parents: [] },
        ],
        edges: [
          { source: 'A', target: 'NonExistent' },
        ],
      }

      const result = validateCausalGraph(invalidGraph)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'edges',
          code: 'INVALID_EDGE_TARGET',
        })
      )
    })

    test('rejects self-loops', () => {
      const invalidGraph: CausalGraph = {
        nodes: [
          { id: 'A', label: 'Node A', type: 'binary', parents: [] },
        ],
        edges: [
          { source: 'A', target: 'A' },
        ],
      }

      const result = validateCausalGraph(invalidGraph)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'edges',
          code: 'SELF_LOOP',
        })
      )
    })

    test('detects cycles', () => {
      const cyclicGraph: CausalGraph = {
        nodes: [
          { id: 'A', label: 'Node A', type: 'binary', parents: [] },
          { id: 'B', label: 'Node B', type: 'binary', parents: [] },
          { id: 'C', label: 'Node C', type: 'binary', parents: [] },
        ],
        edges: [
          { source: 'A', target: 'B' },
          { source: 'B', target: 'C' },
          { source: 'C', target: 'A' }, // Creates cycle
        ],
      }

      const result = validateCausalGraph(cyclicGraph)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'graph',
          code: 'GRAPH_CYCLES',
        })
      )
    })
  })

  describe('validateIntervention', () => {
    const nodes: CausalNode[] = [
      { id: 'binary', label: 'Binary Node', type: 'binary', parents: [] },
      { id: 'continuous', label: 'Continuous Node', type: 'continuous', parents: [] },
      { id: 'categorical', label: 'Categorical Node', type: 'categorical', parents: [] },
    ]

    test('validates correct binary intervention', () => {
      const intervention: Intervention = {
        variable: 'binary',
        value: true,
        type: 'set',
      }

      const result = validateIntervention(intervention, nodes)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    test('validates correct continuous intervention', () => {
      const intervention: Intervention = {
        variable: 'continuous',
        value: 3.14,
        type: 'set',
      }

      const result = validateIntervention(intervention, nodes)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    test('validates correct categorical intervention', () => {
      const intervention: Intervention = {
        variable: 'categorical',
        value: 'category1',
        type: 'set',
      }

      const result = validateIntervention(intervention, nodes)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    test('rejects intervention on non-existent variable', () => {
      const intervention: Intervention = {
        variable: 'nonexistent',
        value: true,
        type: 'set',
      }

      const result = validateIntervention(intervention, nodes)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'variable',
          code: 'VARIABLE_NOT_FOUND',
        })
      )
    })

    test('rejects invalid intervention type', () => {
      const intervention: Intervention = {
        variable: 'binary',
        value: true,
        type: 'invalid' as any,
      }

      const result = validateIntervention(intervention, nodes)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'type',
          code: 'INVALID_INTERVENTION_TYPE',
        })
      )
    })

    test('rejects invalid binary value', () => {
      const intervention: Intervention = {
        variable: 'binary',
        value: 'not boolean',
        type: 'set',
      }

      const result = validateIntervention(intervention, nodes)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'value',
          code: 'INVALID_BINARY_VALUE',
        })
      )
    })

    test('rejects invalid continuous value', () => {
      const intervention: Intervention = {
        variable: 'continuous',
        value: 'not number',
        type: 'set',
      }

      const result = validateIntervention(intervention, nodes)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'value',
          code: 'INVALID_CONTINUOUS_VALUE',
        })
      )
    })

    test('accepts randomize intervention without value validation', () => {
      const intervention: Intervention = {
        variable: 'binary',
        value: undefined,
        type: 'randomize',
      }

      const result = validateIntervention(intervention, nodes)
      expect(result.isValid).toBe(true)
    })
  })

  describe('sanitizeInput', () => {
    test('removes dangerous HTML characters', () => {
      const dangerous = '<script>alert("xss")</script>'
      const sanitized = sanitizeInput(dangerous)
      expect(sanitized).toBe('&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;')
    })

    test('trims whitespace', () => {
      const input = '  hello world  '
      const sanitized = sanitizeInput(input)
      expect(sanitized).toBe('hello world')
    })

    test('limits length', () => {
      const longInput = 'a'.repeat(2000)
      const sanitized = sanitizeInput(longInput)
      expect(sanitized).toHaveLength(1000)
    })

    test('handles non-string input', () => {
      const sanitized = sanitizeInput(null as any)
      expect(sanitized).toBe('')
    })
  })

  describe('validateExperimentConfig', () => {
    const validConfig = {
      environment: {
        graph: {
          nodes: [
            { id: 'A', label: 'Node A', type: 'binary' as const, parents: [] },
          ],
          edges: [],
        },
      },
      interventions: [
        { variable: 'A', value: true, type: 'set' as const },
      ],
      agent_config: {
        provider: 'openai',
        model: 'gpt-4',
      },
    }

    test('validates correct config', () => {
      const result = validateExperimentConfig(validConfig)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    test('rejects missing environment', () => {
      const invalidConfig = {
        ...validConfig,
        environment: undefined,
      }

      const result = validateExperimentConfig(invalidConfig)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'environment',
          code: 'MISSING_ENVIRONMENT',
        })
      )
    })

    test('rejects empty interventions', () => {
      const invalidConfig = {
        ...validConfig,
        interventions: [],
      }

      const result = validateExperimentConfig(invalidConfig)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'interventions',
          code: 'NO_INTERVENTIONS',
        })
      )
    })

    test('rejects missing agent config', () => {
      const invalidConfig = {
        ...validConfig,
        agent_config: undefined,
      }

      const result = validateExperimentConfig(invalidConfig)
      expect(result.isValid).toBe(false)
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          field: 'agent_config',
          code: 'MISSING_AGENT_CONFIG',
        })
      )
    })
  })
})