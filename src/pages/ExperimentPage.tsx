import React, { useState } from 'react'
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  Alert,
  CircularProgress,
} from '@mui/material'
import { CausalGraph } from '../components/CausalGraph'
import { InterventionPanel } from '../components/InterventionPanel'
import { CausalGraph as CausalGraphType, CausalNode, ExperimentResult } from '../types'

// Example causal graph - Simpson's Paradox scenario
const exampleGraph: CausalGraphType = {
  nodes: [
    { id: 'gender', label: 'Gender', type: 'binary', parents: [] },
    { id: 'department', label: 'Department', type: 'categorical', parents: ['gender'] },
    { id: 'qualifications', label: 'Qualifications', type: 'continuous', parents: ['gender'] },
    { id: 'admission', label: 'Admission', type: 'binary', parents: ['department', 'qualifications'] },
  ],
  edges: [
    { source: 'gender', target: 'department' },
    { source: 'gender', target: 'qualifications' },
    { source: 'department', target: 'admission' },
    { source: 'qualifications', target: 'admission' },
  ],
}

export const ExperimentPage: React.FC = () => {
  const [currentGraph] = useState<CausalGraphType>(exampleGraph)
  const [interventions, setInterventions] = useState<Record<string, any>>({})
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<ExperimentResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleNodeClick = (node: CausalNode) => {
    console.log('Node clicked:', node)
  }

  const handleRunExperiment = async () => {
    setIsRunning(true)
    setError(null)
    
    try {
      // Simulate API call to backend
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Mock experiment results
      const mockResults: ExperimentResult = {
        experiment_id: `exp_${Date.now()}`,
        agent_type: 'gpt-4',
        interventions: Object.entries(interventions).map(([variable, config]) => ({
          variable,
          value: config.value,
          type: config.type,
        })),
        initial_beliefs: {
          'P(admission)': 0.45,
          'P(admission|gender=female)': 0.42,
          'P(admission|gender=male)': 0.48,
        },
        intervention_results: [],
        causal_analysis: {
          causal_score: Math.random() * 0.4 + 0.6, // Random score between 0.6-1.0
          intervention_vs_observation: {
            'P(admission|do(gender=female))': {
              score: Math.random() * 0.3 + 0.7,
              intervention_belief: 0.51,
              observational_belief: 0.42,
              difference: 0.09,
            },
          },
        },
      }
      
      setResults(mockResults)
    } catch (err) {
      setError('Failed to run experiment. Please check your configuration.')
      console.error('Experiment error:', err)
    } finally {
      setIsRunning(false)
    }
  }

  const formatScore = (score: number) => (score * 100).toFixed(1)

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Causal Reasoning Experiment
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Design and run experiments to test LLM causal reasoning capabilities. 
        Intervene on variables and measure how agents update their beliefs.
      </Typography>

      <Grid container spacing={4}>
        {/* Left Column - Graph and Results */}
        <Grid item xs={12} lg={8}>
          <Box sx={{ mb: 4 }}>
            <Typography variant="h5" gutterBottom>
              College Admissions Scenario (Simpson's Paradox)
            </Typography>
            <CausalGraph
              graph={currentGraph}
              width={700}
              height={400}
              onNodeClick={handleNodeClick}
              interventions={interventions}
              highlightBackdoors={true}
            />
          </Box>

          {/* Results Section */}
          {isRunning && (
            <Paper sx={{ p: 3, mb: 4 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress size={24} />
                <Typography variant="h6">Running Experiment...</Typography>
              </Box>
              <Typography color="text.secondary" sx={{ mt: 1 }}>
                Testing causal reasoning with {Object.keys(interventions).length} intervention(s)
              </Typography>
            </Paper>
          )}

          {results && (
            <Paper sx={{ p: 3, mb: 4 }}>
              <Typography variant="h6" gutterBottom>
                Experiment Results
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'primary.50', borderRadius: 1 }}>
                    <Typography variant="h4" color="primary.main">
                      {formatScore(results.causal_analysis.causal_score)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Overall Causal Score
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'secondary.50', borderRadius: 1 }}>
                    <Typography variant="h4" color="secondary.main">
                      {results.interventions.length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Interventions Applied
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'success.50', borderRadius: 1 }}>
                    <Typography variant="h4" color="success.main">
                      {results.agent_type}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Agent Model
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              {/* Detailed Analysis */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Intervention vs Observation Analysis
                </Typography>
                {Object.entries(results.causal_analysis.intervention_vs_observation).map(
                  ([belief, data]) => (
                    <Box key={belief} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="body2" fontWeight="bold">
                        {belief}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Score: {formatScore(data.score)}% | 
                        Intervention: {(data.intervention_belief * 100).toFixed(1)}% | 
                        Observation: {(data.observational_belief * 100).toFixed(1)}% | 
                        Difference: {(data.difference * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  )
                )}
              </Box>
            </Paper>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 4 }}>
              {error}
            </Alert>
          )}
        </Grid>

        {/* Right Column - Controls */}
        <Grid item xs={12} lg={4}>
          <InterventionPanel
            nodes={currentGraph.nodes}
            interventions={interventions}
            onInterventionChange={setInterventions}
            onRunExperiment={handleRunExperiment}
            isRunning={isRunning}
          />
        </Grid>
      </Grid>
    </Container>
  )
}