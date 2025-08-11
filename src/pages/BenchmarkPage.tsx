import React, { useState } from 'react'
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Chip,
  Alert,
} from '@mui/material'
import { Assessment, PlayArrow } from '@mui/icons-material'
import Plot from 'react-plotly.js'
import { BenchmarkResult } from '../types'

const mockBenchmarkResults: BenchmarkResult[] = [
  {
    model: 'gpt-4',
    overall_score: 0.82,
    categories: {
      intervention_understanding: 0.89,
      backdoor_identification: 0.78,
      frontdoor_adjustment: 0.75,
      counterfactual_reasoning: 0.86,
    },
    detailed_results: {},
  },
  {
    model: 'claude-3-opus',
    overall_score: 0.79,
    categories: {
      intervention_understanding: 0.85,
      backdoor_identification: 0.82,
      frontdoor_adjustment: 0.71,
      counterfactual_reasoning: 0.78,
    },
    detailed_results: {},
  },
  {
    model: 'gpt-3.5-turbo',
    overall_score: 0.67,
    categories: {
      intervention_understanding: 0.72,
      backdoor_identification: 0.64,
      frontdoor_adjustment: 0.58,
      counterfactual_reasoning: 0.71,
    },
    detailed_results: {},
  },
  {
    model: 'llama-3-70b',
    overall_score: 0.61,
    categories: {
      intervention_understanding: 0.68,
      backdoor_identification: 0.59,
      frontdoor_adjustment: 0.52,
      counterfactual_reasoning: 0.65,
    },
    detailed_results: {},
  },
]

export const BenchmarkPage: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<BenchmarkResult[]>(mockBenchmarkResults)
  const [progress, setProgress] = useState(0)

  const runBenchmark = async () => {
    setIsRunning(true)
    setProgress(0)
    
    // Simulate benchmark progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsRunning(false)
          return 100
        }
        return prev + 10
      })
    }, 500)
  }

  const formatScore = (score: number) => `${(score * 100).toFixed(1)}%`

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'success'
    if (score >= 0.6) return 'warning'
    return 'error'
  }

  // Prepare data for radar chart
  const radarData = results.map(result => ({
    type: 'scatterpolar',
    r: [
      result.categories.intervention_understanding,
      result.categories.backdoor_identification,
      result.categories.frontdoor_adjustment,
      result.categories.counterfactual_reasoning,
      result.categories.intervention_understanding, // Close the loop
    ],
    theta: [
      'Intervention Understanding',
      'Backdoor Identification',
      'Frontdoor Adjustment',
      'Counterfactual Reasoning',
      'Intervention Understanding',
    ],
    fill: 'toself',
    name: result.model,
  }))

  const radarLayout = {
    polar: {
      radialaxis: {
        visible: true,
        range: [0, 1],
      },
    },
    showlegend: true,
    title: 'Causal Reasoning Capabilities by Category',
  }

  // Prepare data for bar chart
  const barData = [
    {
      x: results.map(r => r.model),
      y: results.map(r => r.overall_score),
      type: 'bar',
      marker: {
        color: results.map(r => 
          r.overall_score >= 0.8 ? '#4caf50' :
          r.overall_score >= 0.6 ? '#ff9800' : '#f44336'
        ),
      },
    },
  ]

  const barLayout = {
    title: 'Overall Causal Reasoning Scores',
    xaxis: { title: 'Model' },
    yaxis: { title: 'Score', range: [0, 1] },
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          LLM Causal Reasoning Benchmark
        </Typography>
        
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Comprehensive evaluation of large language models on causal reasoning tasks including 
          intervention understanding, backdoor identification, and counterfactual reasoning.
        </Typography>

        <Button
          variant="contained"
          size="large"
          startIcon={<PlayArrow />}
          onClick={runBenchmark}
          disabled={isRunning}
          sx={{ mb: 2 }}
        >
          {isRunning ? 'Running Benchmark...' : 'Run New Benchmark'}
        </Button>

        {isRunning && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Progress: {progress}%
            </Typography>
            <LinearProgress variant="determinate" value={progress} />
          </Box>
        )}
      </Box>

      {/* Results Overview */}
      <Grid container spacing={4} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, height: '400px' }}>
            <Plot
              data={barData}
              layout={barLayout}
              style={{ width: '100%', height: '100%' }}
              config={{ responsive: true }}
            />
          </Paper>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: '400px' }}>
            <Typography variant="h6" gutterBottom>
              Benchmark Summary
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Models Evaluated: {results.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Test Categories: 4
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Test Cases: 240
              </Typography>
            </Box>

            <Alert severity="info" sx={{ mb: 2 }}>
              Benchmark evaluates models on classic causal scenarios including 
              Simpson's Paradox, confounded treatment, and instrumental variables.
            </Alert>

            <Typography variant="subtitle2" gutterBottom>
              Top Performer
            </Typography>
            {results.length > 0 && (
              <Box sx={{ p: 2, bgcolor: 'success.50', borderRadius: 1 }}>
                <Typography variant="h6" color="success.main">
                  {results[0].model}
                </Typography>
                <Typography variant="body2">
                  {formatScore(results[0].overall_score)}
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Detailed Results Table */}
      <Paper sx={{ mb: 4 }}>
        <Box sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <Assessment sx={{ mr: 2 }} />
            Detailed Results
          </Typography>
        </Box>
        
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell align="center">Overall Score</TableCell>
                <TableCell align="center">Intervention Understanding</TableCell>
                <TableCell align="center">Backdoor Identification</TableCell>
                <TableCell align="center">Frontdoor Adjustment</TableCell>
                <TableCell align="center">Counterfactual Reasoning</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {results.map((result, index) => (
                <TableRow key={result.model} sx={{ '&:nth-of-type(odd)': { bgcolor: 'action.hover' } }}>
                  <TableCell component="th" scope="row">
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {result.model}
                      {index === 0 && (
                        <Chip
                          label="Best"
                          size="small"
                          color="success"
                          sx={{ ml: 1 }}
                        />
                      )}
                    </Box>
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={formatScore(result.overall_score)}
                      color={getScoreColor(result.overall_score)}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="center">
                    {formatScore(result.categories.intervention_understanding)}
                  </TableCell>
                  <TableCell align="center">
                    {formatScore(result.categories.backdoor_identification)}
                  </TableCell>
                  <TableCell align="center">
                    {formatScore(result.categories.frontdoor_adjustment)}
                  </TableCell>
                  <TableCell align="center">
                    {formatScore(result.categories.counterfactual_reasoning)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Category Breakdown */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Category Breakdown
        </Typography>
        <Box sx={{ height: '500px' }}>
          <Plot
            data={radarData}
            layout={radarLayout}
            style={{ width: '100%', height: '100%' }}
            config={{ responsive: true }}
          />
        </Box>
      </Paper>
    </Container>
  )
}