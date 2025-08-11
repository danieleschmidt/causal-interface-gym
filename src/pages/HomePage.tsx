import React from 'react'
import {
  Container,
  Typography,
  Box,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material'
import {
  Psychology,
  Science,
  Assessment,
  Code,
  ArrowForward,
  CheckCircle,
} from '@mui/icons-material'
import { Link } from 'react-router-dom'

const features = [
  {
    icon: <Psychology color="primary" />,
    title: 'Do-Calculus Engine',
    description: 'Implements Pearl\'s causal interventions in interactive UIs with full mathematical rigor.',
  },
  {
    icon: <Science color="secondary" />,
    title: 'LLM Agent Integration',
    description: 'Test any LLM\'s causal reasoning in real-time with comprehensive belief tracking.',
  },
  {
    icon: <Assessment color="primary" />,
    title: 'Interactive Causal Graphs',
    description: 'Visual DAG editor with live intervention effects and backdoor path highlighting.',
  },
  {
    icon: <Code color="secondary" />,
    title: 'A/B Testing Framework',
    description: 'Compare causal vs correlational interface designs with statistical significance.',
  },
]

const quickStart = [
  'Create a causal environment from DAG specification',
  'Build interactive intervention interface components',
  'Connect your LLM agent for belief extraction',
  'Run experiments and measure causal reasoning quality',
  'Export results in publication-ready format',
]

export const HomePage: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Hero Section */}
      <Box sx={{ textAlign: 'center', mb: 8 }}>
        <Typography
          variant="h1"
          component="h1"
          gutterBottom
          sx={{ 
            background: 'linear-gradient(45deg, #2563eb 30%, #7c3aed 90%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 3,
          }}
        >
          Causal Interface Gym
        </Typography>
        
        <Typography variant="h5" color="text.secondary" gutterBottom sx={{ mb: 4 }}>
          Toolkit to embed do-calculus interventions directly into UI prototypes and 
          measure how LLM agents update causal world-models
        </Typography>

        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Chip label="Python 3.10+" color="primary" variant="outlined" />
          <Chip label="React + TypeScript" color="secondary" variant="outlined" />
          <Chip label="NetworkX" color="primary" variant="outlined" />
          <Chip label="Do-Calculus" color="secondary" variant="outlined" />
        </Box>
      </Box>

      {/* Quick Actions */}
      <Grid container spacing={3} sx={{ mb: 8 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography variant="h6" gutterBottom>
                Start Experimenting
              </Typography>
              <Typography color="text.secondary">
                Create causal reasoning experiments with pre-built environments and LLM agents.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                component={Link}
                to="/experiment"
                variant="contained"
                endIcon={<ArrowForward />}
                fullWidth
              >
                Run Experiment
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography variant="h6" gutterBottom>
                Benchmark Models
              </Typography>
              <Typography color="text.secondary">
                Compare causal reasoning capabilities across different LLM providers and models.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                component={Link}
                to="/benchmark"
                variant="outlined"
                endIcon={<ArrowForward />}
                fullWidth
              >
                Start Benchmark
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography variant="h6" gutterBottom>
                Documentation
              </Typography>
              <Typography color="text.secondary">
                Learn about causal inference, do-calculus, and how to build effective interfaces.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                component={Link}
                to="/docs"
                variant="outlined"
                endIcon={<ArrowForward />}
                fullWidth
              >
                Read Docs
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>

      {/* Features Section */}
      <Typography variant="h2" component="h2" gutterBottom sx={{ mb: 4 }}>
        Key Features
      </Typography>

      <Grid container spacing={4} sx={{ mb: 8 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Paper
              sx={{
                p: 3,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'transform 0.2s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-4px)',
                },
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                {feature.icon}
                <Typography variant="h6" sx={{ ml: 2 }}>
                  {feature.title}
                </Typography>
              </Box>
              <Typography color="text.secondary">
                {feature.description}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Quick Start Section */}
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h5" gutterBottom>
              Quick Start Guide
            </Typography>
            <List>
              {quickStart.map((step, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <CheckCircle color="primary" />
                  </ListItemIcon>
                  <ListItemText primary={step} />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%', bgcolor: 'grey.50' }}>
            <Typography variant="h6" gutterBottom>
              Example Code
            </Typography>
            <Box
              component="pre"
              sx={{
                bgcolor: 'grey.900',
                color: 'white',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                fontSize: '0.875rem',
                lineHeight: 1.5,
              }}
            >
{`from causal_interface_gym import CausalEnvironment, InterventionUI

# Create causal environment
env = CausalEnvironment.from_dag({
    "rain": [],
    "sprinkler": ["rain"],
    "wet_grass": ["rain", "sprinkler"],
    "slippery": ["wet_grass"]
})

# Build intervention interface
ui = InterventionUI(env)
ui.add_intervention_button("sprinkler", "Toggle Sprinkler")
ui.add_observation_panel("wet_grass", "Grass Wetness")

# Test LLM causal reasoning
result = ui.run_experiment(
    agent=your_llm,
    interventions=[("sprinkler", True)],
    measure_beliefs=["P(slippery)", "P(rain|wet_grass)"]
)

print(f"Causal Score: {result['causal_analysis']['causal_score']:.3f}")`}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  )
}