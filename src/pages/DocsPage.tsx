import React from 'react'
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material'
import { ExpandMore } from '@mui/icons-material'

const docSections = [
  {
    title: 'Getting Started',
    items: [
      'Installation Guide',
      'Quick Start Tutorial',
      'Basic Concepts',
      'First Experiment',
    ],
  },
  {
    title: 'Causal Inference',
    items: [
      'Do-Calculus Foundations',
      'Backdoor Adjustment',
      'Frontdoor Criterion',
      'Instrumental Variables',
    ],
  },
  {
    title: 'LLM Integration',
    items: [
      'Belief Extraction',
      'Provider Configuration',
      'Response Parsing',
      'Custom Agents',
    ],
  },
  {
    title: 'UI Components',
    items: [
      'Causal Graph Visualization',
      'Intervention Controls',
      'Belief Displays',
      'Custom Components',
    ],
  },
]

const faqs = [
  {
    question: 'What is do-calculus and why is it important?',
    answer: `Do-calculus, developed by Judea Pearl, provides a formal framework for reasoning about causal relationships. 
    It allows us to distinguish between observational correlations P(Y|X) and interventional causation P(Y|do(X)). 
    This is crucial for LLMs because they often confuse correlation with causation, leading to incorrect reasoning 
    about interventions and counterfactuals.`,
  },
  {
    question: 'How does this framework test LLM causal reasoning?',
    answer: `The framework creates interactive environments where LLMs can observe data and perform interventions. 
    We measure how well they update their beliefs when moving from observational to interventional scenarios. 
    Key metrics include intervention vs observation understanding, backdoor path identification, and counterfactual reasoning accuracy.`,
  },
  {
    question: 'What makes a good causal reasoning interface?',
    answer: `Effective causal interfaces should: (1) Clearly distinguish between observation and intervention, 
    (2) Visualize causal relationships through DAGs, (3) Provide interactive controls for testing hypotheses, 
    (4) Show belief updates in real-time, and (5) Highlight potential confounders and backdoor paths.`,
  },
  {
    question: 'Can I use this with any LLM provider?',
    answer: `Yes! The framework supports OpenAI, Anthropic, and local models through a unified provider interface. 
    You can easily add new providers by implementing the LLMProvider interface. The system handles belief extraction 
    and response parsing automatically.`,
  },
]

export const DocsPage: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Documentation
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 6 }}>
        Learn how to build effective causal reasoning interfaces and evaluate LLM capabilities.
      </Typography>

      <Grid container spacing={4}>
        {/* Navigation */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, position: 'sticky', top: 100 }}>
            <Typography variant="h6" gutterBottom>
              Documentation Sections
            </Typography>
            {docSections.map((section, index) => (
              <Box key={index} sx={{ mb: 2 }}>
                <Typography variant="subtitle1" color="primary" gutterBottom>
                  {section.title}
                </Typography>
                <List dense>
                  {section.items.map((item, itemIndex) => (
                    <ListItem key={itemIndex} sx={{ py: 0.5 }}>
                      <ListItemText 
                        primary={item}
                        primaryTypographyProps={{ variant: 'body2' }}
                        sx={{ cursor: 'pointer', '&:hover': { color: 'primary.main' } }}
                      />
                    </ListItem>
                  ))}
                </List>
                {index < docSections.length - 1 && <Divider sx={{ mt: 1 }} />}
              </Box>
            ))}
          </Paper>
        </Grid>

        {/* Content */}
        <Grid item xs={12} md={8}>
          {/* Quick Start */}
          <Paper sx={{ p: 4, mb: 4 }}>
            <Typography variant="h4" gutterBottom>
              Quick Start Guide
            </Typography>
            
            <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
              1. Installation
            </Typography>
            <Box
              component="pre"
              sx={{
                bgcolor: 'grey.900',
                color: 'white',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                mb: 2,
              }}
            >
{`# Install the Python package
pip install causal-interface-gym

# Install with UI components
pip install causal-interface-gym[ui]

# Development installation
git clone https://github.com/yourusername/causal-interface-gym.git
cd causal-interface-gym
pip install -e ".[dev]"`}
            </Box>

            <Typography variant="h6" gutterBottom>
              2. Create Your First Experiment
            </Typography>
            <Box
              component="pre"
              sx={{
                bgcolor: 'grey.900',
                color: 'white',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                mb: 2,
              }}
            >
{`from causal_interface_gym import CausalEnvironment, InterventionUI
import openai

# Define causal structure
dag = {
    "treatment": [],
    "confounder": [],
    "outcome": ["treatment", "confounder"]
}

# Create environment
env = CausalEnvironment.from_dag(dag)

# Build interface
ui = InterventionUI(env)
ui.add_intervention_button("treatment", "Apply Treatment")
ui.add_observation_panel("outcome", "Measure Outcome")

# Test with LLM
agent = openai.ChatCompletion()
results = ui.run_experiment(
    agent=agent,
    interventions=[("treatment", True)],
    measure_beliefs=["P(outcome)", "P(outcome|treatment)"]
)

print(f"Causal reasoning score: {results['causal_analysis']['causal_score']:.3f}")`}
            </Box>

            <Typography variant="h6" gutterBottom>
              3. Analyze Results
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              The framework automatically computes causal reasoning metrics:
            </Typography>
            <List>
              <ListItem>
                <ListItemText 
                  primary="Intervention vs Observation Score"
                  secondary="Measures how well the agent distinguishes P(Y|do(X)) from P(Y|X)"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Backdoor Identification"
                  secondary="Tests ability to identify and adjust for confounding variables"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Belief Update Tracking"
                  secondary="Monitors how beliefs change with interventions vs observations"
                />
              </ListItem>
            </List>
          </Paper>

          {/* Core Concepts */}
          <Paper sx={{ p: 4, mb: 4 }}>
            <Typography variant="h4" gutterBottom>
              Core Concepts
            </Typography>

            <Typography variant="h6" gutterBottom>
              Causal vs Correlational Thinking
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              The fundamental distinction in causal reasoning is between:
            </Typography>
            <List>
              <ListItem>
                <ListItemText 
                  primary="P(Y|X) - Observational"
                  secondary="Probability of Y given that we observe X"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="P(Y|do(X)) - Interventional"
                  secondary="Probability of Y given that we force X to a specific value"
                />
              </ListItem>
            </List>

            <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
              Simpson's Paradox Example
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              Consider college admissions data where:
            </Typography>
            <List>
              <ListItem>
                <ListItemText 
                  primary="Overall: Men have higher admission rates"
                  secondary="This suggests gender discrimination against women"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="By Department: Women have higher rates in each department"
                  secondary="But women apply to more competitive departments"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Causal Question: Would changing someone's gender change their admission probability?"
                  secondary="Answer: No - the effect disappears when controlling for department choice"
                />
              </ListItem>
            </List>
          </Paper>

          {/* FAQ */}
          <Paper sx={{ p: 4 }}>
            <Typography variant="h4" gutterBottom>
              Frequently Asked Questions
            </Typography>

            {faqs.map((faq, index) => (
              <Accordion key={index} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">
                    {faq.question}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                    {faq.answer}
                  </Typography>
                </AccordionDetails>
              </Accordion>
            ))}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  )
}