import React, { useState } from 'react'
import {
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
} from '@mui/material'
import { Add, Clear, PlayArrow } from '@mui/icons-material'
import { CausalNode, Intervention } from '../types'

interface InterventionPanelProps {
  nodes: CausalNode[]
  interventions: Record<string, any>
  onInterventionChange: (interventions: Record<string, any>) => void
  onRunExperiment: () => void
  isRunning?: boolean
}

export const InterventionPanel: React.FC<InterventionPanelProps> = ({
  nodes,
  interventions,
  onInterventionChange,
  onRunExperiment,
  isRunning = false,
}) => {
  const [selectedVariable, setSelectedVariable] = useState('')
  const [interventionValue, setInterventionValue] = useState<any>('')
  const [interventionType, setInterventionType] = useState<'set' | 'randomize' | 'prevent'>('set')

  const addIntervention = () => {
    if (!selectedVariable) return

    const node = nodes.find(n => n.id === selectedVariable)
    if (!node) return

    let value = interventionValue

    // Convert value based on variable type
    if (node.type === 'binary' && typeof value === 'string') {
      value = value.toLowerCase() === 'true' || value === '1'
    } else if (node.type === 'continuous' && typeof value === 'string') {
      value = parseFloat(value) || 0
    }

    const newInterventions = {
      ...interventions,
      [selectedVariable]: { value, type: interventionType }
    }

    onInterventionChange(newInterventions)
    setSelectedVariable('')
    setInterventionValue('')
  }

  const removeIntervention = (variable: string) => {
    const newInterventions = { ...interventions }
    delete newInterventions[variable]
    onInterventionChange(newInterventions)
  }

  const clearAllInterventions = () => {
    onInterventionChange({})
  }

  const renderValueInput = () => {
    const node = nodes.find(n => n.id === selectedVariable)
    if (!node) return null

    switch (node.type) {
      case 'binary':
        return (
          <FormControlLabel
            control={
              <Switch
                checked={Boolean(interventionValue)}
                onChange={(e) => setInterventionValue(e.target.checked)}
              />
            }
            label={interventionValue ? 'True' : 'False'}
          />
        )

      case 'continuous':
        return (
          <Box sx={{ px: 2 }}>
            <Typography gutterBottom>Value: {interventionValue || 0}</Typography>
            <Slider
              value={Number(interventionValue) || 0}
              onChange={(_, newValue) => setInterventionValue(newValue)}
              min={0}
              max={10}
              step={0.1}
              valueLabelDisplay="auto"
            />
          </Box>
        )

      case 'categorical':
        return (
          <TextField
            fullWidth
            label="Category"
            value={interventionValue}
            onChange={(e) => setInterventionValue(e.target.value)}
            placeholder="Enter category value"
            size="small"
          />
        )

      default:
        return (
          <TextField
            fullWidth
            label="Value"
            value={interventionValue}
            onChange={(e) => setInterventionValue(e.target.value)}
            size="small"
          />
        )
    }
  }

  return (
    <Paper elevation={2} sx={{ p: 3, height: 'fit-content' }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        Intervention Controls
      </Typography>

      {/* Add New Intervention */}
      <Box sx={{ mb: 3 }}>
        <FormControl fullWidth size="small" sx={{ mb: 2 }}>
          <InputLabel>Variable</InputLabel>
          <Select
            value={selectedVariable}
            onChange={(e) => setSelectedVariable(e.target.value)}
            label="Variable"
          >
            {nodes
              .filter(node => !interventions[node.id])
              .map(node => (
                <MenuItem key={node.id} value={node.id}>
                  {node.label} ({node.type})
                </MenuItem>
              ))}
          </Select>
        </FormControl>

        {selectedVariable && (
          <>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Intervention Type</InputLabel>
              <Select
                value={interventionType}
                onChange={(e) => setInterventionType(e.target.value as any)}
                label="Intervention Type"
              >
                <MenuItem value="set">Set Value</MenuItem>
                <MenuItem value="randomize">Randomize</MenuItem>
                <MenuItem value="prevent">Prevent/Remove</MenuItem>
              </Select>
            </FormControl>

            {interventionType === 'set' && (
              <Box sx={{ mb: 2 }}>
                {renderValueInput()}
              </Box>
            )}

            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={addIntervention}
              disabled={!selectedVariable || (interventionType === 'set' && interventionValue === '')}
              fullWidth
            >
              Add Intervention
            </Button>
          </>
        )}
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Current Interventions */}
      <Typography variant="subtitle1" gutterBottom>
        Active Interventions ({Object.keys(interventions).length})
      </Typography>

      {Object.keys(interventions).length === 0 ? (
        <Alert severity="info" sx={{ mb: 2 }}>
          No interventions active. Add interventions above to start experimenting.
        </Alert>
      ) : (
        <Box sx={{ mb: 3 }}>
          {Object.entries(interventions).map(([variable, intervention]) => {
            const node = nodes.find(n => n.id === variable)
            return (
              <Chip
                key={variable}
                label={`${node?.label || variable}: ${
                  intervention.type === 'set' 
                    ? `= ${intervention.value}` 
                    : intervention.type
                }`}
                onDelete={() => removeIntervention(variable)}
                deleteIcon={<Clear />}
                color="primary"
                variant="outlined"
                sx={{ mr: 1, mb: 1 }}
              />
            )
          })}
          <Box sx={{ mt: 2 }}>
            <Button
              variant="outlined"
              size="small"
              onClick={clearAllInterventions}
              disabled={isRunning}
            >
              Clear All
            </Button>
          </Box>
        </Box>
      )}

      <Divider sx={{ my: 2 }} />

      {/* Run Experiment */}
      <Button
        variant="contained"
        size="large"
        startIcon={<PlayArrow />}
        onClick={onRunExperiment}
        disabled={isRunning || Object.keys(interventions).length === 0}
        fullWidth
        sx={{ py: 1.5 }}
      >
        {isRunning ? 'Running Experiment...' : 'Run Experiment'}
      </Button>

      {Object.keys(interventions).length === 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
          Add at least one intervention to run experiment
        </Typography>
      )}
    </Paper>
  )
}