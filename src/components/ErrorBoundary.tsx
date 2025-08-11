import React, { Component, ErrorInfo, ReactNode } from 'react'
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material'
import { ExpandMore, Refresh, BugReport } from '@mui/icons-material'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    }
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    })

    // Log error to monitoring service
    console.error('Error caught by boundary:', error, errorInfo)
    
    // Call custom error handler
    this.props.onError?.(error, errorInfo)
  }

  handleReload = () => {
    window.location.reload()
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      const { error, errorInfo } = this.state

      return (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '50vh',
            p: 3,
          }}
        >
          <Paper
            sx={{
              p: 4,
              maxWidth: 600,
              width: '100%',
            }}
          >
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <BugReport color="error" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h4" color="error" gutterBottom>
                Something went wrong
              </Typography>
              <Typography variant="body1" color="text.secondary">
                We apologize for the inconvenience. An unexpected error occurred.
              </Typography>
            </Box>

            <Alert severity="error" sx={{ mb: 3 }}>
              <Typography variant="body2">
                <strong>Error:</strong> {error?.message || 'Unknown error'}
              </Typography>
            </Alert>

            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 3 }}>
              <Button
                variant="contained"
                startIcon={<Refresh />}
                onClick={this.handleRetry}
              >
                Try Again
              </Button>
              <Button
                variant="outlined"
                onClick={this.handleReload}
              >
                Reload Page
              </Button>
            </Box>

            {process.env.NODE_ENV === 'development' && error && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle2">
                    Error Details (Development)
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box
                    component="pre"
                    sx={{
                      bgcolor: 'grey.100',
                      p: 2,
                      borderRadius: 1,
                      overflow: 'auto',
                      fontSize: '0.75rem',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-all',
                    }}
                  >
                    <Typography variant="caption" component="div">
                      <strong>Error Stack:</strong>
                    </Typography>
                    {error.stack}
                    
                    {errorInfo && (
                      <>
                        <Typography variant="caption" component="div" sx={{ mt: 2 }}>
                          <strong>Component Stack:</strong>
                        </Typography>
                        {errorInfo.componentStack}
                      </>
                    )}
                  </Box>
                </AccordionDetails>
              </Accordion>
            )}
          </Paper>
        </Box>
      )
    }

    return this.props.children
  }
}

// HOC version for functional components
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  fallback?: ReactNode,
  onError?: (error: Error, errorInfo: ErrorInfo) => void
) => {
  return React.forwardRef<any, P>((props, ref) => (
    <ErrorBoundary fallback={fallback} onError={onError}>
      <Component {...props} ref={ref} />
    </ErrorBoundary>
  ))
}