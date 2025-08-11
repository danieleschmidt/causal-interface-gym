import React from 'react'
import {
  Box,
  CircularProgress,
  Typography,
  Paper,
  LinearProgress,
} from '@mui/material'

interface LoadingSpinnerProps {
  message?: string
  size?: number
  variant?: 'circular' | 'linear' | 'overlay'
  progress?: number
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Loading...',
  size = 40,
  variant = 'circular',
  progress,
}) => {
  if (variant === 'linear') {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {message}
        </Typography>
        <LinearProgress
          variant={progress !== undefined ? 'determinate' : 'indeterminate'}
          value={progress}
          sx={{ height: 6, borderRadius: 3 }}
        />
        {progress !== undefined && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
            {Math.round(progress)}%
          </Typography>
        )}
      </Box>
    )
  }

  if (variant === 'overlay') {
    return (
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          bgcolor: 'rgba(255, 255, 255, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
        }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 4,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <CircularProgress size={size} />
          <Typography variant="body1" color="text.secondary">
            {message}
          </Typography>
        </Paper>
      </Box>
    )
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
        p: 3,
      }}
    >
      <CircularProgress
        size={size}
        variant={progress !== undefined ? 'determinate' : 'indeterminate'}
        value={progress}
      />
      <Typography variant="body2" color="text.secondary" textAlign="center">
        {message}
      </Typography>
      {progress !== undefined && (
        <Typography variant="caption" color="text.secondary">
          {Math.round(progress)}%
        </Typography>
      )}
    </Box>
  )
}

interface LazyLoadingProps {
  children: React.ReactNode
  fallback?: React.ReactNode
}

export const LazyLoading: React.FC<LazyLoadingProps> = ({
  children,
  fallback = <LoadingSpinner message="Loading component..." />,
}) => {
  return (
    <React.Suspense fallback={fallback}>
      {children}
    </React.Suspense>
  )
}