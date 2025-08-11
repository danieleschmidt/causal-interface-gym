import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'
import { ErrorBoundary } from './components/ErrorBoundary'
import { LazyLoading } from './components/LoadingSpinner'
import { Navbar } from './components/Navbar'

// Lazy load pages for better performance
const HomePage = React.lazy(() => import('./pages/HomePage').then(module => ({ default: module.HomePage })))
const ExperimentPage = React.lazy(() => import('./pages/ExperimentPage').then(module => ({ default: module.ExperimentPage })))
const BenchmarkPage = React.lazy(() => import('./pages/BenchmarkPage').then(module => ({ default: module.BenchmarkPage })))
const DocsPage = React.lazy(() => import('./pages/DocsPage').then(module => ({ default: module.DocsPage })))

function App() {
  return (
    <ErrorBoundary>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Navbar />
        <Box component="main" sx={{ flexGrow: 1, bgcolor: 'background.default' }}>
          <LazyLoading>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/experiment" element={<ExperimentPage />} />
              <Route path="/benchmark" element={<BenchmarkPage />} />
              <Route path="/docs" element={<DocsPage />} />
            </Routes>
          </LazyLoading>
        </Box>
      </Box>
    </ErrorBoundary>
  )
}

export default App