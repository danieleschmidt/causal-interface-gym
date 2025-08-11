import React, { useMemo, useState, useCallback } from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  Paper,
  Box,
  Chip,
  useTheme,
} from '@mui/material'
import { useVirtualList, PerformanceMonitor } from '../utils/performance'
import { BenchmarkResult } from '../types'

interface OptimizedTableProps {
  data: BenchmarkResult[]
  columns: {
    key: keyof BenchmarkResult | keyof BenchmarkResult['categories']
    label: string
    sortable?: boolean
    render?: (value: any, row: BenchmarkResult) => React.ReactNode
  }[]
  itemHeight?: number
  maxHeight?: number
  virtualized?: boolean
  onRowClick?: (row: BenchmarkResult) => void
}

type SortDirection = 'asc' | 'desc'

export const OptimizedTable: React.FC<OptimizedTableProps> = ({
  data,
  columns,
  itemHeight = 60,
  maxHeight = 600,
  virtualized = true,
  onRowClick,
}) => {
  const theme = useTheme()
  const [sortBy, setSortBy] = useState<string>('')
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc')
  const [page, setPage] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(25)

  // Memoized sorting
  const sortedData = useMemo(() => {
    return PerformanceMonitor.measure('sortData', () => {
      if (!sortBy) return data

      return [...data].sort((a, b) => {
        let aValue: any
        let bValue: any

        if (sortBy.startsWith('categories.')) {
          const categoryKey = sortBy.split('.')[1] as keyof BenchmarkResult['categories']
          aValue = a.categories[categoryKey]
          bValue = b.categories[categoryKey]
        } else {
          aValue = a[sortBy as keyof BenchmarkResult]
          bValue = b[sortBy as keyof BenchmarkResult]
        }

        if (typeof aValue === 'string' && typeof bValue === 'string') {
          aValue = aValue.toLowerCase()
          bValue = bValue.toLowerCase()
        }

        if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1
        if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1
        return 0
      })
    })
  }, [data, sortBy, sortDirection])

  // Memoized pagination
  const paginatedData = useMemo(() => {
    const start = page * rowsPerPage
    return sortedData.slice(start, start + rowsPerPage)
  }, [sortedData, page, rowsPerPage])

  // Virtual list for performance
  const { visibleItems, totalHeight, onScroll } = useVirtualList(
    virtualized ? paginatedData : [],
    itemHeight,
    maxHeight
  )

  const handleSort = useCallback((columnKey: string) => {
    if (sortBy === columnKey) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(columnKey)
      setSortDirection('asc')
    }
  }, [sortBy, sortDirection])

  const handlePageChange = useCallback((event: unknown, newPage: number) => {
    setPage(newPage)
  }, [])

  const handleRowsPerPageChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10))
    setPage(0)
  }, [])

  const getCellValue = useCallback((row: BenchmarkResult, column: any) => {
    if (column.key.startsWith('categories.')) {
      const categoryKey = column.key.split('.')[1] as keyof BenchmarkResult['categories']
      return row.categories[categoryKey]
    }
    return row[column.key as keyof BenchmarkResult]
  }, [])

  const renderCell = useCallback((value: any, row: BenchmarkResult, column: any) => {
    if (column.render) {
      return column.render(value, row)
    }

    if (typeof value === 'number') {
      return (value * 100).toFixed(1) + '%'
    }

    if (typeof value === 'string' && column.key === 'model') {
      return (
        <Chip
          label={value}
          variant="outlined"
          color="primary"
          size="small"
        />
      )
    }

    return value
  }, [])

  const renderTableHeader = () => (
    <TableHead>
      <TableRow>
        {columns.map((column) => (
          <TableCell key={column.key}>
            {column.sortable !== false ? (
              <TableSortLabel
                active={sortBy === column.key}
                direction={sortBy === column.key ? sortDirection : 'asc'}
                onClick={() => handleSort(column.key)}
              >
                {column.label}
              </TableSortLabel>
            ) : (
              column.label
            )}
          </TableCell>
        ))}
      </TableRow>
    </TableHead>
  )

  const renderTableRow = useCallback((row: BenchmarkResult, index: number) => (
    <TableRow
      key={`${row.model}-${index}`}
      hover
      onClick={() => onRowClick?.(row)}
      sx={{
        cursor: onRowClick ? 'pointer' : 'default',
        '&:nth-of-type(odd)': {
          backgroundColor: theme.palette.action.hover,
        },
      }}
    >
      {columns.map((column) => {
        const value = getCellValue(row, column)
        return (
          <TableCell key={column.key}>
            {renderCell(value, row, column)}
          </TableCell>
        )
      })}
    </TableRow>
  ), [columns, getCellValue, renderCell, onRowClick, theme])

  if (virtualized && data.length > 100) {
    return (
      <Paper elevation={2}>
        <TableContainer sx={{ maxHeight }}>
          <Table stickyHeader>
            {renderTableHeader()}
            <TableBody>
              <TableRow>
                <TableCell colSpan={columns.length} sx={{ p: 0 }}>
                  <Box
                    sx={{ height: totalHeight, overflow: 'auto' }}
                    onScroll={onScroll}
                  >
                    {visibleItems.map((row, index) => (
                      <Box
                        key={`${row.model}-${index}`}
                        sx={{
                          height: itemHeight,
                          display: 'flex',
                          alignItems: 'center',
                          borderBottom: '1px solid',
                          borderColor: 'divider',
                          px: 2,
                        }}
                      >
                        {columns.map((column) => {
                          const value = getCellValue(row, column)
                          return (
                            <Box
                              key={column.key}
                              sx={{
                                flex: 1,
                                display: 'flex',
                                alignItems: 'center',
                              }}
                            >
                              {renderCell(value, row, column)}
                            </Box>
                          )
                        })}
                      </Box>
                    ))}
                  </Box>
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>

        <TablePagination
          rowsPerPageOptions={[25, 50, 100]}
          component="div"
          count={data.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handlePageChange}
          onRowsPerPageChange={handleRowsPerPageChange}
        />
      </Paper>
    )
  }

  return (
    <Paper elevation={2}>
      <TableContainer sx={{ maxHeight: virtualized ? undefined : maxHeight }}>
        <Table stickyHeader>
          {renderTableHeader()}
          <TableBody>
            {paginatedData.map((row, index) => renderTableRow(row, index))}
          </TableBody>
        </Table>
      </TableContainer>

      {data.length > rowsPerPage && (
        <TablePagination
          rowsPerPageOptions={[25, 50, 100]}
          component="div"
          count={data.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handlePageChange}
          onRowsPerPageChange={handleRowsPerPageChange}
        />
      )}
    </Paper>
  )
}