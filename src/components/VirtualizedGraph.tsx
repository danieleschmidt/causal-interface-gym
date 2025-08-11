import React, { useEffect, useRef, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { Box, Paper } from '@mui/material'
import { CausalGraph as CausalGraphType, CausalNode, CausalEdge } from '../types'
import { PerformanceMonitor, useDebounce } from '../utils/performance'

interface VirtualizedGraphProps {
  graph: CausalGraphType
  width?: number
  height?: number
  onNodeClick?: (node: CausalNode) => void
  interventions?: Record<string, any>
  highlightBackdoors?: boolean
  viewportPadding?: number
}

export const VirtualizedGraph: React.FC<VirtualizedGraphProps> = ({
  graph,
  width = 800,
  height = 600,
  onNodeClick,
  interventions = {},
  highlightBackdoors = false,
  viewportPadding = 100,
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const simulationRef = useRef<d3.Simulation<any, undefined> | null>(null)
  const transformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity)
  const visibleNodesRef = useRef<Set<string>>(new Set())
  const visibleEdgesRef = useRef<Set<string>>(new Set())

  // Memoize node and edge data preparation
  const { processedNodes, processedEdges } = useMemo(() => {
    return PerformanceMonitor.measure('processGraphData', () => {
      const nodes = graph.nodes.map(node => ({
        ...node,
        x: node.position?.x || Math.random() * width,
        y: node.position?.y || Math.random() * height,
        radius: 20,
      }))

      const edges = graph.edges.map((edge, index) => ({
        ...edge,
        id: `${edge.source}-${edge.target}-${index}`,
        source: nodes.find(n => n.id === edge.source),
        target: nodes.find(n => n.id === edge.target),
      })).filter(edge => edge.source && edge.target)

      return { processedNodes: nodes, processedEdges: edges }
    })
  }, [graph.nodes, graph.edges, width, height])

  // Debounced viewport calculation
  const debouncedUpdateViewport = useDebounce((transform: d3.ZoomTransform) => {
    const viewport = {
      x: -transform.x / transform.k - viewportPadding,
      y: -transform.y / transform.k - viewportPadding,
      width: width / transform.k + 2 * viewportPadding,
      height: height / transform.k + 2 * viewportPadding,
    }

    // Calculate visible nodes
    const newVisibleNodes = new Set<string>()
    processedNodes.forEach(node => {
      if (
        node.x >= viewport.x &&
        node.x <= viewport.x + viewport.width &&
        node.y >= viewport.y &&
        node.y <= viewport.y + viewport.height
      ) {
        newVisibleNodes.add(node.id)
      }
    })

    // Calculate visible edges
    const newVisibleEdges = new Set<string>()
    processedEdges.forEach(edge => {
      if (
        newVisibleNodes.has(edge.source.id) ||
        newVisibleNodes.has(edge.target.id)
      ) {
        newVisibleEdges.add(edge.id)
      }
    })

    visibleNodesRef.current = newVisibleNodes
    visibleEdgesRef.current = newVisibleEdges

    // Update rendering
    updateRendering()
  }, 100)

  const updateRendering = useCallback(() => {
    PerformanceMonitor.measure('updateRendering', () => {
      const svg = d3.select(svgRef.current)
      const container = svg.select('.graph-container')

      // Update node visibility
      container.selectAll('.node')
        .style('display', (d: any) => 
          visibleNodesRef.current.has(d.id) ? 'block' : 'none'
        )

      // Update edge visibility
      container.selectAll('.edge')
        .style('display', (d: any) => 
          visibleEdgesRef.current.has(d.id) ? 'block' : 'none'
        )

      // Update labels visibility
      container.selectAll('.label')
        .style('display', (d: any) => 
          visibleNodesRef.current.has(d.id) ? 'block' : 'none'
        )
    })
  }, [])

  const initializeGraph = useCallback(() => {
    if (!svgRef.current || !processedNodes.length) return

    PerformanceMonitor.measure('initializeGraph', () => {
      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      const container = svg.append('g').attr('class', 'graph-container')

      // Create zoom behavior with performance optimization
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 5])
        .on('zoom', (event) => {
          transformRef.current = event.transform
          container.attr('transform', event.transform)
          debouncedUpdateViewport(event.transform)
        })

      svg.call(zoom)

      // Create markers
      const defs = svg.append('defs')
      
      defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 25)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#666')

      // Create force simulation with optimized forces
      simulationRef.current = d3.forceSimulation(processedNodes)
        .force('link', d3.forceLink(processedEdges)
          .id((d: any) => d.id)
          .distance(120)
          .strength(0.5)
        )
        .force('charge', d3.forceManyBody()
          .strength(-300)
          .theta(0.8) // Optimize performance
        )
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide()
          .radius(40)
          .iterations(2) // Reduce iterations for performance
        )
        .alpha(0.3)
        .alphaDecay(0.05)

      // Create edges
      const links = container.append('g')
        .selectAll('.edge')
        .data(processedEdges)
        .enter()
        .append('line')
        .attr('class', 'edge')
        .attr('stroke', '#666')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead)')

      // Create nodes with efficient rendering
      const nodes = container.append('g')
        .selectAll('.node')
        .data(processedNodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('r', 20)
        .attr('fill', (d: any) => {
          if (interventions[d.id] !== undefined) return '#f59e0b'
          switch (d.type) {
            case 'binary': return '#3b82f6'
            case 'continuous': return '#10b981'
            case 'categorical': return '#8b5cf6'
            default: return '#6b7280'
          }
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', 3)
        .style('cursor', onNodeClick ? 'pointer' : 'default')

      // Create labels
      const labels = container.append('g')
        .selectAll('.label')
        .data(processedNodes)
        .enter()
        .append('text')
        .attr('class', 'label')
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('fill', 'white')
        .text((d: any) => d.label)
        .style('pointer-events', 'none')

      // Add drag behavior with performance optimization
      const dragBehavior = d3.drag<SVGCircleElement, any>()
        .on('start', function(event, d) {
          if (!event.active) simulationRef.current?.alphaTarget(0.1).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', function(event, d) {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', function(event, d) {
          if (!event.active) simulationRef.current?.alphaTarget(0)
          d.fx = null
          d.fy = null
        })

      nodes.call(dragBehavior)

      // Add click handlers
      if (onNodeClick) {
        nodes.on('click', (event, d) => {
          event.stopPropagation()
          onNodeClick(d as CausalNode)
        })
      }

      // Optimized tick function
      simulationRef.current?.on('tick', () => {
        // Only update visible elements
        links
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y)

        nodes
          .attr('cx', (d: any) => d.x)
          .attr('cy', (d: any) => d.y)

        labels
          .attr('x', (d: any) => d.x)
          .attr('y', (d: any) => d.y)
      })

      // Initial viewport calculation
      debouncedUpdateViewport(transformRef.current)
    })
  }, [processedNodes, processedEdges, width, height, onNodeClick, interventions, debouncedUpdateViewport])

  useEffect(() => {
    initializeGraph()

    return () => {
      simulationRef.current?.stop()
    }
  }, [initializeGraph])

  return (
    <Paper elevation={2} sx={{ p: 2, bgcolor: 'background.paper' }}>
      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ border: '1px solid #e0e0e0', borderRadius: '4px' }}
        />
      </Box>
    </Paper>
  )
}