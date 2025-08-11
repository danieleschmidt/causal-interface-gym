import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { Box, Paper } from '@mui/material'
import { CausalGraph as CausalGraphType, CausalNode, CausalEdge } from '../types'

interface CausalGraphProps {
  graph: CausalGraphType
  width?: number
  height?: number
  onNodeClick?: (node: CausalNode) => void
  interventions?: Record<string, any>
  highlightBackdoors?: boolean
}

export const CausalGraph: React.FC<CausalGraphProps> = ({
  graph,
  width = 800,
  height = 600,
  onNodeClick,
  interventions = {},
  highlightBackdoors = false,
}) => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current || !graph.nodes.length) return

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
    const g = svg.append('g')

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Prepare data for D3 force simulation
    const nodes = graph.nodes.map(node => ({
      ...node,
      x: node.position?.x || Math.random() * width,
      y: node.position?.y || Math.random() * height,
    }))

    const links = graph.edges.map(edge => ({
      source: edge.source,
      target: edge.target,
      ...edge,
    }))

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40))

    // Create arrow markers for directed edges
    svg.append('defs').append('marker')
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

    // Create intervention marker
    svg.select('defs').append('marker')
      .attr('id', 'intervention-marker')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#f59e0b')

    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', '#666')
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrowhead)')

    // Create nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('r', 20)
      .attr('fill', (d: any) => {
        if (interventions[d.id] !== undefined) return '#f59e0b' // Intervention color
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
      .call(d3.drag<SVGCircleElement, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))

    // Add node labels
    const labels = g.append('g')
      .selectAll('text')
      .data(nodes)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text((d: any) => d.label)
      .style('pointer-events', 'none')

    // Add node titles (tooltips)
    node.append('title')
      .text((d: any) => `${d.label} (${d.type})${interventions[d.id] !== undefined ? ' - Intervened' : ''}`)

    // Handle node clicks
    if (onNodeClick) {
      node.on('click', (event, d) => {
        event.stopPropagation()
        onNodeClick(d as CausalNode)
      })
    }

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y)

      labels
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y)
    })

    // Drag handlers
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      d.fx = d.x
      d.fy = d.y
    }

    function dragged(event: any, d: any) {
      d.fx = event.x
      d.fy = event.y
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0)
      d.fx = null
      d.fy = null
    }

    return () => {
      simulation.stop()
    }
  }, [graph, width, height, onNodeClick, interventions, highlightBackdoors])

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