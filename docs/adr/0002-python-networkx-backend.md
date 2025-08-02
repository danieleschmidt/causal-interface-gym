# ADR-0002: Python Backend with NetworkX for Causal Graphs

## Status

Accepted

## Context

The core of our system needs to represent and manipulate causal graphs, perform do-calculus operations, and compute interventional distributions. We need a robust graph library that can handle:
- Directed Acyclic Graphs (DAGs) with causal constraints
- Graph algorithms for backdoor/frontdoor identification
- Integration with scientific Python ecosystem
- Performance for real-time UI updates

## Decision

We will use Python with NetworkX as our core backend because:
1. NetworkX provides comprehensive graph algorithms needed for causal inference
2. Python integrates well with the broader causal inference ecosystem (PyWhy, scikit-learn)
3. NumPy/SciPy integration for efficient probability computations
4. Easy serialization to JSON for frontend communication
5. Extensive testing and validation capabilities

## Consequences

**Positive:**
- Access to mature causal inference libraries and research code
- NetworkX has algorithms we need (topological sort, path finding, etc.)
- Python's scientific ecosystem enables complex causal computations
- JSON serialization makes frontend integration straightforward

**Negative:**
- Python performance limitations for very large graphs
- NetworkX memory overhead compared to specialized graph libraries
- Threading limitations for concurrent causal computations

**Mitigation:**
- Profile performance and optimize bottlenecks with Cython if needed
- Consider graph partitioning for very large causal models
- Use async patterns for non-blocking UI updates
- Implement caching for repeated causal queries