# ADR-0001: Use React for Frontend Components

## Status

Accepted

## Context

We need to build interactive UI components for causal graph visualization, intervention controls, and belief tracking. The components need to be:
- Highly interactive with real-time updates
- Reusable across different causal scenarios
- Integrable with existing web frameworks
- Maintainable by a research team

## Decision

We will use React for building frontend UI components because:
1. Large ecosystem of visualization libraries (D3, Plotly integration)
2. Component-based architecture fits our modular causal interface needs  
3. Strong TypeScript support for type safety in causal reasoning logic
4. Extensive testing ecosystem (Jest, React Testing Library)
5. Good performance for real-time causal graph updates

## Consequences

**Positive:**
- Reusable components across different causal environments
- Strong community support and documentation
- Easy integration with Python backends via REST APIs
- Component isolation enables easier testing of causal reasoning interfaces

**Negative:**  
- Additional build complexity compared to plain HTML/JS
- Learning curve for team members not familiar with React
- Bundle size considerations for embedding in research papers/demos

**Mitigation:**
- Use Vite for fast development and optimized builds
- Provide CDN-friendly builds for easy embedding
- Document component APIs thoroughly for researchers