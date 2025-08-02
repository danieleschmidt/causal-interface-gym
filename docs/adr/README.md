# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Causal Interface Gym project.

## What are ADRs?

Architecture Decision Records (ADRs) are documents that capture important architectural decisions made along with their context and consequences.

## Format

We use the following format for our ADRs:

```markdown
# ADR-XXXX: [Short title of solved problem and solution]

## Status

[Proposed | Accepted | Deprecated | Superseded]

## Context

What is the issue that we're seeing that is motivating this decision or change?

## Decision

What is the change that we're proposing or have agreed to implement?

## Consequences

What becomes easier or more difficult to do and any risks introduced by the change that will need to be mitigated.
```

## Index

- [ADR-0001: Use React for Frontend Components](0001-react-frontend.md)
- [ADR-0002: Python Backend with NetworkX for Causal Graphs](0002-python-networkx-backend.md)
- [ADR-0003: Pydantic for Data Validation](0003-pydantic-validation.md)