"""Data models for causal interface gym."""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field


@dataclass
class ExperimentModel:
    """Model for causal reasoning experiments."""
    
    experiment_id: str
    agent_type: str
    causal_graph: Dict[str, Any]
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        data['causal_graph'] = json.dumps(self.causal_graph)
        data['metadata'] = json.dumps(self.metadata or {})
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentModel':
        """Create from dictionary (from database)."""
        # Remove database-specific fields that aren't in the model
        clean_data = {k: v for k, v in data.items() if k not in ['id']}
        
        if isinstance(clean_data['causal_graph'], str):
            clean_data['causal_graph'] = json.loads(clean_data['causal_graph'])
        if isinstance(clean_data['metadata'], str):
            clean_data['metadata'] = json.loads(clean_data['metadata'])
        return cls(**clean_data)


@dataclass
class BeliefMeasurement:
    """Model for belief measurements during experiments."""
    
    experiment_id: str
    belief_statement: str
    condition_type: str
    belief_value: float
    timestamp_order: int
    measured_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        data['metadata'] = json.dumps(self.metadata or {})
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeliefMeasurement':
        """Create from dictionary (from database)."""
        # Remove database-specific fields that aren't in the model
        clean_data = {k: v for k, v in data.items() if k not in ['id']}
        
        if isinstance(clean_data['metadata'], str):
            clean_data['metadata'] = json.loads(clean_data['metadata'])
        return cls(**clean_data)


@dataclass
class InterventionRecord:
    """Model for intervention records."""
    
    experiment_id: str
    variable_name: str
    intervention_value: Any
    intervention_type: str = "do"
    applied_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        data['intervention_value'] = json.dumps(self.intervention_value)
        data['result'] = json.dumps(self.result or {})
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterventionRecord':
        """Create from dictionary (from database)."""
        # Remove database-specific fields that aren't in the model
        clean_data = {k: v for k, v in data.items() if k not in ['id']}
        
        if isinstance(clean_data['intervention_value'], str):
            clean_data['intervention_value'] = json.loads(clean_data['intervention_value'])
        if isinstance(clean_data['result'], str):
            clean_data['result'] = json.loads(clean_data['result'])
        return cls(**clean_data)


class CausalGraph(BaseModel):
    """Pydantic model for causal graph validation."""
    
    nodes: List[str] = Field(description="List of variable names")
    edges: List[Dict[str, str]] = Field(description="List of edges with from/to keys")
    node_types: Optional[Dict[str, str]] = Field(default=None, description="Variable types")
    mechanisms: Optional[Dict[str, Any]] = Field(default=None, description="Causal mechanisms")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_networkx_format(self) -> Dict[str, List[str]]:
        """Convert to NetworkX DAG format.
        
        Returns:
            Dictionary in {node: [parents]} format
        """
        dag = {node: [] for node in self.nodes}
        
        for edge in self.edges:
            child = edge['to']
            parent = edge['from']
            if child in dag:
                dag[child].append(parent)
        
        return dag
    
    @classmethod
    def from_networkx_format(cls, dag: Dict[str, List[str]]) -> 'CausalGraph':
        """Create from NetworkX DAG format.
        
        Args:
            dag: Dictionary in {node: [parents]} format
            
        Returns:
            CausalGraph instance
        """
        nodes = list(dag.keys())
        edges = []
        
        for child, parents in dag.items():
            for parent in parents:
                edges.append({"from": parent, "to": child})
        
        return cls(nodes=nodes, edges=edges)


@dataclass
class ExperimentResults:
    """Aggregated results from an experiment."""
    
    experiment: ExperimentModel
    beliefs: List[BeliefMeasurement]
    interventions: List[InterventionRecord]
    causal_analysis: Optional[Dict[str, Any]] = None
    
    def get_belief_trajectory(self, belief_statement: str) -> List[BeliefMeasurement]:
        """Get trajectory for a specific belief.
        
        Args:
            belief_statement: Belief to track
            
        Returns:
            List of belief measurements over time
        """
        return [
            belief for belief in self.beliefs
            if belief.belief_statement == belief_statement
        ]
    
    def get_interventions_by_variable(self, variable: str) -> List[InterventionRecord]:
        """Get all interventions on a specific variable.
        
        Args:
            variable: Variable name
            
        Returns:
            List of intervention records
        """
        return [
            intervention for intervention in self.interventions
            if intervention.variable_name == variable
        ]
    
    def to_export_format(self) -> Dict[str, Any]:
        """Export to dictionary format for analysis.
        
        Returns:
            Dictionary with all experiment data
        """
        return {
            "experiment": asdict(self.experiment),
            "beliefs": [asdict(belief) for belief in self.beliefs],
            "interventions": [asdict(intervention) for intervention in self.interventions],
            "causal_analysis": self.causal_analysis
        }