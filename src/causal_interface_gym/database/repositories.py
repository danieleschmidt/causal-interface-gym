"""Data repositories for causal interface gym."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .connection import DatabaseManager
from .models import (
    ExperimentModel,
    BeliefMeasurement,
    InterventionRecord,
    ExperimentResults
)
from .cache import CacheManager

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common functionality."""
    
    def __init__(self, db_manager: DatabaseManager, cache_manager: Optional[CacheManager] = None):
        """Initialize repository.
        
        Args:
            db_manager: Database manager instance
            cache_manager: Cache manager instance
        """
        self.db = db_manager
        self.cache = cache_manager
    
    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key.
        
        Args:
            prefix: Key prefix
            *args: Additional key components
            
        Returns:
            Cache key
        """
        return f"{prefix}:" + ":".join(str(arg) for arg in args)


class ExperimentRepository(BaseRepository):
    """Repository for experiment data."""
    
    def create_experiment(self, experiment: ExperimentModel) -> str:
        """Create new experiment.
        
        Args:
            experiment: Experiment data
            
        Returns:
            Experiment ID
        """
        data = experiment.to_dict()
        
        if self.db.db_type == 'postgresql':
            query = """
                INSERT INTO experiments (experiment_id, agent_type, causal_graph, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """
            params = (data['experiment_id'], data['agent_type'], 
                     data['causal_graph'], data['metadata'])
        else:
            query = """
                INSERT INTO experiments (experiment_id, agent_type, causal_graph, metadata)
                VALUES (?, ?, ?, ?)
            """
            params = (data['experiment_id'], data['agent_type'],
                     data['causal_graph'], data['metadata'])
        
        self.db.execute_query(query, params)
        
        # Invalidate cache
        if self.cache:
            self.cache.delete(f"experiment:{experiment.experiment_id}")
        
        logger.info(f"Created experiment: {experiment.experiment_id}")
        return experiment.experiment_id
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentModel]:
        """Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment data or None
        """
        cache_key = f"experiment:{experiment_id}"
        
        # Try cache first
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return ExperimentModel.from_dict(cached)
        
        # Query database
        if self.db.db_type == 'postgresql':
            query = "SELECT * FROM experiments WHERE experiment_id = %s"
        else:
            query = "SELECT * FROM experiments WHERE experiment_id = ?"
        
        results = self.db.execute_query(query, (experiment_id,))
        
        if results:
            row = dict(results[0]) if hasattr(results[0], 'keys') else results[0]
            experiment = ExperimentModel.from_dict(row)
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, experiment.to_dict(), ttl=3600)
            
            return experiment
        
        return None
    
    def list_experiments(self, limit: int = 100, offset: int = 0) -> List[ExperimentModel]:
        """List experiments.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of experiments
        """
        if self.db.db_type == 'postgresql':
            query = "SELECT * FROM experiments ORDER BY created_at DESC LIMIT %s OFFSET %s"
        else:
            query = "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ? OFFSET ?"
        
        results = self.db.execute_query(query, (limit, offset))
        
        experiments = []
        for row in results:
            row_dict = dict(row) if hasattr(row, 'keys') else row
            experiments.append(ExperimentModel.from_dict(row_dict))
        
        return experiments
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment and all related data.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if deleted, False if not found
        """
        # Delete related data first
        if self.db.db_type == 'postgresql':
            queries = [
                "DELETE FROM belief_measurements WHERE experiment_id = %s",
                "DELETE FROM intervention_records WHERE experiment_id = %s",
                "DELETE FROM experiments WHERE experiment_id = %s"
            ]
        else:
            queries = [
                "DELETE FROM belief_measurements WHERE experiment_id = ?",
                "DELETE FROM intervention_records WHERE experiment_id = ?",
                "DELETE FROM experiments WHERE experiment_id = ?"
            ]
        
        total_affected = 0
        for query in queries:
            affected = self.db.execute_query(query, (experiment_id,))
            total_affected += affected or 0
        
        # Clear cache
        if self.cache:
            self.cache.delete(f"experiment:{experiment_id}")
            self.cache.clear_cache(f"*{experiment_id}*")
        
        return total_affected > 0


class BeliefRepository(BaseRepository):
    """Repository for belief measurements."""
    
    def record_belief(self, belief: BeliefMeasurement) -> None:
        """Record a belief measurement.
        
        Args:
            belief: Belief measurement data
        """
        data = belief.to_dict()
        
        if self.db.db_type == 'postgresql':
            query = """
                INSERT INTO belief_measurements 
                (experiment_id, belief_statement, condition_type, belief_value, 
                 timestamp_order, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
        else:
            query = """
                INSERT INTO belief_measurements 
                (experiment_id, belief_statement, condition_type, belief_value,
                 timestamp_order, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """
        
        params = (
            data['experiment_id'], data['belief_statement'], data['condition_type'],
            data['belief_value'], data['timestamp_order'], data['metadata']
        )
        
        self.db.execute_query(query, params)
        
        # Invalidate experiment cache
        if self.cache:
            self.cache.delete(f"beliefs:{belief.experiment_id}")
    
    def get_beliefs(self, experiment_id: str) -> List[BeliefMeasurement]:
        """Get all beliefs for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            List of belief measurements
        """
        cache_key = f"beliefs:{experiment_id}"
        
        # Try cache first
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return [BeliefMeasurement.from_dict(b) for b in cached]
        
        # Query database
        if self.db.db_type == 'postgresql':
            query = """
                SELECT * FROM belief_measurements 
                WHERE experiment_id = %s 
                ORDER BY timestamp_order, measured_at
            """
        else:
            query = """
                SELECT * FROM belief_measurements 
                WHERE experiment_id = ? 
                ORDER BY timestamp_order, measured_at
            """
        
        results = self.db.execute_query(query, (experiment_id,))
        
        beliefs = []
        for row in results:
            row_dict = dict(row) if hasattr(row, 'keys') else row
            beliefs.append(BeliefMeasurement.from_dict(row_dict))
        
        # Cache result
        if self.cache:
            cached_data = [b.to_dict() for b in beliefs]
            self.cache.set(cache_key, cached_data, ttl=1800)  # 30 minutes
        
        return beliefs
    
    def get_belief_trajectory(self, experiment_id: str, 
                            belief_statement: str) -> List[BeliefMeasurement]:
        """Get belief trajectory for specific belief.
        
        Args:
            experiment_id: Experiment ID
            belief_statement: Belief statement to track
            
        Returns:
            List of belief measurements over time
        """
        if self.db.db_type == 'postgresql':
            query = """
                SELECT * FROM belief_measurements 
                WHERE experiment_id = %s AND belief_statement = %s
                ORDER BY timestamp_order, measured_at
            """
        else:
            query = """
                SELECT * FROM belief_measurements 
                WHERE experiment_id = ? AND belief_statement = ?
                ORDER BY timestamp_order, measured_at
            """
        
        results = self.db.execute_query(query, (experiment_id, belief_statement))
        
        trajectory = []
        for row in results:
            row_dict = dict(row) if hasattr(row, 'keys') else row
            trajectory.append(BeliefMeasurement.from_dict(row_dict))
        
        return trajectory


class InterventionRepository(BaseRepository):
    """Repository for intervention records."""
    
    def record_intervention(self, intervention: InterventionRecord) -> None:
        """Record an intervention.
        
        Args:
            intervention: Intervention record
        """
        data = intervention.to_dict()
        
        if self.db.db_type == 'postgresql':
            query = """
                INSERT INTO intervention_records 
                (experiment_id, variable_name, intervention_value, intervention_type, result)
                VALUES (%s, %s, %s, %s, %s)
            """
        else:
            query = """
                INSERT INTO intervention_records 
                (experiment_id, variable_name, intervention_value, intervention_type, result)
                VALUES (?, ?, ?, ?, ?)
            """
        
        params = (
            data['experiment_id'], data['variable_name'], data['intervention_value'],
            data['intervention_type'], data['result']
        )
        
        self.db.execute_query(query, params)
        
        # Invalidate cache
        if self.cache:
            self.cache.delete(f"interventions:{intervention.experiment_id}")
    
    def get_interventions(self, experiment_id: str) -> List[InterventionRecord]:
        """Get all interventions for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            List of intervention records
        """
        cache_key = f"interventions:{experiment_id}"
        
        # Try cache first
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return [InterventionRecord.from_dict(i) for i in cached]
        
        # Query database
        if self.db.db_type == 'postgresql':
            query = """
                SELECT * FROM intervention_records 
                WHERE experiment_id = %s 
                ORDER BY applied_at
            """
        else:
            query = """
                SELECT * FROM intervention_records 
                WHERE experiment_id = ? 
                ORDER BY applied_at
            """
        
        results = self.db.execute_query(query, (experiment_id,))
        
        interventions = []
        for row in results:
            row_dict = dict(row) if hasattr(row, 'keys') else row
            interventions.append(InterventionRecord.from_dict(row_dict))
        
        # Cache result
        if self.cache:
            cached_data = [i.to_dict() for i in interventions]
            self.cache.set(cache_key, cached_data, ttl=1800)  # 30 minutes
        
        return interventions


class GraphRepository(BaseRepository):
    """Repository for causal graph operations."""
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Get complete experiment results.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Complete experiment results or None
        """
        # Get experiment
        exp_repo = ExperimentRepository(self.db, self.cache)
        experiment = exp_repo.get_experiment(experiment_id)
        
        if not experiment:
            return None
        
        # Get beliefs and interventions
        belief_repo = BeliefRepository(self.db, self.cache)
        intervention_repo = InterventionRepository(self.db, self.cache)
        
        beliefs = belief_repo.get_beliefs(experiment_id)
        interventions = intervention_repo.get_interventions(experiment_id)
        
        return ExperimentResults(
            experiment=experiment,
            beliefs=beliefs,
            interventions=interventions
        )
    
    def export_experiment_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Export complete experiment data.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Exported data dictionary or None
        """
        results = self.get_experiment_results(experiment_id)
        
        if results:
            return results.to_export_format()
        
        return None