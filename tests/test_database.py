"""Tests for database functionality."""

import pytest
import tempfile
import os
from datetime import datetime

from causal_interface_gym.database import (
    DatabaseManager,
    CacheManager,
    ExperimentModel,
    BeliefMeasurement,
    InterventionRecord,
    ExperimentRepository,
    BeliefRepository,
    InterventionRepository,
    GraphRepository
)


@pytest.mark.database
class TestDatabaseManager:
    """Test database connection and management."""
    
    def test_sqlite_initialization(self):
        """Test SQLite database initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            db_url = f"sqlite:///{db_path}"
            db = DatabaseManager(db_url)
            
            assert db.db_type == "sqlite"
            assert os.path.exists(db_path)
            
            db.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_table_creation(self, test_database):
        """Test database table creation."""
        test_database.create_tables()
        
        # Test that tables exist by querying them
        result = test_database.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in result]
        
        required_tables = ["experiments", "belief_measurements", "intervention_records"]
        for table in required_tables:
            assert table in table_names
    
    def test_query_execution(self, test_database):
        """Test basic query execution."""
        test_database.create_tables()
        
        # Insert test data
        affected = test_database.execute_query(
            "INSERT INTO experiments (experiment_id, agent_type, causal_graph) VALUES (?, ?, ?)",
            ("test_001", "MockAgent", '{"nodes": ["A", "B"]}')
        )
        
        assert affected == 1
        
        # Query test data
        results = test_database.execute_query(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            ("test_001",)
        )
        
        assert len(results) == 1
        assert results[0][1] == "test_001"  # experiment_id column


@pytest.mark.cache
class TestCacheManager:
    """Test caching functionality."""
    
    def test_memory_cache_basic_operations(self, test_cache):
        """Test basic cache operations with memory backend."""
        # Set and get
        test_cache.set("test_key", "test_value")
        value = test_cache.get("test_key")
        assert value == "test_value"
        
        # Delete
        test_cache.delete("test_key")
        value = test_cache.get("test_key")
        assert value is None
    
    def test_cache_expiration(self, test_cache):
        """Test cache expiration."""
        test_cache.set("expiring_key", "expiring_value", ttl=1)
        
        # Should be available immediately
        value = test_cache.get("expiring_key")
        assert value == "expiring_value"
        
        # Should expire after TTL (simulated by clearing)
        test_cache.memory_cache.clear()
        value = test_cache.get("expiring_key")
        assert value is None
    
    def test_cache_computation(self, test_cache):
        """Test cached computation."""
        call_count = 0
        
        def expensive_computation():
            nonlocal call_count
            call_count += 1
            return "computed_result"
        
        # First call should compute
        result1 = test_cache.cache_computation("comp_key", expensive_computation)
        assert result1 == "computed_result"
        assert call_count == 1
        
        # Second call should use cache
        result2 = test_cache.cache_computation("comp_key", expensive_computation)
        assert result2 == "computed_result"
        assert call_count == 1  # Should not have called function again
    
    def test_causal_computation_caching(self, test_cache):
        """Test causal-specific caching."""
        graph = {"A": [], "B": ["A"]}
        
        def compute_effect():
            return {"effect": 0.7, "identifiable": True}
        
        result = test_cache.cache_causal_computation(graph, "A", "B", compute_effect)
        assert result["effect"] == 0.7
        
        # Should be cached for subsequent calls
        result2 = test_cache.cache_causal_computation(graph, "A", "B", lambda: {"should": "not_call"})
        assert result2["effect"] == 0.7
    
    def test_cache_stats(self, test_cache):
        """Test cache statistics."""
        test_cache.set("stat_key", "stat_value")
        
        stats = test_cache.get_cache_stats()
        
        assert "backend" in stats
        assert stats["backend"] == "memory"
        assert "memory_cache_size" in stats
        assert stats["memory_cache_size"] >= 1


@pytest.mark.database
class TestExperimentRepository:
    """Test experiment repository functionality."""
    
    def test_create_experiment(self, experiment_repo):
        """Test experiment creation."""
        experiment = ExperimentModel(
            experiment_id="test_exp_001",
            agent_type="MockAgent",
            causal_graph={"nodes": ["A", "B"], "edges": [{"from": "A", "to": "B"}]},
            metadata={"test": True}
        )
        
        exp_id = experiment_repo.create_experiment(experiment)
        assert exp_id == "test_exp_001"
    
    def test_get_experiment(self, experiment_repo):
        """Test experiment retrieval."""
        # Create experiment
        experiment = ExperimentModel(
            experiment_id="test_exp_002",
            agent_type="TestAgent",
            causal_graph={"nodes": ["X", "Y"]}
        )
        experiment_repo.create_experiment(experiment)
        
        # Retrieve experiment
        retrieved = experiment_repo.get_experiment("test_exp_002")
        
        assert retrieved is not None
        assert retrieved.experiment_id == "test_exp_002"
        assert retrieved.agent_type == "TestAgent"
        assert retrieved.causal_graph == {"nodes": ["X", "Y"]}
    
    def test_list_experiments(self, experiment_repo):
        """Test experiment listing."""
        # Create multiple experiments
        for i in range(3):
            experiment = ExperimentModel(
                experiment_id=f"list_test_{i}",
                agent_type="ListTestAgent",
                causal_graph={"nodes": [f"var_{i}"]}
            )
            experiment_repo.create_experiment(experiment)
        
        # List experiments
        experiments = experiment_repo.list_experiments(limit=10)
        
        list_test_experiments = [
            exp for exp in experiments 
            if exp.experiment_id.startswith("list_test_")
        ]
        
        assert len(list_test_experiments) == 3
    
    def test_delete_experiment(self, experiment_repo):
        """Test experiment deletion."""
        # Create experiment
        experiment = ExperimentModel(
            experiment_id="delete_test",
            agent_type="DeleteTestAgent",
            causal_graph={"nodes": ["A"]}
        )
        experiment_repo.create_experiment(experiment)
        
        # Verify it exists
        retrieved = experiment_repo.get_experiment("delete_test")
        assert retrieved is not None
        
        # Delete experiment
        deleted = experiment_repo.delete_experiment("delete_test")
        assert deleted is True
        
        # Verify it's gone
        retrieved = experiment_repo.get_experiment("delete_test")
        assert retrieved is None


@pytest.mark.database
class TestBeliefRepository:
    """Test belief repository functionality."""
    
    def test_record_belief(self, belief_repo, experiment_repo):
        """Test belief recording."""
        # Create experiment first
        experiment = ExperimentModel(
            experiment_id="belief_test",
            agent_type="BeliefTestAgent",
            causal_graph={"nodes": ["A", "B"]}
        )
        experiment_repo.create_experiment(experiment)
        
        # Record belief
        belief = BeliefMeasurement(
            experiment_id="belief_test",
            belief_statement="P(B|A)",
            condition_type="observational",
            belief_value=0.7,
            timestamp_order=1
        )
        
        belief_repo.record_belief(belief)
        
        # Retrieve beliefs
        beliefs = belief_repo.get_beliefs("belief_test")
        assert len(beliefs) == 1
        assert beliefs[0].belief_statement == "P(B|A)"
        assert beliefs[0].belief_value == 0.7
    
    def test_belief_trajectory(self, belief_repo, experiment_repo):
        """Test belief trajectory tracking."""
        # Create experiment
        experiment = ExperimentModel(
            experiment_id="trajectory_test",
            agent_type="TrajectoryTestAgent",
            causal_graph={"nodes": ["X", "Y"]}
        )
        experiment_repo.create_experiment(experiment)
        
        # Record belief trajectory
        belief_statement = "P(Y|X)"
        for i, condition in enumerate(["observational", "do(X=1)", "do(X=0)"]):
            belief = BeliefMeasurement(
                experiment_id="trajectory_test",
                belief_statement=belief_statement,
                condition_type=condition,
                belief_value=0.3 + i * 0.2,
                timestamp_order=i
            )
            belief_repo.record_belief(belief)
        
        # Get trajectory
        trajectory = belief_repo.get_belief_trajectory("trajectory_test", belief_statement)
        
        assert len(trajectory) == 3
        assert trajectory[0].condition_type == "observational"
        assert trajectory[1].condition_type == "do(X=1)"
        assert trajectory[2].condition_type == "do(X=0)"
        
        # Values should be in order
        values = [b.belief_value for b in trajectory]
        assert values == [0.3, 0.5, 0.7]


@pytest.mark.database
class TestInterventionRepository:
    """Test intervention repository functionality."""
    
    def test_record_intervention(self, intervention_repo, experiment_repo):
        """Test intervention recording."""
        # Create experiment
        experiment = ExperimentModel(
            experiment_id="intervention_test",
            agent_type="InterventionTestAgent",
            causal_graph={"nodes": ["A", "B"]}
        )
        experiment_repo.create_experiment(experiment)
        
        # Record intervention
        intervention = InterventionRecord(
            experiment_id="intervention_test",
            variable_name="A",
            intervention_value=True,
            intervention_type="do",
            result={"effect_on_B": 0.8}
        )
        
        intervention_repo.record_intervention(intervention)
        
        # Retrieve interventions
        interventions = intervention_repo.get_interventions("intervention_test")
        assert len(interventions) == 1
        assert interventions[0].variable_name == "A"
        assert interventions[0].intervention_value is True
        assert interventions[0].result["effect_on_B"] == 0.8


@pytest.mark.database
@pytest.mark.integration
class TestGraphRepository:
    """Test graph repository and complete data integration."""
    
    def test_complete_experiment_results(self, test_database, test_cache):
        """Test retrieving complete experiment results."""
        # Setup repositories
        exp_repo = ExperimentRepository(test_database, test_cache)
        belief_repo = BeliefRepository(test_database, test_cache)
        intervention_repo = InterventionRepository(test_database, test_cache)
        graph_repo = GraphRepository(test_database, test_cache)
        
        # Create experiment
        experiment = ExperimentModel(
            experiment_id="complete_test",
            agent_type="CompleteTestAgent",
            causal_graph={"nodes": ["A", "B"], "edges": [{"from": "A", "to": "B"}]}
        )
        exp_repo.create_experiment(experiment)
        
        # Add beliefs
        belief = BeliefMeasurement(
            experiment_id="complete_test",
            belief_statement="P(B|A)",
            condition_type="observational",
            belief_value=0.6,
            timestamp_order=1
        )
        belief_repo.record_belief(belief)
        
        # Add intervention
        intervention = InterventionRecord(
            experiment_id="complete_test",
            variable_name="A",
            intervention_value=1,
            result={"causal_effect": 0.8}
        )
        intervention_repo.record_intervention(intervention)
        
        # Get complete results
        results = graph_repo.get_experiment_results("complete_test")
        
        assert results is not None
        assert results.experiment.experiment_id == "complete_test"
        assert len(results.beliefs) == 1
        assert len(results.interventions) == 1
        
        # Test export
        exported = graph_repo.export_experiment_data("complete_test")
        assert exported is not None
        assert "experiment" in exported
        assert "beliefs" in exported
        assert "interventions" in exported