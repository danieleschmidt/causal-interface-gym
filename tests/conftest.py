"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import numpy as np
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

from causal_interface_gym.core import CausalEnvironment, InterventionUI
from causal_interface_gym.metrics import CausalMetrics, BeliefTracker
from causal_interface_gym.database import (
    DatabaseManager, 
    CacheManager,
    ExperimentRepository,
    BeliefRepository,
    InterventionRepository
)


@pytest.fixture
def sample_dag() -> Dict[str, List[str]]:
    """Sample DAG for testing."""
    return {
        "rain": [],
        "sprinkler": ["rain"],
        "wet_grass": ["rain", "sprinkler"],
        "slippery": ["wet_grass"]
    }


@pytest.fixture
def causal_environment(sample_dag) -> CausalEnvironment:
    """Create a sample causal environment for testing."""
    return CausalEnvironment.from_dag(sample_dag)


@pytest.fixture
def intervention_ui(causal_environment) -> InterventionUI:
    """Create an intervention UI for testing."""
    ui = InterventionUI(causal_environment)
    ui.add_intervention_button("sprinkler", "Toggle Sprinkler")
    ui.add_observation_panel("wet_grass", "Grass Status")
    return ui


@pytest.fixture
def complex_dag() -> Dict[str, List[str]]:
    """Complex DAG for integration testing."""
    return {
        "smoking": [],
        "genetics": [],
        "tar_deposits": ["smoking"],
        "cancer": ["smoking", "genetics", "tar_deposits"],
        "coughing": ["cancer"],
        "yellow_fingers": ["smoking"]
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return {
        "observations": np.random.random((100, 4)),
        "interventions": np.random.randint(0, 2, (50, 2)),
        "beliefs": np.random.random((75, 3)),
        "belief_trajectories": {
            "P(rain)": [0.3, 0.2, 0.1, 0.15],
            "P(wet_grass|do(sprinkler))": [0.9, 0.85, 0.88, 0.9]
        },
        "intervention_effects": {
            "sprinkler->wet_grass": 0.7,
            "rain->wet_grass": 0.8
        }
    }


class MockLLMAgent:
    """Mock LLM agent for testing."""
    
    def __init__(self, responses: List[str] = None, belief_responses: Dict[str, float] = None):
        self.responses = responses or ["Yes", "No", "Maybe"]
        self.belief_responses = belief_responses or {}
        self.call_count = 0
        self.belief_queries = []
        self.query_history = []
    
    def query(self, prompt: str) -> str:
        """Return mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        self.query_history.append((prompt, response))
        return response
    
    def get_belief(self, variable: str) -> float:
        """Return mock belief probability."""
        belief = self.belief_responses.get(variable, np.random.random())
        self.belief_queries.append((variable, belief))
        return belief
    
    def query_belief(self, belief_statement: str, condition: str) -> float:
        """Query belief with condition."""
        # Simulate different beliefs for different conditions
        if "do(" in condition:
            # Interventional beliefs might be different
            base_prob = self.belief_responses.get(belief_statement, 0.5)
            return np.clip(base_prob + np.random.normal(0, 0.1), 0, 1)
        else:
            # Observational beliefs
            return self.belief_responses.get(belief_statement, np.random.random())
    
    def reset(self):
        """Reset agent state."""
        self.call_count = 0
        self.belief_queries = []
        self.query_history = []


@pytest.fixture
def mock_agent():
    """Create a mock LLM agent for testing."""
    return MockLLMAgent()


@pytest.fixture
def deterministic_agent():
    """Create a deterministic mock agent for consistent testing."""
    belief_responses = {
        "P(rain|wet_grass)": 0.7,
        "P(slippery)": 0.3,
        "P(cancer|smoking)": 0.85,
        "P(wet_grass|do(sprinkler=on))": 0.9
    }
    return MockLLMAgent(belief_responses=belief_responses)


@pytest.fixture
def test_database():
    """Create test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    db_url = f"sqlite:///{db_path}"
    db_manager = DatabaseManager(db_url)
    db_manager.create_tables()
    
    yield db_manager
    
    # Cleanup
    db_manager.close()
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def test_cache():
    """Create test cache manager."""
    return CacheManager(redis_url=None)  # Use memory cache for testing


@pytest.fixture
def experiment_repo(test_database, test_cache):
    """Create experiment repository for testing."""
    return ExperimentRepository(test_database, test_cache)


@pytest.fixture
def belief_repo(test_database, test_cache):
    """Create belief repository for testing."""
    return BeliefRepository(test_database, test_cache)


@pytest.fixture
def intervention_repo(test_database, test_cache):
    """Create intervention repository for testing."""
    return InterventionRepository(test_database, test_cache)


@pytest.fixture
def belief_tracker(deterministic_agent):
    """Create belief tracker for testing."""
    return BeliefTracker(deterministic_agent)


@pytest.fixture
def causal_metrics():
    """Create causal metrics calculator."""
    return CausalMetrics()


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "database: marks tests that require database"
    )
    config.addinivalue_line(
        "markers", "cache: marks tests that require caching"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require LLM integration"
    )
    config.addinivalue_line(
        "markers", "ui: marks tests for UI components"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Auto-mark database tests
        if "database" in item.nodeid or "repo" in item.nodeid:
            item.add_marker(pytest.mark.database)
        
        # Auto-mark integration tests
        if "integration" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Auto-mark slow tests
        if "slow" in item.nodeid or item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)


# Performance test configuration
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 5,
        "max_time": 5.0,
        "timer": "time.perf_counter"
    }


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        "experiment_id": "test_exp_001",
        "agent": "MockAgent",
        "interventions": [("sprinkler", True), ("rain", False)],
        "initial_beliefs": {
            "P(rain|wet_grass)": 0.7,
            "P(slippery)": 0.3
        },
        "intervention_results": [
            {
                "intervention": ("sprinkler", True),
                "agent_beliefs": {
                    "P(rain|wet_grass)": 0.2,
                    "P(slippery)": 0.8
                }
            }
        ],
        "causal_analysis": {
            "causal_score": 0.85,
            "intervention_vs_observation": {
                "rain": {"score": 0.9, "difference": 0.5}
            }
        }
    }


@pytest.fixture
def environment_scenarios():
    """Multiple test scenarios for different environments."""
    return {
        "simple_chain": {
            "dag": {"A": [], "B": ["A"], "C": ["B"]},
            "expected_backdoors": {("A", "C"): []}
        },
        "confounded": {
            "dag": {"X": ["Z"], "Y": ["X", "Z"], "Z": []},
            "expected_backdoors": {("X", "Y"): [["X", "Z", "Y"]]}
        },
        "complex_medical": {
            "dag": {
                "smoking": [],
                "genetics": [],
                "tar_deposits": ["smoking"],
                "cancer": ["smoking", "genetics", "tar_deposits"],
                "coughing": ["cancer"]
            },
            "expected_backdoors": {
                ("smoking", "cancer"): [["smoking", "genetics", "cancer"]]
            }
        }
    }


# Test utilities
def create_test_graph(graph_type: str = "simple"):
    """Create test causal graphs."""
    graphs = {
        "simple": {"rain": [], "sprinkler": ["rain"], "wet_grass": ["rain", "sprinkler"]},
        "complex": {
            "smoking": [], "genetics": [], "tar": ["smoking"],
            "cancer": ["smoking", "genetics", "tar"], "cough": ["cancer"]
        },
        "confounded": {"treatment": ["confounder"], "outcome": ["treatment", "confounder"], "confounder": []}
    }
    return graphs.get(graph_type, graphs["simple"])


def assert_valid_probability(value: float, tolerance: float = 1e-6):
    """Assert that a value is a valid probability."""
    assert 0.0 - tolerance <= value <= 1.0 + tolerance, f"Invalid probability: {value}"


def assert_causal_effect_properties(effect_result: Dict[str, Any]):
    """Assert properties of causal effect computation."""
    assert "identifiable" in effect_result
    assert "strategy" in effect_result
    
    if effect_result["identifiable"]:
        assert "causal_effect" in effect_result
        if "adjustment_set" in effect_result:
            assert isinstance(effect_result["adjustment_set"], list)
        if "formula" in effect_result:
            assert isinstance(effect_result["formula"], str)
    else:
        assert "reason" in effect_result