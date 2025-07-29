"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from causal_interface_gym.core import CausalEnvironment, InterventionUI


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
        "beliefs": np.random.random((75, 3))
    }


class MockLLMAgent:
    """Mock LLM agent for testing."""
    
    def __init__(self, responses: List[str] = None):
        self.responses = responses or ["Yes", "No", "Maybe"]
        self.call_count = 0
    
    def query(self, prompt: str) -> str:
        """Return mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def get_belief(self, variable: str) -> float:
        """Return mock belief probability."""
        return np.random.random()


@pytest.fixture
def mock_agent():
    """Create a mock LLM agent for testing."""
    return MockLLMAgent()


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


# Performance test configuration
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 5,
        "max_time": 5.0,
        "timer": "time.perf_counter"
    }