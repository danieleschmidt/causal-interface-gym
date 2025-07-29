"""Tests for core causal environment functionality."""

import pytest
from causal_interface_gym.core import CausalEnvironment, InterventionUI


class TestCausalEnvironment:
    """Test causal environment functionality."""
    
    def test_environment_creation(self):
        """Test basic environment creation."""
        env = CausalEnvironment()
        assert env.graph.number_of_nodes() == 0
    
    def test_from_dag(self):
        """Test environment creation from DAG."""
        dag = {
            "rain": [],
            "sprinkler": ["rain"],
            "wet_grass": ["rain", "sprinkler"]
        }
        env = CausalEnvironment.from_dag(dag)
        assert env.graph.number_of_nodes() == 3
        assert env.graph.number_of_edges() == 3
    
    def test_intervention(self):
        """Test intervention functionality."""
        env = CausalEnvironment()
        result = env.intervene(sprinkler=True)
        assert "intervention_applied" in result
        assert result["intervention_applied"]["sprinkler"] is True


class TestInterventionUI:
    """Test intervention UI functionality."""
    
    def test_ui_creation(self):
        """Test UI creation."""
        env = CausalEnvironment()
        ui = InterventionUI(env)
        assert ui.environment == env
        assert len(ui.components) == 0
    
    def test_add_components(self):
        """Test adding UI components."""
        env = CausalEnvironment()
        ui = InterventionUI(env)
        
        ui.add_intervention_button("sprinkler", "Toggle Sprinkler")
        ui.add_observation_panel("wet_grass", "Grass Status")
        
        assert len(ui.components) == 2
        assert ui.components[0]["type"] == "button"
        assert ui.components[1]["type"] == "panel"
    
    def test_run_experiment(self):
        """Test experiment execution."""
        env = CausalEnvironment()
        ui = InterventionUI(env)
        
        result = ui.run_experiment(
            agent="test_agent",
            interventions=[("sprinkler", True)],
            measure_beliefs=["P(rain)", "P(wet_grass)"]
        )
        
        assert "agent" in result
        assert "interventions" in result
        assert "beliefs" in result