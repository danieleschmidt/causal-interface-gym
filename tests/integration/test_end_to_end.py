"""End-to-end integration tests for causal reasoning workflows."""

import pytest
import numpy as np
from typing import Dict, List

from causal_interface_gym.core import CausalEnvironment, InterventionUI


@pytest.mark.integration
class TestCausalReasoningWorkflow:
    """Test complete causal reasoning workflows."""
    
    def test_full_experiment_pipeline(self, complex_dag, mock_agent):
        """Test complete experiment from setup to analysis."""
        # Setup environment
        env = CausalEnvironment.from_dag(complex_dag)
        ui = InterventionUI(env)
        
        # Configure interface
        ui.add_intervention_button("smoking", "Smoking Status")
        ui.add_observation_panel("cancer", "Cancer Risk")
        ui.add_observation_panel("coughing", "Coughing Symptoms")
        
        # Run experiment
        interventions = [("smoking", False), ("smoking", True)]
        beliefs = ["P(cancer)", "P(coughing|cancer)", "P(tar_deposits)"]
        
        result = ui.run_experiment(
            agent=mock_agent,
            interventions=interventions,
            measure_beliefs=beliefs
        )
        
        # Verify experiment completion
        assert result is not None
        assert "agent" in result
        assert "interventions" in result
        assert len(result["interventions"]) == 2
        assert "beliefs" in result
        assert len(result["beliefs"]) == 3
    
    def test_multiple_intervention_effects(self, complex_dag):
        """Test multiple interventions on complex causal structure."""
        env = CausalEnvironment.from_dag(complex_dag)
        
        # Apply multiple interventions
        result1 = env.intervene(smoking=True)
        result2 = env.intervene(smoking=False, genetics="high_risk")
        result3 = env.intervene(smoking=True, genetics="low_risk")
        
        # Verify interventions are tracked
        assert result1["intervention_applied"]["smoking"] is True
        assert result2["intervention_applied"]["smoking"] is False
        assert result2["intervention_applied"]["genetics"] == "high_risk"
        assert result3["intervention_applied"]["smoking"] is True
        assert result3["intervention_applied"]["genetics"] == "low_risk"
    
    def test_ui_component_integration(self, sample_dag):
        """Test integration between UI components and environment."""
        env = CausalEnvironment.from_dag(sample_dag)
        ui = InterventionUI(env)
        
        # Add various UI components
        ui.add_intervention_button("rain", "Weather Control")
        ui.add_intervention_button("sprinkler", "Sprinkler Control")
        ui.add_observation_panel("wet_grass", "Grass Moisture")
        ui.add_observation_panel("slippery", "Ground Condition")
        
        # Verify component configuration
        assert len(ui.components) == 4
        
        button_components = [c for c in ui.components if c["type"] == "button"]
        panel_components = [c for c in ui.components if c["type"] == "panel"]
        
        assert len(button_components) == 2
        assert len(panel_components) == 2
        
        # Verify variable mapping
        button_vars = {c["variable"] for c in button_components}
        panel_vars = {c["variable"] for c in panel_components}
        
        assert "rain" in button_vars
        assert "sprinkler" in button_vars
        assert "wet_grass" in panel_vars
        assert "slippery" in panel_vars


@pytest.mark.integration
class TestEnvironmentRobustness:
    """Test environment behavior under various conditions."""
    
    def test_large_dag_handling(self):
        """Test performance with large causal graphs."""
        # Create a large DAG with 50 nodes
        large_dag = {}
        for i in range(50):
            parents = []
            if i > 0:
                # Each node has 1-3 random parents from previous nodes
                num_parents = min(3, i)
                parent_indices = np.random.choice(i, num_parents, replace=False)
                parents = [f"node_{j}" for j in parent_indices]
            large_dag[f"node_{i}"] = parents
        
        # Create environment and verify it handles large DAG
        env = CausalEnvironment.from_dag(large_dag)
        assert env.graph.number_of_nodes() == 50
        
        # Test intervention on large graph
        result = env.intervene(node_25=True, node_40="test_value")
        assert "intervention_applied" in result
        assert len(result["intervention_applied"]) == 2
    
    def test_cyclic_dag_detection(self):
        """Test detection and handling of cyclic graphs."""
        # This would ideally detect cycles and raise appropriate errors
        # For now, we test that the system doesn't crash
        potentially_cyclic_dag = {
            "a": ["b"],
            "b": ["c"],
            "c": ["a"]  # Creates a cycle
        }
        
        # The system should handle this gracefully
        env = CausalEnvironment.from_dag(potentially_cyclic_dag)
        assert env.graph.number_of_nodes() == 3
        assert env.graph.number_of_edges() == 3
    
    def test_empty_and_single_node_dags(self):
        """Test edge cases with minimal DAGs."""
        # Empty DAG
        empty_env = CausalEnvironment.from_dag({})
        assert empty_env.graph.number_of_nodes() == 0
        
        # Single node DAG
        single_env = CausalEnvironment.from_dag({"single_node": []})
        assert single_env.graph.number_of_nodes() == 1
        assert single_env.graph.number_of_edges() == 0
        
        # Test intervention on single node
        result = single_env.intervene(single_node="test_value")
        assert result["intervention_applied"]["single_node"] == "test_value"


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""
    
    def test_intervention_performance_scaling(self, benchmark_config):
        """Test intervention performance with increasing complexity."""
        dag_sizes = [10, 25, 50]
        
        for size in dag_sizes:
            # Create DAG of specified size
            dag = {}
            for i in range(size):
                parents = [f"node_{j}" for j in range(max(0, i-2), i)]
                dag[f"node_{i}"] = parents
            
            env = CausalEnvironment.from_dag(dag)
            
            # Time multiple interventions
            import time
            start_time = time.perf_counter()
            
            for i in range(10):
                env.intervene(**{f"node_{i % size}": f"value_{i}"})
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Performance should scale reasonably
            assert duration < 1.0, f"Interventions too slow for size {size}: {duration}s"
    
    def test_ui_component_scaling(self):
        """Test UI performance with many components."""
        env = CausalEnvironment.from_dag({"root": []})
        ui = InterventionUI(env)
        
        # Add many components
        for i in range(100):
            if i % 2 == 0:
                ui.add_intervention_button(f"var_{i}", f"Button {i}")
            else:
                ui.add_observation_panel(f"var_{i}", f"Panel {i}")
        
        assert len(ui.components) == 100
        
        # Verify component access is efficient
        import time
        start_time = time.perf_counter()
        
        button_count = len([c for c in ui.components if c["type"] == "button"])
        panel_count = len([c for c in ui.components if c["type"] == "panel"])
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        assert button_count == 50
        assert panel_count == 50
        assert duration < 0.1, f"Component access too slow: {duration}s"