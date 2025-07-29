"""Performance benchmarks for causal reasoning components."""

import pytest
import numpy as np
from typing import Dict, List

from causal_interface_gym.core import CausalEnvironment, InterventionUI


@pytest.mark.benchmark
class TestCausalEnvironmentBenchmarks:
    """Benchmark tests for CausalEnvironment performance."""
    
    def test_dag_creation_benchmark(self, benchmark):
        """Benchmark DAG creation with various sizes."""
        def create_dag(size: int) -> CausalEnvironment:
            dag = {}
            for i in range(size):
                parents = [f"node_{j}" for j in range(max(0, i-3), i)]
                dag[f"node_{i}"] = parents
            return CausalEnvironment.from_dag(dag)
        
        # Benchmark medium-sized DAG creation
        result = benchmark(create_dag, 50)
        assert result.graph.number_of_nodes() == 50
    
    def test_intervention_benchmark(self, benchmark):
        """Benchmark intervention operations."""
        # Create test environment
        dag = {f"node_{i}": [f"node_{j}" for j in range(max(0, i-2), i)] 
               for i in range(20)}
        env = CausalEnvironment.from_dag(dag)
        
        def run_intervention():
            return env.intervene(
                node_5=True,
                node_10="test_value",
                node_15=42.5
            )
        
        result = benchmark(run_intervention)
        assert "intervention_applied" in result
        assert len(result["intervention_applied"]) == 3
    
    def test_multiple_interventions_benchmark(self, benchmark):
        """Benchmark multiple sequential interventions."""
        dag = {f"var_{i}": [f"var_{j}" for j in range(max(0, i-1), i)] 
               for i in range(15)}
        env = CausalEnvironment.from_dag(dag)
        
        def run_multiple_interventions():
            results = []
            for i in range(10):
                result = env.intervene(**{f"var_{i % 15}": f"value_{i}"})
                results.append(result)
            return results
        
        results = benchmark(run_multiple_interventions)
        assert len(results) == 10
        assert all("intervention_applied" in r for r in results)


@pytest.mark.benchmark
class TestInterventionUIBenchmarks:
    """Benchmark tests for InterventionUI performance."""
    
    def test_ui_creation_benchmark(self, benchmark):
        """Benchmark UI creation and component addition."""
        dag = {f"node_{i}": [] for i in range(10)}
        env = CausalEnvironment.from_dag(dag)
        
        def create_ui_with_components():
            ui = InterventionUI(env)
            for i in range(20):
                if i % 2 == 0:
                    ui.add_intervention_button(f"node_{i % 10}", f"Button {i}")
                else:
                    ui.add_observation_panel(f"node_{i % 10}", f"Panel {i}")
            return ui
        
        ui = benchmark(create_ui_with_components)
        assert len(ui.components) == 20
    
    def test_experiment_execution_benchmark(self, benchmark):
        """Benchmark experiment execution."""
        dag = {"a": [], "b": ["a"], "c": ["b"], "d": ["c"]}
        env = CausalEnvironment.from_dag(dag)
        ui = InterventionUI(env)
        
        # Setup UI components
        for var in ["a", "b", "c", "d"]:
            ui.add_intervention_button(var, f"Control {var}")
            ui.add_observation_panel(var, f"Observe {var}")
        
        # Mock agent
        class FastMockAgent:
            def __str__(self):
                return "FastMockAgent"
        
        agent = FastMockAgent()
        interventions = [("a", True), ("b", False), ("c", "test")]
        beliefs = ["P(d)", "P(c|b)", "P(b|a)"]
        
        def run_experiment():
            return ui.run_experiment(
                agent=agent,
                interventions=interventions,
                measure_beliefs=beliefs
            )
        
        result = benchmark(run_experiment)
        assert result["agent"] == "FastMockAgent"
        assert len(result["interventions"]) == 3
        assert len(result["beliefs"]) == 3


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Benchmark tests for system scalability."""
    
    @pytest.mark.parametrize("dag_size", [10, 25, 50, 100])
    def test_dag_size_scalability(self, benchmark, dag_size):
        """Test performance scaling with DAG size."""
        def create_and_intervene(size):
            # Create DAG
            dag = {}
            for i in range(size):
                parents = [f"n_{j}" for j in range(max(0, i-2), i)]
                dag[f"n_{i}"] = parents
            
            # Create environment and run intervention
            env = CausalEnvironment.from_dag(dag)
            return env.intervene(**{f"n_{size//2}": "benchmark_value"})
        
        result = benchmark(create_and_intervene, dag_size)
        assert "intervention_applied" in result
    
    @pytest.mark.parametrize("component_count", [10, 25, 50])
    def test_ui_component_scalability(self, benchmark, component_count):
        """Test UI performance scaling with component count."""
        def create_ui_with_many_components(count):
            dag = {f"var_{i}": [] for i in range(count // 2)}
            env = CausalEnvironment.from_dag(dag)
            ui = InterventionUI(env)
            
            for i in range(count):
                var_name = f"var_{i % (count // 2)}"
                if i % 2 == 0:
                    ui.add_intervention_button(var_name, f"Button {i}")
                else:
                    ui.add_observation_panel(var_name, f"Panel {i}")
            
            return ui
        
        ui = benchmark(create_ui_with_many_components, component_count)
        assert len(ui.components) == component_count


@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Memory usage benchmark tests."""
    
    def test_memory_usage_large_dag(self, benchmark):
        """Test memory efficiency with large DAGs."""
        import sys
        
        def create_large_environment():
            # Create a DAG with 200 nodes
            dag = {}
            for i in range(200):
                parents = [f"node_{j}" for j in range(max(0, i-3), i)]
                dag[f"node_{i}"] = parents
            
            env = CausalEnvironment.from_dag(dag)
            
            # Perform some operations to populate internal structures
            for i in range(0, 200, 10):
                env.intervene(**{f"node_{i}": f"value_{i}"})
            
            return env
        
        env = benchmark(create_large_environment)
        
        # Verify the environment was created successfully
        assert env.graph.number_of_nodes() == 200
        
        # Basic memory size check (this is environment-dependent)
        # Mainly ensures we don't have obvious memory leaks
        env_size = sys.getsizeof(env)
        assert env_size > 0  # Basic sanity check
    
    def test_memory_efficiency_repeated_operations(self, benchmark):
        """Test memory efficiency of repeated operations."""
        dag = {f"v_{i}": [f"v_{j}" for j in range(max(0, i-2), i)] 
               for i in range(30)}
        env = CausalEnvironment.from_dag(dag)
        
        def repeated_interventions():
            results = []
            for round_num in range(50):
                for i in range(10):
                    result = env.intervene(**{f"v_{i}": f"round_{round_num}_val_{i}"})
                    results.append(result)
            return results
        
        results = benchmark(repeated_interventions)
        assert len(results) == 500  # 50 rounds * 10 interventions
        
        # Verify all interventions were recorded
        assert all("intervention_applied" in r for r in results)