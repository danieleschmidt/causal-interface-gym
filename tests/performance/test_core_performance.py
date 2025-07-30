"""Performance tests for core causal reasoning algorithms."""

import pytest
import numpy as np
from typing import Dict, Any

from causal_interface_gym.core import CausalEnvironment


class TestCausalEnvironmentPerformance:
    """Performance tests for CausalEnvironment core operations."""
    
    @pytest.mark.benchmark(group="environment_creation")
    def test_environment_creation_small(self, benchmark):
        """Benchmark small causal environment creation."""
        def create_small_env():
            return CausalEnvironment.from_dag({
                "A": [],
                "B": ["A"],
                "C": ["B"],
                "D": ["C"]
            })
        
        result = benchmark(create_small_env)
        assert result is not None
    
    @pytest.mark.benchmark(group="environment_creation")
    def test_environment_creation_medium(self, benchmark):
        """Benchmark medium causal environment creation."""
        def create_medium_env():
            # Create a 20-node DAG
            dag = {}
            for i in range(20):
                parents = [f"node_{j}" for j in range(i) if j % 3 == 0]
                dag[f"node_{i}"] = parents
            return CausalEnvironment.from_dag(dag)
        
        result = benchmark(create_medium_env)
        assert result is not None
    
    @pytest.mark.benchmark(group="environment_creation")  
    def test_environment_creation_large(self, benchmark):
        """Benchmark large causal environment creation."""
        def create_large_env():
            # Create a 100-node DAG
            dag = {}
            for i in range(100):
                parents = [f"node_{j}" for j in range(i) if j % 5 == 0][:3]  # Limit parents
                dag[f"node_{i}"] = parents
            return CausalEnvironment.from_dag(dag)
        
        result = benchmark(create_large_env)
        assert result is not None
    
    @pytest.mark.benchmark(group="intervention")
    def test_intervention_performance(self, benchmark):
        """Benchmark intervention operations."""
        env = CausalEnvironment.from_dag({
            "treatment": [],
            "confounder": [],
            "mediator": ["treatment", "confounder"],
            "outcome": ["mediator", "confounder"]
        })
        
        def run_intervention():
            return env.intervene(treatment=True)
        
        result = benchmark(run_intervention)
        assert result is not None
    
    @pytest.mark.benchmark(group="observation")
    def test_observation_performance(self, benchmark):
        """Benchmark observation operations."""
        env = CausalEnvironment.from_dag({
            "X": [],
            "Y": ["X"],
            "Z": ["Y"]
        })
        
        def run_observation():
            return env.observe(X=1.0)
        
        result = benchmark(run_observation)
        assert result is not None
    
    def test_memory_usage_large_environment(self, memory_profiler, benchmark_config):
        """Test memory usage with large environments."""
        if not memory_profiler:
            pytest.skip("Memory profiler not available")
        
        initial_memory = memory_profiler()
        
        # Create progressively larger environments
        for size in [10, 50, 100, 500]:
            dag = {}
            for i in range(size):
                parents = [f"node_{j}" for j in range(max(0, i-3), i)]
                dag[f"node_{i}"] = parents
            
            env = CausalEnvironment.from_dag(dag)
            current_memory = memory_profiler()
            memory_increase = current_memory - initial_memory
            
            # Assert memory usage is reasonable (< 100MB for 500 nodes)
            assert memory_increase < benchmark_config["memory_threshold_mb"], \
                f"Memory usage {memory_increase:.2f}MB exceeds threshold for size {size}"
    
    @pytest.mark.benchmark(group="batch_operations")
    def test_batch_intervention_performance(self, benchmark):
        """Benchmark batch intervention operations."""
        env = CausalEnvironment.from_dag({
            f"var_{i}": [f"var_{j}" for j in range(max(0, i-2), i)]
            for i in range(20)
        })
        
        def run_batch_interventions():
            interventions = {f"var_{i}": i % 2 for i in range(0, 20, 2)}
            return env.batch_intervene(interventions)
        
        result = benchmark(run_batch_interventions)
        assert result is not None


class TestCausalAlgorithmPerformance:
    """Performance tests for causal reasoning algorithms."""
    
    @pytest.mark.benchmark(group="backdoor_identification")
    def test_backdoor_identification_performance(self, benchmark):
        """Benchmark backdoor path identification."""
        # Create environment with known backdoor paths
        env = CausalEnvironment.from_dag({
            "treatment": [],
            "confounder": [],
            "mediator": ["treatment"],
            "outcome": ["treatment", "mediator", "confounder"],
            "collider": ["mediator", "outcome"]
        })
        
        def identify_backdoor_paths():
            return env.identify_backdoor_paths("treatment", "outcome")
        
        result = benchmark(identify_backdoor_paths)
        assert result is not None
    
    @pytest.mark.benchmark(group="causal_effect")
    def test_causal_effect_estimation_performance(self, benchmark):
        """Benchmark causal effect estimation."""
        env = CausalEnvironment.from_dag({
            "X": [],
            "M": ["X"],
            "Y": ["X", "M"],
            "C": []
        })
        
        # Add some sample data
        data = np.random.randn(1000, 4)
        env.add_data(data, columns=["X", "M", "Y", "C"])
        
        def estimate_causal_effect():
            return env.estimate_causal_effect("X", "Y")
        
        result = benchmark(estimate_causal_effect)
        assert result is not None
    
    @pytest.mark.benchmark(group="graph_algorithms")
    def test_topological_sort_performance(self, benchmark):
        """Benchmark topological sorting of causal graphs."""
        # Create a large DAG
        dag = {}
        for i in range(200):
            parents = [f"node_{j}" for j in range(i) if j % 7 == 0][:4]
            dag[f"node_{i}"] = parents
        
        env = CausalEnvironment.from_dag(dag)
        
        def run_topological_sort():
            return env.get_topological_order()
        
        result = benchmark(run_topological_sort)
        assert len(result) == 200
    
    def test_algorithm_scalability(self, timer, benchmark_config):
        """Test algorithm scalability with increasing graph sizes."""
        sizes = [10, 20, 50, 100]
        times = []
        
        for size in sizes:
            dag = {f"node_{i}": [f"node_{j}" for j in range(max(0, i-3), i)] 
                   for i in range(size)}
            
            env = CausalEnvironment.from_dag(dag)
            
            start_time = timer()
            _ = env.get_topological_order()
            end_time = timer()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            # Ensure reasonable scaling (should be roughly O(V + E))
            if len(times) > 1:
                time_ratio = times[-1] / times[-2]
                size_ratio = sizes[-1] / sizes[-2]
                
                # Time should scale roughly linearly with size
                assert time_ratio < size_ratio * 2, \
                    f"Algorithm scaling is worse than expected: {time_ratio:.2f}x time for {size_ratio:.2f}x size"


@pytest.mark.slow
class TestStressTests:
    """Stress tests for extreme scenarios."""
    
    def test_very_large_environment_creation(self):
        """Test creation of very large causal environments."""
        # Create a 1000-node sparse DAG
        dag = {}
        for i in range(1000):
            # Each node has at most 2 parents from previous 10 nodes
            parents = [f"node_{j}" for j in range(max(0, i-10), i) if j % 5 == 0][:2]
            dag[f"node_{i}"] = parents
        
        # This should complete within reasonable time
        env = CausalEnvironment.from_dag(dag)
        assert len(env.nodes) == 1000
    
    def test_dense_graph_performance(self):
        """Test performance with dense causal graphs."""
        # Create a moderately dense DAG
        size = 50
        dag = {}
        for i in range(size):
            # Each node connected to 30% of previous nodes
            parents = [f"node_{j}" for j in range(i) if j % 3 == 0]
            dag[f"node_{i}"] = parents
        
        env = CausalEnvironment.from_dag(dag)
        
        # Test that basic operations still work efficiently
        start = time.time()
        result = env.identify_backdoor_paths("node_10", "node_40")
        duration = time.time() - start
        
        assert duration < 5.0, f"Dense graph operation took too long: {duration:.2f}s"
        assert result is not None