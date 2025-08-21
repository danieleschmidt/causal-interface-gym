"""Tests for quantum optimization functionality."""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.causal_interface_gym.quantum_optimization import (
    QuantumOptimizationEngine,
    OptimizationConfig,
    OptimizationStrategy,
    PerformanceProfile
)


@pytest.fixture
def quantum_config():
    """Create quantum optimization configuration."""
    return OptimizationConfig(
        strategy=OptimizationStrategy.QUANTUM_ANNEALING,
        max_qubits=10,
        max_iterations=100,
        convergence_threshold=1e-4,
        quantum_backend="simulator",
        classical_fallback=True,
        distributed_computing=False,
        gpu_acceleration=False
    )


@pytest.fixture
def optimization_engine(quantum_config):
    """Create quantum optimization engine."""
    return QuantumOptimizationEngine(quantum_config)


@pytest.fixture
def sample_causal_data():
    """Sample causal data for optimization."""
    np.random.seed(42)
    return {
        'variables': ['X1', 'X2', 'X3', 'X4', 'X5'],
        'data_matrix': np.random.randn(100, 5),
        'correlations': np.random.uniform(0.1, 0.8, (5, 5))
    }


class TestQuantumOptimizationEngine:
    """Test quantum optimization engine functionality."""
    
    def test_initialization(self, quantum_config):
        """Test engine initialization."""
        engine = QuantumOptimizationEngine(quantum_config)
        
        assert engine.config == quantum_config
        assert engine.quantum_circuit_cache == {}
        assert engine.optimization_history == []
        assert engine.performance_profiles == []
    
    def test_different_strategies(self, sample_causal_data):
        """Test different optimization strategies."""
        strategies = [
            OptimizationStrategy.QUANTUM_ANNEALING,
            OptimizationStrategy.VARIATIONAL_QUANTUM,
            OptimizationStrategy.HYBRID_CLASSICAL_QUANTUM
        ]
        
        for strategy in strategies:
            config = OptimizationConfig(
                strategy=strategy,
                max_qubits=8,
                max_iterations=50
            )
            engine = QuantumOptimizationEngine(config)
            
            # Should not raise an exception
            assert engine is not None
            assert engine.config.strategy == strategy
    
    @pytest.mark.asyncio
    async def test_causal_discovery_optimization(self, optimization_engine, sample_causal_data):
        """Test causal discovery optimization."""
        result = await optimization_engine.optimize_causal_discovery(sample_causal_data)
        
        # Check basic result structure
        assert 'optimized_causal_graph' in result
        assert 'optimization_strategy' in result
        assert 'solution_quality' in result
        assert 'performance_profile' in result
        
        # Validate causal graph
        causal_graph = result['optimized_causal_graph']
        assert isinstance(causal_graph, np.ndarray)
        assert causal_graph.shape[0] == causal_graph.shape[1]  # Square matrix
        
        # Validate solution quality
        quality = result['solution_quality']
        assert 0.0 <= quality <= 1.0
    
    @pytest.mark.asyncio
    async def test_problem_profiling(self, optimization_engine, sample_causal_data):
        """Test optimization problem profiling."""
        profile = await optimization_engine._profile_optimization_problem(sample_causal_data)
        
        assert 'problem_size' in profile
        assert 'n_variables' in profile
        assert 'n_samples' in profile
        assert 'quantum_advantage_expected' in profile
        assert 'sparsity_estimate' in profile
        
        assert profile['n_variables'] == 5
        assert profile['n_samples'] == 100
        assert isinstance(profile['quantum_advantage_expected'], bool)
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self, optimization_engine):
        """Test optimization strategy selection."""
        # Small problem
        small_profile = {
            'problem_size': 25,
            'quantum_advantage_expected': False,
            'sparsity_estimate': 0.4
        }
        strategy = await optimization_engine._select_optimization_strategy(small_profile)
        assert strategy in [OptimizationStrategy.HYBRID_CLASSICAL_QUANTUM, OptimizationStrategy.VARIATIONAL_QUANTUM]
        
        # Large problem
        large_profile = {
            'problem_size': 600,
            'quantum_advantage_expected': True,
            'sparsity_estimate': 0.2
        }
        strategy = await optimization_engine._select_optimization_strategy(large_profile)
        assert strategy in [OptimizationStrategy.QUANTUM_ANNEALING, OptimizationStrategy.VARIATIONAL_QUANTUM]
    
    @pytest.mark.asyncio
    async def test_quantum_annealing(self, optimization_engine, sample_causal_data):
        """Test quantum annealing optimization."""
        result = await optimization_engine._quantum_annealing_causal_discovery(sample_causal_data)
        
        assert 'causal_graph' in result
        assert 'solution_quality' in result
        assert 'converged' in result
        
        causal_graph = result['causal_graph']
        assert isinstance(causal_graph, np.ndarray)
        assert causal_graph.shape == (5, 5)
    
    @pytest.mark.asyncio
    async def test_variational_quantum(self, optimization_engine, sample_causal_data):
        """Test variational quantum optimization."""
        result = await optimization_engine._variational_quantum_causal_discovery(sample_causal_data)
        
        assert 'causal_graph' in result
        assert 'solution_quality' in result or 'optimal_energy' in result
        
        if 'causal_graph' in result:
            causal_graph = result['causal_graph']
            assert isinstance(causal_graph, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_hybrid_optimization(self, optimization_engine, sample_causal_data):
        """Test hybrid classical-quantum optimization."""
        result = await optimization_engine._hybrid_causal_discovery(
            sample_causal_data, OptimizationStrategy.HYBRID_CLASSICAL_QUANTUM
        )
        
        assert 'causal_graph' in result
        assert 'hybrid_optimization' in result
        assert result['hybrid_optimization'] is True
    
    def test_solution_quality_calculation(self, optimization_engine, sample_causal_data):
        """Test solution quality calculation."""
        # Create a test causal graph
        causal_graph = np.array([
            [0, 0.5, 0, 0, 0],
            [0, 0, 0.7, 0, 0],
            [0, 0, 0, 0.3, 0],
            [0, 0, 0, 0, 0.8],
            [0, 0, 0, 0, 0]
        ])
        
        quality = optimization_engine._calculate_solution_quality(causal_graph, sample_causal_data)
        
        assert 0.0 <= quality <= 1.0
        assert isinstance(quality, float)
    
    def test_acyclicity_check(self, optimization_engine):
        """Test acyclicity checking."""
        # Acyclic graph
        acyclic_graph = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        assert optimization_engine._is_acyclic(acyclic_graph)
        
        # Cyclic graph
        cyclic_graph = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        assert not optimization_engine._is_acyclic(cyclic_graph)
    
    def test_acyclicity_enforcement(self, optimization_engine):
        """Test acyclicity enforcement."""
        # Create cyclic graph
        cyclic_graph = np.array([
            [0, 0.5, 0],
            [0, 0, 0.7],
            [0.3, 0, 0]
        ])
        
        acyclic_graph = optimization_engine._enforce_acyclicity(cyclic_graph)
        
        assert optimization_engine._is_acyclic(acyclic_graph)
        assert acyclic_graph.shape == cyclic_graph.shape
    
    @pytest.mark.asyncio
    async def test_result_validation(self, optimization_engine, sample_causal_data):
        """Test result validation."""
        # Create mock result
        mock_result = {
            'causal_graph': np.array([
                [0, 0.5, 0],
                [0, 0, 0.7],
                [0.3, 0, 0]  # This creates a cycle
            ]),
            'solution_quality': 0.8
        }
        
        validated_result = await optimization_engine._validate_optimization_result(
            mock_result, sample_causal_data
        )
        
        assert 'causal_graph' in validated_result
        assert 'acyclicity_enforced' in validated_result
        assert optimization_engine._is_acyclic(validated_result['causal_graph'])
    
    @pytest.mark.asyncio
    async def test_performance_profiling(self, optimization_engine):
        """Test performance profiling."""
        import time
        start_time = time.time()
        
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        mock_result = {
            'causal_graph': np.random.random((10, 10)),
            'quantum_time': 0.05
        }
        
        profile = await optimization_engine._profile_optimization_performance(start_time, mock_result)
        
        assert isinstance(profile, PerformanceProfile)
        assert profile.execution_time > 0
        assert profile.memory_peak_mb > 0
        assert 0 <= profile.cpu_utilization_avg <= 1
        assert profile.optimization_effectiveness > 0
    
    def test_optimization_stats(self, optimization_engine):
        """Test optimization statistics."""
        # Initially no optimizations
        stats = optimization_engine.get_optimization_stats()
        assert stats['status'] == 'no_optimizations_performed'
        
        # Add some mock optimization history
        optimization_engine.optimization_history = [
            {
                'timestamp': time.time(),
                'strategy_used': 'quantum_annealing',
                'problem_size': 100,
                'optimization_time': 0.5,
                'solution_quality': 0.8,
                'quantum_advantage_achieved': True
            },
            {
                'timestamp': time.time(),
                'strategy_used': 'variational_quantum',
                'problem_size': 50,
                'optimization_time': 0.3,
                'solution_quality': 0.9,
                'quantum_advantage_achieved': False
            }
        ]
        
        stats = optimization_engine.get_optimization_stats()
        
        assert stats['total_optimizations'] == 2
        assert stats['average_execution_time'] == 0.4
        assert stats['average_solution_quality'] == 0.85
        assert stats['quantum_advantage_rate'] == 0.5
    
    @pytest.mark.asyncio
    async def test_benchmark_performance(self, optimization_engine):
        """Test performance benchmarking."""
        test_sizes = [5, 10, 15]
        
        benchmark_results = await optimization_engine.benchmark_optimization_performance(test_sizes)
        
        assert 'benchmark_results' in benchmark_results
        assert 'scaling_analysis' in benchmark_results
        assert 'performance_summary' in benchmark_results
        
        # Check all test sizes were benchmarked
        for size in test_sizes:
            assert f'size_{size}' in benchmark_results['benchmark_results']
            
        # Check scaling analysis
        scaling = benchmark_results['scaling_analysis']
        assert 'scaling_exponent' in scaling
        assert 'scaling_quality' in scaling
        
        # Check performance summary
        summary = benchmark_results['performance_summary']
        assert 'fastest_time' in summary
        assert 'slowest_time' in summary
        assert 'average_quality' in summary


class TestOptimizationConfig:
    """Test optimization configuration."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = OptimizationConfig(
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            max_qubits=20,
            max_iterations=1000
        )
        
        assert config.strategy == OptimizationStrategy.QUANTUM_ANNEALING
        assert config.max_qubits == 20
        assert config.max_iterations == 1000
        assert config.classical_fallback is True  # Default
    
    def test_config_defaults(self):
        """Test configuration defaults."""
        config = OptimizationConfig(strategy=OptimizationStrategy.VARIATIONAL_QUANTUM)
        
        assert config.max_qubits == 20
        assert config.convergence_threshold == 1e-6
        assert config.quantum_backend == "simulator"
        assert config.classical_fallback is True


class TestPerformanceProfile:
    """Test performance profiling."""
    
    def test_profile_creation(self):
        """Test performance profile creation."""
        profile = PerformanceProfile(
            execution_time=1.5,
            memory_peak_mb=500.0,
            cpu_utilization_avg=0.75,
            gpu_utilization_avg=0.60,
            quantum_gates_used=1000,
            energy_consumption_joules=50.0,
            scalability_score=0.8,
            optimization_effectiveness=2.5
        )
        
        assert profile.execution_time == 1.5
        assert profile.memory_peak_mb == 500.0
        assert profile.cpu_utilization_avg == 0.75
        assert profile.optimization_effectiveness == 2.5


@pytest.mark.asyncio
async def test_end_to_end_optimization(sample_causal_data):
    """Test end-to-end optimization workflow."""
    # Test different strategies
    strategies = [
        OptimizationStrategy.QUANTUM_ANNEALING,
        OptimizationStrategy.VARIATIONAL_QUANTUM,
        OptimizationStrategy.HYBRID_CLASSICAL_QUANTUM
    ]
    
    for strategy in strategies:
        config = OptimizationConfig(
            strategy=strategy,
            max_qubits=8,
            max_iterations=50,
            quantum_backend="simulator"
        )
        
        engine = QuantumOptimizationEngine(config)
        result = await engine.optimize_causal_discovery(sample_causal_data)
        
        # Should not fail and return valid results
        assert 'optimized_causal_graph' in result
        assert 'solution_quality' in result
        assert isinstance(result['solution_quality'], (int, float))
        assert 0.0 <= result['solution_quality'] <= 1.0


@pytest.mark.performance
@pytest.mark.asyncio
async def test_optimization_performance():
    """Test optimization performance with larger problems."""
    # Test with larger problem size
    large_data = {
        'variables': [f'X{i}' for i in range(20)],
        'data_matrix': np.random.randn(1000, 20),
        'correlations': np.random.uniform(0.1, 0.8, (20, 20))
    }
    
    config = OptimizationConfig(
        strategy=OptimizationStrategy.HYBRID_CLASSICAL_QUANTUM,
        max_qubits=10,
        max_iterations=100
    )
    
    engine = QuantumOptimizationEngine(config)
    
    import time
    start_time = time.time()
    result = await engine.optimize_causal_discovery(large_data)
    execution_time = time.time() - start_time
    
    # Should complete within reasonable time (adjust as needed)
    assert execution_time < 30.0  # 30 seconds max
    assert 'optimized_causal_graph' in result
    assert result['optimized_causal_graph'].shape == (20, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])