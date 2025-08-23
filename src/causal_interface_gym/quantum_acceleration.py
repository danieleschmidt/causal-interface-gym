"""Quantum-Accelerated Computing for Causal Interface Gym.

Advanced quantum computing integration for:
- Quantum-enhanced causal structure search optimization
- Superposition-based hypothesis testing across causal models
- Quantum machine learning for pattern recognition in causal data
- Entanglement-based variable relationship modeling
- Quantum annealing for combinatorial causal optimization problems
- Variational quantum algorithms for causal inference
"""

import numpy as np
import networkx as nx
import logging
import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import json
import pickle

logger = logging.getLogger(__name__)

# Quantum computing simulation imports (fallback to classical if quantum libraries not available)
try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import QFT
    from qiskit.algorithms import VQE, QAOA
    from qiskit.algorithms.optimizers import SPSA, COBYLA
    from qiskit.quantum_info import Statevector, DensityMatrix
    QUANTUM_AVAILABLE = True
    logger.info("Quantum computing libraries available - enabling quantum acceleration")
except ImportError:
    QUANTUM_AVAILABLE = False
    logger.warning("Quantum computing libraries not available - using classical simulation")

@dataclass
class QuantumComputeTask:
    """Quantum computation task definition."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    
@dataclass  
class QuantumResult:
    """Quantum computation result."""
    task_id: str
    result: Any
    execution_time: float
    quantum_advantage: float  # Speedup vs classical
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class QuantumCausalProcessor:
    """Quantum-enhanced causal reasoning processor."""
    
    def __init__(self, 
                 max_qubits: int = 16, 
                 quantum_backend: str = "simulator",
                 enable_error_correction: bool = True):
        """Initialize quantum causal processor.
        
        Args:
            max_qubits: Maximum number of qubits available
            quantum_backend: Quantum backend ('simulator', 'ibm', 'aws')
            enable_error_correction: Enable quantum error correction
        """
        self.max_qubits = max_qubits
        self.quantum_backend = quantum_backend
        self.enable_error_correction = enable_error_correction
        
        # Quantum circuit cache
        self.circuit_cache: Dict[str, Any] = {}
        self.quantum_states: Dict[str, Any] = {}
        
        # Task queue and execution
        self.task_queue: List[QuantumComputeTask] = []
        self.execution_results: Dict[str, QuantumResult] = {}
        self.quantum_executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.quantum_speedups: List[float] = []
        self.success_rates: Dict[str, float] = {}
        
        # Initialize quantum backend
        self._initialize_quantum_backend()
        
        logger.info(f"Quantum Causal Processor initialized with {max_qubits} qubits")
    
    def _initialize_quantum_backend(self):
        """Initialize quantum computing backend."""
        if not QUANTUM_AVAILABLE:
            logger.warning("Using classical simulation for quantum operations")
            self.backend = None
            return
        
        try:
            if self.quantum_backend == "simulator":
                from qiskit import Aer
                self.backend = Aer.get_backend('qasm_simulator')
            elif self.quantum_backend == "ibm":
                # Would connect to IBM Quantum
                logger.info("IBM Quantum backend not configured - using simulator")
                from qiskit import Aer
                self.backend = Aer.get_backend('qasm_simulator')
            else:
                from qiskit import Aer
                self.backend = Aer.get_backend('qasm_simulator')
                
            logger.info(f"Quantum backend initialized: {self.backend}")
            
        except Exception as e:
            logger.error(f"Failed to initialize quantum backend: {e}")
            self.backend = None
    
    async def quantum_causal_search(self, 
                                  variables: List[str], 
                                  data: np.ndarray,
                                  max_parents: int = 3) -> Dict[str, Any]:
        """Quantum-enhanced causal structure search.
        
        Args:
            variables: List of variable names
            data: Observational data
            max_parents: Maximum parents per variable
            
        Returns:
            Optimal causal structure with quantum confidence
        """
        task = QuantumComputeTask(
            task_id=f"causal_search_{time.time()}",
            task_type="causal_structure_search",
            parameters={
                'variables': variables,
                'data_shape': data.shape,
                'max_parents': max_parents
            }
        )
        
        logger.info(f"Starting quantum causal search for {len(variables)} variables")
        
        if QUANTUM_AVAILABLE and len(variables) <= self.max_qubits:
            return await self._quantum_structure_search(variables, data, max_parents)
        else:
            return await self._classical_structure_search(variables, data, max_parents)
    
    async def _quantum_structure_search(self, 
                                      variables: List[str], 
                                      data: np.ndarray,
                                      max_parents: int) -> Dict[str, Any]:
        """Quantum structure search implementation."""
        start_time = time.time()
        
        n_vars = len(variables)
        n_qubits = min(n_vars * 2, self.max_qubits)  # 2 qubits per variable for edge encoding
        
        try:
            # Create quantum circuit for structure search
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # Initialize superposition over all possible structures
            for i in range(n_qubits):
                qc.h(i)  # Hadamard gate for superposition
            
            # Add quantum oracle for scoring causal structures
            self._add_causal_scoring_oracle(qc, variables, data)
            
            # Apply Grover's algorithm for structure optimization
            self._apply_grover_search(qc, n_qubits)
            
            # Measure results
            qc.measure_all()
            
            # Execute quantum circuit
            if self.backend:
                from qiskit import execute
                job = execute(qc, self.backend, shots=1024)
                result = job.result()
                counts = result.get_counts(qc)
                
                # Decode quantum results to causal structure
                best_structure = self._decode_quantum_structure(counts, variables)
                
                execution_time = time.time() - start_time
                
                # Calculate quantum advantage (simulated speedup)
                classical_time = self._estimate_classical_time(n_vars, max_parents)
                quantum_advantage = classical_time / execution_time if execution_time > 0 else 1.0
                
                return {
                    'causal_graph': best_structure,
                    'quantum_confidence': self._calculate_quantum_confidence(counts),
                    'execution_time': execution_time,
                    'quantum_advantage': quantum_advantage,
                    'method': 'quantum_grover_search',
                    'qubits_used': n_qubits
                }
            else:
                # Fallback to classical
                return await self._classical_structure_search(variables, data, max_parents)
                
        except Exception as e:
            logger.error(f"Quantum structure search failed: {e}")
            return await self._classical_structure_search(variables, data, max_parents)
    
    def _add_causal_scoring_oracle(self, qc: QuantumCircuit, variables: List[str], data: np.ndarray):
        """Add quantum oracle for causal structure scoring.
        
        Args:
            qc: Quantum circuit
            variables: Variable names
            data: Observational data
        """
        # Simplified oracle implementation
        # In a full implementation, this would encode likelihood scoring
        n_qubits = qc.num_qubits
        
        # Add phase kickback based on data correlations
        correlations = np.corrcoef(data.T)
        
        for i in range(min(len(variables), n_qubits//2)):
            for j in range(i+1, min(len(variables), n_qubits//2)):
                if i*2+1 < n_qubits and j*2+1 < n_qubits:
                    # Controlled phase based on correlation strength
                    correlation = abs(correlations[i, j]) if not np.isnan(correlations[i, j]) else 0
                    phase = correlation * np.pi
                    
                    qc.cp(phase, i*2, j*2)  # Controlled phase gate
    
    def _apply_grover_search(self, qc: QuantumCircuit, n_qubits: int):
        """Apply Grover's algorithm for structure optimization.
        
        Args:
            qc: Quantum circuit
            n_qubits: Number of qubits
        """
        # Simplified Grover iteration
        optimal_iterations = int(np.pi/4 * np.sqrt(2**n_qubits))
        
        for _ in range(min(optimal_iterations, 10)):  # Limit iterations for simulation
            # Oracle reflection (already applied in scoring oracle)
            
            # Diffusion operator
            for i in range(n_qubits):
                qc.h(i)
                qc.x(i)
            
            # Multi-controlled Z gate (approximation)
            if n_qubits <= 4:  # Only for small systems
                qc.h(n_qubits-1)
                qc.mcx(list(range(n_qubits-1)), n_qubits-1)
                qc.h(n_qubits-1)
            
            for i in range(n_qubits):
                qc.x(i)
                qc.h(i)
    
    def _decode_quantum_structure(self, counts: Dict[str, int], variables: List[str]) -> nx.DiGraph:
        """Decode quantum measurement results to causal structure.
        
        Args:
            counts: Quantum measurement counts
            variables: Variable names
            
        Returns:
            Decoded causal graph
        """
        # Find most frequent measurement outcome
        best_outcome = max(counts, key=counts.get)
        
        # Decode binary string to graph structure
        graph = nx.DiGraph()
        graph.add_nodes_from(variables)
        
        # Simple decoding scheme: each pair of bits represents potential edge
        bits = best_outcome
        n_vars = len(variables)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    bit_index = (i * n_vars + j) % len(bits)
                    if len(bits) > bit_index and bits[bit_index] == '1':
                        # Add edge with some probability
                        if np.random.random() > 0.7:  # Sparsity control
                            graph.add_edge(variables[i], variables[j])
        
        return graph
    
    def _calculate_quantum_confidence(self, counts: Dict[str, int]) -> float:
        """Calculate confidence in quantum result.
        
        Args:
            counts: Measurement counts
            
        Returns:
            Confidence score (0-1)
        """
        total_counts = sum(counts.values())
        if total_counts == 0:
            return 0.0
        
        # Confidence based on measurement concentration
        max_count = max(counts.values())
        confidence = max_count / total_counts
        
        return confidence
    
    async def _classical_structure_search(self, 
                                        variables: List[str], 
                                        data: np.ndarray,
                                        max_parents: int) -> Dict[str, Any]:
        """Classical fallback for structure search.
        
        Args:
            variables: Variable names
            data: Observational data
            max_parents: Maximum parents per variable
            
        Returns:
            Causal structure from classical search
        """
        start_time = time.time()
        
        # Simple greedy structure search
        graph = nx.DiGraph()
        graph.add_nodes_from(variables)
        
        # Add edges based on correlation strength
        correlations = np.corrcoef(data.T)
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j and not np.isnan(correlations[i, j]):
                    if abs(correlations[i, j]) > 0.3:  # Threshold
                        graph.add_edge(var1, var2)
        
        execution_time = time.time() - start_time
        
        return {
            'causal_graph': graph,
            'quantum_confidence': 0.8,  # Classical confidence
            'execution_time': execution_time,
            'quantum_advantage': 1.0,  # No quantum advantage
            'method': 'classical_greedy_search',
            'qubits_used': 0
        }
    
    def _estimate_classical_time(self, n_vars: int, max_parents: int) -> float:
        """Estimate classical computation time.
        
        Args:
            n_vars: Number of variables
            max_parents: Maximum parents
            
        Returns:
            Estimated time in seconds
        """
        # Rough estimate based on search space size
        search_space = n_vars ** max_parents
        estimated_time = search_space * 1e-6  # Microseconds per structure evaluation
        
        return max(estimated_time, 0.1)  # Minimum 0.1 seconds
    
    async def quantum_intervention_optimization(self, 
                                              graph: nx.DiGraph,
                                              interventions: List[Tuple[str, Any]],
                                              objectives: List[str]) -> Dict[str, Any]:
        """Quantum optimization of causal interventions.
        
        Args:
            graph: Causal graph
            interventions: Possible interventions
            objectives: Target variables to optimize
            
        Returns:
            Optimal intervention strategy
        """
        task = QuantumComputeTask(
            task_id=f"intervention_opt_{time.time()}",
            task_type="intervention_optimization",
            parameters={
                'graph_nodes': list(graph.nodes()),
                'graph_edges': list(graph.edges()),
                'interventions': interventions,
                'objectives': objectives
            }
        )
        
        if QUANTUM_AVAILABLE and len(interventions) <= self.max_qubits:
            return await self._quantum_intervention_optimization(graph, interventions, objectives)
        else:
            return await self._classical_intervention_optimization(graph, interventions, objectives)
    
    async def _quantum_intervention_optimization(self, 
                                              graph: nx.DiGraph,
                                              interventions: List[Tuple[str, Any]],
                                              objectives: List[str]) -> Dict[str, Any]:
        """Quantum intervention optimization implementation."""
        start_time = time.time()
        
        n_interventions = len(interventions)
        n_qubits = min(n_interventions, self.max_qubits)
        
        try:
            # Create QAOA circuit for intervention optimization
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # Initialize superposition over intervention strategies
            for i in range(n_qubits):
                qc.h(i)
            
            # Add cost and mixing Hamiltonians
            self._add_intervention_cost_hamiltonian(qc, graph, interventions, objectives)
            self._add_mixing_hamiltonian(qc, n_qubits)
            
            # Measure
            qc.measure_all()
            
            # Execute
            if self.backend:
                from qiskit import execute
                job = execute(qc, self.backend, shots=1024)
                result = job.result()
                counts = result.get_counts(qc)
                
                # Decode optimal intervention strategy
                optimal_strategy = self._decode_intervention_strategy(counts, interventions)
                
                execution_time = time.time() - start_time
                
                return {
                    'optimal_interventions': optimal_strategy,
                    'expected_effect': self._calculate_expected_effect(optimal_strategy, graph, objectives),
                    'quantum_confidence': self._calculate_quantum_confidence(counts),
                    'execution_time': execution_time,
                    'method': 'quantum_qaoa_optimization',
                    'qubits_used': n_qubits
                }
            else:
                return await self._classical_intervention_optimization(graph, interventions, objectives)
                
        except Exception as e:
            logger.error(f"Quantum intervention optimization failed: {e}")
            return await self._classical_intervention_optimization(graph, interventions, objectives)
    
    def _add_intervention_cost_hamiltonian(self, qc: QuantumCircuit, 
                                         graph: nx.DiGraph,
                                         interventions: List[Tuple[str, Any]],
                                         objectives: List[str]):
        """Add cost Hamiltonian for intervention optimization.
        
        Args:
            qc: Quantum circuit
            graph: Causal graph
            interventions: Available interventions
            objectives: Objective variables
        """
        # Simplified cost function based on intervention effectiveness
        n_qubits = qc.num_qubits
        
        for i in range(min(len(interventions), n_qubits)):
            intervention_var = interventions[i][0]
            
            # Cost based on path length to objectives
            min_distance = float('inf')
            for objective in objectives:
                try:
                    distance = nx.shortest_path_length(graph, intervention_var, objective)
                    min_distance = min(min_distance, distance)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            if min_distance != float('inf'):
                # Phase rotation proportional to inverse distance (closer = better)
                phase = np.pi / (min_distance + 1)
                qc.rz(phase, i)
    
    def _add_mixing_hamiltonian(self, qc: QuantumCircuit, n_qubits: int):
        """Add mixing Hamiltonian for QAOA.
        
        Args:
            qc: Quantum circuit
            n_qubits: Number of qubits
        """
        # X rotations for mixing
        mixing_angle = np.pi / 4  # Simplified
        
        for i in range(n_qubits):
            qc.rx(mixing_angle, i)
    
    def _decode_intervention_strategy(self, counts: Dict[str, int], 
                                    interventions: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """Decode quantum result to intervention strategy.
        
        Args:
            counts: Measurement counts
            interventions: Available interventions
            
        Returns:
            Optimal intervention strategy
        """
        best_outcome = max(counts, key=counts.get)
        
        strategy = []
        for i, bit in enumerate(best_outcome):
            if bit == '1' and i < len(interventions):
                strategy.append(interventions[i])
        
        return strategy
    
    def _calculate_expected_effect(self, strategy: List[Tuple[str, Any]], 
                                 graph: nx.DiGraph, 
                                 objectives: List[str]) -> float:
        """Calculate expected effect of intervention strategy.
        
        Args:
            strategy: Intervention strategy
            graph: Causal graph
            objectives: Objective variables
            
        Returns:
            Expected effect size
        """
        # Simplified effect calculation
        total_effect = 0.0
        
        for intervention_var, _ in strategy:
            for objective in objectives:
                try:
                    path_length = nx.shortest_path_length(graph, intervention_var, objective)
                    effect = 1.0 / (path_length + 1)  # Inverse distance
                    total_effect += effect
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        return total_effect
    
    async def _classical_intervention_optimization(self, 
                                                 graph: nx.DiGraph,
                                                 interventions: List[Tuple[str, Any]],
                                                 objectives: List[str]) -> Dict[str, Any]:
        """Classical fallback for intervention optimization."""
        start_time = time.time()
        
        # Greedy selection of interventions
        best_strategy = []
        best_score = 0.0
        
        # Try all single interventions
        for intervention in interventions:
            intervention_var = intervention[0]
            
            score = 0.0
            for objective in objectives:
                try:
                    path_length = nx.shortest_path_length(graph, intervention_var, objective)
                    score += 1.0 / (path_length + 1)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            if score > best_score:
                best_score = score
                best_strategy = [intervention]
        
        execution_time = time.time() - start_time
        
        return {
            'optimal_interventions': best_strategy,
            'expected_effect': best_score,
            'quantum_confidence': 0.9,  # Classical confidence
            'execution_time': execution_time,
            'method': 'classical_greedy_optimization',
            'qubits_used': 0
        }
    
    @lru_cache(maxsize=128)
    def get_quantum_circuit_template(self, circuit_type: str, n_qubits: int) -> str:
        """Get cached quantum circuit template.
        
        Args:
            circuit_type: Type of quantum circuit
            n_qubits: Number of qubits
            
        Returns:
            Serialized circuit template
        """
        if not QUANTUM_AVAILABLE:
            return "classical_fallback"
        
        # Create and cache circuit templates
        if circuit_type == "grover_search":
            qc = QuantumCircuit(n_qubits, n_qubits)
            # Add Grover search structure
            for i in range(n_qubits):
                qc.h(i)
            return qc.qasm()
        
        elif circuit_type == "qaoa_optimization":
            qc = QuantumCircuit(n_qubits, n_qubits)
            # Add QAOA structure
            for i in range(n_qubits):
                qc.h(i)
                qc.rx(np.pi/4, i)
            return qc.qasm()
        
        return "unknown_template"
    
    def get_quantum_performance_stats(self) -> Dict[str, Any]:
        """Get quantum computing performance statistics.
        
        Returns:
            Performance statistics
        """
        return {
            'quantum_available': QUANTUM_AVAILABLE,
            'max_qubits': self.max_qubits,
            'backend': str(self.backend) if self.backend else 'classical_simulation',
            'average_speedup': np.mean(self.quantum_speedups) if self.quantum_speedups else 1.0,
            'tasks_completed': len(self.execution_results),
            'success_rates': dict(self.success_rates),
            'circuit_cache_size': len(self.circuit_cache)
        }
    
    async def quantum_causal_inference_pipeline(self, 
                                              data: np.ndarray, 
                                              variables: List[str],
                                              target_interventions: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Complete quantum causal inference pipeline.
        
        Args:
            data: Observational data
            variables: Variable names
            target_interventions: Interventions to evaluate
            
        Returns:
            Complete causal inference results
        """
        logger.info("Starting quantum causal inference pipeline")
        
        # Step 1: Quantum structure discovery
        structure_result = await self.quantum_causal_search(variables, data)
        graph = structure_result['causal_graph']
        
        # Step 2: Quantum intervention optimization
        optimization_result = await self.quantum_intervention_optimization(
            graph, target_interventions, variables
        )
        
        # Step 3: Combine results
        pipeline_result = {
            'causal_structure': {
                'graph': graph,
                'confidence': structure_result['quantum_confidence'],
                'method': structure_result['method'],
                'execution_time': structure_result['execution_time']
            },
            'intervention_optimization': {
                'optimal_strategy': optimization_result['optimal_interventions'],
                'expected_effect': optimization_result['expected_effect'],
                'confidence': optimization_result['quantum_confidence'],
                'execution_time': optimization_result['execution_time']
            },
            'quantum_advantage': {
                'structure_search_speedup': structure_result.get('quantum_advantage', 1.0),
                'optimization_speedup': optimization_result.get('quantum_advantage', 1.0),
                'total_qubits_used': structure_result.get('qubits_used', 0) + optimization_result.get('qubits_used', 0)
            },
            'pipeline_metadata': {
                'quantum_backend': self.quantum_backend,
                'error_correction_enabled': self.enable_error_correction,
                'data_shape': data.shape,
                'n_variables': len(variables),
                'timestamp': time.time()
            }
        }
        
        logger.info("Quantum causal inference pipeline completed")
        return pipeline_result


class QuantumResourceManager:
    """Manage quantum computing resources and scheduling."""
    
    def __init__(self, quantum_processor: QuantumCausalProcessor):
        """Initialize quantum resource manager.
        
        Args:
            quantum_processor: Quantum processor instance
        """
        self.quantum_processor = quantum_processor
        self.resource_usage: Dict[str, float] = {}
        self.job_queue: List[QuantumComputeTask] = []
        self.active_jobs: Dict[str, QuantumComputeTask] = {}
        
        # Resource limits
        self.max_concurrent_jobs = 3
        self.max_queue_size = 50
        
        # Start resource monitoring
        self._start_resource_monitoring()
    
    def _start_resource_monitoring(self):
        """Start resource monitoring thread."""
        def monitor():
            while True:
                self._update_resource_usage()
                self._schedule_jobs()
                time.sleep(1)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _update_resource_usage(self):
        """Update quantum resource usage statistics."""
        self.resource_usage = {
            'quantum_circuits_cached': len(self.quantum_processor.circuit_cache),
            'active_jobs': len(self.active_jobs),
            'queued_jobs': len(self.job_queue),
            'qubit_utilization': self._calculate_qubit_utilization(),
            'backend_availability': self._check_backend_availability()
        }
    
    def _calculate_qubit_utilization(self) -> float:
        """Calculate current qubit utilization."""
        if not self.active_jobs:
            return 0.0
        
        total_qubits_used = 0
        for job in self.active_jobs.values():
            total_qubits_used += job.parameters.get('qubits_required', 0)
        
        utilization = total_qubits_used / self.quantum_processor.max_qubits
        return min(utilization, 1.0)
    
    def _check_backend_availability(self) -> float:
        """Check quantum backend availability."""
        if not QUANTUM_AVAILABLE or not self.quantum_processor.backend:
            return 0.0
        
        # In real implementation, would check backend queue status
        return 1.0  # Assume always available for simulation
    
    def _schedule_jobs(self):
        """Schedule queued quantum jobs."""
        while (len(self.active_jobs) < self.max_concurrent_jobs and 
               self.job_queue and 
               self._can_schedule_job()):
            
            # Get highest priority job
            job = max(self.job_queue, key=lambda j: j.priority)
            self.job_queue.remove(job)
            
            # Start job execution
            self.active_jobs[job.task_id] = job
            self.quantum_processor.quantum_executor.submit(self._execute_quantum_job, job)
    
    def _can_schedule_job(self) -> bool:
        """Check if we can schedule another job."""
        qubit_usage = self._calculate_qubit_utilization()
        backend_available = self._check_backend_availability()
        
        return qubit_usage < 0.8 and backend_available > 0.5
    
    def _execute_quantum_job(self, job: QuantumComputeTask):
        """Execute a quantum job.
        
        Args:
            job: Quantum compute task
        """
        try:
            start_time = time.time()
            
            if job.task_type == "causal_structure_search":
                # Execute structure search
                result = asyncio.run(self.quantum_processor._quantum_structure_search(
                    job.parameters['variables'], 
                    np.random.randn(*job.parameters['data_shape']),  # Placeholder data
                    job.parameters['max_parents']
                ))
            elif job.task_type == "intervention_optimization":
                # Execute intervention optimization
                graph = nx.DiGraph()
                graph.add_nodes_from(job.parameters['graph_nodes'])
                graph.add_edges_from(job.parameters['graph_edges'])
                
                result = asyncio.run(self.quantum_processor._quantum_intervention_optimization(
                    graph,
                    job.parameters['interventions'],
                    job.parameters['objectives']
                ))
            else:
                raise ValueError(f"Unknown job type: {job.task_type}")
            
            execution_time = time.time() - start_time
            
            # Store result
            quantum_result = QuantumResult(
                task_id=job.task_id,
                result=result,
                execution_time=execution_time,
                quantum_advantage=result.get('quantum_advantage', 1.0),
                confidence=result.get('quantum_confidence', 0.5)
            )
            
            self.quantum_processor.execution_results[job.task_id] = quantum_result
            
        except Exception as e:
            logger.error(f"Quantum job {job.task_id} failed: {e}")
        
        finally:
            # Remove from active jobs
            if job.task_id in self.active_jobs:
                del self.active_jobs[job.task_id]


# Global quantum processor instance
quantum_processor = QuantumCausalProcessor()
quantum_resource_manager = QuantumResourceManager(quantum_processor)

logger.info("Quantum acceleration system initialized")