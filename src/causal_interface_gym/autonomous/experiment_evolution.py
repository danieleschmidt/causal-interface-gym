"""Real-time adaptive experiment design system with evolutionary optimization."""

import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from scipy import stats
from scipy.optimize import differential_evolution
import warnings

from ..core import CausalEnvironment
from ..metrics import CausalMetrics

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of adaptive experiments."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    ADAPTING = "adapting"
    CONVERGED = "converged"
    FAILED = "failed"
    PAUSED = "paused"


class AdaptationStrategy(Enum):
    """Strategies for experiment adaptation."""
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN_OPT = "bayesian_optimization"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


@dataclass
class ExperimentParameter:
    """Parameter for adaptive experiments."""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Tuple[Any, Any]
    current_value: Any
    adaptation_rate: float = 0.1
    sensitivity: float = 1.0
    constraints: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Result from a single experiment trial."""
    trial_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    sample_size: int = 0


@dataclass
class AdaptiveExperiment:
    """Adaptive experiment with real-time optimization."""
    experiment_id: str
    name: str
    objective_function: str
    parameters: Dict[str, ExperimentParameter]
    adaptation_strategy: AdaptationStrategy
    status: ExperimentStatus = ExperimentStatus.INITIALIZING
    created_at: datetime = field(default_factory=datetime.now)
    
    # Experiment tracking
    trials: List[ExperimentResult] = field(default_factory=list)
    best_result: Optional[ExperimentResult] = None
    convergence_history: List[float] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Optimization state
    current_generation: int = 0
    max_generations: int = 100
    convergence_threshold: float = 1e-6
    patience_counter: int = 0
    max_patience: int = 10
    
    # Resource management
    max_parallel_trials: int = 5
    resource_budget: float = 1000.0
    resource_consumed: float = 0.0


class ExperimentEvolution:
    """Real-time adaptive experiment design with evolutionary optimization."""
    
    def __init__(self,
                 max_concurrent_experiments: int = 10,
                 default_adaptation_strategy: AdaptationStrategy = AdaptationStrategy.EVOLUTIONARY):
        """Initialize experiment evolution system.
        
        Args:
            max_concurrent_experiments: Maximum concurrent experiments
            default_adaptation_strategy: Default adaptation strategy
        """
        self.max_concurrent_experiments = max_concurrent_experiments
        self.default_adaptation_strategy = default_adaptation_strategy
        
        self.active_experiments: Dict[str, AdaptiveExperiment] = {}
        self.experiment_history: List[AdaptiveExperiment] = []
        self.global_knowledge: Dict[str, Any] = {}
        
        # Performance tracking
        self.evolution_metrics = {
            "experiments_launched": 0,
            "experiments_completed": 0,
            "avg_convergence_time": 0.0,
            "best_global_objective": 0.0,
            "adaptation_efficiency": 0.0
        }
        
        # Optimization algorithms
        self.optimizers = {
            AdaptationStrategy.EVOLUTIONARY: self._evolutionary_optimization,
            AdaptationStrategy.GRADIENT_BASED: self._gradient_based_optimization,
            AdaptationStrategy.BAYESIAN_OPT: self._bayesian_optimization,
            AdaptationStrategy.MULTI_ARMED_BANDIT: self._bandit_optimization,
            AdaptationStrategy.REINFORCEMENT_LEARNING: self._rl_optimization
        }
    
    async def create_adaptive_experiment(self,
                                       name: str,
                                       objective_function: str,
                                       parameter_space: Dict[str, Dict[str, Any]],
                                       adaptation_strategy: Optional[AdaptationStrategy] = None,
                                       **kwargs) -> str:
        """Create a new adaptive experiment.
        
        Args:
            name: Experiment name
            objective_function: Objective function to optimize
            parameter_space: Parameter definitions
            adaptation_strategy: Adaptation strategy to use
            **kwargs: Additional experiment configuration
            
        Returns:
            Experiment ID
        """
        if len(self.active_experiments) >= self.max_concurrent_experiments:
            raise RuntimeError("Maximum concurrent experiments reached")
        
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        strategy = adaptation_strategy or self.default_adaptation_strategy
        
        # Create experiment parameters
        parameters = {}
        for param_name, param_config in parameter_space.items():
            parameters[param_name] = ExperimentParameter(
                name=param_name,
                param_type=param_config.get("type", "continuous"),
                bounds=param_config.get("bounds", (0, 1)),
                current_value=param_config.get("initial_value"),
                adaptation_rate=param_config.get("adaptation_rate", 0.1),
                sensitivity=param_config.get("sensitivity", 1.0),
                constraints=param_config.get("constraints", [])
            )
        
        # Create experiment
        experiment = AdaptiveExperiment(
            experiment_id=experiment_id,
            name=name,
            objective_function=objective_function,
            parameters=parameters,
            adaptation_strategy=strategy,
            max_generations=kwargs.get("max_generations", 100),
            convergence_threshold=kwargs.get("convergence_threshold", 1e-6),
            max_parallel_trials=kwargs.get("max_parallel_trials", 5),
            resource_budget=kwargs.get("resource_budget", 1000.0)
        )
        
        self.active_experiments[experiment_id] = experiment
        self.evolution_metrics["experiments_launched"] += 1
        
        logger.info(f"Created adaptive experiment {experiment_id}: {name}")
        return experiment_id
    
    async def run_experiment(self, 
                           experiment_id: str,
                           evaluation_function: Callable,
                           auto_adapt: bool = True) -> Dict[str, Any]:
        """Run adaptive experiment with real-time optimization.
        
        Args:
            experiment_id: Experiment to run
            evaluation_function: Function to evaluate parameter combinations
            auto_adapt: Enable automatic adaptation
            
        Returns:
            Experiment results and final state
        """
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        
        try:
            # Initialize experiment
            await self._initialize_experiment(experiment, evaluation_function)
            
            if auto_adapt:
                # Run adaptive optimization
                return await self._run_adaptive_optimization(experiment, evaluation_function)
            else:
                # Run single evaluation
                return await self._run_single_evaluation(experiment, evaluation_function)
                
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            experiment.status = ExperimentStatus.FAILED
            return {"status": "failed", "error": str(e)}
    
    async def _initialize_experiment(self,
                                   experiment: AdaptiveExperiment,
                                   evaluation_function: Callable) -> None:
        """Initialize experiment with initial parameter values."""
        # Set initial parameter values if not specified
        for param_name, param in experiment.parameters.items():
            if param.current_value is None:
                if param.param_type == "continuous":
                    param.current_value = np.random.uniform(param.bounds[0], param.bounds[1])
                elif param.param_type == "discrete":
                    param.current_value = np.random.randint(param.bounds[0], param.bounds[1] + 1)
                elif param.param_type == "categorical":
                    param.current_value = np.random.choice(param.bounds)
        
        # Run initial evaluation
        initial_params = {name: param.current_value for name, param in experiment.parameters.items()}
        initial_result = await self._evaluate_parameters(
            experiment, initial_params, evaluation_function
        )
        
        if initial_result.success:
            experiment.best_result = initial_result
            logger.info(f"Initial evaluation for {experiment.experiment_id}: "
                       f"objective = {initial_result.metrics.get(experiment.objective_function, 0):.4f}")
    
    async def _run_adaptive_optimization(self,
                                       experiment: AdaptiveExperiment,
                                       evaluation_function: Callable) -> Dict[str, Any]:
        """Run adaptive optimization with the specified strategy."""
        optimizer = self.optimizers.get(experiment.adaptation_strategy)
        if not optimizer:
            raise ValueError(f"Unknown adaptation strategy: {experiment.adaptation_strategy}")
        
        start_time = datetime.now()
        
        # Run optimization
        final_result = await optimizer(experiment, evaluation_function)
        
        # Update metrics
        optimization_time = (datetime.now() - start_time).total_seconds()
        self._update_experiment_metrics(experiment, optimization_time)
        
        # Mark experiment as completed
        experiment.status = ExperimentStatus.CONVERGED
        self.experiment_history.append(experiment)
        del self.active_experiments[experiment.experiment_id]
        
        return {
            "status": "completed",
            "experiment_id": experiment.experiment_id,
            "best_parameters": {name: param.current_value for name, param in experiment.parameters.items()},
            "best_objective": experiment.best_result.metrics.get(experiment.objective_function, 0) if experiment.best_result else 0,
            "total_trials": len(experiment.trials),
            "convergence_generation": experiment.current_generation,
            "optimization_time": optimization_time,
            "final_result": final_result
        }
    
    async def _evolutionary_optimization(self,
                                       experiment: AdaptiveExperiment,
                                       evaluation_function: Callable) -> Dict[str, Any]:
        """Evolutionary optimization strategy."""
        population_size = min(20, experiment.max_parallel_trials * 4)
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Initialize population
        population = await self._create_initial_population(experiment, population_size)
        
        for generation in range(experiment.max_generations):
            experiment.current_generation = generation
            experiment.status = ExperimentStatus.ADAPTING
            
            # Evaluate population in parallel
            population_results = await self._evaluate_population_parallel(
                experiment, population, evaluation_function
            )
            
            # Update best result
            self._update_best_result(experiment, population_results)
            
            # Check convergence
            current_best = max(population_results, 
                             key=lambda r: r.metrics.get(experiment.objective_function, 0))
            objective_value = current_best.metrics.get(experiment.objective_function, 0)
            experiment.convergence_history.append(objective_value)
            
            if self._check_convergence(experiment):
                logger.info(f"Experiment {experiment.experiment_id} converged at generation {generation}")
                break
            
            # Create next generation
            population = await self._create_next_generation(
                experiment, population_results, mutation_rate, crossover_rate
            )
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: best objective = {objective_value:.4f}")
        
        return {
            "strategy": "evolutionary",
            "final_generation": experiment.current_generation,
            "population_size": population_size,
            "convergence_history": experiment.convergence_history
        }
    
    async def _create_initial_population(self,
                                       experiment: AdaptiveExperiment,
                                       population_size: int) -> List[Dict[str, Any]]:
        """Create initial population for evolutionary optimization."""
        population = []
        
        for _ in range(population_size):
            individual = {}
            for param_name, param in experiment.parameters.items():
                if param.param_type == "continuous":
                    individual[param_name] = np.random.uniform(param.bounds[0], param.bounds[1])
                elif param.param_type == "discrete":
                    individual[param_name] = np.random.randint(param.bounds[0], param.bounds[1] + 1)
                elif param.param_type == "categorical":
                    individual[param_name] = np.random.choice(param.bounds)
            
            population.append(individual)
        
        return population
    
    async def _evaluate_population_parallel(self,
                                          experiment: AdaptiveExperiment,
                                          population: List[Dict[str, Any]],
                                          evaluation_function: Callable) -> List[ExperimentResult]:
        """Evaluate population in parallel."""
        # Limit parallel evaluations
        max_parallel = experiment.max_parallel_trials
        results = []
        
        # Process population in chunks
        for i in range(0, len(population), max_parallel):
            chunk = population[i:i + max_parallel]
            
            # Create evaluation tasks
            tasks = [
                asyncio.create_task(
                    self._evaluate_parameters(experiment, individual, evaluation_function)
                )
                for individual in chunk
            ]
            
            # Wait for chunk completion
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            for result in chunk_results:
                if isinstance(result, ExperimentResult) and result.success:
                    results.append(result)
        
        return results
    
    async def _create_next_generation(self,
                                    experiment: AdaptiveExperiment,
                                    population_results: List[ExperimentResult],
                                    mutation_rate: float,
                                    crossover_rate: float) -> List[Dict[str, Any]]:
        """Create next generation using selection, crossover, and mutation."""
        if not population_results:
            return await self._create_initial_population(experiment, 20)
        
        # Selection: tournament selection
        selected_parents = self._tournament_selection(
            population_results, len(population_results)
        )
        
        next_generation = []
        
        # Create offspring
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[0]
            
            # Crossover
            if np.random.random() < crossover_rate:
                child1, child2 = self._crossover(experiment, parent1.parameters, parent2.parameters)
            else:
                child1, child2 = parent1.parameters.copy(), parent2.parameters.copy()
            
            # Mutation
            child1 = self._mutate(experiment, child1, mutation_rate)
            child2 = self._mutate(experiment, child2, mutation_rate)
            
            next_generation.extend([child1, child2])
        
        return next_generation[:len(population_results)]
    
    def _tournament_selection(self, 
                            population_results: List[ExperimentResult],
                            num_selected: int,
                            tournament_size: int = 3) -> List[ExperimentResult]:
        """Tournament selection for evolutionary algorithm."""
        selected = []
        
        for _ in range(num_selected):
            # Random tournament
            tournament = np.random.choice(population_results, tournament_size, replace=False)
            
            # Select best from tournament
            winner = max(tournament, 
                        key=lambda r: r.metrics.get("objective_function", 0))
            selected.append(winner)
        
        return selected
    
    def _crossover(self, 
                  experiment: AdaptiveExperiment,
                  parent1: Dict[str, Any],
                  parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for creating offspring."""
        child1, child2 = {}, {}
        
        for param_name in experiment.parameters:
            if param_name in parent1 and param_name in parent2:
                if np.random.random() < 0.5:
                    child1[param_name] = parent1[param_name]
                    child2[param_name] = parent2[param_name]
                else:
                    child1[param_name] = parent2[param_name]
                    child2[param_name] = parent1[param_name]
        
        return child1, child2
    
    def _mutate(self,
               experiment: AdaptiveExperiment,
               individual: Dict[str, Any],
               mutation_rate: float) -> Dict[str, Any]:
        """Mutation operation for introducing variation."""
        mutated = individual.copy()
        
        for param_name, param in experiment.parameters.items():
            if np.random.random() < mutation_rate:
                if param.param_type == "continuous":
                    # Gaussian mutation
                    current_value = mutated[param_name]
                    mutation_strength = (param.bounds[1] - param.bounds[0]) * 0.1
                    mutated[param_name] = np.clip(
                        current_value + np.random.normal(0, mutation_strength),
                        param.bounds[0], param.bounds[1]
                    )
                elif param.param_type == "discrete":
                    mutated[param_name] = np.random.randint(param.bounds[0], param.bounds[1] + 1)
                elif param.param_type == "categorical":
                    mutated[param_name] = np.random.choice(param.bounds)
        
        return mutated
    
    async def _gradient_based_optimization(self,
                                         experiment: AdaptiveExperiment,
                                         evaluation_function: Callable) -> Dict[str, Any]:
        """Gradient-based optimization strategy."""
        learning_rate = 0.01
        momentum = 0.9
        velocity = {name: 0.0 for name in experiment.parameters}
        
        for iteration in range(experiment.max_generations):
            experiment.current_generation = iteration
            
            # Compute gradients numerically
            gradients = await self._compute_numerical_gradients(
                experiment, evaluation_function
            )
            
            # Update parameters with momentum
            for param_name, param in experiment.parameters.items():
                if param.param_type == "continuous":
                    gradient = gradients.get(param_name, 0)
                    velocity[param_name] = momentum * velocity[param_name] + learning_rate * gradient
                    new_value = param.current_value + velocity[param_name]
                    
                    # Apply bounds
                    param.current_value = np.clip(new_value, param.bounds[0], param.bounds[1])
            
            # Evaluate current parameters
            current_params = {name: param.current_value for name, param in experiment.parameters.items()}
            result = await self._evaluate_parameters(experiment, current_params, evaluation_function)
            
            if result.success:
                self._update_best_result(experiment, [result])
                objective_value = result.metrics.get(experiment.objective_function, 0)
                experiment.convergence_history.append(objective_value)
                
                if self._check_convergence(experiment):
                    break
        
        return {
            "strategy": "gradient_based",
            "final_iteration": experiment.current_generation,
            "learning_rate": learning_rate,
            "momentum": momentum
        }
    
    async def _compute_numerical_gradients(self,
                                         experiment: AdaptiveExperiment,
                                         evaluation_function: Callable,
                                         epsilon: float = 1e-5) -> Dict[str, float]:
        """Compute numerical gradients for continuous parameters."""
        gradients = {}
        current_params = {name: param.current_value for name, param in experiment.parameters.items()}
        
        # Evaluate at current point
        current_result = await self._evaluate_parameters(experiment, current_params, evaluation_function)
        if not current_result.success:
            return gradients
        
        current_objective = current_result.metrics.get(experiment.objective_function, 0)
        
        # Compute gradients for continuous parameters
        for param_name, param in experiment.parameters.items():
            if param.param_type == "continuous":
                # Forward difference
                perturbed_params = current_params.copy()
                perturbed_params[param_name] += epsilon
                
                # Apply bounds
                perturbed_params[param_name] = np.clip(
                    perturbed_params[param_name], param.bounds[0], param.bounds[1]
                )
                
                perturbed_result = await self._evaluate_parameters(
                    experiment, perturbed_params, evaluation_function
                )
                
                if perturbed_result.success:
                    perturbed_objective = perturbed_result.metrics.get(experiment.objective_function, 0)
                    gradients[param_name] = (perturbed_objective - current_objective) / epsilon
        
        return gradients
    
    async def _bayesian_optimization(self,
                                   experiment: AdaptiveExperiment,
                                   evaluation_function: Callable) -> Dict[str, Any]:
        """Bayesian optimization strategy (simplified implementation)."""
        # This is a simplified version - in practice would use libraries like scikit-optimize
        n_initial_samples = 5
        n_iterations = experiment.max_generations - n_initial_samples
        
        # Initial random sampling
        initial_samples = await self._create_initial_population(experiment, n_initial_samples)
        initial_results = await self._evaluate_population_parallel(
            experiment, initial_samples, evaluation_function
        )
        
        self._update_best_result(experiment, initial_results)
        
        # Bayesian optimization iterations
        for iteration in range(n_iterations):
            experiment.current_generation = n_initial_samples + iteration
            
            # In a full implementation, would build Gaussian Process surrogate model
            # and use acquisition function (EI, UCB, etc.) to select next point
            
            # Simplified: use expected improvement heuristic
            next_point = await self._select_next_point_simple_ei(
                experiment, initial_results
            )
            
            # Evaluate next point
            result = await self._evaluate_parameters(experiment, next_point, evaluation_function)
            
            if result.success:
                initial_results.append(result)  # Add to training data
                self._update_best_result(experiment, [result])
                
                objective_value = result.metrics.get(experiment.objective_function, 0)
                experiment.convergence_history.append(objective_value)
                
                if self._check_convergence(experiment):
                    break
        
        return {
            "strategy": "bayesian_optimization",
            "n_initial_samples": n_initial_samples,
            "n_bo_iterations": experiment.current_generation - n_initial_samples
        }
    
    async def _select_next_point_simple_ei(self,
                                         experiment: AdaptiveExperiment,
                                         historical_results: List[ExperimentResult]) -> Dict[str, Any]:
        """Simple expected improvement heuristic for next point selection."""
        # Extract objective values
        objectives = [r.metrics.get(experiment.objective_function, 0) for r in historical_results]
        best_objective = max(objectives)
        
        # Generate candidate points
        n_candidates = 100
        candidates = await self._create_initial_population(experiment, n_candidates)
        
        # Simple EI approximation: distance from best point + random exploration
        best_params = max(historical_results, 
                         key=lambda r: r.metrics.get(experiment.objective_function, 0)).parameters
        
        best_candidate = None
        best_ei = -float('inf')
        
        for candidate in candidates:
            # Distance from best (simplified)
            distance = sum(
                abs(candidate[name] - best_params.get(name, 0))
                for name in candidate
            )
            
            # Simple EI heuristic
            exploration = np.random.exponential(0.1)
            ei = -distance + exploration  # Negative distance + exploration term
            
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate
        
        return best_candidate or candidates[0]
    
    async def _bandit_optimization(self,
                                 experiment: AdaptiveExperiment,
                                 evaluation_function: Callable) -> Dict[str, Any]:
        """Multi-armed bandit optimization strategy."""
        # Discretize continuous parameters into arms
        arms = await self._create_bandit_arms(experiment, n_arms_per_param=5)
        
        # Initialize arm statistics
        arm_counts = np.zeros(len(arms))
        arm_rewards = np.zeros(len(arms))
        
        epsilon = 0.1  # Exploration rate
        
        for trial in range(experiment.max_generations):
            experiment.current_generation = trial
            
            # Select arm using epsilon-greedy
            if np.random.random() < epsilon:
                # Exploration
                selected_arm = np.random.randint(len(arms))
            else:
                # Exploitation
                avg_rewards = np.divide(arm_rewards, arm_counts + 1e-8)
                selected_arm = np.argmax(avg_rewards)
            
            # Evaluate selected arm
            arm_params = arms[selected_arm]
            result = await self._evaluate_parameters(experiment, arm_params, evaluation_function)
            
            if result.success:
                # Update arm statistics
                arm_counts[selected_arm] += 1
                reward = result.metrics.get(experiment.objective_function, 0)
                arm_rewards[selected_arm] += reward
                
                self._update_best_result(experiment, [result])
                experiment.convergence_history.append(reward)
                
                if self._check_convergence(experiment):
                    break
        
        return {
            "strategy": "multi_armed_bandit",
            "n_arms": len(arms),
            "epsilon": epsilon,
            "arm_selection_history": arm_counts.tolist()
        }
    
    async def _create_bandit_arms(self,
                                experiment: AdaptiveExperiment,
                                n_arms_per_param: int = 5) -> List[Dict[str, Any]]:
        """Create discrete arms for bandit optimization."""
        param_values = {}
        
        for param_name, param in experiment.parameters.items():
            if param.param_type == "continuous":
                param_values[param_name] = np.linspace(
                    param.bounds[0], param.bounds[1], n_arms_per_param
                )
            elif param.param_type == "discrete":
                param_values[param_name] = list(range(param.bounds[0], param.bounds[1] + 1))
            elif param.param_type == "categorical":
                param_values[param_name] = param.bounds
        
        # Create all combinations (Cartesian product)
        import itertools
        param_names = list(param_values.keys())
        param_combinations = list(itertools.product(*[param_values[name] for name in param_names]))
        
        arms = []
        for combination in param_combinations:
            arm = dict(zip(param_names, combination))
            arms.append(arm)
        
        return arms
    
    async def _rl_optimization(self,
                             experiment: AdaptiveExperiment,
                             evaluation_function: Callable) -> Dict[str, Any]:
        """Reinforcement learning optimization strategy (simplified Q-learning)."""
        # This is a simplified RL approach
        # In practice, would use more sophisticated RL algorithms
        
        learning_rate = 0.1
        discount_factor = 0.9
        epsilon = 0.1
        
        # Discretize state and action spaces
        states = await self._create_bandit_arms(experiment, n_arms_per_param=3)  # State space
        actions = await self._create_bandit_arms(experiment, n_arms_per_param=3)  # Action space
        
        # Initialize Q-table
        q_table = np.zeros((len(states), len(actions)))
        
        current_state_idx = 0  # Start with first state
        
        for episode in range(experiment.max_generations):
            experiment.current_generation = episode
            
            # Select action using epsilon-greedy
            if np.random.random() < epsilon:
                action_idx = np.random.randint(len(actions))
            else:
                action_idx = np.argmax(q_table[current_state_idx])
            
            # Execute action (evaluate parameters)
            action_params = actions[action_idx]
            result = await self._evaluate_parameters(experiment, action_params, evaluation_function)
            
            if result.success:
                # Get reward
                reward = result.metrics.get(experiment.objective_function, 0)
                
                # Update Q-value
                next_state_idx = (current_state_idx + 1) % len(states)  # Simplified state transition
                q_table[current_state_idx, action_idx] += learning_rate * (
                    reward + discount_factor * np.max(q_table[next_state_idx]) - 
                    q_table[current_state_idx, action_idx]
                )
                
                current_state_idx = next_state_idx
                
                self._update_best_result(experiment, [result])
                experiment.convergence_history.append(reward)
                
                if self._check_convergence(experiment):
                    break
        
        return {
            "strategy": "reinforcement_learning",
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "final_q_table_max": np.max(q_table)
        }
    
    async def _evaluate_parameters(self,
                                 experiment: AdaptiveExperiment,
                                 parameters: Dict[str, Any],
                                 evaluation_function: Callable) -> ExperimentResult:
        """Evaluate a set of parameters using the evaluation function."""
        trial_id = f"trial_{len(experiment.trials)}"
        start_time = datetime.now()
        
        try:
            # Call evaluation function
            if asyncio.iscoroutinefunction(evaluation_function):
                metrics = await evaluation_function(parameters)
            else:
                metrics = evaluation_function(parameters)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ExperimentResult(
                trial_id=trial_id,
                parameters=parameters.copy(),
                metrics=metrics if isinstance(metrics, dict) else {"objective": metrics},
                timestamp=start_time,
                success=True,
                execution_time=execution_time,
                sample_size=metrics.get("sample_size", 1) if isinstance(metrics, dict) else 1
            )
            
            experiment.trials.append(result)
            experiment.resource_consumed += execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ExperimentResult(
                trial_id=trial_id,
                parameters=parameters.copy(),
                metrics={},
                timestamp=start_time,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
            
            experiment.trials.append(result)
            return result
    
    def _update_best_result(self,
                          experiment: AdaptiveExperiment,
                          results: List[ExperimentResult]) -> None:
        """Update the best result for the experiment."""
        for result in results:
            if result.success:
                objective_value = result.metrics.get(experiment.objective_function, 0)
                
                if (experiment.best_result is None or 
                    objective_value > experiment.best_result.metrics.get(experiment.objective_function, 0)):
                    experiment.best_result = result
                    
                    # Update parameter values to best found
                    for param_name, param in experiment.parameters.items():
                        if param_name in result.parameters:
                            param.current_value = result.parameters[param_name]
    
    def _check_convergence(self, experiment: AdaptiveExperiment) -> bool:
        """Check if experiment has converged."""
        if len(experiment.convergence_history) < 5:
            return False
        
        # Check if improvement has stagnated
        recent_improvements = np.diff(experiment.convergence_history[-5:])
        avg_improvement = np.mean(recent_improvements)
        
        if avg_improvement < experiment.convergence_threshold:
            experiment.patience_counter += 1
        else:
            experiment.patience_counter = 0
        
        return experiment.patience_counter >= experiment.max_patience
    
    def _update_experiment_metrics(self, experiment: AdaptiveExperiment, optimization_time: float) -> None:
        """Update global experiment metrics."""
        self.evolution_metrics["experiments_completed"] += 1
        
        # Update average convergence time
        current_avg = self.evolution_metrics["avg_convergence_time"]
        completed = self.evolution_metrics["experiments_completed"]
        
        self.evolution_metrics["avg_convergence_time"] = (
            (current_avg * (completed - 1) + optimization_time) / completed
        )
        
        # Update best global objective
        if experiment.best_result:
            best_objective = experiment.best_result.metrics.get(experiment.objective_function, 0)
            self.evolution_metrics["best_global_objective"] = max(
                self.evolution_metrics["best_global_objective"],
                best_objective
            )
        
        # Compute adaptation efficiency
        total_trials = len(experiment.trials)
        successful_trials = sum(1 for t in experiment.trials if t.success)
        efficiency = successful_trials / max(total_trials, 1)
        
        current_efficiency = self.evolution_metrics["adaptation_efficiency"]
        self.evolution_metrics["adaptation_efficiency"] = (
            (current_efficiency * (completed - 1) + efficiency) / completed
        )
    
    async def _run_single_evaluation(self,
                                   experiment: AdaptiveExperiment,
                                   evaluation_function: Callable) -> Dict[str, Any]:
        """Run single evaluation without adaptation."""
        current_params = {name: param.current_value for name, param in experiment.parameters.items()}
        result = await self._evaluate_parameters(experiment, current_params, evaluation_function)
        
        experiment.status = ExperimentStatus.CONVERGED
        if result.success:
            experiment.best_result = result
        
        return {
            "status": "completed",
            "experiment_id": experiment.experiment_id,
            "parameters": current_params,
            "result": result.metrics if result.success else {"error": result.error_message},
            "success": result.success
        }
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment."""
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
        else:
            # Look in history
            experiment = next((e for e in self.experiment_history if e.experiment_id == experiment_id), None)
            if not experiment:
                return {"error": "Experiment not found"}
        
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "current_generation": experiment.current_generation,
            "total_trials": len(experiment.trials),
            "best_objective": (experiment.best_result.metrics.get(experiment.objective_function, 0) 
                             if experiment.best_result else None),
            "resource_consumed": experiment.resource_consumed,
            "resource_budget": experiment.resource_budget,
            "convergence_history": experiment.convergence_history[-10:],  # Last 10 values
            "adaptation_strategy": experiment.adaptation_strategy.value
        }
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of experiment evolution system."""
        return {
            "active_experiments": len(self.active_experiments),
            "completed_experiments": len(self.experiment_history),
            "total_trials_run": sum(len(e.trials) for e in 
                                  list(self.active_experiments.values()) + self.experiment_history),
            **self.evolution_metrics,
            "experiment_summaries": [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "status": exp.status.value,
                    "best_objective": (exp.best_result.metrics.get(exp.objective_function, 0) 
                                     if exp.best_result else None),
                    "trials": len(exp.trials)
                }
                for exp in list(self.active_experiments.values()) + self.experiment_history[-5:]
            ]
        }