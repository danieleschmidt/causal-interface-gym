"""Novel causal discovery algorithms with adaptive learning capabilities."""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
import itertools

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Result of causal discovery process."""
    discovered_graph: nx.DiGraph
    confidence_scores: Dict[Tuple[str, str], float]
    statistical_tests: Dict[str, Dict[str, float]]
    algorithm_performance: Dict[str, float]
    convergence_metrics: Dict[str, Any]


class AdaptiveCausalDiscovery:
    """Advanced causal discovery with adaptive algorithm selection."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 max_iterations: int = 1000,
                 convergence_tolerance: float = 1e-6):
        """Initialize adaptive causal discovery system.
        
        Args:
            confidence_threshold: Minimum confidence for edge inclusion
            max_iterations: Maximum iterations for convergence
            convergence_tolerance: Tolerance for convergence detection
        """
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.algorithm_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
    def discover_structure(self, 
                          data: np.ndarray, 
                          variable_names: List[str],
                          prior_knowledge: Optional[nx.DiGraph] = None) -> DiscoveryResult:
        """Discover causal structure using adaptive ensemble approach.
        
        Args:
            data: Observational data matrix (n_samples x n_variables)
            variable_names: Names of variables
            prior_knowledge: Optional prior causal structure
            
        Returns:
            DiscoveryResult with discovered structure and metrics
        """
        logger.info(f"Starting adaptive causal discovery for {len(variable_names)} variables")
        
        # Run multiple algorithms
        pc_result = self._run_pc_algorithm(data, variable_names)
        ges_result = self._run_ges_algorithm(data, variable_names) 
        novel_result = self._run_novel_discovery(data, variable_names)
        
        # Ensemble combination with adaptive weighting
        ensemble_graph = self._combine_results([pc_result, ges_result, novel_result],
                                              variable_names, prior_knowledge)
        
        # Compute confidence scores
        confidence_scores = self._compute_edge_confidence(
            [pc_result, ges_result, novel_result], ensemble_graph)
            
        # Statistical validation
        statistical_tests = self._validate_discovered_edges(data, variable_names, ensemble_graph)
        
        # Performance metrics
        algorithm_performance = self._evaluate_algorithm_performance(
            [pc_result, ges_result, novel_result])
            
        # Convergence analysis
        convergence_metrics = self._analyze_convergence(
            [pc_result, ges_result, novel_result])
            
        result = DiscoveryResult(
            discovered_graph=ensemble_graph,
            confidence_scores=confidence_scores,
            statistical_tests=statistical_tests,
            algorithm_performance=algorithm_performance,
            convergence_metrics=convergence_metrics
        )
        
        self._update_adaptive_weights(result)
        
        logger.info(f"Discovery completed. Found {ensemble_graph.number_of_edges()} edges")
        return result
        
    def _run_pc_algorithm(self, data: np.ndarray, variable_names: List[str]) -> nx.DiGraph:
        """Run PC algorithm for causal discovery."""
        graph = nx.DiGraph()
        graph.add_nodes_from(variable_names)
        
        n_vars = len(variable_names)
        
        # Start with complete graph
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Conditional independence test
                    if self._conditional_independence_test(data[:, i], data[:, j], 
                                                         data, significance_level=0.05):
                        continue
                    graph.add_edge(variable_names[i], variable_names[j])
        
        # Orientation rules
        graph = self._orient_edges_pc(graph, data, variable_names)
        
        return graph
        
    def _run_ges_algorithm(self, data: np.ndarray, variable_names: List[str]) -> nx.DiGraph:
        """Run GES (Greedy Equivalence Search) algorithm."""
        graph = nx.DiGraph()
        graph.add_nodes_from(variable_names)
        
        # Score-based approach with BIC scoring
        best_score = self._compute_bic_score(graph, data, variable_names)
        improved = True
        
        while improved:
            improved = False
            best_operation = None
            
            # Try edge additions
            for i, j in itertools.permutations(range(len(variable_names)), 2):
                if not graph.has_edge(variable_names[i], variable_names[j]):
                    test_graph = graph.copy()
                    test_graph.add_edge(variable_names[i], variable_names[j])
                    
                    if not self._creates_cycle(test_graph):
                        score = self._compute_bic_score(test_graph, data, variable_names)
                        if score > best_score:
                            best_score = score
                            best_operation = ('add', variable_names[i], variable_names[j])
                            improved = True
            
            # Try edge deletions
            for edge in list(graph.edges()):
                test_graph = graph.copy()
                test_graph.remove_edge(*edge)
                score = self._compute_bic_score(test_graph, data, variable_names)
                if score > best_score:
                    best_score = score
                    best_operation = ('delete', edge[0], edge[1])
                    improved = True
                    
            # Try edge reversals
            for edge in list(graph.edges()):
                test_graph = graph.copy()
                test_graph.remove_edge(*edge)
                test_graph.add_edge(edge[1], edge[0])
                
                if not self._creates_cycle(test_graph):
                    score = self._compute_bic_score(test_graph, data, variable_names)
                    if score > best_score:
                        best_score = score
                        best_operation = ('reverse', edge[0], edge[1])
                        improved = True
                        
            # Apply best operation
            if best_operation:
                op, var1, var2 = best_operation
                if op == 'add':
                    graph.add_edge(var1, var2)
                elif op == 'delete':
                    graph.remove_edge(var1, var2)
                elif op == 'reverse':
                    graph.remove_edge(var1, var2)
                    graph.add_edge(var2, var1)
                    
        return graph
        
    def _run_novel_discovery(self, data: np.ndarray, variable_names: List[str]) -> nx.DiGraph:
        """Run novel adaptive discovery algorithm."""
        graph = nx.DiGraph()
        graph.add_nodes_from(variable_names)
        
        # Information-theoretic approach with adaptive thresholding
        n_vars = len(variable_names)
        mutual_info_matrix = np.zeros((n_vars, n_vars))
        
        # Compute pairwise mutual information
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    mi = mutual_info_regression(data[:, [i]], data[:, j])
                    mutual_info_matrix[i, j] = mi[0]
        
        # Adaptive threshold based on distribution
        threshold = np.percentile(mutual_info_matrix[mutual_info_matrix > 0], 75)
        
        # Add edges based on thresholding and directionality inference
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and mutual_info_matrix[i, j] > threshold:
                    # Infer direction using entropy-based criterion
                    if self._infer_direction(data[:, i], data[:, j]):
                        graph.add_edge(variable_names[i], variable_names[j])
                        
        return graph
        
    def _combine_results(self, 
                        results: List[nx.DiGraph], 
                        variable_names: List[str],
                        prior_knowledge: Optional[nx.DiGraph] = None) -> nx.DiGraph:
        """Combine multiple discovery results using ensemble voting."""
        ensemble_graph = nx.DiGraph()
        ensemble_graph.add_nodes_from(variable_names)
        
        # Weighted voting based on algorithm performance
        edge_votes: Dict[Tuple[str, str], float] = {}
        
        weights = self._compute_algorithm_weights(results)
        
        for graph, weight in zip(results, weights):
            for edge in graph.edges():
                if edge not in edge_votes:
                    edge_votes[edge] = 0.0
                edge_votes[edge] += weight
                
        # Add edges that meet confidence threshold
        for edge, vote_weight in edge_votes.items():
            if vote_weight >= self.confidence_threshold:
                ensemble_graph.add_edge(edge[0], edge[1])
                
        # Incorporate prior knowledge if available
        if prior_knowledge:
            for edge in prior_knowledge.edges():
                if not ensemble_graph.has_edge(*edge):
                    # Add with reduced confidence if supported by at least one algorithm
                    if edge in edge_votes and edge_votes[edge] > 0.3:
                        ensemble_graph.add_edge(edge[0], edge[1])
                        
        return ensemble_graph
        
    def _conditional_independence_test(self, 
                                     X: np.ndarray, 
                                     Y: np.ndarray,
                                     conditioning_set: np.ndarray,
                                     significance_level: float = 0.05) -> bool:
        """Test conditional independence using partial correlation."""
        try:
            if conditioning_set.shape[1] == 0:
                # Unconditional independence test
                corr, p_value = stats.pearsonr(X, Y)
                return p_value > significance_level
            else:
                # Partial correlation test
                from scipy.stats import pearsonr
                
                # Simple partial correlation approximation
                # For production, use more sophisticated methods
                corr_xy = np.corrcoef(X, Y)[0, 1]
                
                if abs(corr_xy) < 0.1:  # Threshold for weak correlation
                    return True
                    
                return False
                
        except Exception as e:
            logger.warning(f"CI test failed: {e}")
            return False
            
    def _orient_edges_pc(self, graph: nx.DiGraph, data: np.ndarray, 
                        variable_names: List[str]) -> nx.DiGraph:
        """Apply PC orientation rules."""
        # Convert to undirected for orientation
        undirected = graph.to_undirected()
        oriented = nx.DiGraph()
        oriented.add_nodes_from(variable_names)
        
        # Rule 1: Orient v-structures
        for node in undirected.nodes():
            neighbors = list(undirected.neighbors(node))
            for i, j in itertools.combinations(neighbors, 2):
                if not undirected.has_edge(i, j):
                    # Found v-structure: i -> node <- j
                    oriented.add_edge(i, node)
                    oriented.add_edge(j, node)
                    
        # Add remaining edges with arbitrary orientation
        for edge in undirected.edges():
            if not oriented.has_edge(*edge) and not oriented.has_edge(edge[1], edge[0]):
                oriented.add_edge(*edge)
                
        return oriented
        
    def _compute_bic_score(self, graph: nx.DiGraph, data: np.ndarray, 
                          variable_names: List[str]) -> float:
        """Compute BIC score for graph structure."""
        n_samples, n_vars = data.shape
        log_likelihood = 0.0
        n_parameters = 0
        
        for i, var in enumerate(variable_names):
            parents = list(graph.predecessors(var))
            n_parents = len(parents)
            
            if n_parents == 0:
                # Marginal distribution
                variance = np.var(data[:, i])
                log_likelihood += -0.5 * n_samples * np.log(2 * np.pi * variance)
                log_likelihood += -0.5 * np.sum((data[:, i] - np.mean(data[:, i]))**2) / variance
                n_parameters += 2  # mean and variance
            else:
                # Linear regression on parents
                parent_indices = [variable_names.index(p) for p in parents]
                X = data[:, parent_indices]
                y = data[:, i]
                
                # Add intercept
                X_with_intercept = np.column_stack([np.ones(n_samples), X])
                
                try:
                    # Least squares solution
                    beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                    residuals = y - X_with_intercept @ beta
                    variance = np.var(residuals)
                    
                    if variance > 0:
                        log_likelihood += -0.5 * n_samples * np.log(2 * np.pi * variance)
                        log_likelihood += -0.5 * np.sum(residuals**2) / variance
                        
                    n_parameters += n_parents + 2  # coefficients + intercept + variance
                    
                except np.linalg.LinAlgError:
                    # Singular matrix, penalize heavily
                    log_likelihood -= 1000
                    
        # BIC = log likelihood - 0.5 * k * log(n)
        bic = log_likelihood - 0.5 * n_parameters * np.log(n_samples)
        return bic
        
    def _creates_cycle(self, graph: nx.DiGraph) -> bool:
        """Check if graph contains cycles."""
        try:
            nx.find_cycle(graph, orientation='original')
            return True
        except nx.NetworkXNoCycle:
            return False
            
    def _infer_direction(self, X: np.ndarray, Y: np.ndarray) -> bool:
        """Infer causal direction using entropy-based criterion."""
        try:
            # Information-theoretic approach to direction inference
            # Based on additive noise models
            
            # Forward direction: X -> Y
            X_std = (X - np.mean(X)) / np.std(X)
            residuals_forward = Y - np.mean(Y) - np.std(Y) * X_std
            
            # Backward direction: Y -> X  
            Y_std = (Y - np.mean(Y)) / np.std(Y)
            residuals_backward = X - np.mean(X) - np.std(X) * Y_std
            
            # Test independence of cause and residuals
            corr_forward = abs(np.corrcoef(X, residuals_forward)[0, 1])
            corr_backward = abs(np.corrcoef(Y, residuals_backward)[0, 1])
            
            # Lower correlation indicates better fit (more independence)
            return corr_forward < corr_backward
            
        except Exception as e:
            logger.warning(f"Direction inference failed: {e}")
            return True  # Default direction
            
    def _compute_algorithm_weights(self, results: List[nx.DiGraph]) -> List[float]:
        """Compute adaptive weights for ensemble combination."""
        # Start with equal weights
        weights = [1.0] * len(results)
        
        # Adjust based on performance history
        if hasattr(self, 'performance_metrics'):
            for i, (algo_name, performance) in enumerate([
                ('pc', self.performance_metrics.get('pc_performance', 1.0)),
                ('ges', self.performance_metrics.get('ges_performance', 1.0)), 
                ('novel', self.performance_metrics.get('novel_performance', 1.0))
            ]):
                if i < len(weights):
                    weights[i] *= performance
                    
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(results)] * len(results)
            
        return weights
        
    def _compute_edge_confidence(self, 
                                results: List[nx.DiGraph],
                                ensemble_graph: nx.DiGraph) -> Dict[Tuple[str, str], float]:
        """Compute confidence scores for discovered edges."""
        confidence_scores = {}
        
        for edge in ensemble_graph.edges():
            votes = sum(1 for graph in results if graph.has_edge(*edge))
            confidence = votes / len(results)
            confidence_scores[edge] = confidence
            
        return confidence_scores
        
    def _validate_discovered_edges(self, 
                                  data: np.ndarray,
                                  variable_names: List[str],
                                  graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """Validate discovered edges using statistical tests."""
        validation_results = {}
        
        for edge in graph.edges():
            source_idx = variable_names.index(edge[0])
            target_idx = variable_names.index(edge[1])
            
            # Granger causality test approximation
            X = data[:, source_idx]
            Y = data[:, target_idx]
            
            try:
                # Simple correlation test as approximation
                corr, p_value = stats.pearsonr(X, Y)
                
                validation_results[f"{edge[0]}->{edge[1]}"] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
            except Exception as e:
                logger.warning(f"Validation failed for edge {edge}: {e}")
                validation_results[f"{edge[0]}->{edge[1]}"] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
                
        return validation_results
        
    def _evaluate_algorithm_performance(self, results: List[nx.DiGraph]) -> Dict[str, float]:
        """Evaluate performance of different algorithms."""
        algo_names = ['pc', 'ges', 'novel']
        performance = {}
        
        for i, (name, graph) in enumerate(zip(algo_names, results)):
            # Simple performance metrics
            n_edges = graph.number_of_edges()
            n_nodes = graph.number_of_nodes()
            
            # Edge density
            max_edges = n_nodes * (n_nodes - 1)
            edge_density = n_edges / max_edges if max_edges > 0 else 0
            
            # Connectivity
            if n_nodes > 1:
                connectivity = nx.is_connected(graph.to_undirected())
            else:
                connectivity = True
                
            performance[f"{name}_performance"] = 0.7 + 0.3 * edge_density + (0.1 if connectivity else 0)
            
        return performance
        
    def _analyze_convergence(self, results: List[nx.DiGraph]) -> Dict[str, Any]:
        """Analyze convergence properties of discovery process."""
        convergence_metrics = {
            'stability': 0.0,
            'consensus': 0.0,
            'iterations': len(results)
        }
        
        if len(results) > 1:
            # Edge agreement between algorithms
            all_edges = set()
            for graph in results:
                all_edges.update(graph.edges())
                
            if all_edges:
                edge_agreements = []
                for edge in all_edges:
                    agreement = sum(1 for graph in results if graph.has_edge(*edge))
                    edge_agreements.append(agreement / len(results))
                    
                convergence_metrics['consensus'] = np.mean(edge_agreements)
                convergence_metrics['stability'] = 1.0 - np.std(edge_agreements)
                
        return convergence_metrics
        
    def _update_adaptive_weights(self, result: DiscoveryResult) -> None:
        """Update algorithm weights based on discovery result quality."""
        # Update performance metrics for future use
        self.performance_metrics.update(result.algorithm_performance)
        
        # Store result for learning
        self.algorithm_history.append({
            'result': result,
            'timestamp': np.datetime64('now'),
            'performance': result.algorithm_performance
        })
        
        # Keep only recent history for adaptivity
        if len(self.algorithm_history) > 100:
            self.algorithm_history = self.algorithm_history[-100:]


class CausalStructureLearner:
    """Advanced structure learning with bootstrap validation."""
    
    def __init__(self, n_bootstrap_samples: int = 100):
        """Initialize structure learner.
        
        Args:
            n_bootstrap_samples: Number of bootstrap samples for stability
        """
        self.n_bootstrap_samples = n_bootstrap_samples
        self.discovery_engine = AdaptiveCausalDiscovery()
        
    def learn_with_uncertainty(self, 
                             data: np.ndarray,
                             variable_names: List[str]) -> Tuple[nx.DiGraph, Dict[Tuple[str, str], float]]:
        """Learn structure with uncertainty quantification via bootstrapping.
        
        Args:
            data: Input data matrix
            variable_names: Variable names
            
        Returns:
            Tuple of (consensus_graph, edge_probabilities)
        """
        n_samples = data.shape[0]
        edge_counts: Dict[Tuple[str, str], int] = {}
        
        logger.info(f"Learning structure with {self.n_bootstrap_samples} bootstrap samples")
        
        # Bootstrap sampling and structure discovery
        for i in range(self.n_bootstrap_samples):
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_data = data[bootstrap_indices]
            
            # Discover structure
            result = self.discovery_engine.discover_structure(bootstrap_data, variable_names)
            
            # Count edge occurrences
            for edge in result.discovered_graph.edges():
                if edge not in edge_counts:
                    edge_counts[edge] = 0
                edge_counts[edge] += 1
                
        # Compute edge probabilities
        edge_probabilities = {
            edge: count / self.n_bootstrap_samples 
            for edge, count in edge_counts.items()
        }
        
        # Create consensus graph with high-probability edges
        consensus_graph = nx.DiGraph()
        consensus_graph.add_nodes_from(variable_names)
        
        # Add edges with probability > 0.5
        for edge, prob in edge_probabilities.items():
            if prob > 0.5:
                consensus_graph.add_edge(edge[0], edge[1])
                
        logger.info(f"Consensus graph has {consensus_graph.number_of_edges()} edges")
        
        return consensus_graph, edge_probabilities


class MetaCausalLearner:
    """Meta-learning approach for adaptive causal discovery algorithm selection."""
    
    def __init__(self, 
                 meta_learning_samples: int = 1000,
                 performance_memory: int = 50):
        """Initialize meta-causal learner.
        
        Args:
            meta_learning_samples: Number of samples for meta-learning
            performance_memory: Number of past performance records to keep
        """
        self.meta_learning_samples = meta_learning_samples
        self.performance_memory = performance_memory
        
        # Algorithm performance history
        self.algorithm_performance_history: Dict[str, List[float]] = {
            'pc': [], 'ges': [], 'novel': [], 'quantum': []
        }
        
        # Data characteristics history
        self.data_characteristics_history: List[Dict[str, float]] = []
        
        # Meta-features for algorithm selection
        self.meta_features = [
            'n_samples', 'n_variables', 'sparsity', 'linearity', 
            'gaussianity', 'edge_density_prior'
        ]
        
        logger.info("Meta-causal learner initialized")
    
    def extract_data_characteristics(self, data: np.ndarray) -> Dict[str, float]:
        """Extract characteristics from data for algorithm selection.
        
        Args:
            data: Input data matrix
            
        Returns:
            Dictionary of data characteristics
        """
        n_samples, n_variables = data.shape
        
        characteristics = {
            'n_samples': float(n_samples),
            'n_variables': float(n_variables),
            'sample_to_var_ratio': float(n_samples / n_variables) if n_variables > 0 else 0.0
        }
        
        # Statistical properties
        try:
            # Sparsity: measure of zero/near-zero correlations
            corr_matrix = np.corrcoef(data.T)
            corr_matrix = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
            characteristics['sparsity'] = float(np.mean(np.abs(corr_matrix) < 0.1))
            
            # Linearity: measure of linear relationships
            linear_scores = []
            for i in range(min(n_variables, 5)):  # Sample a few variables
                for j in range(i+1, min(n_variables, 5)):
                    try:
                        # Linear vs nonlinear fit comparison
                        linear_corr = abs(np.corrcoef(data[:, i], data[:, j])[0, 1])
                        linear_scores.append(linear_corr)
                    except:
                        continue
            
            characteristics['linearity'] = float(np.mean(linear_scores) if linear_scores else 0.5)
            
            # Gaussianity: test for normal distribution
            gaussianity_scores = []
            for i in range(min(n_variables, 5)):
                from scipy.stats import normaltest
                try:
                    _, p_value = normaltest(data[:, i])
                    gaussianity_scores.append(p_value)
                except:
                    continue
            
            characteristics['gaussianity'] = float(np.mean(gaussianity_scores) if gaussianity_scores else 0.5)
            
            # Edge density estimate from mutual information
            edge_density = self._estimate_edge_density(data)
            characteristics['edge_density_prior'] = edge_density
            
        except Exception as e:
            logger.warning(f"Failed to extract some characteristics: {e}")
            # Fill with defaults
            for key in ['sparsity', 'linearity', 'gaussianity', 'edge_density_prior']:
                if key not in characteristics:
                    characteristics[key] = 0.5
        
        return characteristics
    
    def _estimate_edge_density(self, data: np.ndarray) -> float:
        """Estimate expected edge density from data characteristics."""
        try:
            n_vars = data.shape[1]
            if n_vars < 2:
                return 0.0
            
            # Use mutual information to estimate connections
            from sklearn.feature_selection import mutual_info_regression
            
            total_connections = 0
            total_possible = n_vars * (n_vars - 1)
            
            for i in range(min(n_vars, 10)):  # Sample to avoid computational burden
                for j in range(n_vars):
                    if i != j:
                        mi = mutual_info_regression(data[:, [i]], data[:, j])
                        if mi[0] > 0.1:  # Threshold for meaningful connection
                            total_connections += 1
            
            # Scale estimate
            if n_vars > 10:
                scale_factor = n_vars / 10
                total_connections *= scale_factor
                
            edge_density = total_connections / total_possible
            return min(edge_density, 1.0)
            
        except Exception as e:
            logger.warning(f"Edge density estimation failed: {e}")
            return 0.1  # Conservative estimate
    
    def select_best_algorithm(self, data: np.ndarray) -> str:
        """Select best algorithm based on data characteristics and past performance.
        
        Args:
            data: Input data matrix
            
        Returns:
            Name of recommended algorithm
        """
        characteristics = self.extract_data_characteristics(data)
        
        # Simple rule-based selection enhanced with performance history
        algorithm_scores = {}
        
        # Base scores from data characteristics
        n_samples = characteristics['n_samples']
        n_variables = characteristics['n_variables']
        sparsity = characteristics['sparsity']
        linearity = characteristics['linearity']
        
        # PC algorithm scoring
        pc_score = 0.5
        if n_samples > 1000 and sparsity > 0.7:  # PC works well with sparse, large datasets
            pc_score += 0.3
        if linearity > 0.6:  # PC assumes linear relationships
            pc_score += 0.2
        
        # GES algorithm scoring  
        ges_score = 0.5
        if n_variables < 20:  # GES is computationally intensive
            ges_score += 0.3
        if linearity > 0.5:  # GES works well with linear models
            ges_score += 0.2
        
        # Novel algorithm scoring
        novel_score = 0.5
        if sparsity < 0.5:  # Novel method good for dense relationships
            novel_score += 0.2
        if n_samples < 500:  # Works well with smaller datasets
            novel_score += 0.3
        
        # Quantum algorithm scoring
        quantum_score = 0.3  # Lower base score due to experimental nature
        if n_variables <= 10 and n_samples > 100:  # Quantum has qubit limitations
            quantum_score += 0.4
        
        algorithm_scores = {
            'pc': pc_score,
            'ges': ges_score, 
            'novel': novel_score,
            'quantum': quantum_score
        }
        
        # Adjust scores based on historical performance
        for algo_name, base_score in algorithm_scores.items():
            if self.algorithm_performance_history[algo_name]:
                avg_performance = np.mean(self.algorithm_performance_history[algo_name][-10:])
                algorithm_scores[algo_name] = 0.7 * base_score + 0.3 * avg_performance
        
        # Select algorithm with highest score
        best_algorithm = max(algorithm_scores, key=algorithm_scores.get)
        
        logger.info(f"Selected algorithm: {best_algorithm} (scores: {algorithm_scores})")
        return best_algorithm
    
    def update_performance(self, algorithm: str, performance_score: float, 
                          data_characteristics: Dict[str, float]):
        """Update algorithm performance history.
        
        Args:
            algorithm: Algorithm name
            performance_score: Performance score (0-1)
            data_characteristics: Characteristics of the data used
        """
        if algorithm in self.algorithm_performance_history:
            self.algorithm_performance_history[algorithm].append(performance_score)
            
            # Keep only recent performance records
            if len(self.algorithm_performance_history[algorithm]) > self.performance_memory:
                self.algorithm_performance_history[algorithm] = \
                    self.algorithm_performance_history[algorithm][-self.performance_memory:]
        
        # Store data characteristics
        self.data_characteristics_history.append(data_characteristics)
        if len(self.data_characteristics_history) > self.performance_memory:
            self.data_characteristics_history = self.data_characteristics_history[-self.performance_memory:]
    
    def get_algorithm_recommendations(self, data: np.ndarray) -> Dict[str, float]:
        """Get recommendations for all algorithms with confidence scores.
        
        Args:
            data: Input data matrix
            
        Returns:
            Dictionary of algorithm recommendations with confidence scores
        """
        characteristics = self.extract_data_characteristics(data)
        
        recommendations = {}
        
        # Get scores for all algorithms
        for algo in ['pc', 'ges', 'novel', 'quantum']:
            base_confidence = 0.5
            
            # Adjust based on data characteristics
            if algo == 'pc':
                if characteristics['sparsity'] > 0.7 and characteristics['n_samples'] > 1000:
                    base_confidence += 0.3
            elif algo == 'ges':
                if characteristics['n_variables'] < 20 and characteristics['linearity'] > 0.6:
                    base_confidence += 0.3
            elif algo == 'novel':
                if characteristics['edge_density_prior'] > 0.3:
                    base_confidence += 0.2
            elif algo == 'quantum':
                if characteristics['n_variables'] <= 10:
                    base_confidence += 0.2
            
            # Factor in historical performance
            if self.algorithm_performance_history[algo]:
                historical_perf = np.mean(self.algorithm_performance_history[algo][-5:])
                confidence = 0.6 * base_confidence + 0.4 * historical_perf
            else:
                confidence = base_confidence
            
            recommendations[algo] = min(confidence, 1.0)
        
        return recommendations


class HybridCausalDiscovery:
    """Hybrid approach combining multiple discovery paradigms."""
    
    def __init__(self):
        """Initialize hybrid discovery system."""
        self.adaptive_discovery = AdaptiveCausalDiscovery()
        self.structure_learner = CausalStructureLearner(n_bootstrap_samples=50)
        self.meta_learner = MetaCausalLearner()
        
        # Import quantum and distributed components if available
        try:
            from ..quantum_acceleration import quantum_processor
            self.quantum_processor = quantum_processor
            self.quantum_available = True
        except ImportError:
            self.quantum_available = False
            logger.warning("Quantum acceleration not available")
        
        try:
            from ..distributed_computing import task_scheduler
            self.task_scheduler = task_scheduler
            self.distributed_available = True
        except ImportError:
            self.distributed_available = False
            logger.warning("Distributed computing not available")
    
    async def comprehensive_discovery(self, 
                                    data: np.ndarray, 
                                    variable_names: List[str],
                                    use_quantum: bool = True,
                                    use_distributed: bool = True) -> Dict[str, Any]:
        """Perform comprehensive causal discovery using all available methods.
        
        Args:
            data: Input data matrix
            variable_names: Variable names
            use_quantum: Whether to use quantum acceleration
            use_distributed: Whether to use distributed computing
            
        Returns:
            Comprehensive discovery results
        """
        results = {
            'timestamp': time.time(),
            'data_characteristics': {},
            'algorithm_results': {},
            'ensemble_result': None,
            'uncertainty_analysis': {},
            'recommendations': {}
        }
        
        # Extract data characteristics
        data_chars = self.meta_learner.extract_data_characteristics(data)
        results['data_characteristics'] = data_chars
        
        # Get algorithm recommendations
        recommendations = self.meta_learner.get_algorithm_recommendations(data)
        results['recommendations'] = recommendations
        
        logger.info(f"Starting comprehensive discovery with recommendations: {recommendations}")
        
        # Run selected algorithms
        algorithms_to_run = [algo for algo, confidence in recommendations.items() 
                           if confidence > 0.6]
        
        if not algorithms_to_run:
            algorithms_to_run = ['pc', 'novel']  # Fallback
        
        # Classical algorithms
        for algo in algorithms_to_run:
            if algo in ['pc', 'ges', 'novel']:
                try:
                    if algo == 'pc':
                        graph = self.adaptive_discovery._run_pc_algorithm(data, variable_names)
                    elif algo == 'ges':
                        graph = self.adaptive_discovery._run_ges_algorithm(data, variable_names)
                    elif algo == 'novel':
                        graph = self.adaptive_discovery._run_novel_discovery(data, variable_names)
                    
                    results['algorithm_results'][algo] = {
                        'graph': graph,
                        'edges': list(graph.edges()),
                        'n_edges': graph.number_of_edges(),
                        'success': True
                    }
                    
                except Exception as e:
                    logger.error(f"Algorithm {algo} failed: {e}")
                    results['algorithm_results'][algo] = {
                        'success': False,
                        'error': str(e)
                    }
        
        # Quantum algorithm (if available and requested)
        if use_quantum and self.quantum_available and 'quantum' in algorithms_to_run:
            try:
                quantum_result = await self.quantum_processor.quantum_causal_search(
                    variable_names, data, max_parents=3
                )
                results['algorithm_results']['quantum'] = {
                    'graph': quantum_result['causal_graph'],
                    'edges': list(quantum_result['causal_graph'].edges()),
                    'n_edges': quantum_result['causal_graph'].number_of_edges(),
                    'quantum_confidence': quantum_result['quantum_confidence'],
                    'execution_time': quantum_result['execution_time'],
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"Quantum algorithm failed: {e}")
                results['algorithm_results']['quantum'] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Bootstrap uncertainty analysis
        try:
            consensus_graph, edge_probabilities = self.structure_learner.learn_with_uncertainty(
                data, variable_names
            )
            
            results['uncertainty_analysis'] = {
                'consensus_graph': consensus_graph,
                'edge_probabilities': {str(k): v for k, v in edge_probabilities.items()},
                'high_confidence_edges': [
                    str(edge) for edge, prob in edge_probabilities.items() if prob > 0.8
                ],
                'uncertain_edges': [
                    str(edge) for edge, prob in edge_probabilities.items() 
                    if 0.3 < prob < 0.7
                ]
            }
            
        except Exception as e:
            logger.error(f"Uncertainty analysis failed: {e}")
            results['uncertainty_analysis'] = {'error': str(e)}
        
        # Ensemble combination
        successful_graphs = []
        for algo_result in results['algorithm_results'].values():
            if algo_result.get('success') and 'graph' in algo_result:
                successful_graphs.append(algo_result['graph'])
        
        if successful_graphs:
            ensemble_graph = self.adaptive_discovery._combine_results(
                successful_graphs, variable_names
            )
            
            results['ensemble_result'] = {
                'graph': ensemble_graph,
                'edges': list(ensemble_graph.edges()),
                'n_edges': ensemble_graph.number_of_edges(),
                'contributing_algorithms': len(successful_graphs)
            }
        
        # Update meta-learner performance
        for algo, result in results['algorithm_results'].items():
            if result.get('success'):
                # Simple performance metric based on edge count and data size
                performance = min(1.0, result['n_edges'] / (len(variable_names) * 0.5))
                self.meta_learner.update_performance(algo, performance, data_chars)
        
        logger.info("Comprehensive discovery completed")
        return results