"""Advanced LLM benchmarking suite for causal reasoning evaluation."""

import asyncio
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import statistics
from scipy import stats
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result for an LLM."""
    model_name: str
    overall_causal_score: float
    category_scores: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    statistical_significance: Dict[str, float]
    response_times: List[float]
    error_analysis: Dict[str, Any]
    comparative_ranking: Optional[int] = None


@dataclass
class CausalReasoningTask:
    """Individual causal reasoning task."""
    task_id: str
    category: str
    description: str
    causal_graph: nx.DiGraph
    intervention_scenario: Dict[str, Any]
    ground_truth: Dict[str, Any]
    difficulty_level: int  # 1-5 scale
    required_concepts: List[str]


class AdvancedLLMBenchmarker:
    """Comprehensive LLM causal reasoning benchmarking system."""
    
    def __init__(self, 
                 timeout_seconds: int = 120,
                 max_concurrent_requests: int = 5,
                 statistical_significance_threshold: float = 0.05):
        """Initialize advanced benchmarker.
        
        Args:
            timeout_seconds: Request timeout
            max_concurrent_requests: Max concurrent LLM requests
            statistical_significance_threshold: P-value threshold for significance
        """
        self.timeout_seconds = timeout_seconds
        self.max_concurrent_requests = max_concurrent_requests
        self.significance_threshold = statistical_significance_threshold
        self.task_suite = self._initialize_comprehensive_task_suite()
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        
    def _initialize_comprehensive_task_suite(self) -> List[CausalReasoningTask]:
        """Initialize comprehensive task suite covering all causal reasoning aspects."""
        tasks = []
        
        # Category 1: Intervention vs Observation
        tasks.extend(self._create_intervention_observation_tasks())
        
        # Category 2: Backdoor Path Identification
        tasks.extend(self._create_backdoor_identification_tasks())
        
        # Category 3: Frontdoor Adjustment
        tasks.extend(self._create_frontdoor_adjustment_tasks())
        
        # Category 4: Instrumental Variables
        tasks.extend(self._create_instrumental_variable_tasks())
        
        # Category 5: Counterfactual Reasoning
        tasks.extend(self._create_counterfactual_tasks())
        
        # Category 6: Simpson's Paradox
        tasks.extend(self._create_simpsons_paradox_tasks())
        
        # Category 7: Mediation Analysis
        tasks.extend(self._create_mediation_tasks())
        
        # Category 8: Collider Bias
        tasks.extend(self._create_collider_bias_tasks())
        
        logger.info(f"Initialized {len(tasks)} causal reasoning tasks")
        return tasks
        
    async def benchmark_model(self, 
                            model_client: Any,
                            model_name: str,
                            num_trials: int = 3) -> BenchmarkResult:
        """Benchmark a single LLM across all causal reasoning tasks.
        
        Args:
            model_client: Client for LLM API calls
            model_name: Name identifier for the model
            num_trials: Number of trials per task for statistical robustness
            
        Returns:
            Comprehensive benchmark result
        """
        logger.info(f"Starting comprehensive benchmark for {model_name}")
        
        task_results = {}
        response_times = []
        error_count = 0
        
        # Process tasks with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_task_with_trials(task: CausalReasoningTask):
            trial_results = []
            trial_times = []
            
            for trial in range(num_trials):
                try:
                    async with semaphore:
                        start_time = time.time()
                        result = await self._evaluate_task(model_client, task)
                        end_time = time.time()
                        
                        trial_results.append(result)
                        trial_times.append(end_time - start_time)
                        
                except Exception as e:
                    logger.error(f"Task {task.task_id} trial {trial} failed: {e}")
                    trial_results.append({"success": False, "error": str(e)})
                    
            return task.task_id, trial_results, trial_times
            
        # Execute all tasks
        tasks = [process_task_with_trials(task) for task in self.task_suite]
        
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in completed_tasks:
            if isinstance(result, Exception):
                error_count += 1
                logger.error(f"Task failed: {result}")
                continue
                
            task_id, trial_results, trial_times = result
            task_results[task_id] = {
                'results': trial_results,
                'response_times': trial_times
            }
            response_times.extend(trial_times)
            
        # Compute comprehensive metrics
        category_scores = self._compute_category_scores(task_results)
        overall_score = self._compute_overall_causal_score(category_scores)
        detailed_metrics = self._compute_detailed_metrics(task_results)
        statistical_tests = self._perform_statistical_analysis(task_results)
        error_analysis = self._analyze_errors(task_results, error_count)
        
        result = BenchmarkResult(
            model_name=model_name,
            overall_causal_score=overall_score,
            category_scores=category_scores,
            detailed_metrics=detailed_metrics,
            statistical_significance=statistical_tests,
            response_times=response_times,
            error_analysis=error_analysis
        )
        
        logger.info(f"Benchmark completed for {model_name}. Overall score: {overall_score:.3f}")
        return result
        
    async def comparative_benchmark(self,
                                  model_configs: List[Dict[str, Any]],
                                  num_trials: int = 3) -> Dict[str, BenchmarkResult]:
        """Run comparative benchmark across multiple LLMs.
        
        Args:
            model_configs: List of model configuration dictionaries
            num_trials: Number of trials per task
            
        Returns:
            Dictionary mapping model names to benchmark results
        """
        logger.info(f"Starting comparative benchmark for {len(model_configs)} models")
        
        results = {}
        
        # Benchmark each model
        for config in model_configs:
            model_name = config['name']
            model_client = config['client']
            
            try:
                result = await self.benchmark_model(model_client, model_name, num_trials)
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Benchmark failed for {model_name}: {e}")
                
        # Add comparative rankings
        if len(results) > 1:
            ranked_models = sorted(results.items(), 
                                 key=lambda x: x[1].overall_causal_score, 
                                 reverse=True)
            
            for rank, (model_name, result) in enumerate(ranked_models, 1):
                result.comparative_ranking = rank
                
        # Store as baseline for future comparisons
        self.baseline_results.update(results)
        
        return results
        
    async def _evaluate_task(self, model_client: Any, task: CausalReasoningTask) -> Dict[str, Any]:
        """Evaluate a single causal reasoning task."""
        
        # Construct prompt based on task type
        prompt = self._construct_task_prompt(task)
        
        try:
            # Get model response
            response = await asyncio.wait_for(
                model_client.complete(prompt),
                timeout=self.timeout_seconds
            )
            
            # Parse and evaluate response
            parsed_response = self._parse_model_response(response, task)
            evaluation = self._evaluate_response_accuracy(parsed_response, task)
            
            return {
                'success': True,
                'response': response,
                'parsed_response': parsed_response,
                'evaluation': evaluation,
                'accuracy': evaluation.get('accuracy', 0.0),
                'reasoning_quality': evaluation.get('reasoning_quality', 0.0)
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': 'timeout',
                'accuracy': 0.0,
                'reasoning_quality': 0.0
            }
        except Exception as e:
            return {
                'success': False, 
                'error': str(e),
                'accuracy': 0.0,
                'reasoning_quality': 0.0
            }
            
    def _construct_task_prompt(self, task: CausalReasoningTask) -> str:
        """Construct a detailed prompt for the causal reasoning task."""
        
        prompt_templates = {
            'intervention_vs_observation': """
You are evaluating causal relationships. Consider the following scenario:

Causal Structure: {graph_description}
Scenario: {scenario}

Question: {question}

Please provide:
1. Your reasoning process
2. The specific causal mechanism involved  
3. Your final answer with confidence level (0-1)

Focus on distinguishing between interventional and observational relationships.
            """,
            
            'backdoor_identification': """
You are analyzing confounding in causal relationships. Given:

Variables: {variables}
Relationships: {relationships}
Query: {query}

Task: Identify all backdoor paths and determine if adjustment is needed.

Provide:
1. All backdoor paths from cause to effect
2. Minimal adjustment set (if any)
3. Explanation of your reasoning
4. Confidence in your answer (0-1)
            """,
            
            'counterfactual': """
You are reasoning about counterfactual scenarios. Consider:

Observed Situation: {observed}
Causal Model: {model}
Counterfactual Question: {counterfactual_query}

Determine:
1. What would have happened if circumstances were different?
2. Your step-by-step reasoning
3. Confidence level (0-1)

Be precise about the counterfactual inference rules you apply.
            """
        }
        
        template = prompt_templates.get(task.category, prompt_templates['intervention_vs_observation'])
        
        # Fill in task-specific details
        return template.format(
            graph_description=self._describe_causal_graph(task.causal_graph),
            scenario=task.intervention_scenario.get('description', ''),
            question=task.intervention_scenario.get('question', ''),
            variables=list(task.causal_graph.nodes()),
            relationships=list(task.causal_graph.edges()),
            query=task.intervention_scenario.get('query', ''),
            observed=task.intervention_scenario.get('observed_situation', ''),
            model=self._describe_causal_graph(task.causal_graph),
            counterfactual_query=task.intervention_scenario.get('counterfactual_query', '')
        )
        
    def _describe_causal_graph(self, graph: nx.DiGraph) -> str:
        """Convert graph to natural language description."""
        descriptions = []
        
        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            if parents:
                if len(parents) == 1:
                    descriptions.append(f"{parents[0]} causes {node}")
                else:
                    parent_str = ", ".join(parents[:-1]) + f", and {parents[-1]}"
                    descriptions.append(f"{parent_str} cause {node}")
            else:
                descriptions.append(f"{node} is an exogenous variable")
                
        return ". ".join(descriptions)
        
    def _parse_model_response(self, response: str, task: CausalReasoningTask) -> Dict[str, Any]:
        """Parse model response into structured format."""
        
        parsed = {
            'raw_response': response,
            'reasoning': '',
            'answer': None,
            'confidence': 0.0,
            'mentioned_concepts': []
        }
        
        # Extract reasoning (look for numbered lists or structured explanations)
        lines = response.split('\n')
        reasoning_lines = []
        
        for line in lines:
            line = line.strip()
            if any(starter in line.lower() for starter in ['reason', 'because', 'since', 'therefore']):
                reasoning_lines.append(line)
                
        parsed['reasoning'] = ' '.join(reasoning_lines)
        
        # Extract confidence (look for patterns like "confidence: 0.8" or "80% confident")
        confidence_patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9]+)%\s*confident',
            r'probability[:\s]+([0-9.]+)'
        ]
        
        import re
        for pattern in confidence_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    conf_val = float(match.group(1))
                    if conf_val > 1:  # Convert percentage to decimal
                        conf_val /= 100
                    parsed['confidence'] = min(1.0, max(0.0, conf_val))
                    break
                except (ValueError, IndexError):
                    continue
                    
        # Extract mentioned causal concepts
        causal_concepts = [
            'intervention', 'observation', 'confounding', 'backdoor',
            'frontdoor', 'collider', 'mediator', 'instrumental variable',
            'counterfactual', 'simpson\'s paradox', 'causal effect'
        ]
        
        for concept in causal_concepts:
            if concept.lower() in response.lower():
                parsed['mentioned_concepts'].append(concept)
                
        # Extract final answer (task-specific logic)
        if task.category == 'intervention_vs_observation':
            if any(word in response.lower() for word in ['intervention', 'do(', 'manipulate']):
                parsed['answer'] = 'intervention'
            elif any(word in response.lower() for word in ['observe', 'correlation', 'association']):
                parsed['answer'] = 'observation'
                
        return parsed
        
    def _evaluate_response_accuracy(self, parsed_response: Dict[str, Any], 
                                  task: CausalReasoningTask) -> Dict[str, float]:
        """Evaluate accuracy of model response against ground truth."""
        
        evaluation = {
            'accuracy': 0.0,
            'reasoning_quality': 0.0,
            'concept_coverage': 0.0,
            'confidence_calibration': 0.0
        }
        
        ground_truth = task.ground_truth
        
        # Accuracy score
        if 'answer' in ground_truth and parsed_response['answer'] is not None:
            if parsed_response['answer'] == ground_truth['answer']:
                evaluation['accuracy'] = 1.0
            elif self._answers_similar(parsed_response['answer'], ground_truth['answer']):
                evaluation['accuracy'] = 0.7
                
        # Reasoning quality (based on mentioned concepts and explanation depth)
        required_concepts = set(task.required_concepts)
        mentioned_concepts = set(parsed_response['mentioned_concepts'])
        
        concept_overlap = len(required_concepts.intersection(mentioned_concepts))
        if len(required_concepts) > 0:
            evaluation['concept_coverage'] = concept_overlap / len(required_concepts)
            
        # Reasoning depth score
        reasoning_length = len(parsed_response['reasoning'].split())
        reasoning_score = min(1.0, reasoning_length / 50)  # Normalize around 50 words
        
        evaluation['reasoning_quality'] = 0.5 * evaluation['concept_coverage'] + 0.5 * reasoning_score
        
        # Confidence calibration (how well confidence matches accuracy)
        if parsed_response['confidence'] > 0:
            conf_diff = abs(parsed_response['confidence'] - evaluation['accuracy'])
            evaluation['confidence_calibration'] = 1.0 - conf_diff
        else:
            evaluation['confidence_calibration'] = 0.5  # Neutral for no confidence given
            
        return evaluation
        
    def _answers_similar(self, answer1: Any, answer2: Any) -> bool:
        """Check if two answers are semantically similar."""
        if isinstance(answer1, str) and isinstance(answer2, str):
            # Simple string similarity
            answer1, answer2 = answer1.lower().strip(), answer2.lower().strip()
            
            # Check for synonyms
            synonyms = {
                'intervention': ['manipulate', 'do', 'force', 'control'],
                'observation': ['observe', 'see', 'correlate', 'associate'],
                'yes': ['true', 'correct', 'right'],
                'no': ['false', 'incorrect', 'wrong']
            }
            
            for key, values in synonyms.items():
                if (answer1 == key and answer2 in values) or (answer2 == key and answer1 in values):
                    return True
                    
            return answer1 == answer2
            
        return answer1 == answer2
        
    def _compute_category_scores(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute scores by causal reasoning category."""
        category_scores = {}
        category_tasks = {}
        
        # Group tasks by category
        for task in self.task_suite:
            if task.category not in category_tasks:
                category_tasks[task.category] = []
            category_tasks[task.category].append(task.task_id)
            
        # Compute average scores per category
        for category, task_ids in category_tasks.items():
            scores = []
            
            for task_id in task_ids:
                if task_id in task_results:
                    results = task_results[task_id]['results']
                    
                    # Average across trials
                    trial_scores = []
                    for result in results:
                        if result.get('success', False):
                            eval_score = result.get('evaluation', {})
                            # Weighted combination of accuracy and reasoning
                            combined_score = (0.6 * eval_score.get('accuracy', 0) + 
                                           0.4 * eval_score.get('reasoning_quality', 0))
                            trial_scores.append(combined_score)
                            
                    if trial_scores:
                        scores.append(statistics.mean(trial_scores))
                        
            category_scores[category] = statistics.mean(scores) if scores else 0.0
            
        return category_scores
        
    def _compute_overall_causal_score(self, category_scores: Dict[str, float]) -> float:
        """Compute overall causal reasoning score with weighted categories."""
        
        # Category weights based on fundamental importance
        weights = {
            'intervention_vs_observation': 0.25,  # Core distinction
            'backdoor_identification': 0.20,     # Confounding control
            'counterfactual': 0.15,              # Advanced reasoning
            'frontdoor_adjustment': 0.10,        # Mediation understanding
            'instrumental_variables': 0.10,      # Advanced causal ID
            'simpsons_paradox': 0.10,           # Paradox resolution
            'mediation_analysis': 0.05,          # Effect decomposition
            'collider_bias': 0.05               # Selection bias
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = weights.get(category, 0.05)  # Default weight for unlisted categories
            weighted_sum += weight * score
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0
        
    def _compute_detailed_metrics(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute detailed performance metrics."""
        
        metrics = {
            'total_tasks': len(task_results),
            'successful_tasks': 0,
            'average_accuracy': 0.0,
            'average_reasoning_quality': 0.0,
            'confidence_calibration': 0.0,
            'difficulty_breakdown': {},
            'response_consistency': 0.0
        }
        
        all_accuracies = []
        all_reasoning_scores = []
        all_confidence_calibrations = []
        difficulty_scores = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        for task in self.task_suite:
            if task.task_id in task_results:
                results = task_results[task.task_id]['results']
                
                # Process all trials for this task
                task_accuracies = []
                task_reasoning = []
                task_calibrations = []
                
                for result in results:
                    if result.get('success', False):
                        metrics['successful_tasks'] += 1
                        eval_data = result.get('evaluation', {})
                        
                        acc = eval_data.get('accuracy', 0.0)
                        reasoning = eval_data.get('reasoning_quality', 0.0)
                        calibration = eval_data.get('confidence_calibration', 0.0)
                        
                        task_accuracies.append(acc)
                        task_reasoning.append(reasoning)
                        task_calibrations.append(calibration)
                        
                        all_accuracies.append(acc)
                        all_reasoning_scores.append(reasoning)
                        all_confidence_calibrations.append(calibration)
                        
                        # Group by difficulty
                        difficulty_scores[task.difficulty_level].append(acc)
                        
                # Response consistency across trials
                if len(task_accuracies) > 1:
                    consistency = 1.0 - np.std(task_accuracies)
                    metrics['response_consistency'] += max(0, consistency)
                    
        # Compute averages
        if all_accuracies:
            metrics['average_accuracy'] = statistics.mean(all_accuracies)
            metrics['average_reasoning_quality'] = statistics.mean(all_reasoning_scores)
            metrics['confidence_calibration'] = statistics.mean(all_confidence_calibrations)
            
        # Difficulty breakdown
        for level, scores in difficulty_scores.items():
            if scores:
                metrics['difficulty_breakdown'][f'level_{level}'] = statistics.mean(scores)
                
        metrics['response_consistency'] /= len(task_results) if task_results else 1
        
        return metrics
        
    def _perform_statistical_analysis(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Perform statistical significance tests."""
        
        statistical_tests = {}
        
        # Collect scores for analysis
        all_scores = []
        for task_results_data in task_results.values():
            for result in task_results_data['results']:
                if result.get('success', False):
                    eval_data = result.get('evaluation', {})
                    score = 0.6 * eval_data.get('accuracy', 0) + 0.4 * eval_data.get('reasoning_quality', 0)
                    all_scores.append(score)
                    
        if len(all_scores) >= 10:  # Minimum sample size for meaningful tests
            # Test if performance significantly different from random (0.5)
            t_stat, p_value = stats.ttest_1samp(all_scores, 0.5)
            statistical_tests['better_than_random_p'] = p_value
            statistical_tests['better_than_random'] = p_value < self.significance_threshold
            
            # Test normality of scores
            shapiro_stat, shapiro_p = stats.shapiro(all_scores[:min(50, len(all_scores))])
            statistical_tests['normality_p'] = shapiro_p
            
            # Confidence intervals
            mean_score = statistics.mean(all_scores)
            std_error = stats.sem(all_scores)
            ci_lower, ci_upper = stats.t.interval(
                0.95, len(all_scores)-1, loc=mean_score, scale=std_error)
            statistical_tests['confidence_interval_95'] = [ci_lower, ci_upper]
            
        return statistical_tests
        
    def _analyze_errors(self, task_results: Dict[str, Any], total_errors: int) -> Dict[str, Any]:
        """Analyze error patterns and failure modes."""
        
        error_analysis = {
            'total_errors': total_errors,
            'error_rate': 0.0,
            'common_error_types': {},
            'timeout_rate': 0.0,
            'parsing_failures': 0
        }
        
        total_attempts = 0
        timeout_count = 0
        parsing_errors = 0
        error_types = {}
        
        for task_results_data in task_results.values():
            for result in task_results_data['results']:
                total_attempts += 1
                
                if not result.get('success', False):
                    error_type = result.get('error', 'unknown')
                    
                    if error_type not in error_types:
                        error_types[error_type] = 0
                    error_types[error_type] += 1
                    
                    if error_type == 'timeout':
                        timeout_count += 1
                    elif 'parsing' in error_type.lower():
                        parsing_errors += 1
                        
        if total_attempts > 0:
            error_analysis['error_rate'] = total_errors / total_attempts
            error_analysis['timeout_rate'] = timeout_count / total_attempts
            
        error_analysis['common_error_types'] = error_types
        error_analysis['parsing_failures'] = parsing_errors
        
        return error_analysis
        
    # Task creation methods
    def _create_intervention_observation_tasks(self) -> List[CausalReasoningTask]:
        """Create tasks testing intervention vs observation understanding."""
        tasks = []
        
        # Classic Simpson's paradox scenario
        graph = nx.DiGraph()
        graph.add_edges_from([('Department', 'Gender'), ('Department', 'Admission'), ('Gender', 'Admission')])
        
        task = CausalReasoningTask(
            task_id="intervention_obs_1",
            category="intervention_vs_observation",
            description="University admission bias analysis",
            causal_graph=graph,
            intervention_scenario={
                "description": "A university shows overall lower admission rates for women, but within each department, women have higher admission rates.",
                "question": "If we want to measure gender discrimination in admissions, should we look at P(Admission|Gender) or P(Admission|do(Gender))?"
            },
            ground_truth={"answer": "intervention"},
            difficulty_level=3,
            required_concepts=["intervention", "confounding", "simpson's paradox"]
        )
        tasks.append(task)
        
        # Add more intervention vs observation tasks...
        
        return tasks
        
    def _create_backdoor_identification_tasks(self) -> List[CausalReasoningTask]:
        """Create backdoor path identification tasks."""
        tasks = []
        
        # Classic confounding scenario
        graph = nx.DiGraph()
        graph.add_edges_from([('Z', 'X'), ('Z', 'Y'), ('X', 'Y')])
        
        task = CausalReasoningTask(
            task_id="backdoor_1",
            category="backdoor_identification", 
            description="Basic confounding control",
            causal_graph=graph,
            intervention_scenario={
                "query": "Causal effect of X on Y",
                "relationships": "Z causes both X and Y, X causes Y"
            },
            ground_truth={"answer": ["Z"], "explanation": "Z is a confounder that opens a backdoor path"},
            difficulty_level=2,
            required_concepts=["backdoor", "confounding", "adjustment"]
        )
        tasks.append(task)
        
        return tasks
        
    def _create_frontdoor_adjustment_tasks(self) -> List[CausalReasoningTask]:
        """Create frontdoor adjustment tasks."""
        # Implementation for frontdoor criterion tasks
        return []
        
    def _create_instrumental_variable_tasks(self) -> List[CausalReasoningTask]:
        """Create instrumental variable tasks."""
        # Implementation for IV tasks
        return []
        
    def _create_counterfactual_tasks(self) -> List[CausalReasoningTask]:
        """Create counterfactual reasoning tasks."""
        # Implementation for counterfactual tasks
        return []
        
    def _create_simpsons_paradox_tasks(self) -> List[CausalReasoningTask]:
        """Create Simpson's paradox tasks."""
        # Implementation for Simpson's paradox tasks
        return []
        
    def _create_mediation_tasks(self) -> List[CausalReasoningTask]:
        """Create mediation analysis tasks."""
        # Implementation for mediation tasks
        return []
        
    def _create_collider_bias_tasks(self) -> List[CausalReasoningTask]:
        """Create collider bias tasks."""
        # Implementation for collider bias tasks
        return []


class CausalReasoningMetrics:
    """Comprehensive metrics for evaluating causal reasoning capabilities."""
    
    def __init__(self):
        self.benchmark_history: List[BenchmarkResult] = []
        
    def compute_meta_metrics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compute meta-level metrics across multiple benchmark runs."""
        
        if not results:
            return {}
            
        meta_metrics = {
            'mean_overall_score': statistics.mean([r.overall_causal_score for r in results]),
            'score_variance': statistics.variance([r.overall_causal_score for r in results]) if len(results) > 1 else 0,
            'best_model': max(results, key=lambda x: x.overall_causal_score).model_name,
            'worst_model': min(results, key=lambda x: x.overall_causal_score).model_name,
            'category_rankings': self._compute_category_rankings(results),
            'performance_gaps': self._compute_performance_gaps(results)
        }
        
        return meta_metrics
        
    def _compute_category_rankings(self, results: List[BenchmarkResult]) -> Dict[str, List[str]]:
        """Compute model rankings for each category."""
        rankings = {}
        
        # Get all categories
        all_categories = set()
        for result in results:
            all_categories.update(result.category_scores.keys())
            
        # Rank models in each category
        for category in all_categories:
            category_results = [(r.model_name, r.category_scores.get(category, 0)) 
                              for r in results]
            category_results.sort(key=lambda x: x[1], reverse=True)
            rankings[category] = [name for name, score in category_results]
            
        return rankings
        
    def _compute_performance_gaps(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """Compute performance gaps between models."""
        if len(results) < 2:
            return {}
            
        scores = [r.overall_causal_score for r in results]
        
        return {
            'max_gap': max(scores) - min(scores),
            'std_dev': statistics.stdev(scores),
            'score_range': [min(scores), max(scores)]
        }