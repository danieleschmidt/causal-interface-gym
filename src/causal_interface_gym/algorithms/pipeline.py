"""Data analysis pipelines for causal reasoning experiments."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from abc import ABC, abstractmethod

from ..core import CausalEnvironment
from ..metrics import BeliefState
from .scoring import CausalScore, CausalScorer

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    DATA_COLLECTION = "data_collection"
    PREPROCESSING = "preprocessing"
    ANALYSIS = "analysis"
    SCORING = "scoring"
    AGGREGATION = "aggregation"
    REPORTING = "reporting"


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    pipeline_id: str
    stage: PipelineStage
    data: Any
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentData:
    """Structured experiment data for pipeline processing."""
    experiment_id: str
    agent_id: str
    environment: CausalEnvironment
    belief_history: List[BeliefState]
    interventions: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineStageProcessor(ABC):
    """Abstract base class for pipeline stage processors."""
    
    @abstractmethod
    async def process(self, data: Any, context: Dict[str, Any]) -> PipelineResult:
        """Process data through this pipeline stage.
        
        Args:
            data: Input data
            context: Processing context
            
        Returns:
            Pipeline result
        """
        pass


class DataCollectionProcessor(PipelineStageProcessor):
    """Collects and validates experiment data."""
    
    async def process(self, data: ExperimentData, context: Dict[str, Any]) -> PipelineResult:
        """Collect and validate experiment data.
        
        Args:
            data: Experiment data
            context: Processing context
            
        Returns:
            Processing result
        """
        start_time = datetime.now()
        
        try:
            # Validate required fields
            if not data.experiment_id:
                raise ValueError("Missing experiment_id")
            
            if not data.belief_history:
                raise ValueError("No belief history available")
            
            if not data.interventions:
                logger.warning(f"No interventions found for experiment {data.experiment_id}")
            
            # Collect additional metrics
            collection_data = {
                "experiment_id": data.experiment_id,
                "agent_id": data.agent_id,
                "num_beliefs": len(data.belief_history),
                "num_interventions": len(data.interventions),
                "environment_variables": list(data.environment.variables.keys()),
                "belief_timestamps": [b.timestamp.isoformat() for b in data.belief_history],
                "intervention_timestamps": [i.get("timestamp", "") for i in data.interventions],
                "data_quality_score": self._assess_data_quality(data)
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                pipeline_id=context.get("pipeline_id", "unknown"),
                stage=PipelineStage.DATA_COLLECTION,
                data=collection_data,
                success=True,
                processing_time=processing_time,
                metadata={"validation_passed": True}
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PipelineResult(
                pipeline_id=context.get("pipeline_id", "unknown"),
                stage=PipelineStage.DATA_COLLECTION,
                data=None,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _assess_data_quality(self, data: ExperimentData) -> float:
        """Assess quality of experiment data.
        
        Args:
            data: Experiment data
            
        Returns:
            Quality score (0-1)
        """
        quality_factors = []
        
        # Belief consistency
        if len(data.belief_history) >= 2:
            belief_changes = []
            for i in range(1, len(data.belief_history)):
                prev_beliefs = data.belief_history[i-1].causal_beliefs
                curr_beliefs = data.belief_history[i].causal_beliefs
                
                # Calculate change in beliefs
                total_change = 0
                for key in set(prev_beliefs.keys()) | set(curr_beliefs.keys()):
                    prev_val = prev_beliefs.get(key, 0)
                    curr_val = curr_beliefs.get(key, 0)
                    total_change += abs(curr_val - prev_val)
                
                belief_changes.append(total_change)
            
            # Reasonable belief evolution (not too erratic)
            avg_change = np.mean(belief_changes)
            consistency_score = max(0, 1 - avg_change / 2)  # Normalize
            quality_factors.append(consistency_score)
        
        # Intervention quality
        if data.interventions:
            intervention_quality = 1.0  # Default good quality
            for intervention in data.interventions:
                if "variable" not in intervention or "value" not in intervention:
                    intervention_quality *= 0.8  # Penalize incomplete data
            quality_factors.append(intervention_quality)
        
        # Temporal ordering
        belief_times = [b.timestamp for b in data.belief_history]
        if len(belief_times) > 1:
            temporal_score = 1.0 if all(belief_times[i] <= belief_times[i+1] 
                                      for i in range(len(belief_times)-1)) else 0.5
            quality_factors.append(temporal_score)
        
        return np.mean(quality_factors) if quality_factors else 0.5


class PreprocessingProcessor(PipelineStageProcessor):
    """Preprocesses experiment data for analysis."""
    
    async def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        """Preprocess experiment data.
        
        Args:
            data: Data from collection stage
            context: Processing context
            
        Returns:
            Processing result
        """
        start_time = datetime.now()
        
        try:
            # Extract experiment data from context
            experiment_data = context.get("experiment_data")
            if not experiment_data:
                raise ValueError("No experiment data in context")
            
            # Normalize belief values
            normalized_beliefs = self._normalize_beliefs(experiment_data.belief_history)
            
            # Extract features for analysis
            features = self._extract_features(experiment_data)
            
            # Clean intervention data
            cleaned_interventions = self._clean_interventions(experiment_data.interventions)
            
            # Create time series data
            time_series = self._create_time_series(experiment_data)
            
            preprocessed_data = {
                "normalized_beliefs": normalized_beliefs,
                "features": features,
                "cleaned_interventions": cleaned_interventions,
                "time_series": time_series,
                "original_data": data
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                pipeline_id=context.get("pipeline_id", "unknown"),
                stage=PipelineStage.PREPROCESSING,
                data=preprocessed_data,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PipelineResult(
                pipeline_id=context.get("pipeline_id", "unknown"),
                stage=PipelineStage.PREPROCESSING,
                data=None,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _normalize_beliefs(self, beliefs: List[BeliefState]) -> List[Dict[str, float]]:
        """Normalize belief values.
        
        Args:
            beliefs: Raw belief states
            
        Returns:
            Normalized belief data
        """
        normalized = []
        
        for belief_state in beliefs:
            normalized_belief = {}
            
            # Ensure all belief values are in [0, 1] range
            for relationship, strength in belief_state.causal_beliefs.items():
                normalized_belief[relationship] = max(0, min(1, strength))
            
            normalized.append(normalized_belief)
        
        return normalized
    
    def _extract_features(self, experiment_data: ExperimentData) -> Dict[str, Any]:
        """Extract features for analysis.
        
        Args:
            experiment_data: Experiment data
            
        Returns:
            Extracted features
        """
        features = {}
        
        # Basic features
        features["num_variables"] = len(experiment_data.environment.variables)
        features["num_interventions"] = len(experiment_data.interventions)
        features["belief_evolution_length"] = len(experiment_data.belief_history)
        
        # Belief stability features
        if len(experiment_data.belief_history) >= 2:
            changes = []
            for i in range(1, len(experiment_data.belief_history)):
                prev = experiment_data.belief_history[i-1].causal_beliefs
                curr = experiment_data.belief_history[i].causal_beliefs
                
                total_change = sum(abs(curr.get(k, 0) - prev.get(k, 0)) 
                                 for k in set(prev.keys()) | set(curr.keys()))
                changes.append(total_change)
            
            features["belief_stability"] = 1 - (np.mean(changes) if changes else 0)
            features["belief_volatility"] = np.std(changes) if len(changes) > 1 else 0
        
        # Intervention features
        if experiment_data.interventions:
            intervention_values = [i.get("value", 0) for i in experiment_data.interventions]
            features["intervention_magnitude_mean"] = np.mean(intervention_values)
            features["intervention_magnitude_std"] = np.std(intervention_values)
            
            # Intervention timing
            timestamps = [i.get("timestamp") for i in experiment_data.interventions 
                         if i.get("timestamp")]
            if len(timestamps) > 1:
                # Convert to relative timing if timestamps available
                features["intervention_frequency"] = len(timestamps)
        
        return features
    
    def _clean_interventions(self, interventions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean intervention data.
        
        Args:
            interventions: Raw intervention data
            
        Returns:
            Cleaned interventions
        """
        cleaned = []
        
        for intervention in interventions:
            if "variable" in intervention and "value" in intervention:
                clean_intervention = {
                    "variable": str(intervention["variable"]),
                    "value": float(intervention["value"]),
                    "timestamp": intervention.get("timestamp", ""),
                    "scope": intervention.get("scope", "single"),
                    "method": intervention.get("method", "direct")
                }
                cleaned.append(clean_intervention)
        
        return cleaned
    
    def _create_time_series(self, experiment_data: ExperimentData) -> Dict[str, List]:
        """Create time series representation of experiment.
        
        Args:
            experiment_data: Experiment data
            
        Returns:
            Time series data
        """
        time_series = {
            "timestamps": [],
            "belief_vectors": [],
            "intervention_events": []
        }
        
        # Extract belief evolution timeline
        for belief_state in experiment_data.belief_history:
            time_series["timestamps"].append(belief_state.timestamp.isoformat())
            
            # Create belief vector (ordered by relationship names)
            relationships = sorted(belief_state.causal_beliefs.keys())
            belief_vector = [belief_state.causal_beliefs.get(rel, 0) for rel in relationships]
            time_series["belief_vectors"].append(belief_vector)
        
        # Add intervention events
        for intervention in experiment_data.interventions:
            time_series["intervention_events"].append({
                "timestamp": intervention.get("timestamp", ""),
                "variable": intervention.get("variable"),
                "value": intervention.get("value")
            })
        
        return time_series


class AnalysisProcessor(PipelineStageProcessor):
    """Performs advanced analysis on preprocessed data."""
    
    async def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        """Perform analysis on preprocessed data.
        
        Args:
            data: Preprocessed data
            context: Processing context
            
        Returns:
            Analysis results
        """
        start_time = datetime.now()
        
        try:
            experiment_data = context.get("experiment_data")
            
            # Perform different types of analysis
            analysis_results = {}
            
            # Causal learning analysis
            learning_analysis = await self._analyze_causal_learning(data, experiment_data)
            analysis_results["causal_learning"] = learning_analysis
            
            # Intervention effectiveness analysis
            intervention_analysis = await self._analyze_interventions(data, experiment_data)
            analysis_results["intervention_effectiveness"] = intervention_analysis
            
            # Belief evolution analysis
            evolution_analysis = await self._analyze_belief_evolution(data)
            analysis_results["belief_evolution"] = evolution_analysis
            
            # Performance patterns
            pattern_analysis = await self._analyze_performance_patterns(data, experiment_data)
            analysis_results["performance_patterns"] = pattern_analysis
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                pipeline_id=context.get("pipeline_id", "unknown"),
                stage=PipelineStage.ANALYSIS,
                data=analysis_results,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PipelineResult(
                pipeline_id=context.get("pipeline_id", "unknown"),
                stage=PipelineStage.ANALYSIS,
                data=None,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def _analyze_causal_learning(self, data: Dict[str, Any], 
                                     experiment_data: ExperimentData) -> Dict[str, Any]:
        """Analyze causal learning progression.
        
        Args:
            data: Preprocessed data
            experiment_data: Original experiment data
            
        Returns:
            Causal learning analysis
        """
        # Track discovery of true causal relationships
        true_relationships = set(experiment_data.ground_truth.get("causal_graph", {}).keys())
        discovery_timeline = []
        
        for i, belief_dict in enumerate(data["normalized_beliefs"]):
            discovered = set()
            for relationship, strength in belief_dict.items():
                if strength > 0.7:  # High confidence threshold
                    discovered.add(relationship)
            
            # Calculate discovery metrics
            true_positives = len(discovered.intersection(true_relationships))
            false_positives = len(discovered - true_relationships)
            false_negatives = len(true_relationships - discovered)
            
            precision = true_positives / max(len(discovered), 1)
            recall = true_positives / max(len(true_relationships), 1)
            f1 = 2 * precision * recall / max(precision + recall, 0.001)
            
            discovery_timeline.append({
                "step": i,
                "discovered_count": len(discovered),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
        
        return {
            "discovery_timeline": discovery_timeline,
            "final_precision": discovery_timeline[-1]["precision"] if discovery_timeline else 0,
            "final_recall": discovery_timeline[-1]["recall"] if discovery_timeline else 0,
            "final_f1": discovery_timeline[-1]["f1_score"] if discovery_timeline else 0,
            "learning_efficiency": self._calculate_learning_efficiency(discovery_timeline)
        }
    
    async def _analyze_interventions(self, data: Dict[str, Any], 
                                   experiment_data: ExperimentData) -> Dict[str, Any]:
        """Analyze intervention effectiveness.
        
        Args:
            data: Preprocessed data
            experiment_data: Original experiment data
            
        Returns:
            Intervention analysis
        """
        interventions = data["cleaned_interventions"]
        
        if not interventions:
            return {"message": "No interventions to analyze"}
        
        # Analyze intervention impact on belief evolution
        impact_scores = []
        
        for i, intervention in enumerate(interventions):
            # Find beliefs before and after intervention
            before_beliefs = data["normalized_beliefs"][min(i, len(data["normalized_beliefs"])-1)]
            after_beliefs = data["normalized_beliefs"][min(i+1, len(data["normalized_beliefs"])-1)]
            
            # Calculate belief change magnitude
            total_change = sum(abs(after_beliefs.get(rel, 0) - before_beliefs.get(rel, 0))
                             for rel in set(before_beliefs.keys()) | set(after_beliefs.keys()))
            
            impact_scores.append({
                "intervention_index": i,
                "variable": intervention["variable"],
                "value": intervention["value"],
                "belief_change_magnitude": total_change,
                "scope": intervention.get("scope", "unknown")
            })
        
        # Calculate intervention statistics
        change_magnitudes = [score["belief_change_magnitude"] for score in impact_scores]
        
        return {
            "intervention_impacts": impact_scores,
            "avg_impact": np.mean(change_magnitudes) if change_magnitudes else 0,
            "impact_variance": np.var(change_magnitudes) if len(change_magnitudes) > 1 else 0,
            "most_effective_intervention": max(impact_scores, 
                                             key=lambda x: x["belief_change_magnitude"]) if impact_scores else None,
            "intervention_efficiency": np.mean(change_magnitudes) / len(interventions) if interventions else 0
        }
    
    async def _analyze_belief_evolution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze belief evolution patterns.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Belief evolution analysis
        """
        beliefs = data["normalized_beliefs"]
        
        if len(beliefs) < 2:
            return {"message": "Insufficient belief history for evolution analysis"}
        
        # Track evolution of each relationship
        relationships = set()
        for belief_dict in beliefs:
            relationships.update(belief_dict.keys())
        
        evolution_patterns = {}
        
        for relationship in relationships:
            values = [belief_dict.get(relationship, 0) for belief_dict in beliefs]
            
            # Calculate evolution metrics
            if len(values) > 1:
                evolution_patterns[relationship] = {
                    "initial_value": values[0],
                    "final_value": values[-1],
                    "change": values[-1] - values[0],
                    "volatility": np.std(values),
                    "trend": np.polyfit(range(len(values)), values, 1)[0] if len(values) > 2 else 0,
                    "convergence": self._calculate_convergence(values)
                }
        
        # Overall evolution metrics
        overall_volatility = np.mean([pattern["volatility"] for pattern in evolution_patterns.values()])
        overall_change = np.mean([abs(pattern["change"]) for pattern in evolution_patterns.values()])
        
        return {
            "relationship_patterns": evolution_patterns,
            "overall_volatility": overall_volatility,
            "overall_change": overall_change,
            "convergence_score": np.mean([pattern["convergence"] for pattern in evolution_patterns.values()]),
            "learning_stability": 1 - overall_volatility  # Inverse of volatility
        }
    
    async def _analyze_performance_patterns(self, data: Dict[str, Any], 
                                          experiment_data: ExperimentData) -> Dict[str, Any]:
        """Analyze performance patterns and trends.
        
        Args:
            data: Preprocessed data
            experiment_data: Original experiment data
            
        Returns:
            Performance pattern analysis
        """
        features = data["features"]
        
        # Identify performance characteristics
        patterns = {
            "learning_speed": self._assess_learning_speed(data),
            "exploration_strategy": self._assess_exploration_strategy(data),
            "adaptation_ability": self._assess_adaptation_ability(data),
            "consistency": features.get("belief_stability", 0)
        }
        
        # Performance classification
        performance_class = self._classify_performance(patterns)
        
        return {
            "patterns": patterns,
            "performance_class": performance_class,
            "strengths": self._identify_strengths(patterns),
            "weaknesses": self._identify_weaknesses(patterns),
            "recommendations": self._generate_recommendations(patterns)
        }
    
    def _calculate_learning_efficiency(self, discovery_timeline: List[Dict]) -> float:
        """Calculate learning efficiency metric.
        
        Args:
            discovery_timeline: Timeline of discovery progress
            
        Returns:
            Learning efficiency score
        """
        if not discovery_timeline:
            return 0.0
        
        # Calculate area under the F1 curve (learning efficiency)
        f1_scores = [step["f1_score"] for step in discovery_timeline]
        
        if len(f1_scores) <= 1:
            return f1_scores[0] if f1_scores else 0.0
        
        # Trapezoidal integration
        area = np.trapz(f1_scores, dx=1) / len(f1_scores)
        return area
    
    def _calculate_convergence(self, values: List[float]) -> float:
        """Calculate convergence score for a value sequence.
        
        Args:
            values: Sequence of values
            
        Returns:
            Convergence score (0-1, higher = more convergent)
        """
        if len(values) < 3:
            return 0.5
        
        # Calculate rate of change over time
        changes = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        
        # Convergence = decreasing rate of change
        if len(changes) < 2:
            return 0.5
        
        # Check if changes are decreasing (converging)
        convergence_trend = -np.polyfit(range(len(changes)), changes, 1)[0]
        
        # Normalize to 0-1 range
        return max(0, min(1, convergence_trend + 0.5))
    
    def _assess_learning_speed(self, data: Dict[str, Any]) -> float:
        """Assess learning speed from data.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Learning speed score (0-1)
        """
        beliefs = data["normalized_beliefs"]
        
        if len(beliefs) < 2:
            return 0.5
        
        # Calculate how quickly beliefs stabilize
        changes = []
        for i in range(1, len(beliefs)):
            total_change = sum(abs(beliefs[i].get(rel, 0) - beliefs[i-1].get(rel, 0))
                             for rel in set(beliefs[i].keys()) | set(beliefs[i-1].keys()))
            changes.append(total_change)
        
        # Fast learning = large early changes that decrease quickly
        if len(changes) >= 3:
            early_change = np.mean(changes[:len(changes)//3])
            late_change = np.mean(changes[-len(changes)//3:])
            
            speed_score = (early_change - late_change) / max(early_change, 0.1)
            return max(0, min(1, speed_score))
        
        return 0.5
    
    def _assess_exploration_strategy(self, data: Dict[str, Any]) -> str:
        """Assess exploration strategy from intervention patterns.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Exploration strategy type
        """
        interventions = data["cleaned_interventions"]
        
        if not interventions:
            return "none"
        
        # Analyze intervention diversity
        variables = [i["variable"] for i in interventions]
        unique_variables = set(variables)
        
        if len(unique_variables) / max(len(variables), 1) > 0.7:
            return "broad"  # Explores many different variables
        elif len(unique_variables) / max(len(variables), 1) < 0.3:
            return "focused"  # Focuses on few variables
        else:
            return "balanced"  # Mix of broad and focused
    
    def _assess_adaptation_ability(self, data: Dict[str, Any]) -> float:
        """Assess adaptation ability from belief evolution.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Adaptation score (0-1)
        """
        beliefs = data["normalized_beliefs"]
        interventions = data["cleaned_interventions"]
        
        if len(beliefs) < 2 or not interventions:
            return 0.5
        
        # Look for belief changes after interventions
        adaptation_scores = []
        
        for i, intervention in enumerate(interventions):
            if i < len(beliefs) - 1:
                before = beliefs[i]
                after = beliefs[i + 1]
                
                # Calculate belief change magnitude
                change = sum(abs(after.get(rel, 0) - before.get(rel, 0))
                           for rel in set(before.keys()) | set(after.keys()))
                
                adaptation_scores.append(change)
        
        # Good adaptation = appropriate response to interventions
        return np.mean(adaptation_scores) if adaptation_scores else 0.5
    
    def _classify_performance(self, patterns: Dict[str, Any]) -> str:
        """Classify overall performance based on patterns.
        
        Args:
            patterns: Performance patterns
            
        Returns:
            Performance classification
        """
        learning_speed = patterns["learning_speed"]
        consistency = patterns["consistency"]
        adaptation = patterns["adaptation_ability"]
        
        avg_score = (learning_speed + consistency + adaptation) / 3
        
        if avg_score > 0.8:
            return "excellent"
        elif avg_score > 0.6:
            return "good"
        elif avg_score > 0.4:
            return "average"
        else:
            return "needs_improvement"
    
    def _identify_strengths(self, patterns: Dict[str, Any]) -> List[str]:
        """Identify performance strengths.
        
        Args:
            patterns: Performance patterns
            
        Returns:
            List of strengths
        """
        strengths = []
        
        if patterns["learning_speed"] > 0.7:
            strengths.append("fast_learning")
        
        if patterns["consistency"] > 0.7:
            strengths.append("stable_beliefs")
        
        if patterns["adaptation_ability"] > 0.7:
            strengths.append("good_adaptation")
        
        if patterns["exploration_strategy"] == "balanced":
            strengths.append("balanced_exploration")
        
        return strengths
    
    def _identify_weaknesses(self, patterns: Dict[str, Any]) -> List[str]:
        """Identify performance weaknesses.
        
        Args:
            patterns: Performance patterns
            
        Returns:
            List of weaknesses
        """
        weaknesses = []
        
        if patterns["learning_speed"] < 0.3:
            weaknesses.append("slow_learning")
        
        if patterns["consistency"] < 0.3:
            weaknesses.append("unstable_beliefs")
        
        if patterns["adaptation_ability"] < 0.3:
            weaknesses.append("poor_adaptation")
        
        if patterns["exploration_strategy"] == "none":
            weaknesses.append("no_exploration")
        
        return weaknesses
    
    def _generate_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations.
        
        Args:
            patterns: Performance patterns
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if patterns["learning_speed"] < 0.5:
            recommendations.append("Consider more aggressive belief updates")
        
        if patterns["consistency"] < 0.5:
            recommendations.append("Implement belief smoothing mechanisms")
        
        if patterns["adaptation_ability"] < 0.5:
            recommendations.append("Improve intervention response sensitivity")
        
        if patterns["exploration_strategy"] == "focused":
            recommendations.append("Increase exploration diversity")
        elif patterns["exploration_strategy"] == "broad":
            recommendations.append("Focus exploration on promising variables")
        
        return recommendations


class ScoringProcessor(PipelineStageProcessor):
    """Scores experiment performance."""
    
    def __init__(self):
        """Initialize scoring processor."""
        self.scorer = CausalScorer()
    
    async def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> PipelineResult:
        """Score experiment performance.
        
        Args:
            data: Analysis results
            context: Processing context
            
        Returns:
            Scoring results
        """
        start_time = datetime.now()
        
        try:
            experiment_data = context.get("experiment_data")
            
            # Calculate causal score
            score = self.scorer.score_causal_reasoning(
                experiment_data.environment,
                experiment_data.belief_history,
                experiment_data.interventions,
                experiment_data.ground_truth
            )
            
            # Add analysis insights to metadata
            score.metadata.update({
                "analysis_results": data,
                "pipeline_processed": True
            })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                pipeline_id=context.get("pipeline_id", "unknown"),
                stage=PipelineStage.SCORING,
                data=score,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return PipelineResult(
                pipeline_id=context.get("pipeline_id", "unknown"),
                stage=PipelineStage.SCORING,
                data=None,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )


class AnalysisPipeline:
    """Complete analysis pipeline for causal reasoning experiments."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize analysis pipeline.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        
        # Initialize processors
        self.processors = {
            PipelineStage.DATA_COLLECTION: DataCollectionProcessor(),
            PipelineStage.PREPROCESSING: PreprocessingProcessor(),
            PipelineStage.ANALYSIS: AnalysisProcessor(),
            PipelineStage.SCORING: ScoringProcessor()
        }
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_experiment(self, experiment_data: ExperimentData) -> Dict[str, Any]:
        """Process experiment through complete pipeline.
        
        Args:
            experiment_data: Experiment data to process
            
        Returns:
            Complete pipeline results
        """
        pipeline_id = f"pipeline_{experiment_data.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        context = {
            "pipeline_id": pipeline_id,
            "experiment_data": experiment_data
        }
        
        results = {}
        current_data = experiment_data
        
        # Process through each stage sequentially
        for stage in [PipelineStage.DATA_COLLECTION, 
                     PipelineStage.PREPROCESSING, 
                     PipelineStage.ANALYSIS, 
                     PipelineStage.SCORING]:
            
            processor = self.processors[stage]
            result = await processor.process(current_data, context)
            
            results[stage.value] = result
            
            if not result.success:
                logger.error(f"Pipeline stage {stage.value} failed: {result.error_message}")
                break
            
            # Pass result to next stage
            current_data = result.data
        
        # Calculate total processing time
        total_time = sum(result.processing_time for result in results.values())
        
        return {
            "pipeline_id": pipeline_id,
            "experiment_id": experiment_data.experiment_id,
            "agent_id": experiment_data.agent_id,
            "total_processing_time": total_time,
            "stage_results": results,
            "success": all(result.success for result in results.values()),
            "final_score": current_data if isinstance(current_data, CausalScore) else None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def process_multiple_experiments(self, 
                                          experiments: List[ExperimentData]) -> List[Dict[str, Any]]:
        """Process multiple experiments in parallel.
        
        Args:
            experiments: List of experiments to process
            
        Returns:
            List of pipeline results
        """
        # Process experiments in parallel
        tasks = [self.process_experiment(exp) for exp in experiments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process experiment {experiments[i].experiment_id}: {result}")
                processed_results.append({
                    "pipeline_id": f"failed_{experiments[i].experiment_id}",
                    "experiment_id": experiments[i].experiment_id,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics.
        
        Returns:
            Pipeline statistics
        """
        return {
            "max_workers": self.max_workers,
            "available_stages": list(self.processors.keys()),
            "processor_status": {
                stage.value: "active" for stage in self.processors.keys()
            }
        }