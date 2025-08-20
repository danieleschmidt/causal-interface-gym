"""Autonomous meta-learning system that improves causal reasoning over time."""

import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime
import logging
from pathlib import Path

from ..core import CausalEnvironment
from ..llm.client import LLMClient
from ..metrics import CausalMetrics

logger = logging.getLogger(__name__)


@dataclass
class MetaKnowledge:
    """Represents learned meta-knowledge about causal reasoning patterns."""
    pattern_id: str
    description: str
    success_rate: float
    contexts: List[str]
    interventions: List[Dict[str, Any]]
    learned_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    confidence: float = 0.0


@dataclass
class LearningSession:
    """Tracks a single meta-learning session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    experiments_run: int = 0
    patterns_discovered: int = 0
    improvement_metrics: Dict[str, float] = field(default_factory=dict)


class AutoMetaLearner:
    """Autonomous meta-learning system for causal reasoning improvement."""
    
    def __init__(self, 
                 knowledge_path: Optional[Path] = None,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.3):
        """Initialize the meta-learner.
        
        Args:
            knowledge_path: Path to store learned knowledge
            learning_rate: Rate of meta-learning adaptation
            exploration_rate: Rate of exploration vs exploitation
        """
        self.knowledge_path = knowledge_path or Path("autonomous_knowledge.json")
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        self.meta_knowledge: Dict[str, MetaKnowledge] = {}
        self.active_sessions: Dict[str, LearningSession] = {}
        self.performance_history: List[Dict[str, float]] = []
        
        self._load_knowledge()
        
    def _load_knowledge(self) -> None:
        """Load previously learned meta-knowledge."""
        if self.knowledge_path.exists():
            try:
                with open(self.knowledge_path) as f:
                    data = json.load(f)
                    for pattern_id, pattern_data in data.get("patterns", {}).items():
                        self.meta_knowledge[pattern_id] = MetaKnowledge(
                            pattern_id=pattern_data["pattern_id"],
                            description=pattern_data["description"],
                            success_rate=pattern_data["success_rate"],
                            contexts=pattern_data["contexts"],
                            interventions=pattern_data["interventions"],
                            learned_at=datetime.fromisoformat(pattern_data["learned_at"]),
                            usage_count=pattern_data.get("usage_count", 0),
                            confidence=pattern_data.get("confidence", 0.0)
                        )
                logger.info(f"Loaded {len(self.meta_knowledge)} meta-patterns")
            except Exception as e:
                logger.warning(f"Failed to load meta-knowledge: {e}")
    
    def _save_knowledge(self) -> None:
        """Save learned meta-knowledge to disk."""
        try:
            data = {
                "patterns": {
                    pid: {
                        "pattern_id": pattern.pattern_id,
                        "description": pattern.description,
                        "success_rate": pattern.success_rate,
                        "contexts": pattern.contexts,
                        "interventions": pattern.interventions,
                        "learned_at": pattern.learned_at.isoformat(),
                        "usage_count": pattern.usage_count,
                        "confidence": pattern.confidence
                    }
                    for pid, pattern in self.meta_knowledge.items()
                },
                "performance_history": self.performance_history[-100:]  # Keep last 100
            }
            
            self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.knowledge_path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save meta-knowledge: {e}")
    
    async def start_learning_session(self, session_id: str) -> LearningSession:
        """Start a new meta-learning session."""
        session = LearningSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        self.active_sessions[session_id] = session
        logger.info(f"Started meta-learning session: {session_id}")
        return session
    
    async def observe_experiment_outcome(self, 
                                       session_id: str,
                                       experiment_context: Dict[str, Any],
                                       interventions: List[Dict[str, Any]],
                                       success_metrics: Dict[str, float]) -> None:
        """Observe and learn from experiment outcomes."""
        if session_id not in self.active_sessions:
            logger.warning(f"Unknown session: {session_id}")
            return
            
        session = self.active_sessions[session_id]
        session.experiments_run += 1
        
        # Analyze patterns in successful experiments
        if success_metrics.get("causal_accuracy", 0) > 0.8:
            await self._extract_success_patterns(
                experiment_context, interventions, success_metrics
            )
        
        # Update performance history
        self.performance_history.append({
            "timestamp": datetime.now().timestamp(),
            "session_id": session_id,
            **success_metrics
        })
        
        # Adaptive learning rate adjustment
        await self._adjust_learning_parameters()
    
    async def _extract_success_patterns(self,
                                      context: Dict[str, Any],
                                      interventions: List[Dict[str, Any]],
                                      metrics: Dict[str, float]) -> None:
        """Extract successful patterns from experiment data."""
        # Generate pattern ID based on context and interventions
        pattern_signature = self._generate_pattern_signature(context, interventions)
        pattern_id = f"pattern_{hash(pattern_signature) % 10000:04d}"
        
        if pattern_id in self.meta_knowledge:
            # Update existing pattern
            pattern = self.meta_knowledge[pattern_id]
            pattern.usage_count += 1
            # Exponential moving average for success rate
            pattern.success_rate = (
                0.9 * pattern.success_rate + 
                0.1 * metrics.get("causal_accuracy", 0)
            )
            pattern.confidence = min(1.0, pattern.confidence + 0.1)
        else:
            # Create new pattern
            pattern = MetaKnowledge(
                pattern_id=pattern_id,
                description=self._generate_pattern_description(context, interventions),
                success_rate=metrics.get("causal_accuracy", 0),
                contexts=[str(context)],
                interventions=interventions.copy(),
                confidence=0.1
            )
            self.meta_knowledge[pattern_id] = pattern
            
            # Update session stats
            for session in self.active_sessions.values():
                session.patterns_discovered += 1
        
        logger.debug(f"Updated pattern {pattern_id}: success_rate={pattern.success_rate:.3f}")
    
    def _generate_pattern_signature(self,
                                  context: Dict[str, Any],
                                  interventions: List[Dict[str, Any]]) -> str:
        """Generate unique signature for a pattern."""
        context_features = sorted([
            f"{k}:{str(v)[:50]}" for k, v in context.items()
            if k in ["graph_structure", "variable_types", "complexity"]
        ])
        
        intervention_features = sorted([
            f"{i.get('target', 'unknown')}:{i.get('type', 'set')}"
            for i in interventions
        ])
        
        return "|".join(context_features + intervention_features)
    
    def _generate_pattern_description(self,
                                    context: Dict[str, Any],
                                    interventions: List[Dict[str, Any]]) -> str:
        """Generate human-readable pattern description."""
        graph_type = context.get("graph_structure", "unknown")
        intervention_types = [i.get("type", "set") for i in interventions]
        
        return (
            f"Successful causal reasoning on {graph_type} graphs "
            f"using {', '.join(set(intervention_types))} interventions"
        )
    
    async def recommend_interventions(self,
                                    current_context: Dict[str, Any],
                                    exploration_bonus: float = 0.0) -> List[Dict[str, Any]]:
        """Recommend interventions based on learned patterns."""
        context_signature = str(current_context)
        
        # Find matching patterns
        matching_patterns = []
        for pattern in self.meta_knowledge.values():
            # Simple context similarity based on shared features
            similarity = self._compute_context_similarity(
                current_context, pattern.contexts[0] if pattern.contexts else "{}"
            )
            if similarity > 0.3:  # Similarity threshold
                matching_patterns.append((pattern, similarity))
        
        if not matching_patterns:
            # No patterns found, use exploration
            return await self._generate_exploratory_interventions(current_context)
        
        # Sort by success rate and similarity
        matching_patterns.sort(
            key=lambda x: x[0].success_rate * x[1] * x[0].confidence,
            reverse=True
        )
        
        # Select best pattern with exploration bonus
        if np.random.random() < self.exploration_rate + exploration_bonus:
            # Exploration: try something different
            return await self._generate_exploratory_interventions(current_context)
        else:
            # Exploitation: use best known pattern
            best_pattern = matching_patterns[0][0]
            best_pattern.usage_count += 1
            return best_pattern.interventions.copy()
    
    def _compute_context_similarity(self, context1: Dict[str, Any], context2_str: str) -> float:
        """Compute similarity between contexts."""
        try:
            context2 = eval(context2_str) if isinstance(context2_str, str) else context2_str
            if not isinstance(context2, dict):
                return 0.0
                
            shared_keys = set(context1.keys()) & set(context2.keys())
            if not shared_keys:
                return 0.0
                
            similarity_scores = []
            for key in shared_keys:
                if context1[key] == context2[key]:
                    similarity_scores.append(1.0)
                elif isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
                    # Numerical similarity
                    diff = abs(context1[key] - context2[key])
                    max_val = max(abs(context1[key]), abs(context2[key]), 1)
                    similarity_scores.append(max(0, 1 - diff / max_val))
                else:
                    similarity_scores.append(0.0)
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception:
            return 0.0
    
    async def _generate_exploratory_interventions(self,
                                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate exploratory interventions for unknown contexts."""
        variables = context.get("variables", [])
        if not variables:
            return []
        
        # Generate diverse intervention types
        intervention_types = ["set", "randomize", "prevent"]
        interventions = []
        
        # Sample variables and intervention types
        num_interventions = min(3, len(variables))
        selected_vars = np.random.choice(variables, num_interventions, replace=False)
        
        for var in selected_vars:
            intervention_type = np.random.choice(intervention_types)
            intervention = {
                "target": var,
                "type": intervention_type
            }
            
            if intervention_type == "set":
                # Random value for the variable
                var_type = context.get("variable_types", {}).get(var, "binary")
                if var_type == "binary":
                    intervention["value"] = bool(np.random.random() > 0.5)
                elif var_type == "continuous":
                    intervention["value"] = np.random.normal(0, 1)
                else:
                    intervention["value"] = np.random.randint(0, 3)
            
            interventions.append(intervention)
        
        logger.debug(f"Generated {len(interventions)} exploratory interventions")
        return interventions
    
    async def _adjust_learning_parameters(self) -> None:
        """Adaptively adjust learning parameters based on performance."""
        if len(self.performance_history) < 10:
            return
        
        # Compute recent performance trend
        recent_performance = [
            h.get("causal_accuracy", 0) 
            for h in self.performance_history[-10:]
        ]
        
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Adjust exploration rate based on performance trend
        if trend < -0.01:  # Performance decreasing
            self.exploration_rate = min(0.8, self.exploration_rate + 0.05)
        elif trend > 0.01:  # Performance improving
            self.exploration_rate = max(0.1, self.exploration_rate - 0.02)
        
        # Adjust learning rate based on pattern stability
        pattern_confidences = [p.confidence for p in self.meta_knowledge.values()]
        if pattern_confidences:
            avg_confidence = np.mean(pattern_confidences)
            self.learning_rate = 0.05 + 0.15 * (1 - avg_confidence)
        
        logger.debug(f"Adjusted parameters: exploration_rate={self.exploration_rate:.3f}, "
                    f"learning_rate={self.learning_rate:.3f}")
    
    async def end_learning_session(self, session_id: str) -> Dict[str, Any]:
        """End a meta-learning session and return summary."""
        if session_id not in self.active_sessions:
            logger.warning(f"Unknown session: {session_id}")
            return {}
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        
        # Compute session metrics
        session_duration = (session.end_time - session.start_time).total_seconds()
        patterns_per_hour = session.patterns_discovered / max(session_duration / 3600, 0.1)
        experiments_per_hour = session.experiments_run / max(session_duration / 3600, 0.1)
        
        summary = {
            "session_id": session_id,
            "duration_seconds": session_duration,
            "experiments_run": session.experiments_run,
            "patterns_discovered": session.patterns_discovered,
            "patterns_per_hour": patterns_per_hour,
            "experiments_per_hour": experiments_per_hour,
            "total_patterns": len(self.meta_knowledge),
            "avg_pattern_confidence": np.mean([p.confidence for p in self.meta_knowledge.values()]) 
                                    if self.meta_knowledge else 0.0
        }
        
        # Save learned knowledge
        self._save_knowledge()
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"Ended session {session_id}: {summary}")
        return summary
    
    def get_meta_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of current meta-knowledge."""
        if not self.meta_knowledge:
            return {"total_patterns": 0}
        
        patterns = list(self.meta_knowledge.values())
        
        return {
            "total_patterns": len(patterns),
            "avg_success_rate": np.mean([p.success_rate for p in patterns]),
            "avg_confidence": np.mean([p.confidence for p in patterns]),
            "total_usage": sum(p.usage_count for p in patterns),
            "top_patterns": [
                {
                    "id": p.pattern_id,
                    "description": p.description,
                    "success_rate": p.success_rate,
                    "confidence": p.confidence,
                    "usage_count": p.usage_count
                }
                for p in sorted(patterns, 
                              key=lambda x: x.success_rate * x.confidence,
                              reverse=True)[:5]
            ]
        }