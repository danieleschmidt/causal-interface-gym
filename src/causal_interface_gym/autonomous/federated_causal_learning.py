"""Federated learning system for collaborative causal model development."""

import asyncio
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor
import pickle
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..core import CausalEnvironment
from .causal_discovery_ai import QuantumCausalDiscovery, CausalHypothesis

logger = logging.getLogger(__name__)


class ParticipantRole(Enum):
    """Roles in federated learning system."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"


class FederationPhase(Enum):
    """Phases of federated learning process."""
    INITIALIZATION = "initialization"
    LOCAL_TRAINING = "local_training"
    MODEL_SHARING = "model_sharing"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    CONSENSUS = "consensus"
    COMPLETED = "completed"


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_AGGREGATION = "secure_aggregation"
    ZERO_KNOWLEDGE = "zero_knowledge"


@dataclass
class FederatedParticipant:
    """Represents a participant in federated learning."""
    participant_id: str
    name: str
    role: ParticipantRole
    data_characteristics: Dict[str, Any]
    privacy_requirements: PrivacyLevel
    compute_capacity: Dict[str, float]
    trust_score: float = 0.5
    contribution_history: List[Dict[str, Any]] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.now)
    
    # Cryptographic keys for secure communication
    public_key: Optional[str] = None
    private_key: Optional[str] = None


@dataclass
class LocalCausalModel:
    """Local causal model maintained by each participant."""
    model_id: str
    participant_id: str
    causal_graph: nx.DiGraph
    parameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    data_summary: Dict[str, Any]  # Privacy-preserving summary
    model_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class GlobalCausalModel:
    """Global causal model aggregated from local models."""
    model_id: str
    version: int
    aggregated_graph: nx.DiGraph
    consensus_parameters: Dict[str, Any]
    participant_contributions: Dict[str, float]
    validation_metrics: Dict[str, float]
    confidence_scores: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FederatedSession:
    """Represents a federated learning session."""
    session_id: str
    coordinator_id: str
    participants: Dict[str, FederatedParticipant]
    global_model: Optional[GlobalCausalModel] = None
    current_phase: FederationPhase = FederationPhase.INITIALIZATION
    round_number: int = 0
    max_rounds: int = 100
    convergence_threshold: float = 0.01
    privacy_budget: float = 1.0
    privacy_consumed: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    # Session state
    local_models: Dict[str, LocalCausalModel] = field(default_factory=dict)
    aggregation_results: List[Dict[str, Any]] = field(default_factory=list)
    consensus_history: List[float] = field(default_factory=list)


class FederatedCausalLearning:
    """Federated learning system for collaborative causal discovery."""
    
    def __init__(self,
                 node_id: str,
                 encryption_key: Optional[bytes] = None,
                 default_privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY):
        """Initialize federated causal learning system.
        
        Args:
            node_id: Unique identifier for this node
            encryption_key: Key for encrypting communications
            default_privacy_level: Default privacy protection level
        """
        self.node_id = node_id
        self.default_privacy_level = default_privacy_level
        
        # Initialize encryption
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key)
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.cipher_suite = Fernet(key)
        
        # Federation state
        self.active_sessions: Dict[str, FederatedSession] = {}
        self.session_history: List[FederatedSession] = []
        self.trusted_participants: Dict[str, FederatedParticipant] = {}
        
        # Local causal discovery engine
        self.local_discovery = QuantumCausalDiscovery()
        
        # Privacy mechanisms
        self.privacy_mechanisms = {
            PrivacyLevel.DIFFERENTIAL_PRIVACY: self._apply_differential_privacy,
            PrivacyLevel.SECURE_AGGREGATION: self._apply_secure_aggregation,
            PrivacyLevel.HOMOMORPHIC_ENCRYPTION: self._apply_homomorphic_encryption
        }
        
        # Performance metrics
        self.federation_metrics = {
            "sessions_participated": 0,
            "models_contributed": 0,
            "aggregations_performed": 0,
            "consensus_achieved": 0,
            "avg_convergence_rounds": 0.0,
            "privacy_budget_consumed": 0.0
        }
    
    async def create_federation_session(self,
                                      session_name: str,
                                      objective: str,
                                      privacy_requirements: Dict[str, Any],
                                      participant_constraints: Optional[Dict[str, Any]] = None) -> str:
        """Create a new federated learning session.
        
        Args:
            session_name: Name of the federation session
            objective: Learning objective description
            privacy_requirements: Privacy and security requirements
            participant_constraints: Constraints on participants
            
        Returns:
            Session ID
        """
        session_id = f"fed_session_{uuid.uuid4().hex[:8]}"
        
        # Create coordinator participant (self)
        coordinator = FederatedParticipant(
            participant_id=self.node_id,
            name=f"Coordinator_{self.node_id}",
            role=ParticipantRole.COORDINATOR,
            data_characteristics={"role": "coordinator"},
            privacy_requirements=self.default_privacy_level,
            compute_capacity={"cpu": 1.0, "memory": 1.0, "network": 1.0}
        )
        
        session = FederatedSession(
            session_id=session_id,
            coordinator_id=self.node_id,
            participants={self.node_id: coordinator},
            max_rounds=privacy_requirements.get("max_rounds", 100),
            convergence_threshold=privacy_requirements.get("convergence_threshold", 0.01),
            privacy_budget=privacy_requirements.get("privacy_budget", 1.0)
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Created federation session {session_id}: {session_name}")
        return session_id
    
    async def join_federation_session(self,
                                    session_id: str,
                                    participant_info: Dict[str, Any]) -> bool:
        """Join an existing federation session as a participant.
        
        Args:
            session_id: ID of session to join
            participant_info: Information about this participant
            
        Returns:
            True if successfully joined
        """
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.active_sessions[session_id]
        
        # Create participant record
        participant = FederatedParticipant(
            participant_id=self.node_id,
            name=participant_info.get("name", f"Participant_{self.node_id}"),
            role=ParticipantRole.PARTICIPANT,
            data_characteristics=participant_info.get("data_characteristics", {}),
            privacy_requirements=PrivacyLevel(participant_info.get("privacy_level", "differential_privacy")),
            compute_capacity=participant_info.get("compute_capacity", {"cpu": 0.5, "memory": 0.5, "network": 0.5})
        )
        
        # Add to session
        session.participants[self.node_id] = participant
        
        logger.info(f"Joined federation session {session_id} as {participant.name}")
        return True
    
    async def contribute_local_model(self,
                                   session_id: str,
                                   local_data: np.ndarray,
                                   variable_names: List[str],
                                   privacy_level: Optional[PrivacyLevel] = None) -> Dict[str, Any]:
        """Contribute a locally trained causal model to the federation.
        
        Args:
            session_id: Federation session ID
            local_data: Local training data
            variable_names: Names of variables
            privacy_level: Privacy protection level for this contribution
            
        Returns:
            Contribution result and metadata
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        privacy_level = privacy_level or self.default_privacy_level
        
        # Train local causal model
        local_model = await self._train_local_causal_model(
            local_data, variable_names, privacy_level
        )
        
        # Apply privacy protection
        protected_model = await self._apply_privacy_protection(
            local_model, privacy_level, session.privacy_budget
        )
        
        # Add to session
        session.local_models[self.node_id] = protected_model
        
        # Update metrics
        self.federation_metrics["models_contributed"] += 1
        
        logger.info(f"Contributed local model to session {session_id}")
        
        return {
            "status": "success",
            "model_id": protected_model.model_id,
            "privacy_cost": self._calculate_privacy_cost(protected_model, privacy_level),
            "contribution_metrics": {
                "graph_size": len(protected_model.causal_graph.nodes()),
                "edge_count": len(protected_model.causal_graph.edges()),
                "training_score": protected_model.training_metrics.get("accuracy", 0.0)
            }
        }
    
    async def _train_local_causal_model(self,
                                      data: np.ndarray,
                                      variable_names: List[str],
                                      privacy_level: PrivacyLevel) -> LocalCausalModel:
        """Train a local causal model on private data."""
        # Use quantum causal discovery
        hypothesis = await self.local_discovery.discover_causal_structure(
            data, variable_names
        )
        
        # Create data summary (privacy-preserving)
        data_summary = await self._create_privacy_preserving_summary(data, privacy_level)
        
        # Calculate model hash for integrity
        model_content = {
            "graph": nx.to_dict_of_lists(hypothesis.graph),
            "parameters": hypothesis.evidence_strength,
            "summary": data_summary
        }
        model_hash = hashlib.sha256(json.dumps(model_content, sort_keys=True).encode()).hexdigest()
        
        return LocalCausalModel(
            model_id=f"local_model_{uuid.uuid4().hex[:8]}",
            participant_id=self.node_id,
            causal_graph=hypothesis.graph,
            parameters=hypothesis.evidence_strength,
            training_metrics={
                "likelihood": hypothesis.likelihood,
                "coherence": hypothesis.quantum_coherence,
                "stability": hypothesis.temporal_stability
            },
            data_summary=data_summary,
            model_hash=model_hash
        )
    
    async def _create_privacy_preserving_summary(self,
                                               data: np.ndarray,
                                               privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """Create privacy-preserving summary of local data."""
        n_samples, n_features = data.shape
        
        if privacy_level == PrivacyLevel.NONE:
            # Full statistics
            return {
                "n_samples": n_samples,
                "n_features": n_features,
                "means": data.mean(axis=0).tolist(),
                "stds": data.std(axis=0).tolist(),
                "correlations": np.corrcoef(data.T).tolist()
            }
        
        elif privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
            # Add noise to statistics
            epsilon = 0.1  # Privacy parameter
            noise_scale = 1.0 / epsilon
            
            return {
                "n_samples": n_samples,
                "n_features": n_features,
                "means": (data.mean(axis=0) + np.random.laplace(0, noise_scale, n_features)).tolist(),
                "stds": np.maximum(data.std(axis=0) + np.random.laplace(0, noise_scale, n_features), 0.1).tolist(),
                "correlations": "privacy_protected"  # Don't share correlations
            }
        
        else:
            # Minimal summary for high privacy
            return {
                "n_samples": min(n_samples, 1000),  # Cap reported size
                "n_features": n_features,
                "data_type": "privacy_protected"
            }
    
    async def _apply_privacy_protection(self,
                                      local_model: LocalCausalModel,
                                      privacy_level: PrivacyLevel,
                                      privacy_budget: float) -> LocalCausalModel:
        """Apply privacy protection mechanisms to local model."""
        if privacy_level == PrivacyLevel.NONE:
            return local_model
        
        # Apply appropriate privacy mechanism
        privacy_mechanism = self.privacy_mechanisms.get(privacy_level)
        if privacy_mechanism:
            protected_model = await privacy_mechanism(local_model, privacy_budget)
            return protected_model
        
        return local_model
    
    async def _apply_differential_privacy(self,
                                        model: LocalCausalModel,
                                        privacy_budget: float) -> LocalCausalModel:
        """Apply differential privacy to model parameters."""
        epsilon = min(privacy_budget * 0.1, 0.1)  # Use portion of budget
        
        # Add noise to edge weights
        noisy_parameters = {}
        for edge, weight in model.parameters.items():
            noise = np.random.laplace(0, 1.0 / epsilon)
            noisy_parameters[edge] = max(0, weight + noise)  # Ensure non-negative
        
        # Create protected model
        protected_model = LocalCausalModel(
            model_id=model.model_id,
            participant_id=model.participant_id,
            causal_graph=model.causal_graph.copy(),
            parameters=noisy_parameters,
            training_metrics=model.training_metrics.copy(),
            data_summary=model.data_summary.copy(),
            model_hash=model.model_hash
        )
        
        return protected_model
    
    async def _apply_secure_aggregation(self,
                                      model: LocalCausalModel,
                                      privacy_budget: float) -> LocalCausalModel:
        """Apply secure aggregation protection."""
        # In practice, would implement secure multi-party computation
        # For now, add minimal noise to prevent exact reconstruction
        
        protected_parameters = {}
        for edge, weight in model.parameters.items():
            # Add small random perturbation
            noise = np.random.normal(0, 0.01)  # Small noise
            protected_parameters[edge] = weight + noise
        
        protected_model = LocalCausalModel(
            model_id=model.model_id,
            participant_id=model.participant_id,
            causal_graph=model.causal_graph.copy(),
            parameters=protected_parameters,
            training_metrics=model.training_metrics.copy(),
            data_summary={"secure_aggregation": True},  # Hide details
            model_hash=model.model_hash
        )
        
        return protected_model
    
    async def _apply_homomorphic_encryption(self,
                                          model: LocalCausalModel,
                                          privacy_budget: float) -> LocalCausalModel:
        """Apply homomorphic encryption (simplified implementation)."""
        # In practice, would use proper homomorphic encryption libraries
        # For demonstration, use simple encryption of serialized parameters
        
        encrypted_parameters = {}
        for edge, weight in model.parameters.items():
            # Convert to bytes and encrypt
            weight_bytes = str(weight).encode()
            encrypted_weight = self.cipher_suite.encrypt(weight_bytes)
            encrypted_parameters[edge] = base64.b64encode(encrypted_weight).decode()
        
        protected_model = LocalCausalModel(
            model_id=model.model_id,
            participant_id=model.participant_id,
            causal_graph=model.causal_graph.copy(),
            parameters=encrypted_parameters,
            training_metrics={"encrypted": True},
            data_summary={"homomorphic_encryption": True},
            model_hash=model.model_hash
        )
        
        return protected_model
    
    def _calculate_privacy_cost(self, model: LocalCausalModel, privacy_level: PrivacyLevel) -> float:
        """Calculate privacy budget cost for model contribution."""
        base_cost = 0.1
        
        if privacy_level == PrivacyLevel.NONE:
            return 0.0
        elif privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
            # Cost proportional to graph complexity
            complexity_factor = len(model.causal_graph.edges()) / 10.0
            return base_cost * (1 + complexity_factor)
        elif privacy_level in [PrivacyLevel.SECURE_AGGREGATION, PrivacyLevel.HOMOMORPHIC_ENCRYPTION]:
            return base_cost * 0.5  # Lower privacy cost
        else:
            return base_cost
    
    async def aggregate_models(self, session_id: str) -> Dict[str, Any]:
        """Aggregate local models into a global model.
        
        Args:
            session_id: Federation session ID
            
        Returns:
            Aggregation results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if len(session.local_models) < 2:
            raise ValueError("Need at least 2 local models for aggregation")
        
        # Update session phase
        session.current_phase = FederationPhase.AGGREGATION
        session.round_number += 1
        
        # Perform aggregation
        aggregated_graph, consensus_params = await self._federated_graph_aggregation(
            list(session.local_models.values())
        )
        
        # Calculate participant contributions
        contributions = await self._calculate_contributions(session.local_models)
        
        # Validate aggregated model
        validation_metrics = await self._validate_global_model(
            aggregated_graph, consensus_params
        )
        
        # Create global model
        global_model = GlobalCausalModel(
            model_id=f"global_model_{session_id}_{session.round_number}",
            version=session.round_number,
            aggregated_graph=aggregated_graph,
            consensus_parameters=consensus_params,
            participant_contributions=contributions,
            validation_metrics=validation_metrics,
            confidence_scores=await self._calculate_confidence_scores(session.local_models)
        )
        
        session.global_model = global_model
        session.current_phase = FederationPhase.VALIDATION
        
        # Update metrics
        self.federation_metrics["aggregations_performed"] += 1
        
        # Record aggregation result
        aggregation_result = {
            "round": session.round_number,
            "participants": len(session.local_models),
            "consensus_score": validation_metrics.get("consensus_score", 0.0),
            "graph_complexity": len(aggregated_graph.edges()),
            "timestamp": datetime.now()
        }
        session.aggregation_results.append(aggregation_result)
        
        logger.info(f"Aggregated models for session {session_id}, round {session.round_number}")
        
        return {
            "status": "success",
            "global_model_id": global_model.model_id,
            "round": session.round_number,
            "participants": len(session.local_models),
            "aggregation_metrics": validation_metrics,
            "consensus_achieved": validation_metrics.get("consensus_score", 0.0) > 0.8
        }
    
    async def _federated_graph_aggregation(self,
                                         local_models: List[LocalCausalModel]) -> Tuple[nx.DiGraph, Dict[str, float]]:
        """Aggregate causal graphs from multiple participants."""
        # Collect all edges from local models
        edge_votes = {}  # edge -> list of weights
        all_nodes = set()
        
        for model in local_models:
            all_nodes.update(model.causal_graph.nodes())
            
            for edge in model.causal_graph.edges():
                if edge not in edge_votes:
                    edge_votes[edge] = []
                
                # Get edge weight from parameters
                edge_key = f"{edge[0]}→{edge[1]}"
                weight = model.parameters.get(edge_key, 0.5)
                
                # Handle encrypted parameters
                if isinstance(weight, str):
                    try:
                        # Try to decrypt if encrypted
                        encrypted_bytes = base64.b64decode(weight.encode())
                        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
                        weight = float(decrypted_bytes.decode())
                    except Exception:
                        weight = 0.5  # Default if can't decrypt
                
                edge_votes[edge].append(weight)
        
        # Aggregate edges using consensus mechanism
        aggregated_graph = nx.DiGraph()
        aggregated_graph.add_nodes_from(all_nodes)
        consensus_params = {}
        
        # Use majority voting with weighted consensus
        for edge, weights in edge_votes.items():
            if len(weights) >= len(local_models) * 0.5:  # Majority threshold
                # Calculate consensus weight
                consensus_weight = self._calculate_consensus_weight(weights)
                
                if consensus_weight > 0.3:  # Minimum confidence threshold
                    aggregated_graph.add_edge(edge[0], edge[1])
                    consensus_params[f"{edge[0]}→{edge[1]}"] = consensus_weight
        
        return aggregated_graph, consensus_params
    
    def _calculate_consensus_weight(self, weights: List[float]) -> float:
        """Calculate consensus weight from multiple participant weights."""
        if not weights:
            return 0.0
        
        # Use weighted average with confidence weighting
        weights = np.array(weights)
        
        # Calculate variance to assess consensus
        variance = np.var(weights)
        confidence = 1.0 / (1.0 + variance)  # High variance = low confidence
        
        # Weighted mean
        consensus = np.mean(weights) * confidence
        
        return float(consensus)
    
    async def _calculate_contributions(self,
                                     local_models: Dict[str, LocalCausalModel]) -> Dict[str, float]:
        """Calculate contribution scores for each participant."""
        contributions = {}
        total_edges = sum(len(model.causal_graph.edges()) for model in local_models.values())
        
        for participant_id, model in local_models.items():
            # Base contribution from graph size
            graph_contribution = len(model.causal_graph.edges()) / max(total_edges, 1)
            
            # Quality contribution from training metrics
            quality_score = (
                model.training_metrics.get("likelihood", 0.5) * 0.4 +
                model.training_metrics.get("coherence", 0.5) * 0.4 +
                model.training_metrics.get("stability", 0.5) * 0.2
            )
            
            # Combine contributions
            total_contribution = (graph_contribution * 0.6 + quality_score * 0.4)
            contributions[participant_id] = total_contribution
        
        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {
                pid: contrib / total_contribution 
                for pid, contrib in contributions.items()
            }
        
        return contributions
    
    async def _validate_global_model(self,
                                   aggregated_graph: nx.DiGraph,
                                   consensus_params: Dict[str, float]) -> Dict[str, float]:
        """Validate the aggregated global model."""
        metrics = {}
        
        # Graph structure validation
        metrics["is_dag"] = 1.0 if nx.is_directed_acyclic_graph(aggregated_graph) else 0.0
        metrics["node_count"] = len(aggregated_graph.nodes())
        metrics["edge_count"] = len(aggregated_graph.edges())
        metrics["density"] = nx.density(aggregated_graph)
        
        # Consensus validation
        if consensus_params:
            weights = list(consensus_params.values())
            metrics["avg_consensus"] = np.mean(weights)
            metrics["consensus_variance"] = np.var(weights)
            metrics["consensus_score"] = np.mean(weights) * (1.0 - min(np.var(weights), 1.0))
        else:
            metrics["consensus_score"] = 0.0
        
        # Structural validity
        try:
            # Check for reasonable causal structure
            if aggregated_graph.number_of_nodes() > 0:
                avg_degree = sum(dict(aggregated_graph.degree()).values()) / aggregated_graph.number_of_nodes()
                metrics["avg_degree"] = avg_degree
                metrics["structural_validity"] = min(1.0, avg_degree / 3.0)  # Reasonable connectivity
            else:
                metrics["structural_validity"] = 0.0
        except Exception:
            metrics["structural_validity"] = 0.0
        
        return metrics
    
    async def _calculate_confidence_scores(self,
                                         local_models: Dict[str, LocalCausalModel]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects of the global model."""
        if not local_models:
            return {}
        
        # Collect training metrics from all models
        all_likelihoods = [m.training_metrics.get("likelihood", 0.5) for m in local_models.values()]
        all_coherences = [m.training_metrics.get("coherence", 0.5) for m in local_models.values()]
        all_stabilities = [m.training_metrics.get("stability", 0.5) for m in local_models.values()]
        
        return {
            "likelihood_confidence": np.mean(all_likelihoods),
            "coherence_confidence": np.mean(all_coherences),
            "stability_confidence": np.mean(all_stabilities),
            "participant_agreement": 1.0 - np.var(all_likelihoods),  # Low variance = high agreement
            "overall_confidence": np.mean([np.mean(all_likelihoods), np.mean(all_coherences), np.mean(all_stabilities)])
        }
    
    async def check_convergence(self, session_id: str) -> bool:
        """Check if federated learning has converged.
        
        Args:
            session_id: Federation session ID
            
        Returns:
            True if converged
        """
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if len(session.aggregation_results) < 2:
            return False
        
        # Check consensus score convergence
        recent_consensus = [r["consensus_score"] for r in session.aggregation_results[-3:]]
        consensus_variance = np.var(recent_consensus)
        
        convergence_check = consensus_variance < session.convergence_threshold
        
        if convergence_check:
            # Update session status
            session.current_phase = FederationPhase.CONSENSUS
            session.consensus_history.append(recent_consensus[-1])
            
            # Update metrics
            self.federation_metrics["consensus_achieved"] += 1
            
            # Calculate average convergence rounds
            current_avg = self.federation_metrics["avg_convergence_rounds"]
            total_consensus = self.federation_metrics["consensus_achieved"]
            
            self.federation_metrics["avg_convergence_rounds"] = (
                (current_avg * (total_consensus - 1) + session.round_number) / total_consensus
            )
        
        return convergence_check
    
    async def finalize_federation_session(self, session_id: str) -> Dict[str, Any]:
        """Finalize a federated learning session.
        
        Args:
            session_id: Session to finalize
            
        Returns:
            Final session results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.current_phase = FederationPhase.COMPLETED
        
        # Calculate final metrics
        final_metrics = {
            "session_id": session_id,
            "total_rounds": session.round_number,
            "participants": len(session.participants),
            "models_contributed": len(session.local_models),
            "final_consensus_score": session.aggregation_results[-1]["consensus_score"] if session.aggregation_results else 0.0,
            "privacy_budget_consumed": session.privacy_consumed,
            "convergence_achieved": session.current_phase == FederationPhase.CONSENSUS,
            "global_model": {
                "model_id": session.global_model.model_id if session.global_model else None,
                "graph_size": len(session.global_model.aggregated_graph.nodes()) if session.global_model else 0,
                "edge_count": len(session.global_model.aggregated_graph.edges()) if session.global_model else 0,
                "validation_metrics": session.global_model.validation_metrics if session.global_model else {}
            }
        }
        
        # Move to history
        self.session_history.append(session)
        del self.active_sessions[session_id]
        
        # Update global metrics
        self.federation_metrics["sessions_participated"] += 1
        
        logger.info(f"Finalized federation session {session_id}")
        
        return final_metrics
    
    def get_federation_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a federation session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        else:
            # Look in history
            session = next((s for s in self.session_history if s.session_id == session_id), None)
            if not session:
                return {"error": "Session not found"}
        
        return {
            "session_id": session.session_id,
            "current_phase": session.current_phase.value,
            "round_number": session.round_number,
            "max_rounds": session.max_rounds,
            "participants": len(session.participants),
            "models_received": len(session.local_models),
            "privacy_budget_remaining": session.privacy_budget - session.privacy_consumed,
            "convergence_history": session.consensus_history,
            "latest_consensus_score": session.aggregation_results[-1]["consensus_score"] if session.aggregation_results else None,
            "global_model_available": session.global_model is not None
        }
    
    def get_federation_summary(self) -> Dict[str, Any]:
        """Get summary of federated learning activities."""
        return {
            "node_id": self.node_id,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.session_history),
            "trusted_participants": len(self.trusted_participants),
            **self.federation_metrics,
            "recent_sessions": [
                {
                    "session_id": s.session_id,
                    "phase": s.current_phase.value,
                    "participants": len(s.participants),
                    "rounds": s.round_number
                }
                for s in list(self.active_sessions.values()) + self.session_history[-3:]
            ]
        }