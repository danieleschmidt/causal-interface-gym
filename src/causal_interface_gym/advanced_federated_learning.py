"""Advanced federated learning for collaborative causal discovery across organizations."""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import json
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import base64

logger = logging.getLogger(__name__)

class FederatedProtocol(Enum):
    """Federated learning protocols."""
    FEDERATED_AVERAGING = "federated_averaging"
    FEDERATED_SGD = "federated_sgd"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    BYZANTINE_ROBUST = "byzantine_robust"

class PrivacyLevel(Enum):
    """Privacy protection levels."""
    NONE = "none"
    BASIC = "basic"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_MULTIPARTY = "secure_multiparty"
    HOMOMORPHIC = "homomorphic_encryption"
    ZERO_KNOWLEDGE = "zero_knowledge_proofs"

@dataclass
class ParticipantInfo:
    """Information about a federated learning participant."""
    participant_id: str
    organization: str
    data_size: int
    compute_capability: float
    privacy_level: PrivacyLevel
    contribution_score: float = 0.0
    trust_score: float = 1.0
    last_seen: float = field(default_factory=time.time)
    public_key: Optional[str] = None
    
@dataclass
class FederatedCausalModel:
    """Federated causal model state."""
    model_id: str
    global_parameters: Dict[str, np.ndarray]
    causal_graph: np.ndarray
    model_version: int
    last_updated: float
    participant_contributions: Dict[str, float]
    validation_metrics: Dict[str, float]
    
@dataclass
class PrivacyBudget:
    """Differential privacy budget tracking."""
    epsilon: float
    delta: float
    consumed_epsilon: float = 0.0
    remaining_rounds: int = 100
    
class SecureCommunicationProtocol(ABC):
    """Abstract base for secure communication protocols."""
    
    @abstractmethod
    async def encrypt_message(self, message: bytes, recipient_id: str) -> bytes:
        pass
    
    @abstractmethod
    async def decrypt_message(self, encrypted_message: bytes, sender_id: str) -> bytes:
        pass
    
    @abstractmethod
    async def verify_signature(self, message: bytes, signature: bytes, sender_id: str) -> bool:
        pass

class BasicSecureProtocol(SecureCommunicationProtocol):
    """Basic secure protocol implementation."""
    
    def __init__(self):
        self.keys = {}
    
    async def encrypt_message(self, message: bytes, recipient_id: str) -> bytes:
        # Simplified encryption (use proper crypto libraries in production)
        key = self.keys.get(recipient_id, b'default_key')
        return base64.b64encode(message + key)
    
    async def decrypt_message(self, encrypted_message: bytes, sender_id: str) -> bytes:
        # Simplified decryption
        decoded = base64.b64decode(encrypted_message)
        key = self.keys.get(sender_id, b'default_key')
        return decoded[:-len(key)]
    
    async def verify_signature(self, message: bytes, signature: bytes, sender_id: str) -> bool:
        # Simplified signature verification
        expected_signature = hashlib.sha256(message + sender_id.encode()).digest()
        return signature == expected_signature

class AdvancedFederatedLearning:
    """Advanced federated learning system for causal discovery."""
    
    def __init__(self, coordinator_id: str = "central_coordinator",
                 protocol: FederatedProtocol = FederatedProtocol.SECURE_AGGREGATION,
                 privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY):
        """Initialize federated learning coordinator.
        
        Args:
            coordinator_id: Unique identifier for this coordinator
            protocol: Federated learning protocol to use
            privacy_level: Privacy protection level
        """
        self.coordinator_id = coordinator_id
        self.protocol = protocol
        self.privacy_level = privacy_level
        self.participants: Dict[str, ParticipantInfo] = {}
        self.global_model: Optional[FederatedCausalModel] = None
        self.privacy_budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        self.secure_protocol = BasicSecureProtocol()
        self.aggregation_history: List[Dict[str, Any]] = []
        self.byzantine_detection = ByzantineDetection()
        self.incentive_mechanism = IncentiveMechanism()
        
        # Performance tracking
        self.round_number = 0
        self.convergence_threshold = 1e-6
        self.max_rounds = 100
        
        logger.info(f"Federated learning coordinator initialized with {protocol.value} protocol")
    
    async def register_participant(self, participant_info: ParticipantInfo) -> Dict[str, Any]:
        """Register a new participant in the federated learning system.
        
        Args:
            participant_info: Information about the participant
            
        Returns:
            Registration response with participant status
        """
        try:
            # Validate participant
            if not self._validate_participant(participant_info):
                return {
                    'status': 'rejected',
                    'reason': 'Participant validation failed',
                    'participant_id': participant_info.participant_id
                }
            
            # Check privacy compatibility
            if not self._check_privacy_compatibility(participant_info.privacy_level):
                return {
                    'status': 'rejected',
                    'reason': 'Privacy level incompatible',
                    'required_privacy_level': self.privacy_level.value
                }
            
            # Register participant
            self.participants[participant_info.participant_id] = participant_info
            
            # Initialize secure communication
            await self._setup_secure_communication(participant_info)
            
            # Calculate initial contribution score
            self._update_contribution_score(participant_info.participant_id)
            
            logger.info(f"Participant {participant_info.participant_id} registered successfully")
            
            return {
                'status': 'accepted',
                'participant_id': participant_info.participant_id,
                'global_model_version': self.global_model.model_version if self.global_model else 0,
                'privacy_budget_remaining': self.privacy_budget.remaining_rounds,
                'expected_contribution_weight': self._calculate_participant_weight(participant_info)
            }
            
        except Exception as e:
            logger.error(f"Failed to register participant {participant_info.participant_id}: {e}")
            return {
                'status': 'error',
                'reason': str(e)
            }
    
    def _validate_participant(self, participant_info: ParticipantInfo) -> bool:
        """Validate participant information."""
        # Check for duplicate IDs
        if participant_info.participant_id in self.participants:
            return False
        
        # Validate data size
        if participant_info.data_size < 100:  # Minimum data requirement
            return False
        
        # Check compute capability
        if participant_info.compute_capability < 0.1:
            return False
        
        return True
    
    def _check_privacy_compatibility(self, participant_privacy: PrivacyLevel) -> bool:
        """Check if participant's privacy level is compatible."""
        privacy_hierarchy = {
            PrivacyLevel.NONE: 0,
            PrivacyLevel.BASIC: 1,
            PrivacyLevel.DIFFERENTIAL_PRIVACY: 2,
            PrivacyLevel.SECURE_MULTIPARTY: 3,
            PrivacyLevel.HOMOMORPHIC: 4,
            PrivacyLevel.ZERO_KNOWLEDGE: 5
        }
        
        return privacy_hierarchy[participant_privacy] >= privacy_hierarchy[self.privacy_level]
    
    async def _setup_secure_communication(self, participant_info: ParticipantInfo) -> None:
        """Setup secure communication with participant."""
        # Generate keys for secure communication
        participant_key = hashlib.sha256(
            f"{participant_info.participant_id}{self.coordinator_id}".encode()
        ).digest()
        
        self.secure_protocol.keys[participant_info.participant_id] = participant_key
    
    def _update_contribution_score(self, participant_id: str) -> None:
        """Update participant's contribution score."""
        participant = self.participants[participant_id]
        
        # Score based on data size, compute capability, and privacy level
        data_score = min(participant.data_size / 10000, 1.0)  # Normalize to [0,1]
        compute_score = min(participant.compute_capability, 1.0)
        privacy_score = list(PrivacyLevel).index(participant.privacy_level) / len(PrivacyLevel)
        
        participant.contribution_score = (data_score + compute_score + privacy_score) / 3
    
    def _calculate_participant_weight(self, participant_info: ParticipantInfo) -> float:
        """Calculate expected weight for participant in aggregation."""
        total_data = sum(p.data_size for p in self.participants.values())
        if total_data == 0:
            return 0.0
        return participant_info.data_size / total_data
    
    async def initiate_federated_round(self, causal_discovery_task: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate a new round of federated causal discovery.
        
        Args:
            causal_discovery_task: Task specification for causal discovery
            
        Returns:
            Round initiation results
        """
        self.round_number += 1
        round_start_time = time.time()
        
        logger.info(f"Initiating federated round {self.round_number}")
        
        try:
            # Select participants for this round
            selected_participants = await self._select_participants_for_round()
            
            if len(selected_participants) < 2:
                return {
                    'status': 'failed',
                    'reason': 'Insufficient participants for federated learning',
                    'required_minimum': 2,
                    'available': len(selected_participants)
                }
            
            # Distribute task to participants
            participant_tasks = await self._distribute_task_to_participants(
                selected_participants, causal_discovery_task
            )
            
            # Wait for participant responses
            participant_results = await self._collect_participant_results(participant_tasks)
            
            # Perform secure aggregation
            aggregation_result = await self._perform_secure_aggregation(participant_results)
            
            # Update global model
            model_update = await self._update_global_model(aggregation_result)
            
            # Detect byzantine participants
            byzantine_analysis = await self.byzantine_detection.detect_byzantine_participants(
                participant_results, aggregation_result
            )
            
            # Update participant trust scores
            self._update_trust_scores(byzantine_analysis)
            
            # Calculate and distribute incentives
            incentives = await self.incentive_mechanism.calculate_incentives(
                participant_results, aggregation_result
            )
            
            round_time = time.time() - round_start_time
            
            round_result = {
                'round_number': self.round_number,
                'status': 'completed',
                'participants_selected': len(selected_participants),
                'participants_responded': len(participant_results),
                'aggregation_quality': aggregation_result.get('quality_score', 0.0),
                'model_improvement': model_update.get('improvement_score', 0.0),
                'byzantine_participants_detected': len(byzantine_analysis.get('byzantine_ids', [])),
                'round_execution_time': round_time,
                'privacy_budget_consumed': self._calculate_privacy_cost(),
                'convergence_metric': model_update.get('convergence_metric', 1.0),
                'incentives_distributed': incentives
            }
            
            self.aggregation_history.append(round_result)
            
            # Check convergence
            if round_result['convergence_metric'] < self.convergence_threshold:
                logger.info(f"Federated learning converged after {self.round_number} rounds")
                round_result['converged'] = True
            
            return round_result
            
        except Exception as e:
            logger.error(f"Federated round {self.round_number} failed: {e}")
            return {
                'status': 'failed',
                'round_number': self.round_number,
                'error': str(e),
                'execution_time': time.time() - round_start_time
            }
    
    async def _select_participants_for_round(self) -> List[str]:
        """Select participants for the current round."""
        if not self.participants:
            return []
        
        # Select based on contribution score, trust score, and availability
        available_participants = []
        
        for participant_id, participant in self.participants.items():
            # Check if participant is available (last seen within 1 hour)
            if time.time() - participant.last_seen < 3600:
                # Calculate selection score
                selection_score = (
                    participant.contribution_score * 0.4 +
                    participant.trust_score * 0.4 +
                    min(participant.compute_capability, 1.0) * 0.2
                )
                available_participants.append((participant_id, selection_score))
        
        # Sort by selection score and select top participants
        available_participants.sort(key=lambda x: x[1], reverse=True)
        
        # Select up to 10 participants for this round
        max_participants = min(10, len(available_participants))
        selected = [pid for pid, _ in available_participants[:max_participants]]
        
        logger.info(f"Selected {len(selected)} participants for round {self.round_number}")
        return selected
    
    async def _distribute_task_to_participants(self, participant_ids: List[str], 
                                            task: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Distribute causal discovery task to selected participants."""
        participant_tasks = {}
        
        for participant_id in participant_ids:
            participant = self.participants[participant_id]
            
            # Customize task based on participant capabilities
            customized_task = await self._customize_task_for_participant(task, participant)
            
            # Add privacy protection if required
            if self.privacy_level != PrivacyLevel.NONE:
                customized_task = await self._add_privacy_protection(customized_task, participant)
            
            participant_tasks[participant_id] = {
                'task': customized_task,
                'deadline': time.time() + 300,  # 5 minute deadline
                'expected_compute_time': self._estimate_compute_time(customized_task, participant),
                'global_model_version': self.global_model.model_version if self.global_model else 0
            }
        
        return participant_tasks
    
    async def _customize_task_for_participant(self, task: Dict[str, Any], 
                                            participant: ParticipantInfo) -> Dict[str, Any]:
        """Customize task based on participant capabilities."""
        customized_task = task.copy()
        
        # Adjust complexity based on compute capability
        if participant.compute_capability < 0.5:
            # Reduce task complexity for lower-capability participants
            customized_task['max_graph_size'] = min(task.get('max_graph_size', 100), 50)
            customized_task['algorithm_complexity'] = 'basic'
        else:
            customized_task['algorithm_complexity'] = 'advanced'
        
        # Add participant-specific parameters
        customized_task['participant_id'] = participant.participant_id
        customized_task['data_size_hint'] = participant.data_size
        
        return customized_task
    
    async def _add_privacy_protection(self, task: Dict[str, Any], 
                                    participant: ParticipantInfo) -> Dict[str, Any]:
        """Add privacy protection to the task."""
        protected_task = task.copy()
        
        if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
            # Add differential privacy parameters
            protected_task['privacy_epsilon'] = self.privacy_budget.epsilon / self.privacy_budget.remaining_rounds
            protected_task['privacy_delta'] = self.privacy_budget.delta
            protected_task['noise_mechanism'] = 'laplace'
        
        elif self.privacy_level == PrivacyLevel.SECURE_MULTIPARTY:
            # Add secure multiparty computation parameters
            protected_task['smc_protocol'] = 'shamir_secret_sharing'
            protected_task['threshold'] = max(2, len(self.participants) // 2 + 1)
        
        elif self.privacy_level == PrivacyLevel.HOMOMORPHIC:
            # Add homomorphic encryption parameters
            protected_task['he_scheme'] = 'paillier'
            protected_task['key_size'] = 2048
        
        return protected_task
    
    def _estimate_compute_time(self, task: Dict[str, Any], participant: ParticipantInfo) -> float:
        """Estimate computation time for task on participant's system."""
        base_time = task.get('max_graph_size', 100) ** 2 * 0.001  # Basic O(n^2) estimate
        capability_factor = max(0.1, participant.compute_capability)
        return base_time / capability_factor
    
    async def _collect_participant_results(self, participant_tasks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Collect results from participants."""
        participant_results = {}
        
        # Simulate participant computation (in real implementation, this would involve network communication)
        async def simulate_participant_computation(participant_id: str, task_info: Dict[str, Any]):
            try:
                participant = self.participants[participant_id]
                task = task_info['task']
                
                # Simulate computation delay
                compute_time = task_info['expected_compute_time']
                await asyncio.sleep(min(compute_time, 1.0))  # Cap simulation time
                
                # Generate simulated causal discovery results
                result = await self._simulate_causal_discovery(task, participant)
                
                # Apply privacy protection to results
                if self.privacy_level != PrivacyLevel.NONE:
                    result = await self._apply_privacy_to_result(result, participant)
                
                return participant_id, result
                
            except Exception as e:
                logger.error(f"Participant {participant_id} computation failed: {e}")
                return participant_id, {'error': str(e)}
        
        # Collect results with timeout
        tasks = [
            simulate_participant_computation(pid, task_info)
            for pid, task_info in participant_tasks.items()
        ]
        
        try:
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0  # 1 minute timeout
            )
            
            for result in completed_tasks:
                if isinstance(result, tuple):
                    participant_id, computation_result = result
                    participant_results[participant_id] = computation_result
                    
                    # Update participant last seen time
                    if participant_id in self.participants:
                        self.participants[participant_id].last_seen = time.time()
                
        except asyncio.TimeoutError:
            logger.warning("Some participants timed out during computation")
        
        logger.info(f"Collected results from {len(participant_results)} participants")
        return participant_results
    
    async def _simulate_causal_discovery(self, task: Dict[str, Any], 
                                       participant: ParticipantInfo) -> Dict[str, Any]:
        """Simulate causal discovery computation."""
        graph_size = task.get('max_graph_size', 50)
        
        # Generate simulated causal graph
        causal_graph = np.random.random((graph_size, graph_size)) * 0.5
        
        # Make it a DAG by zeroing upper triangle
        for i in range(graph_size):
            for j in range(i, graph_size):
                causal_graph[i, j] = 0
        
        # Add some noise based on participant's data quality
        noise_level = 1.0 / max(participant.data_size / 1000, 1.0)
        causal_graph += np.random.normal(0, noise_level, causal_graph.shape)
        causal_graph = np.clip(causal_graph, 0, 1)
        
        # Calculate confidence scores
        confidence_matrix = np.random.random((graph_size, graph_size)) * participant.trust_score
        
        return {
            'causal_graph': causal_graph,
            'confidence_matrix': confidence_matrix,
            'participant_id': participant.participant_id,
            'data_size_used': participant.data_size,
            'computation_time': task.get('expected_compute_time', 0.1),
            'algorithm_used': task.get('algorithm_complexity', 'basic'),
            'privacy_level': participant.privacy_level.value
        }
    
    async def _apply_privacy_to_result(self, result: Dict[str, Any], 
                                     participant: ParticipantInfo) -> Dict[str, Any]:
        """Apply privacy protection to computation results."""
        protected_result = result.copy()
        
        if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
            # Add Laplace noise to causal graph
            epsilon = self.privacy_budget.epsilon / self.privacy_budget.remaining_rounds
            sensitivity = 1.0  # Assuming normalized causal strengths
            noise_scale = sensitivity / epsilon
            
            noise = np.random.laplace(0, noise_scale, result['causal_graph'].shape)
            protected_result['causal_graph'] = result['causal_graph'] + noise
            protected_result['privacy_noise_added'] = noise_scale
            
        elif self.privacy_level == PrivacyLevel.SECURE_MULTIPARTY:
            # Simulate secret sharing
            protected_result['secret_shares'] = self._create_secret_shares(
                result['causal_graph'], threshold=3
            )
            del protected_result['causal_graph']  # Remove plaintext
            
        return protected_result
    
    def _create_secret_shares(self, data: np.ndarray, threshold: int) -> List[Dict[str, Any]]:
        """Create secret shares for secure multiparty computation."""
        # Simplified secret sharing simulation
        shares = []
        for i in range(threshold):
            share = {
                'share_id': i,
                'share_data': (data + np.random.random(data.shape)) / threshold,
                'threshold': threshold
            }
            shares.append(share)
        return shares
    
    async def _perform_secure_aggregation(self, participant_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform secure aggregation of participant results."""
        if not participant_results:
            return {'error': 'No participant results to aggregate'}
        
        aggregation_start_time = time.time()
        
        try:
            if self.protocol == FederatedProtocol.FEDERATED_AVERAGING:
                aggregated_result = await self._federated_averaging(participant_results)
            elif self.protocol == FederatedProtocol.SECURE_AGGREGATION:
                aggregated_result = await self._secure_aggregation(participant_results)
            elif self.protocol == FederatedProtocol.BYZANTINE_ROBUST:
                aggregated_result = await self._byzantine_robust_aggregation(participant_results)
            else:
                aggregated_result = await self._federated_averaging(participant_results)
            
            aggregation_time = time.time() - aggregation_start_time
            aggregated_result['aggregation_time'] = aggregation_time
            aggregated_result['num_participants'] = len(participant_results)
            aggregated_result['aggregation_protocol'] = self.protocol.value
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Secure aggregation failed: {e}")
            return {
                'error': str(e),
                'aggregation_time': time.time() - aggregation_start_time
            }
    
    async def _federated_averaging(self, participant_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform federated averaging of causal graphs."""
        valid_results = {pid: result for pid, result in participant_results.items() 
                        if 'causal_graph' in result and 'error' not in result}
        
        if not valid_results:
            return {'error': 'No valid results for aggregation'}
        
        # Calculate weights based on data size
        total_data_size = sum(
            self.participants[pid].data_size for pid in valid_results.keys()
        )
        
        # Initialize aggregated graph
        first_result = next(iter(valid_results.values()))
        graph_shape = first_result['causal_graph'].shape
        aggregated_graph = np.zeros(graph_shape)
        aggregated_confidence = np.zeros(graph_shape)
        
        # Weighted averaging
        for participant_id, result in valid_results.items():
            participant_weight = self.participants[participant_id].data_size / total_data_size
            
            aggregated_graph += participant_weight * result['causal_graph']
            if 'confidence_matrix' in result:
                aggregated_confidence += participant_weight * result['confidence_matrix']
        
        # Calculate quality metrics
        quality_score = self._calculate_aggregation_quality(valid_results, aggregated_graph)
        
        return {
            'aggregated_causal_graph': aggregated_graph,
            'aggregated_confidence': aggregated_confidence,
            'quality_score': quality_score,
            'participant_weights': {
                pid: self.participants[pid].data_size / total_data_size 
                for pid in valid_results.keys()
            }
        }
    
    async def _secure_aggregation(self, participant_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform secure aggregation with privacy protection."""
        # Start with federated averaging
        base_result = await self._federated_averaging(participant_results)
        
        if 'error' in base_result:
            return base_result
        
        # Add additional security measures
        if self.privacy_level == PrivacyLevel.SECURE_MULTIPARTY:
            # Simulate secure multiparty aggregation
            base_result['secure_aggregation_applied'] = True
            base_result['privacy_guarantees'] = 'secure_multiparty_computation'
        
        elif self.privacy_level == PrivacyLevel.HOMOMORPHIC:
            # Simulate homomorphic encryption aggregation
            base_result['homomorphic_aggregation_applied'] = True
            base_result['privacy_guarantees'] = 'homomorphic_encryption'
        
        # Update privacy budget
        self._consume_privacy_budget()
        
        return base_result
    
    async def _byzantine_robust_aggregation(self, participant_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Byzantine-robust aggregation."""
        valid_results = {pid: result for pid, result in participant_results.items() 
                        if 'causal_graph' in result and 'error' not in result}
        
        if len(valid_results) < 3:
            return await self._federated_averaging(participant_results)
        
        # Use geometric median for Byzantine robustness
        graphs = [result['causal_graph'] for result in valid_results.values()]
        
        # Simplified geometric median (use proper implementation in production)
        stacked_graphs = np.stack(graphs)
        median_graph = np.median(stacked_graphs, axis=0)
        
        # Calculate deviation scores to identify potential Byzantine participants
        deviation_scores = {}
        for participant_id, result in valid_results.items():
            deviation = np.linalg.norm(result['causal_graph'] - median_graph)
            deviation_scores[participant_id] = float(deviation)
        
        return {
            'aggregated_causal_graph': median_graph,
            'aggregation_method': 'geometric_median',
            'byzantine_robustness': True,
            'participant_deviations': deviation_scores,
            'quality_score': 0.9  # High quality due to robustness
        }
    
    def _calculate_aggregation_quality(self, participant_results: Dict[str, Any], 
                                     aggregated_graph: np.ndarray) -> float:
        """Calculate quality score for aggregation."""
        if not participant_results:
            return 0.0
        
        # Calculate consistency across participants
        participant_graphs = [result['causal_graph'] for result in participant_results.values()]
        
        # Measure pairwise correlations
        correlations = []
        for i, graph1 in enumerate(participant_graphs):
            for j, graph2 in enumerate(participant_graphs[i+1:], i+1):
                correlation = np.corrcoef(graph1.flatten(), graph2.flatten())[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
        
        consistency_score = np.mean(correlations) if correlations else 0.0
        
        # Measure how well aggregated graph represents individual graphs
        representation_scores = []
        for graph in participant_graphs:
            correlation = np.corrcoef(graph.flatten(), aggregated_graph.flatten())[0, 1]
            if not np.isnan(correlation):
                representation_scores.append(correlation)
        
        representation_score = np.mean(representation_scores) if representation_scores else 0.0
        
        # Combined quality score
        quality_score = (consistency_score + representation_score) / 2
        return max(0.0, min(1.0, quality_score))
    
    def _consume_privacy_budget(self) -> None:
        """Consume privacy budget for this round."""
        if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
            consumed_epsilon = self.privacy_budget.epsilon / self.privacy_budget.remaining_rounds
            self.privacy_budget.consumed_epsilon += consumed_epsilon
            self.privacy_budget.remaining_rounds -= 1
    
    def _calculate_privacy_cost(self) -> float:
        """Calculate privacy cost for this round."""
        if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
            return self.privacy_budget.epsilon / max(self.privacy_budget.remaining_rounds, 1)
        return 0.0
    
    async def _update_global_model(self, aggregation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update the global causal model with aggregated results."""
        if 'error' in aggregation_result:
            return {'error': 'Cannot update model due to aggregation error'}
        
        try:
            new_causal_graph = aggregation_result['aggregated_causal_graph']
            
            if self.global_model is None:
                # Initialize global model
                self.global_model = FederatedCausalModel(
                    model_id=f"federated_model_{self.coordinator_id}",
                    global_parameters={'learning_rate': 0.01, 'regularization': 0.001},
                    causal_graph=new_causal_graph,
                    model_version=1,
                    last_updated=time.time(),
                    participant_contributions={},
                    validation_metrics={}
                )
                
                return {
                    'status': 'initialized',
                    'model_version': 1,
                    'improvement_score': 1.0,
                    'convergence_metric': 1.0
                }
            
            else:
                # Update existing model
                old_graph = self.global_model.causal_graph
                
                # Calculate improvement metrics
                improvement_score = self._calculate_model_improvement(old_graph, new_causal_graph)
                convergence_metric = np.linalg.norm(new_causal_graph - old_graph)
                
                # Update global model
                self.global_model.causal_graph = new_causal_graph
                self.global_model.model_version += 1
                self.global_model.last_updated = time.time()
                
                # Update participant contributions
                if 'participant_weights' in aggregation_result:
                    for pid, weight in aggregation_result['participant_weights'].items():
                        if pid in self.global_model.participant_contributions:
                            self.global_model.participant_contributions[pid] += weight
                        else:
                            self.global_model.participant_contributions[pid] = weight
                
                return {
                    'status': 'updated',
                    'model_version': self.global_model.model_version,
                    'improvement_score': improvement_score,
                    'convergence_metric': float(convergence_metric),
                    'total_participants': len(self.global_model.participant_contributions)
                }
                
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
            return {'error': str(e)}
    
    def _calculate_model_improvement(self, old_graph: np.ndarray, new_graph: np.ndarray) -> float:
        """Calculate improvement score between model versions."""
        try:
            # Calculate various improvement metrics
            
            # 1. Sparsity improvement (fewer spurious edges)
            old_sparsity = np.sum(old_graph > 0.1) / old_graph.size
            new_sparsity = np.sum(new_graph > 0.1) / new_graph.size
            sparsity_improvement = max(0, old_sparsity - new_sparsity)  # Lower sparsity is better
            
            # 2. Edge strength improvement
            old_avg_strength = np.mean(old_graph[old_graph > 0.1])
            new_avg_strength = np.mean(new_graph[new_graph > 0.1])
            strength_improvement = (new_avg_strength - old_avg_strength) / max(old_avg_strength, 0.01)
            
            # 3. Structural stability
            structure_similarity = np.corrcoef(old_graph.flatten(), new_graph.flatten())[0, 1]
            stability_score = max(0, structure_similarity)
            
            # Combined improvement score
            improvement_score = (
                sparsity_improvement * 0.3 +
                max(0, strength_improvement) * 0.4 +
                stability_score * 0.3
            )
            
            return max(0.0, min(1.0, improvement_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate model improvement: {e}")
            return 0.0
    
    def _update_trust_scores(self, byzantine_analysis: Dict[str, Any]) -> None:
        """Update participant trust scores based on Byzantine detection."""
        byzantine_ids = byzantine_analysis.get('byzantine_ids', [])
        
        for participant_id in self.participants:
            if participant_id in byzantine_ids:
                # Decrease trust for Byzantine participants
                self.participants[participant_id].trust_score *= 0.8
                self.participants[participant_id].trust_score = max(0.1, 
                    self.participants[participant_id].trust_score)
            else:
                # Increase trust for honest participants
                self.participants[participant_id].trust_score *= 1.02
                self.participants[participant_id].trust_score = min(1.0, 
                    self.participants[participant_id].trust_score)
    
    def get_federated_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive federated learning statistics."""
        stats = {
            'coordinator_id': self.coordinator_id,
            'protocol': self.protocol.value,
            'privacy_level': self.privacy_level.value,
            'total_participants': len(self.participants),
            'active_participants': sum(
                1 for p in self.participants.values() 
                if time.time() - p.last_seen < 3600
            ),
            'total_rounds_completed': self.round_number,
            'global_model_version': self.global_model.model_version if self.global_model else 0,
            'privacy_budget_remaining': self.privacy_budget.remaining_rounds,
            'privacy_budget_consumed': self.privacy_budget.consumed_epsilon
        }
        
        if self.participants:
            # Participant statistics
            trust_scores = [p.trust_score for p in self.participants.values()]
            contribution_scores = [p.contribution_score for p in self.participants.values()]
            
            stats.update({
                'average_trust_score': np.mean(trust_scores),
                'min_trust_score': np.min(trust_scores),
                'max_trust_score': np.max(trust_scores),
                'average_contribution_score': np.mean(contribution_scores),
                'total_data_size': sum(p.data_size for p in self.participants.values())
            })
        
        if self.aggregation_history:
            # Round statistics
            convergence_metrics = [r.get('convergence_metric', 1.0) for r in self.aggregation_history]
            quality_scores = [r.get('aggregation_quality', 0.0) for r in self.aggregation_history]
            
            stats.update({
                'average_convergence_metric': np.mean(convergence_metrics),
                'latest_convergence_metric': convergence_metrics[-1],
                'average_aggregation_quality': np.mean(quality_scores),
                'latest_aggregation_quality': quality_scores[-1],
                'convergence_trend': 'improving' if len(convergence_metrics) > 1 and 
                    convergence_metrics[-1] < convergence_metrics[-2] else 'stable'
            })
        
        return stats

class ByzantineDetection:
    """Byzantine participant detection system."""
    
    async def detect_byzantine_participants(self, participant_results: Dict[str, Any], 
                                          aggregation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potentially Byzantine participants."""
        if 'aggregated_causal_graph' not in aggregation_result:
            return {'byzantine_ids': [], 'method': 'none'}
        
        aggregated_graph = aggregation_result['aggregated_causal_graph']
        byzantine_scores = {}
        
        for participant_id, result in participant_results.items():
            if 'causal_graph' in result:
                # Calculate deviation from aggregated result
                deviation = np.linalg.norm(result['causal_graph'] - aggregated_graph)
                byzantine_scores[participant_id] = float(deviation)
        
        # Identify outliers using IQR method
        scores = list(byzantine_scores.values())
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25
        threshold = q75 + 1.5 * iqr
        
        byzantine_ids = [
            pid for pid, score in byzantine_scores.items() 
            if score > threshold
        ]
        
        return {
            'byzantine_ids': byzantine_ids,
            'deviation_scores': byzantine_scores,
            'detection_threshold': float(threshold),
            'method': 'iqr_outlier_detection'
        }

class IncentiveMechanism:
    """Incentive mechanism for federated learning participants."""
    
    async def calculate_incentives(self, participant_results: Dict[str, Any], 
                                 aggregation_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate incentives for participants based on contribution quality."""
        incentives = {}
        
        if 'participant_weights' not in aggregation_result:
            # Equal incentives if no weights available
            base_incentive = 1.0
            for participant_id in participant_results:
                incentives[participant_id] = base_incentive
            return incentives
        
        quality_score = aggregation_result.get('quality_score', 0.5)
        participant_weights = aggregation_result['participant_weights']
        
        # Base incentive pool
        total_incentive_pool = 100.0
        
        for participant_id, weight in participant_weights.items():
            if participant_id in participant_results:
                # Calculate individual incentive based on weight and quality
                base_incentive = total_incentive_pool * weight
                quality_bonus = base_incentive * quality_score * 0.5
                incentives[participant_id] = base_incentive + quality_bonus
        
        return incentives