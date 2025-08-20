"""Multi-agent LLM orchestration system for advanced causal reasoning."""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import uuid

from ..llm.client import LLMClient
from ..core import CausalEnvironment

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles for specialized LLM agents."""
    CAUSAL_REASONER = "causal_reasoner"
    INTERVENTION_DESIGNER = "intervention_designer"
    EVIDENCE_EVALUATOR = "evidence_evaluator"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    DOMAIN_EXPERT = "domain_expert"
    METHODOLOGIST = "methodologist"


@dataclass
class AgentCapabilities:
    """Defines capabilities of an LLM agent."""
    role: AgentRole
    expertise_areas: List[str]
    reasoning_strengths: List[str]
    preferred_tasks: List[str]
    confidence_threshold: float = 0.7
    max_concurrent_tasks: int = 3
    response_time_limit: float = 30.0
    

@dataclass
class AgentMessage:
    """Message between agents in the orchestration system."""
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    requires_response: bool = False
    response_deadline: Optional[datetime] = None


@dataclass
class CollaborativeTask:
    """Task that requires collaboration between multiple agents."""
    task_id: str
    task_type: str
    description: str
    required_roles: List[AgentRole]
    assigned_agents: Dict[AgentRole, str] = field(default_factory=dict)
    subtasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None


class LLMAgent:
    """Individual LLM agent with specialized capabilities."""
    
    def __init__(self,
                 agent_id: str,
                 capabilities: AgentCapabilities,
                 llm_client: LLMClient,
                 memory_size: int = 1000):
        """Initialize LLM agent.
        
        Args:
            agent_id: Unique identifier for the agent
            capabilities: Agent's capabilities and specializations
            llm_client: LLM client for inference
            memory_size: Size of agent's working memory
        """
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.llm_client = llm_client
        self.memory_size = memory_size
        
        self.working_memory: List[Dict[str, Any]] = []
        self.long_term_memory: Dict[str, Any] = {}
        self.current_tasks: Dict[str, CollaborativeTask] = {}
        self.message_queue: List[AgentMessage] = []
        
        self.performance_metrics = {
            "tasks_completed": 0,
            "avg_response_time": 0.0,
            "accuracy_score": 0.0,
            "collaboration_rating": 0.0
        }
        
        # Specialized prompts for different roles
        self.role_prompts = self._initialize_role_prompts()
    
    def _initialize_role_prompts(self) -> Dict[str, str]:
        """Initialize role-specific prompts."""
        prompts = {
            AgentRole.CAUSAL_REASONER: """
            You are an expert in causal reasoning and Pearl's causal hierarchy.
            Your role is to analyze causal relationships, identify confounders,
            and reason about interventions vs observations.
            Focus on do-calculus, backdoor criteria, and causal identification.
            """,
            
            AgentRole.INTERVENTION_DESIGNER: """
            You are an expert in designing causal interventions.
            Your role is to propose optimal interventions to test causal hypotheses,
            considering feasibility, ethics, and statistical power.
            Focus on experimental design and intervention strategies.
            """,
            
            AgentRole.EVIDENCE_EVALUATOR: """
            You are an expert in evaluating causal evidence.
            Your role is to assess the quality and strength of causal claims,
            identify potential biases, and evaluate study designs.
            Focus on evidence hierarchies and methodological rigor.
            """,
            
            AgentRole.HYPOTHESIS_GENERATOR: """
            You are an expert in generating causal hypotheses.
            Your role is to propose plausible causal mechanisms and alternative
            explanations for observed phenomena.
            Focus on creativity and theoretical grounding.
            """,
            
            AgentRole.CRITIC: """
            You are a critical evaluator of causal claims and reasoning.
            Your role is to identify flaws, challenge assumptions, and propose
            alternative explanations.
            Focus on constructive criticism and devil's advocate perspectives.
            """,
            
            AgentRole.SYNTHESIZER: """
            You are an expert in synthesizing multiple perspectives on causation.
            Your role is to integrate diverse viewpoints, resolve conflicts,
            and create coherent conclusions from agent discussions.
            Focus on consensus building and integration.
            """,
            
            AgentRole.DOMAIN_EXPERT: """
            You are a domain-specific expert with deep knowledge in your field.
            Your role is to provide domain-specific insights, constraints,
            and contextual knowledge relevant to causal questions.
            Focus on domain expertise and practical considerations.
            """,
            
            AgentRole.METHODOLOGIST: """
            You are an expert in causal inference methodology.
            Your role is to ensure methodological rigor, recommend appropriate
            statistical techniques, and guide research design decisions.
            Focus on methodological best practices and statistical validity.
            """
        }
        
        return prompts
    
    async def process_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Process incoming message and generate response if needed."""
        # Add to working memory
        self._add_to_memory({
            "type": "message",
            "content": message.content,
            "sender": message.sender_id,
            "timestamp": message.timestamp
        })
        
        if message.requires_response:
            response = await self._generate_response(message)
            return response
        else:
            # Process message without response
            await self._process_information(message.content)
            return None
    
    async def _generate_response(self, message: AgentMessage) -> Dict[str, Any]:
        """Generate response to a message based on agent's role."""
        start_time = datetime.now()
        
        try:
            # Construct context-aware prompt
            context = self._build_context(message)
            role_prompt = self.role_prompts.get(self.capabilities.role, "")
            
            full_prompt = f"""
            {role_prompt}
            
            Context: {context}
            
            Message: {json.dumps(message.content, indent=2)}
            
            Please provide your expert analysis and recommendations based on your role
            as a {self.capabilities.role.value}.
            """
            
            # Generate response using LLM
            response_text = await self.llm_client.generate_response(
                prompt=full_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            response = {
                "agent_id": self.agent_id,
                "role": self.capabilities.role.value,
                "response": response_text,
                "confidence": self._assess_response_confidence(response_text),
                "response_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Update performance metrics
            self._update_performance_metrics(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate response: {e}")
            return {
                "agent_id": self.agent_id,
                "role": self.capabilities.role.value,
                "response": f"Error generating response: {str(e)}",
                "confidence": 0.0,
                "response_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _build_context(self, message: AgentMessage) -> str:
        """Build context from working memory and current tasks."""
        recent_memory = self.working_memory[-10:]  # Last 10 memories
        context_parts = []
        
        # Add recent memory
        if recent_memory:
            context_parts.append("Recent context:")
            for memory in recent_memory:
                context_parts.append(f"- {memory.get('type', 'unknown')}: {str(memory.get('content', ''))[:200]}...")
        
        # Add current tasks
        if self.current_tasks:
            context_parts.append("Current tasks:")
            for task_id, task in self.current_tasks.items():
                context_parts.append(f"- {task.task_type}: {task.description[:100]}...")
        
        return "\n".join(context_parts)
    
    def _assess_response_confidence(self, response: str) -> float:
        """Assess confidence in the generated response."""
        # Simple heuristic-based confidence assessment
        confidence_indicators = [
            "confident", "certain", "clearly", "definitely", "strongly",
            "evidence shows", "research indicates", "studies demonstrate"
        ]
        
        uncertainty_indicators = [
            "might", "could", "possibly", "perhaps", "uncertain",
            "unclear", "ambiguous", "difficult to determine"
        ]
        
        response_lower = response.lower()
        
        confidence_score = sum(1 for indicator in confidence_indicators 
                             if indicator in response_lower)
        uncertainty_score = sum(1 for indicator in uncertainty_indicators 
                              if indicator in response_lower)
        
        # Normalize and combine scores
        total_words = len(response.split())
        confidence_ratio = confidence_score / max(total_words / 50, 1)
        uncertainty_ratio = uncertainty_score / max(total_words / 50, 1)
        
        final_confidence = max(0.1, min(0.9, 0.5 + confidence_ratio - uncertainty_ratio))
        return final_confidence
    
    def _add_to_memory(self, memory_item: Dict[str, Any]) -> None:
        """Add item to working memory with size management."""
        self.working_memory.append(memory_item)
        
        # Manage memory size
        if len(self.working_memory) > self.memory_size:
            # Remove oldest items
            self.working_memory = self.working_memory[-self.memory_size:]
    
    async def _process_information(self, information: Dict[str, Any]) -> None:
        """Process information without generating a response."""
        # Add to memory for future reference
        self._add_to_memory({
            "type": "information",
            "content": information,
            "timestamp": datetime.now()
        })
        
        # Extract and store relevant patterns
        if "causal_pattern" in information:
            pattern = information["causal_pattern"]
            pattern_key = f"pattern_{len(self.long_term_memory)}"
            self.long_term_memory[pattern_key] = pattern
    
    def _update_performance_metrics(self, response: Dict[str, Any]) -> None:
        """Update agent performance metrics."""
        self.performance_metrics["tasks_completed"] += 1
        
        # Update average response time
        current_avg = self.performance_metrics["avg_response_time"]
        new_time = response["response_time"]
        task_count = self.performance_metrics["tasks_completed"]
        
        self.performance_metrics["avg_response_time"] = (
            (current_avg * (task_count - 1) + new_time) / task_count
        )
        
        # Update accuracy score (simplified)
        confidence = response["confidence"]
        current_accuracy = self.performance_metrics["accuracy_score"]
        self.performance_metrics["accuracy_score"] = (
            (current_accuracy * (task_count - 1) + confidence) / task_count
        )


class MultiAgentOrchestrator:
    """Orchestrates collaboration between multiple LLM agents."""
    
    def __init__(self, max_agents: int = 20, collaboration_timeout: float = 300.0):
        """Initialize multi-agent orchestrator.
        
        Args:
            max_agents: Maximum number of agents to manage
            collaboration_timeout: Timeout for collaborative tasks
        """
        self.max_agents = max_agents
        self.collaboration_timeout = collaboration_timeout
        
        self.agents: Dict[str, LLMAgent] = {}
        self.active_tasks: Dict[str, CollaborativeTask] = {}
        self.message_bus: List[AgentMessage] = []
        self.collaboration_history: List[Dict[str, Any]] = []
        
        self.orchestration_metrics = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "avg_collaboration_time": 0.0,
            "agent_utilization": 0.0
        }
    
    def register_agent(self, agent: LLMAgent) -> bool:
        """Register an agent with the orchestrator."""
        if len(self.agents) >= self.max_agents:
            logger.warning(f"Cannot register agent {agent.agent_id}: max agents reached")
            return False
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} with role {agent.capabilities.role.value}")
        return True
    
    async def create_agent_ensemble(self,
                                  roles: List[AgentRole],
                                  llm_clients: Dict[str, LLMClient]) -> Dict[str, str]:
        """Create an ensemble of agents for collaborative causal reasoning."""
        agent_assignments = {}
        
        for role in roles:
            # Select appropriate LLM client for the role
            client_name = self._select_client_for_role(role, llm_clients)
            client = llm_clients.get(client_name)
            
            if client:
                # Create capabilities for the role
                capabilities = AgentCapabilities(
                    role=role,
                    expertise_areas=self._get_expertise_areas(role),
                    reasoning_strengths=self._get_reasoning_strengths(role),
                    preferred_tasks=self._get_preferred_tasks(role)
                )
                
                # Create agent
                agent_id = f"{role.value}_{uuid.uuid4().hex[:8]}"
                agent = LLMAgent(agent_id, capabilities, client)
                
                if self.register_agent(agent):
                    agent_assignments[role.value] = agent_id
                    logger.info(f"Created agent {agent_id} for role {role.value}")
        
        return agent_assignments
    
    def _select_client_for_role(self, role: AgentRole, clients: Dict[str, LLMClient]) -> str:
        """Select the best LLM client for a specific role."""
        # Role-specific client preferences (can be made configurable)
        role_preferences = {
            AgentRole.CAUSAL_REASONER: ["gpt-4", "claude-3", "gpt-3.5-turbo"],
            AgentRole.INTERVENTION_DESIGNER: ["gpt-4", "claude-3"],
            AgentRole.EVIDENCE_EVALUATOR: ["claude-3", "gpt-4"],
            AgentRole.HYPOTHESIS_GENERATOR: ["gpt-4", "claude-3"],
            AgentRole.CRITIC: ["claude-3", "gpt-4"],
            AgentRole.SYNTHESIZER: ["gpt-4", "claude-3"],
            AgentRole.DOMAIN_EXPERT: ["gpt-4", "claude-3"],
            AgentRole.METHODOLOGIST: ["claude-3", "gpt-4"]
        }
        
        preferred_clients = role_preferences.get(role, list(clients.keys()))
        
        # Return first available preferred client
        for client_name in preferred_clients:
            if client_name in clients:
                return client_name
        
        # Fallback to any available client
        return list(clients.keys())[0] if clients else "default"
    
    def _get_expertise_areas(self, role: AgentRole) -> List[str]:
        """Get expertise areas for a role."""
        expertise_map = {
            AgentRole.CAUSAL_REASONER: ["causal_inference", "do_calculus", "pearl_hierarchy"],
            AgentRole.INTERVENTION_DESIGNER: ["experimental_design", "randomized_trials", "quasi_experiments"],
            AgentRole.EVIDENCE_EVALUATOR: ["study_design", "bias_detection", "evidence_quality"],
            AgentRole.HYPOTHESIS_GENERATOR: ["theory_generation", "mechanism_design", "creative_reasoning"],
            AgentRole.CRITIC: ["logical_reasoning", "argumentation", "error_detection"],
            AgentRole.SYNTHESIZER: ["integration", "consensus_building", "meta_analysis"],
            AgentRole.DOMAIN_EXPERT: ["domain_knowledge", "practical_constraints", "contextual_factors"],
            AgentRole.METHODOLOGIST: ["statistical_methods", "research_design", "validity"]
        }
        return expertise_map.get(role, [])
    
    def _get_reasoning_strengths(self, role: AgentRole) -> List[str]:
        """Get reasoning strengths for a role."""
        strengths_map = {
            AgentRole.CAUSAL_REASONER: ["deductive", "graphical_reasoning", "counterfactual"],
            AgentRole.INTERVENTION_DESIGNER: ["creative", "practical", "strategic"],
            AgentRole.EVIDENCE_EVALUATOR: ["critical", "analytical", "systematic"],
            AgentRole.HYPOTHESIS_GENERATOR: ["abductive", "creative", "analogical"],
            AgentRole.CRITIC: ["critical", "logical", "contrarian"],
            AgentRole.SYNTHESIZER: ["integrative", "holistic", "diplomatic"],
            AgentRole.DOMAIN_EXPERT: ["contextual", "practical", "experiential"],
            AgentRole.METHODOLOGIST: ["systematic", "rigorous", "technical"]
        }
        return strengths_map.get(role, [])
    
    def _get_preferred_tasks(self, role: AgentRole) -> List[str]:
        """Get preferred tasks for a role."""
        tasks_map = {
            AgentRole.CAUSAL_REASONER: ["causal_analysis", "confound_identification", "mechanism_inference"],
            AgentRole.INTERVENTION_DESIGNER: ["experiment_design", "intervention_planning", "feasibility_analysis"],
            AgentRole.EVIDENCE_EVALUATOR: ["study_evaluation", "bias_assessment", "quality_rating"],
            AgentRole.HYPOTHESIS_GENERATOR: ["hypothesis_generation", "theory_building", "explanation_creation"],
            AgentRole.CRITIC: ["critique", "error_detection", "alternative_explanation"],
            AgentRole.SYNTHESIZER: ["result_integration", "consensus_building", "summary_creation"],
            AgentRole.DOMAIN_EXPERT: ["context_provision", "constraint_identification", "practical_guidance"],
            AgentRole.METHODOLOGIST: ["method_selection", "analysis_design", "validity_assessment"]
        }
        return tasks_map.get(role, [])
    
    async def orchestrate_causal_analysis(self,
                                        problem: Dict[str, Any],
                                        required_roles: Optional[List[AgentRole]] = None) -> Dict[str, Any]:
        """Orchestrate a collaborative causal analysis using multiple agents."""
        if required_roles is None:
            required_roles = [
                AgentRole.CAUSAL_REASONER,
                AgentRole.EVIDENCE_EVALUATOR,
                AgentRole.HYPOTHESIS_GENERATOR,
                AgentRole.CRITIC,
                AgentRole.SYNTHESIZER
            ]
        
        # Create collaborative task
        task_id = f"causal_analysis_{uuid.uuid4().hex[:8]}"
        task = CollaborativeTask(
            task_id=task_id,
            task_type="causal_analysis",
            description=problem.get("description", "Collaborative causal analysis"),
            required_roles=required_roles,
            deadline=datetime.now() + timedelta(seconds=self.collaboration_timeout)
        )
        
        self.active_tasks[task_id] = task
        
        try:
            # Phase 1: Individual analysis
            individual_results = await self._phase_individual_analysis(task, problem)
            
            # Phase 2: Peer review and critique
            critique_results = await self._phase_peer_critique(task, individual_results)
            
            # Phase 3: Synthesis and consensus
            final_result = await self._phase_synthesis(task, critique_results)
            
            # Update metrics
            self.orchestration_metrics["total_collaborations"] += 1
            if final_result.get("status") == "success":
                self.orchestration_metrics["successful_collaborations"] += 1
            
            task.status = "completed"
            task.results = final_result
            
            # Store collaboration history
            self.collaboration_history.append({
                "task_id": task_id,
                "problem": problem,
                "results": final_result,
                "timestamp": datetime.now(),
                "agents_involved": list(task.assigned_agents.values())
            })
            
            return final_result
            
        except Exception as e:
            logger.error(f"Orchestration failed for task {task_id}: {e}")
            task.status = "failed"
            return {
                "status": "error",
                "error": str(e),
                "task_id": task_id
            }
        finally:
            # Clean up
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _phase_individual_analysis(self,
                                       task: CollaborativeTask,
                                       problem: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Individual analysis by each agent."""
        results = {}
        
        # Assign agents to roles
        available_agents = {
            agent.capabilities.role: agent_id
            for agent_id, agent in self.agents.items()
            if agent.capabilities.role in task.required_roles
        }
        
        task.assigned_agents = available_agents
        
        # Send analysis requests to agents
        analysis_tasks = []
        for role in task.required_roles:
            agent_id = available_agents.get(role)
            if agent_id and agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Create analysis message
                message = AgentMessage(
                    sender_id="orchestrator",
                    recipient_id=agent_id,
                    message_type="analysis_request",
                    content={
                        "task_id": task.task_id,
                        "problem": problem,
                        "role_specific_focus": self._get_role_focus(role)
                    },
                    timestamp=datetime.now(),
                    requires_response=True,
                    response_deadline=datetime.now() + timedelta(seconds=60)
                )
                
                # Process message asynchronously
                analysis_task = asyncio.create_task(
                    agent.process_message(message)
                )
                analysis_tasks.append((role.value, analysis_task))
        
        # Wait for all analyses
        for role_name, task_future in analysis_tasks:
            try:
                result = await asyncio.wait_for(task_future, timeout=60)
                if result:
                    results[role_name] = result
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for {role_name} analysis")
                results[role_name] = {"error": "timeout"}
        
        return results
    
    async def _phase_peer_critique(self,
                                 task: CollaborativeTask,
                                 individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Peer critique and review."""
        critique_results = {}
        
        # Each agent critiques other agents' work
        critic_agent_id = task.assigned_agents.get(AgentRole.CRITIC)
        if critic_agent_id and critic_agent_id in self.agents:
            critic = self.agents[critic_agent_id]
            
            critique_message = AgentMessage(
                sender_id="orchestrator",
                recipient_id=critic_agent_id,
                message_type="critique_request",
                content={
                    "task_id": task.task_id,
                    "analyses_to_critique": individual_results,
                    "focus": "identify_weaknesses_and_alternatives"
                },
                timestamp=datetime.now(),
                requires_response=True,
                response_deadline=datetime.now() + timedelta(seconds=60)
            )
            
            try:
                critique_result = await asyncio.wait_for(
                    critic.process_message(critique_message),
                    timeout=60
                )
                if critique_result:
                    critique_results["critique"] = critique_result
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for critique")
        
        return critique_results
    
    async def _phase_synthesis(self,
                             task: CollaborativeTask,
                             all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Synthesis and consensus building."""
        synthesizer_agent_id = task.assigned_agents.get(AgentRole.SYNTHESIZER)
        
        if not (synthesizer_agent_id and synthesizer_agent_id in self.agents):
            # Fallback: use first available agent for synthesis
            synthesizer_agent_id = list(task.assigned_agents.values())[0]
        
        if synthesizer_agent_id in self.agents:
            synthesizer = self.agents[synthesizer_agent_id]
            
            synthesis_message = AgentMessage(
                sender_id="orchestrator",
                recipient_id=synthesizer_agent_id,
                message_type="synthesis_request",
                content={
                    "task_id": task.task_id,
                    "all_analyses": all_results,
                    "goal": "integrate_perspectives_and_build_consensus"
                },
                timestamp=datetime.now(),
                requires_response=True,
                response_deadline=datetime.now() + timedelta(seconds=90)
            )
            
            try:
                synthesis_result = await asyncio.wait_for(
                    synthesizer.process_message(synthesis_message),
                    timeout=90
                )
                
                if synthesis_result:
                    return {
                        "status": "success",
                        "synthesis": synthesis_result,
                        "individual_analyses": all_results,
                        "collaboration_quality": self._assess_collaboration_quality(all_results)
                    }
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for synthesis")
        
        # Fallback synthesis
        return {
            "status": "partial",
            "message": "Synthesis phase incomplete",
            "individual_analyses": all_results,
            "collaboration_quality": 0.5
        }
    
    def _get_role_focus(self, role: AgentRole) -> str:
        """Get role-specific focus for analysis."""
        focus_map = {
            AgentRole.CAUSAL_REASONER: "Identify causal relationships, confounders, and apply do-calculus",
            AgentRole.EVIDENCE_EVALUATOR: "Assess evidence quality, identify biases, evaluate methodology",
            AgentRole.HYPOTHESIS_GENERATOR: "Generate alternative hypotheses and causal mechanisms",
            AgentRole.INTERVENTION_DESIGNER: "Design interventions to test causal hypotheses",
            AgentRole.CRITIC: "Identify flaws, weaknesses, and alternative explanations",
            AgentRole.SYNTHESIZER: "Integrate multiple perspectives into coherent conclusions",
            AgentRole.DOMAIN_EXPERT: "Provide domain-specific context and constraints",
            AgentRole.METHODOLOGIST: "Ensure methodological rigor and statistical validity"
        }
        return focus_map.get(role, "Provide your expert perspective")
    
    def _assess_collaboration_quality(self, results: Dict[str, Any]) -> float:
        """Assess the quality of collaboration based on results."""
        if not results:
            return 0.0
        
        # Metrics for collaboration quality
        num_participants = len([r for r in results.values() if isinstance(r, dict) and "response" in r])
        avg_confidence = np.mean([
            r.get("confidence", 0) for r in results.values()
            if isinstance(r, dict) and "confidence" in r
        ])
        
        response_quality = len([
            r for r in results.values()
            if isinstance(r, dict) and len(str(r.get("response", ""))) > 100
        ]) / max(len(results), 1)
        
        # Combine metrics
        participation_score = min(1.0, num_participants / 5)  # Expect ~5 agents
        confidence_score = avg_confidence if not np.isnan(avg_confidence) else 0.5
        quality_score = response_quality
        
        collaboration_quality = (participation_score + confidence_score + quality_score) / 3
        return collaboration_quality
    
    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Get summary of orchestration performance and metrics."""
        active_agent_count = len(self.agents)
        total_tasks = self.orchestration_metrics["total_collaborations"]
        
        success_rate = (
            self.orchestration_metrics["successful_collaborations"] / max(total_tasks, 1)
        )
        
        return {
            "active_agents": active_agent_count,
            "total_collaborations": total_tasks,
            "success_rate": success_rate,
            "avg_collaboration_time": self.orchestration_metrics["avg_collaboration_time"],
            "agent_performance": {
                agent_id: {
                    "role": agent.capabilities.role.value,
                    "tasks_completed": agent.performance_metrics["tasks_completed"],
                    "avg_response_time": agent.performance_metrics["avg_response_time"],
                    "accuracy_score": agent.performance_metrics["accuracy_score"]
                }
                for agent_id, agent in self.agents.items()
            },
            "recent_collaborations": self.collaboration_history[-5:]  # Last 5
        }