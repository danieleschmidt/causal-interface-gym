"""Comprehensive tests for causal reasoning functionality."""

import pytest
import numpy as np
from typing import Dict, Any

from causal_interface_gym.core import CausalEnvironment, InterventionUI
from causal_interface_gym.metrics import CausalMetrics, BeliefTracker
from tests.conftest import assert_valid_probability, assert_causal_effect_properties


class TestCausalEnvironmentAdvanced:
    """Advanced tests for causal environment functionality."""
    
    def test_do_calculus_comprehensive(self, environment_scenarios):
        """Test do-calculus across different scenarios."""
        for scenario_name, scenario_data in environment_scenarios.items():
            env = CausalEnvironment.from_dag(scenario_data["dag"])
            
            # Test all possible treatment-outcome pairs
            nodes = list(scenario_data["dag"].keys())
            for treatment in nodes:
                for outcome in nodes:
                    if treatment != outcome:
                        result = env.do_calculus(treatment, outcome, 1.0)
                        assert_causal_effect_properties(result)
    
    def test_backdoor_identification_accuracy(self, environment_scenarios):
        """Test accuracy of backdoor path identification."""
        for scenario_name, scenario_data in environment_scenarios.items():
            env = CausalEnvironment.from_dag(scenario_data["dag"])
            
            if "expected_backdoors" in scenario_data:
                for (treatment, outcome), expected_paths in scenario_data["expected_backdoors"].items():
                    backdoor_paths = env.get_backdoor_paths(treatment, outcome)
                    
                    # Check that we find the expected number of backdoor paths
                    assert len(backdoor_paths) >= len(expected_paths)
    
    def test_causal_effect_consistency(self, sample_dag):
        """Test that causal effects are consistent."""
        env = CausalEnvironment.from_dag(sample_dag)
        
        # Test same intervention multiple times
        results = []
        for _ in range(5):
            result = env.do_calculus("sprinkler", "wet_grass", True)
            if result["identifiable"]:
                results.append(result["causal_effect"])
        
        if results:
            # Results should be deterministic (same each time)
            assert all(abs(r - results[0]) < 0.1 for r in results)
    
    def test_intervention_vs_observation_distinction(self, sample_dag):
        """Test that intervention and observation are properly distinguished."""
        env = CausalEnvironment.from_dag(sample_dag)
        
        # Create belief trajectory that should show intervention understanding
        belief_trajectory = {
            "observational_beliefs": {"wet_grass": 0.6},
            "intervention_beliefs": {"wet_grass": 0.9}  # Higher after sprinkler intervention
        }
        
        analysis = env.analyze_causal_reasoning(belief_trajectory)
        
        assert "intervention_vs_observation" in analysis
        assert "wet_grass" in analysis["intervention_vs_observation"]
        
        # Should recognize the difference
        wet_grass_analysis = analysis["intervention_vs_observation"]["wet_grass"]
        assert wet_grass_analysis["difference"] > 0.2  # Significant difference
        assert wet_grass_analysis["score"] > 0.5  # Good understanding


class TestBeliefTracking:
    """Tests for belief tracking functionality."""
    
    def test_belief_recording(self, belief_tracker):
        """Test belief recording and retrieval."""
        belief_tracker.record("P(rain)", "observational", 0.3)
        belief_tracker.record("P(rain)", "do(sprinkler=on)", 0.1)
        
        assert len(belief_tracker.beliefs) == 1
        assert len(belief_tracker.beliefs["P(rain)"]) == 2
        
        # Check values
        beliefs = belief_tracker.beliefs["P(rain)"]
        assert beliefs[0]["value"] == 0.3
        assert beliefs[1]["value"] == 0.1
        assert beliefs[0]["condition"] == "observational"
        assert beliefs[1]["condition"] == "do(sprinkler=on)"
    
    def test_belief_evolution_plot(self, belief_tracker):
        """Test belief evolution plotting."""
        # Record belief trajectory
        for i, condition in enumerate(["obs", "do(sprinkler=on)", "do(rain=off)"]):
            belief_tracker.record("P(wet_grass)", condition, 0.5 + i * 0.2)
        
        plot_data = belief_tracker.plot_belief_evolution(
            "wet_grass",
            ["obs", "do(sprinkler=on)"],
            "increase"
        )
        
        assert "belief_trajectories" in plot_data
        assert "analysis" in plot_data
        assert "expected_change" in plot_data
    
    def test_belief_summary(self, belief_tracker):
        """Test belief summary generation."""
        # Add multiple beliefs
        belief_tracker.record("P(rain)", "observational", 0.3)
        belief_tracker.record("P(rain)", "do(sprinkler=on)", 0.1)
        belief_tracker.record("P(wet_grass)", "observational", 0.6)
        
        summary = belief_tracker.get_belief_summary()
        
        assert summary["total_beliefs_tracked"] == 2
        assert summary["total_measurements"] == 3
        assert "P(rain)" in summary["belief_summaries"]
        assert "P(wet_grass)" in summary["belief_summaries"]


class TestCausalMetrics:
    """Tests for causal reasoning metrics."""
    
    def test_intervention_test_scoring(self, causal_metrics):
        """Test intervention vs observation scoring."""
        # Agent correctly distinguishes intervention from observation
        agent_responses = [
            {"intervention_belief": 0.9, "observational_belief": 0.6}
        ]
        ground_truth = [
            {"intervention_belief": 0.85, "observational_belief": 0.55}
        ]
        
        score = causal_metrics.intervention_test(agent_responses, ground_truth)
        assert_valid_probability(score)
        assert score > 0.7  # Should score well
    
    def test_backdoor_test_scoring(self, causal_metrics):
        """Test backdoor identification scoring."""
        import networkx as nx
        
        # Create test graphs
        true_graph = nx.DiGraph()
        true_graph.add_edges_from([("Z", "X"), ("Z", "Y"), ("X", "Y")])
        
        agent_graph = nx.DiGraph()
        agent_graph.add_edges_from([("Z", "X"), ("Z", "Y"), ("X", "Y")])
        
        score = causal_metrics.backdoor_test(agent_graph, true_graph)
        assert_valid_probability(score)
        assert score > 0.8  # Should score well for correct graph
    
    def test_counterfactual_test_scoring(self, causal_metrics):
        """Test counterfactual reasoning scoring."""
        agent_predictions = [0.3, 0.7, 0.5, 0.8]
        true_counterfactuals = [0.25, 0.75, 0.45, 0.85]
        
        score = causal_metrics.counterfactual_test(agent_predictions, true_counterfactuals)
        assert_valid_probability(score)
        assert score > 0.8  # Should score well for correlated predictions
    
    def test_comprehensive_evaluation(self, causal_metrics, sample_experiment_data):
        """Test comprehensive evaluation metrics."""
        evaluation = causal_metrics.comprehensive_evaluation(sample_experiment_data)
        
        required_metrics = [
            "overall_score",
            "intervention_understanding",
            "backdoor_identification",
            "counterfactual_reasoning",
            "belief_consistency"
        ]
        
        for metric in required_metrics:
            assert metric in evaluation
            if evaluation[metric] > 0:  # Only test if metric was computed
                assert_valid_probability(evaluation[metric])


class TestExperimentWorkflows:
    """Test complete experiment workflows."""
    
    @pytest.mark.integration
    def test_complete_causal_experiment(self, sample_dag, deterministic_agent):
        """Test a complete causal reasoning experiment."""
        # Setup environment and UI
        env = CausalEnvironment.from_dag(sample_dag)
        ui = InterventionUI(env)
        metrics = CausalMetrics()
        tracker = BeliefTracker(deterministic_agent)
        
        # Configure UI
        ui.add_intervention_button("sprinkler", "Sprinkler Control")
        ui.add_observation_panel("wet_grass", "Grass Status")
        ui.add_belief_display(["P(rain)", "P(wet_grass)"])
        ui.add_graph_visualization(show_backdoors=True)
        
        # Run experiment
        result = ui.run_experiment(
            agent=deterministic_agent,
            interventions=[("sprinkler", True), ("rain", False)],
            measure_beliefs=["P(rain|wet_grass)", "P(slippery)"]
        )
        
        # Analyze results
        evaluation = metrics.comprehensive_evaluation(result)
        
        # Validate complete workflow
        assert "experiment_id" in result
        assert len(result["intervention_results"]) == 2
        assert "causal_analysis" in result
        assert "overall_score" in evaluation
        
        # Export results
        exported = ui.export_results("json")
        assert len(exported) > 100  # Should be substantial JSON
    
    @pytest.mark.integration
    def test_multi_agent_comparison(self, sample_dag):
        """Test comparing multiple agents."""
        from tests.conftest import MockLLMAgent
        
        env = CausalEnvironment.from_dag(sample_dag)
        ui = InterventionUI(env)
        metrics = CausalMetrics()
        
        # Create different agents with different capabilities
        good_agent = MockLLMAgent(belief_responses={
            "P(rain|wet_grass)": 0.3,  # Correctly lower after sprinkler intervention
            "P(slippery)": 0.8
        })
        
        bad_agent = MockLLMAgent(belief_responses={
            "P(rain|wet_grass)": 0.7,  # Incorrectly same as observational
            "P(slippery)": 0.3
        })
        
        # Test both agents
        results = {}
        for agent_name, agent in [("good", good_agent), ("bad", bad_agent)]:
            result = ui.run_experiment(
                agent=agent,
                interventions=[("sprinkler", True)],
                measure_beliefs=["P(rain|wet_grass)", "P(slippery)"]
            )
            
            evaluation = metrics.comprehensive_evaluation(result)
            results[agent_name] = evaluation["overall_score"]
        
        # Good agent should outperform bad agent
        assert results["good"] > results["bad"]
    
    @pytest.mark.slow
    def test_longitudinal_experiment(self, sample_dag, deterministic_agent):
        """Test experiment with many time points."""
        env = CausalEnvironment.from_dag(sample_dag)
        ui = InterventionUI(env)
        tracker = BeliefTracker(deterministic_agent)
        
        # Simulate longitudinal study
        time_points = 10
        for t in range(time_points):
            # Alternate interventions
            intervention = ("sprinkler", t % 2 == 0)
            
            tracker.record(f"P(wet_grass_t{t})", f"do(sprinkler={intervention[1]})", 
                         0.9 if intervention[1] else 0.3)
        
        # Analyze evolution
        summary = tracker.get_belief_summary()
        assert summary["total_measurements"] == time_points
        
        # Should show pattern over time
        beliefs = list(tracker.beliefs.values())[0] if tracker.beliefs else []
        if beliefs:
            values = [b["value"] for b in beliefs]
            # Should alternate between high and low values
            assert max(values) - min(values) > 0.3