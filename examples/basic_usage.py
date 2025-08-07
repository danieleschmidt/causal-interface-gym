#!/usr/bin/env python3
"""Basic usage example for Causal Interface Gym."""

from causal_interface_gym import CausalEnvironment, InterventionUI, CausalMetrics


def main():
    """Demonstrate basic causal reasoning experiment."""
    
    # Create a simple causal environment
    # Rain → Sprinkler ← Season
    # Rain → Wet Grass ← Sprinkler
    dag = {
        "season": [],
        "rain": ["season"],
        "sprinkler": ["season", "rain"],
        "wet_grass": ["rain", "sprinkler"]
    }
    
    env = CausalEnvironment.from_dag(dag)
    print("Created causal environment with DAG:")
    print(f"Nodes: {list(env.graph.nodes())}")
    print(f"Edges: {list(env.graph.edges())}")
    
    # Build an intervention interface
    ui = InterventionUI(env)
    ui.add_intervention_button("sprinkler", "Turn Sprinkler On/Off")
    ui.add_observation_panel("wet_grass", "Grass Wetness")
    ui.add_observation_panel("rain", "Rain Status")
    
    print(f"\nCreated UI with {len(ui.components)} components")
    
    # Simulate an experiment
    print("\n--- Running Experiment ---")
    
    # Test intervention understanding
    interventions = [("sprinkler", True)]
    beliefs_to_measure = [
        "P(wet_grass|do(sprinkler=on))",
        "P(rain|wet_grass,do(sprinkler=on))"
    ]
    
    # Mock agent for demonstration
    class MockAgent:
        def __init__(self, name):
            self.name = name
        
        def __str__(self):
            return f"MockAgent({self.name})"
    
    agent = MockAgent("demo-agent")
    
    result = ui.run_experiment(
        agent=agent,
        interventions=interventions,
        measure_beliefs=beliefs_to_measure
    )
    
    print(f"Experiment completed for {result['agent']}")
    print(f"Applied interventions: {result['interventions']}")
    print(f"Measured beliefs: {result['measured_beliefs']}")
    print(f"Initial beliefs: {result['initial_beliefs']}")
    if 'causal_analysis' in result and 'causal_score' in result['causal_analysis']:
        print(f"Causal reasoning score: {result['causal_analysis']['causal_score']:.3f}")
    
    # Apply intervention to environment
    intervention_result = env.intervene(sprinkler=True)
    print(f"\nIntervention result: {intervention_result}")
    
    # Evaluate causal reasoning (mock data)
    metrics = CausalMetrics()
    
    # Mock some agent responses and ground truth
    mock_agent_responses = [
        {"belief": "P(wet_grass)", "value": 0.8, "condition": "do(sprinkler=on)"},
        {"belief": "P(rain)", "value": 0.3, "condition": "observational"}
    ]
    
    mock_ground_truth = [
        {"belief": "P(wet_grass)", "value": 0.9, "condition": "do(sprinkler=on)"},
        {"belief": "P(rain)", "value": 0.3, "condition": "observational"}
    ]
    
    intervention_score = metrics.intervention_test(
        agent_responses=mock_agent_responses,
        ground_truth=mock_ground_truth
    )
    
    print(f"\n--- Evaluation Results ---")
    print(f"Intervention understanding score: {intervention_score:.3f}")
    
    # Demonstrate belief tracking
    from causal_interface_gym.metrics import BeliefTracker
    
    tracker = BeliefTracker(agent)
    
    # Record some belief evolution
    tracker.record("P(rain|wet_grass)", "observational", 0.7)
    tracker.record("P(rain|wet_grass)", "do(sprinkler=on)", 0.3)
    
    print(f"\nBelief tracking data:")
    for belief, records in tracker.beliefs.items():
        print(f"  {belief}:")
        for record in records:
            print(f"    {record['condition']}: {record['value']:.3f}")
    
    print("\n--- Example Complete ---")
    print("This demonstrates the basic workflow:")
    print("1. Define causal environment")
    print("2. Build intervention interface")
    print("3. Run experiments with agents")
    print("4. Evaluate causal reasoning")
    print("5. Track belief evolution")


if __name__ == "__main__":
    main()