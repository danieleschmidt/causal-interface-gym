#!/usr/bin/env python3
"""Complete demonstration of Causal Interface Gym functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from causal_interface_gym import CausalEnvironment, InterventionUI, CausalMetrics
from causal_interface_gym.llm.providers import LocalProvider, get_provider


def create_sprinkler_scenario():
    """Create the classic sprinkler causal scenario."""
    # Define causal structure: rain → sprinkler → wet_grass ← rain
    env = CausalEnvironment.from_dag({
        "rain": [],
        "sprinkler": ["rain"],
        "wet_grass": ["rain", "sprinkler"],
        "slippery": ["wet_grass"]
    })
    
    # Add variable types
    env.add_variable("rain", "binary")
    env.add_variable("sprinkler", "binary")
    env.add_variable("wet_grass", "binary")
    env.add_variable("slippery", "binary")
    
    return env


def demo_causal_analysis():
    """Demonstrate causal analysis capabilities."""
    print("🧠 CAUSAL ANALYSIS DEMO")
    print("=" * 50)
    
    # Create environment
    env = create_sprinkler_scenario()
    print(f"Created environment with {len(env.graph.nodes())} variables")
    print(f"Variables: {list(env.graph.nodes())}")
    print(f"Edges: {list(env.graph.edges())}")
    
    # Test backdoor path identification
    print("\n🔍 BACKDOOR ANALYSIS")
    backdoor_paths = env.get_backdoor_paths("sprinkler", "wet_grass")
    print(f"Backdoor paths from sprinkler to wet_grass: {backdoor_paths}")
    
    backdoor_set = env.identify_backdoor_set("sprinkler", "wet_grass")
    print(f"Backdoor adjustment set: {backdoor_set}")
    
    # Test do-calculus
    print("\n⚡ DO-CALCULUS ANALYSIS")
    causal_effect = env.do_calculus("sprinkler", "wet_grass", True)
    print(f"Causal effect of setting sprinkler=True on wet_grass:")
    for key, value in causal_effect.items():
        print(f"  {key}: {value}")


def demo_llm_integration():
    """Demonstrate LLM integration for belief querying."""
    print("\n🤖 LLM INTEGRATION DEMO")
    print("=" * 50)
    
    # Create a local provider (simulated)
    provider = LocalProvider(model_name="test-model")
    
    # Test belief querying
    belief = "P(wet_grass|sprinkler=on)"
    observational_condition = "observational"
    interventional_condition = "do(sprinkler=on)"
    
    obs_belief = provider.query_belief(belief, observational_condition)
    int_belief = provider.query_belief(belief, interventional_condition)
    
    print(f"Belief: {belief}")
    print(f"Observational P(wet_grass|sprinkler=on): {obs_belief:.3f}")
    print(f"Interventional P(wet_grass|do(sprinkler=on)): {int_belief:.3f}")
    print(f"Difference: {abs(int_belief - obs_belief):.3f}")


def demo_intervention_experiment():
    """Demonstrate running a causal reasoning experiment."""
    print("\n🧪 INTERVENTION EXPERIMENT DEMO")
    print("=" * 50)
    
    # Create environment and UI
    env = create_sprinkler_scenario()
    ui = InterventionUI(env)
    
    # Add UI components
    ui.add_intervention_button("sprinkler", "Turn Sprinkler On/Off")
    ui.add_observation_panel("wet_grass", "Grass Wetness")
    ui.add_belief_display(["P(slippery)", "P(rain|wet_grass)"])
    ui.add_graph_visualization()
    
    # Create mock agent
    agent = LocalProvider(model_name="test-agent")
    
    # Run experiment
    interventions = [("sprinkler", True)]
    measure_beliefs = ["P(slippery)", "P(rain|wet_grass)"]
    
    print(f"Running experiment with interventions: {interventions}")
    print(f"Measuring beliefs: {measure_beliefs}")
    
    try:
        results = ui.run_experiment(
            agent=agent,
            interventions=interventions,
            measure_beliefs=measure_beliefs
        )
        
        print(f"\n📊 EXPERIMENT RESULTS")
        print(f"Experiment ID: {results['experiment_id']}")
        print(f"Agent: {results['agent']}")
        print(f"Causal Score: {results['causal_analysis']['causal_score']:.3f}")
        
        # Show intervention vs observation analysis
        print(f"\n🔄 INTERVENTION VS OBSERVATION ANALYSIS")
        for var, analysis in results['causal_analysis']['intervention_vs_observation'].items():
            print(f"  {var}:")
            print(f"    Score: {analysis['score']:.3f}")
            print(f"    Intervention belief: {analysis['intervention_belief']:.3f}")
            print(f"    Observational belief: {analysis['observational_belief']:.3f}")
            print(f"    Difference: {analysis['difference']:.3f}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")


def demo_metrics_evaluation():
    """Demonstrate causal reasoning metrics."""
    print("\n📈 METRICS EVALUATION DEMO")
    print("=" * 50)
    
    metrics = CausalMetrics()
    
    # Simulate agent responses for intervention test
    agent_responses = [
        {
            "intervention_belief": 0.8,
            "observational_belief": 0.6
        },
        {
            "intervention_belief": 0.3,
            "observational_belief": 0.7
        }
    ]
    
    ground_truth = [
        {
            "intervention_belief": 0.9,
            "observational_belief": 0.5
        },
        {
            "intervention_belief": 0.2,
            "observational_belief": 0.8
        }
    ]
    
    intervention_score = metrics.intervention_test(agent_responses, ground_truth)
    print(f"Intervention understanding score: {intervention_score:.3f}")
    
    # Simulate counterfactual test
    agent_predictions = [0.7, 0.3, 0.8, 0.2]
    true_counterfactuals = [0.8, 0.2, 0.9, 0.1]
    
    counterfactual_score = metrics.counterfactual_test(agent_predictions, true_counterfactuals)
    print(f"Counterfactual reasoning score: {counterfactual_score:.3f}")


def demo_ui_generation():
    """Demonstrate UI generation capabilities."""
    print("\n🖥️  UI GENERATION DEMO")
    print("=" * 50)
    
    env = create_sprinkler_scenario()
    ui = InterventionUI(env)
    
    # Add components
    ui.add_intervention_button("sprinkler", "Toggle Sprinkler")
    ui.add_observation_panel("wet_grass", "Grass Status")
    ui.add_graph_visualization(layout="hierarchical", show_backdoors=True)
    
    # Generate HTML
    html = ui.generate_html()
    
    print(f"Generated HTML interface ({len(html)} characters)")
    print("HTML preview (first 200 chars):")
    print(html[:200] + "...")
    
    # Save to file
    output_file = "/root/repo/demo_interface.html"
    with open(output_file, "w") as f:
        f.write(html)
    print(f"Saved interface to: {output_file}")


def main():
    """Run complete demonstration."""
    print("🚀 CAUSAL INTERFACE GYM - COMPLETE DEMO")
    print("=" * 60)
    print()
    
    try:
        demo_causal_analysis()
        demo_llm_integration()
        demo_intervention_experiment()
        demo_metrics_evaluation()
        demo_ui_generation()
        
        print("\n✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Open demo_interface.html in a browser")
        print("2. Set up LLM API keys for real experiments")
        print("3. Explore the generated UI components")
        print("4. Run custom causal reasoning experiments")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())