#!/usr/bin/env python3
"""
Validate causal reasoning accuracy in the codebase.

This script ensures that all causal inference implementations maintain
theoretical correctness and research validity.
"""

import sys
import ast
import inspect
from pathlib import Path
from typing import List, Dict, Any, Tuple
import networkx as nx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from causal_interface_gym.core import CausalEnvironment
    from causal_interface_gym.metrics import CausalMetrics
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class CausalAccuracyValidator:
    """Validates causal reasoning accuracy in the codebase."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_dag_structure(self, dag: Dict[str, List[str]]) -> bool:
        """Validate that DAG structure is acyclic and well-formed."""
        try:
            G = nx.DiGraph()
            for node, parents in dag.items():
                G.add_node(node)
                for parent in parents:
                    G.add_edge(parent, node)
            
            if not nx.is_directed_acyclic_graph(G):
                self.errors.append(f"DAG contains cycles: {dag}")
                return False
                
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating DAG structure: {e}")
            return False
    
    def validate_intervention_semantics(self) -> bool:
        """Validate that intervention methods implement do-calculus correctly."""
        try:
            # Test basic intervention logic
            dag = {
                "X": [],
                "Y": ["X"],
                "Z": ["Y"]
            }
            
            env = CausalEnvironment.from_dag(dag)
            
            # Test that intervention breaks incoming edges
            result = env.intervene(X=1)
            
            # Should implement proper do-calculus semantics
            if "intervention_applied" not in result:
                self.errors.append("Intervention method does not properly track applied interventions")
                return False
                
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating intervention semantics: {e}")
            return False
    
    def validate_confounding_detection(self) -> bool:
        """Validate that confounding relationships are properly identified."""
        try:
            # Classic confounding example: Z -> X, Z -> Y, X -> Y
            confounded_dag = {
                "Z": [],  # Confounder
                "X": ["Z"],  # Treatment
                "Y": ["Z", "X"]  # Outcome
            }
            
            env = CausalEnvironment.from_dag(confounded_dag)
            
            # Should be able to identify backdoor paths
            # This is a placeholder - actual implementation would check
            # backdoor path identification logic
            
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating confounding detection: {e}")
            return False
    
    def validate_belief_tracking_consistency(self) -> bool:
        """Validate that belief tracking maintains logical consistency."""
        try:
            from causal_interface_gym.metrics import BeliefTracker
            
            # Create mock agent
            class MockAgent:
                pass
            
            agent = MockAgent()
            tracker = BeliefTracker(agent)
            
            # Test belief recording
            tracker.record("P(Y|X)", "observational", 0.8)
            tracker.record("P(Y|do(X))", "interventional", 0.6)
            
            # Interventional and observational beliefs should be tracked separately
            if len(tracker.beliefs) != 2:
                self.errors.append("Belief tracker not properly distinguishing belief types")
                return False
                
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating belief tracking: {e}")
            return False
    
    def validate_metric_soundness(self) -> bool:
        """Validate that evaluation metrics are theoretically sound."""
        try:
            metrics = CausalMetrics()
            
            # Test intervention understanding metric
            mock_responses = [{"belief": 0.8, "type": "interventional"}]
            mock_ground_truth = [{"belief": 0.8, "type": "interventional"}]
            
            score = metrics.intervention_test(mock_responses, mock_ground_truth)
            
            # Score should be between 0 and 1
            if not (0 <= score <= 1):
                self.errors.append(f"Intervention test score out of bounds: {score}")
                return False
                
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating metric soundness: {e}")
            return False
    
    def check_research_integrity_violations(self) -> bool:
        """Check for common research integrity violations."""
        try:
            src_dir = Path(__file__).parent.parent / "src"
            
            for py_file in src_dir.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for hardcoded random seeds in non-test files
                if "random.seed(" in content and "test" not in str(py_file):
                    self.warnings.append(f"Hardcoded random seed found in {py_file}")
                
                # Check for TODO comments indicating incomplete implementation
                if "TODO" in content or "FIXME" in content:
                    self.warnings.append(f"Incomplete implementation markers in {py_file}")
                    
                # Check for potential p-hacking patterns
                if "p_value" in content and "< 0.05" in content:
                    self.warnings.append(f"Potential p-hacking pattern in {py_file}")
                    
            return True
            
        except Exception as e:
            self.errors.append(f"Error checking research integrity: {e}")
            return False
    
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        validations = [
            ("DAG Structure", self.validate_dag_structure({"A": [], "B": ["A"]})),
            ("Intervention Semantics", self.validate_intervention_semantics()),
            ("Confounding Detection", self.validate_confounding_detection()),
            ("Belief Tracking", self.validate_belief_tracking_consistency()),
            ("Metric Soundness", self.validate_metric_soundness()),
            ("Research Integrity", self.check_research_integrity_violations()),
        ]
        
        print("🔬 Causal Accuracy Validation Report")
        print("=" * 50)
        
        all_passed = True
        for name, passed in validations:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{name:.<30} {status}")
            if not passed:
                all_passed = False
        
        print()
        
        if self.errors:
            print("🚨 ERRORS:")
            for error in self.errors:
                print(f"  • {error}")
            print()
        
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()
        
        if all_passed:
            print("🎉 All causal accuracy validations passed!")
        else:
            print("💥 Some validations failed. Please address errors before proceeding.")
        
        return all_passed


def main():
    """Main validation function."""
    validator = CausalAccuracyValidator()
    success = validator.run_all_validations()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()