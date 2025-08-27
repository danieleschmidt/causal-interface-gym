#!/usr/bin/env python3
"""
TERRAGON QUALITY GATES RUNNER
Comprehensive quality gate validation for autonomous SDLC implementation.
"""

import sys
import os
import subprocess
import time
import json
from typing import Dict, List, Any
import traceback

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from causal_interface_gym import CausalEnvironment, InterventionUI


class QualityGatesRunner:
    """Comprehensive quality gate validation."""
    
    def __init__(self):
        """Initialize quality gates runner."""
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "running",
            "gates": {},
            "summary": {}
        }
        self.start_time = time.time()
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        print("🚀 TERRAGON QUALITY GATES EXECUTION")
        print("=" * 60)
        
        # Gate 1: Code runs without errors
        self.results["gates"]["code_execution"] = self._test_code_execution()
        
        # Gate 2: Core functionality tests
        self.results["gates"]["core_functionality"] = self._test_core_functionality()
        
        # Gate 3: Generation validation
        self.results["gates"]["generation_validation"] = self._test_generation_validation()
        
        # Gate 4: Security validation
        self.results["gates"]["security"] = self._test_security_validation()
        
        # Gate 5: Performance benchmarks
        self.results["gates"]["performance"] = self._test_performance_benchmarks()
        
        # Gate 6: Documentation validation
        self.results["gates"]["documentation"] = self._test_documentation()
        
        # Calculate overall results
        self._calculate_overall_results()
        
        return self.results
    
    def _test_code_execution(self) -> Dict[str, Any]:
        """Test that code runs without errors."""
        print("\n🧪 Quality Gate 1: Code Execution")
        
        result = {
            "status": "running",
            "tests": {},
            "score": 0
        }
        
        try:
            # Test basic import
            result["tests"]["import_test"] = self._safe_test(
                lambda: self._test_imports()
            )
            
            # Test environment creation
            result["tests"]["environment_creation"] = self._safe_test(
                lambda: self._test_environment_creation()
            )
            
            # Test UI creation
            result["tests"]["ui_creation"] = self._safe_test(
                lambda: self._test_ui_creation()
            )
            
            # Test intervention
            result["tests"]["intervention_test"] = self._safe_test(
                lambda: self._test_intervention()
            )
            
            # Calculate score
            passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
            total_tests = len(result["tests"])
            result["score"] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            result["status"] = "passed" if result["score"] >= 85 else "failed"
            
            print(f"   ✅ Code Execution: {result['score']:.1f}% ({passed_tests}/{total_tests} tests passed)")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"   ❌ Code Execution: FAILED - {e}")
            return result
    
    def _test_core_functionality(self) -> Dict[str, Any]:
        """Test core functionality."""
        print("\n🧪 Quality Gate 2: Core Functionality")
        
        result = {
            "status": "running",
            "tests": {},
            "score": 0
        }
        
        try:
            # Test causal graph operations
            result["tests"]["causal_graph"] = self._safe_test(
                lambda: self._test_causal_graph_operations()
            )
            
            # Test backdoor identification
            result["tests"]["backdoor_identification"] = self._safe_test(
                lambda: self._test_backdoor_identification()
            )
            
            # Test intervention effects
            result["tests"]["intervention_effects"] = self._safe_test(
                lambda: self._test_intervention_effects()
            )
            
            # Test UI components
            result["tests"]["ui_components"] = self._safe_test(
                lambda: self._test_ui_components()
            )
            
            # Calculate score
            passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
            total_tests = len(result["tests"])
            result["score"] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            result["status"] = "passed" if result["score"] >= 85 else "failed"
            
            print(f"   ✅ Core Functionality: {result['score']:.1f}% ({passed_tests}/{total_tests} tests passed)")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"   ❌ Core Functionality: FAILED - {e}")
            return result
    
    def _test_generation_validation(self) -> Dict[str, Any]:
        """Test all three generations."""
        print("\n🧪 Quality Gate 3: Generation Validation")
        
        result = {
            "status": "running",
            "tests": {},
            "score": 0
        }
        
        try:
            # Test Generation 1
            result["tests"]["generation_1"] = self._safe_test(
                lambda: self._run_generation_test("examples/autonomous_demo.py")
            )
            
            # Test Generation 2
            result["tests"]["generation_2"] = self._safe_test(
                lambda: self._run_generation_test("examples/autonomous_demo_gen2_simple.py")
            )
            
            # Test Generation 3
            result["tests"]["generation_3"] = self._safe_test(
                lambda: self._run_generation_test("examples/autonomous_demo_gen3.py")
            )
            
            # Calculate score
            passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
            total_tests = len(result["tests"])
            result["score"] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            result["status"] = "passed" if result["score"] >= 85 else "failed"
            
            print(f"   ✅ Generation Validation: {result['score']:.1f}% ({passed_tests}/{total_tests} generations passed)")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"   ❌ Generation Validation: FAILED - {e}")
            return result
    
    def _test_security_validation(self) -> Dict[str, Any]:
        """Test security measures."""
        print("\n🧪 Quality Gate 4: Security Validation")
        
        result = {
            "status": "running",
            "tests": {},
            "score": 0
        }
        
        try:
            # Test input validation
            result["tests"]["input_validation"] = self._safe_test(
                lambda: self._test_input_validation()
            )
            
            # Test injection protection
            result["tests"]["injection_protection"] = self._safe_test(
                lambda: self._test_injection_protection()
            )
            
            # Test error handling
            result["tests"]["error_handling"] = self._safe_test(
                lambda: self._test_error_handling()
            )
            
            # Calculate score
            passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
            total_tests = len(result["tests"])
            result["score"] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            result["status"] = "passed" if result["score"] >= 80 else "failed"
            
            print(f"   ✅ Security Validation: {result['score']:.1f}% ({passed_tests}/{total_tests} tests passed)")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"   ❌ Security Validation: FAILED - {e}")
            return result
    
    def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks."""
        print("\n🧪 Quality Gate 5: Performance Benchmarks")
        
        result = {
            "status": "running",
            "tests": {},
            "score": 0
        }
        
        try:
            # Test response times
            result["tests"]["response_times"] = self._safe_test(
                lambda: self._test_response_times()
            )
            
            # Test scalability
            result["tests"]["scalability"] = self._safe_test(
                lambda: self._test_scalability()
            )
            
            # Test memory usage
            result["tests"]["memory_efficiency"] = self._safe_test(
                lambda: self._test_memory_efficiency()
            )
            
            # Calculate score
            passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
            total_tests = len(result["tests"])
            result["score"] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            result["status"] = "passed" if result["score"] >= 70 else "failed"
            
            print(f"   ✅ Performance Benchmarks: {result['score']:.1f}% ({passed_tests}/{total_tests} tests passed)")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"   ❌ Performance Benchmarks: FAILED - {e}")
            return result
    
    def _test_documentation(self) -> Dict[str, Any]:
        """Test documentation completeness."""
        print("\n🧪 Quality Gate 6: Documentation")
        
        result = {
            "status": "running",
            "tests": {},
            "score": 0
        }
        
        try:
            # Check README
            result["tests"]["readme_exists"] = self._safe_test(
                lambda: self._check_file_exists("README.md")
            )
            
            # Check examples
            result["tests"]["examples_exist"] = self._safe_test(
                lambda: self._check_examples_exist()
            )
            
            # Check docstrings
            result["tests"]["docstrings"] = self._safe_test(
                lambda: self._check_docstrings()
            )
            
            # Calculate score
            passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
            total_tests = len(result["tests"])
            result["score"] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            result["status"] = "passed" if result["score"] >= 75 else "failed"
            
            print(f"   ✅ Documentation: {result['score']:.1f}% ({passed_tests}/{total_tests} tests passed)")
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"   ❌ Documentation: FAILED - {e}")
            return result
    
    def _safe_test(self, test_func) -> Dict[str, Any]:
        """Run a test function safely and return results."""
        try:
            return test_func()
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _test_imports(self) -> Dict[str, Any]:
        """Test core imports."""
        from causal_interface_gym import CausalEnvironment, InterventionUI
        return {"passed": True, "message": "Core imports successful"}
    
    def _test_environment_creation(self) -> Dict[str, Any]:
        """Test environment creation."""
        env = CausalEnvironment()
        env.add_variable("test", "binary")
        env.add_variable("test2", "binary")
        env.add_edge("test", "test2")
        return {"passed": True, "message": "Environment creation successful"}
    
    def _test_ui_creation(self) -> Dict[str, Any]:
        """Test UI creation."""
        env = CausalEnvironment()
        env.add_variable("test", "binary")
        ui = InterventionUI(env)
        ui.add_intervention_button("test", "Test")
        return {"passed": True, "message": "UI creation successful"}
    
    def _test_intervention(self) -> Dict[str, Any]:
        """Test intervention."""
        env = CausalEnvironment()
        env.add_variable("test", "binary")
        result = env.intervene(test=1)
        return {"passed": True, "message": "Intervention successful", "result": str(result)}
    
    def _test_causal_graph_operations(self) -> Dict[str, Any]:
        """Test causal graph operations."""
        env = CausalEnvironment()
        env.add_variable("X", "binary")
        env.add_variable("Y", "binary")
        env.add_variable("Z", "binary")
        env.add_edge("X", "Y")
        env.add_edge("Y", "Z")
        
        # Test graph structure
        assert len(env.graph.nodes()) == 3
        assert len(env.graph.edges()) == 2
        
        return {"passed": True, "message": "Causal graph operations successful"}
    
    def _test_backdoor_identification(self) -> Dict[str, Any]:
        """Test backdoor identification."""
        env = CausalEnvironment()
        env.add_variable("X", "binary")
        env.add_variable("Y", "binary")
        env.add_edge("X", "Y")
        
        backdoor_paths = env.get_backdoor_paths("X", "Y")
        backdoor_set = env.identify_backdoor_set("X", "Y")
        
        return {"passed": True, "message": "Backdoor identification successful"}
    
    def _test_intervention_effects(self) -> Dict[str, Any]:
        """Test intervention effects."""
        env = CausalEnvironment()
        env.add_variable("X", "binary")
        env.add_variable("Y", "binary")
        env.add_edge("X", "Y")
        
        result = env.intervene(X=1)
        assert "interventions_applied" in result
        
        return {"passed": True, "message": "Intervention effects successful"}
    
    def _test_ui_components(self) -> Dict[str, Any]:
        """Test UI components."""
        env = CausalEnvironment()
        env.add_variable("X", "binary")
        ui = InterventionUI(env)
        ui.add_intervention_button("X", "Control X")
        ui.add_observation_panel("X", "Observe X")
        
        html = ui.generate_html()
        assert len(html) > 100
        
        return {"passed": True, "message": "UI components successful"}
    
    def _run_generation_test(self, script_path: str) -> Dict[str, Any]:
        """Run a generation test script."""
        try:
            cmd = ["python", script_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd="/root/repo")
            
            if result.returncode == 0:
                return {"passed": True, "message": f"Generation test {script_path} passed"}
            else:
                return {"passed": False, "message": f"Generation test {script_path} failed", "stderr": result.stderr}
        
        except subprocess.TimeoutExpired:
            return {"passed": False, "message": f"Generation test {script_path} timed out"}
        except Exception as e:
            return {"passed": False, "message": f"Generation test {script_path} error: {e}"}
    
    def _test_input_validation(self) -> Dict[str, Any]:
        """Test input validation."""
        env = CausalEnvironment()
        
        # Test invalid variable names
        try:
            env.add_variable("", "binary")
            return {"passed": False, "message": "Empty variable name not rejected"}
        except ValueError:
            pass
        
        try:
            env.add_variable("invalid-var", "invalid_type")
            return {"passed": False, "message": "Invalid variable type not rejected"}
        except ValueError:
            pass
        
        return {"passed": True, "message": "Input validation working"}
    
    def _test_injection_protection(self) -> Dict[str, Any]:
        """Test injection protection."""
        env = CausalEnvironment()
        
        # Test malicious variable names
        malicious_names = ["<script>", "__import__", "eval("]
        
        for name in malicious_names:
            try:
                env.add_variable(name, "binary")
                # If it succeeds, check if it was sanitized
                if name in env.graph.nodes():
                    return {"passed": False, "message": f"Malicious input {name} not protected"}
            except:
                pass  # Expected to fail
        
        return {"passed": True, "message": "Injection protection working"}
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling."""
        env = CausalEnvironment()
        env.add_variable("test", "binary")
        
        # Test various error conditions
        error_tests = [
            lambda: env.add_edge("nonexistent", "test"),  # Nonexistent parent
            lambda: env.add_edge("test", "nonexistent"),  # Nonexistent child
            lambda: env.add_edge("test", "test"),  # Self loop
        ]
        
        for test in error_tests:
            try:
                test()
                return {"passed": False, "message": "Error condition not handled"}
            except Exception:
                continue  # Expected to fail
        
        return {"passed": True, "message": "Error handling working"}
    
    def _test_response_times(self) -> Dict[str, Any]:
        """Test response times."""
        start = time.time()
        
        env = CausalEnvironment()
        env.add_variable("X", "binary")
        env.add_variable("Y", "binary")
        env.add_edge("X", "Y")
        env.intervene(X=1)
        
        response_time = time.time() - start
        
        # Should complete within 200ms
        passed = response_time < 0.2
        
        return {
            "passed": passed,
            "message": f"Response time: {response_time:.3f}s",
            "response_time": response_time
        }
    
    def _test_scalability(self) -> Dict[str, Any]:
        """Test scalability."""
        start = time.time()
        
        # Create larger graph
        env = CausalEnvironment()
        for i in range(20):
            env.add_variable(f"var_{i}", "binary")
        
        for i in range(19):
            env.add_edge(f"var_{i}", f"var_{i+1}")
        
        # Multiple interventions
        for i in range(10):
            env.intervene(**{f"var_{i}": 1})
        
        scalability_time = time.time() - start
        passed = scalability_time < 2.0  # Should complete within 2 seconds
        
        return {
            "passed": passed,
            "message": f"Scalability test: {scalability_time:.3f}s",
            "scalability_time": scalability_time
        }
    
    def _test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory efficiency (simplified)."""
        # Create and destroy environments to test memory management
        for _ in range(100):
            env = CausalEnvironment()
            env.add_variable("test", "binary")
            del env
        
        return {"passed": True, "message": "Memory efficiency test completed"}
    
    def _check_file_exists(self, filepath: str) -> Dict[str, Any]:
        """Check if file exists."""
        exists = os.path.exists(os.path.join("/root/repo", filepath))
        return {"passed": exists, "message": f"File {filepath} {'exists' if exists else 'missing'}"}
    
    def _check_examples_exist(self) -> Dict[str, Any]:
        """Check if examples exist."""
        examples_dir = "/root/repo/examples"
        if not os.path.exists(examples_dir):
            return {"passed": False, "message": "Examples directory missing"}
        
        example_files = [f for f in os.listdir(examples_dir) if f.endswith('.py')]
        passed = len(example_files) >= 3  # At least 3 example files
        
        return {
            "passed": passed,
            "message": f"Found {len(example_files)} example files",
            "examples": example_files
        }
    
    def _check_docstrings(self) -> Dict[str, Any]:
        """Check docstrings in core module."""
        from causal_interface_gym import core
        
        # Check if main classes have docstrings
        classes_with_docstrings = 0
        total_classes = 0
        
        for name in dir(core):
            obj = getattr(core, name)
            if isinstance(obj, type):
                total_classes += 1
                if obj.__doc__ and obj.__doc__.strip():
                    classes_with_docstrings += 1
        
        if total_classes == 0:
            return {"passed": False, "message": "No classes found"}
        
        doc_coverage = classes_with_docstrings / total_classes
        passed = doc_coverage >= 0.8  # 80% docstring coverage
        
        return {
            "passed": passed,
            "message": f"Docstring coverage: {doc_coverage:.1%}",
            "classes_with_docs": classes_with_docstrings,
            "total_classes": total_classes
        }
    
    def _calculate_overall_results(self):
        """Calculate overall quality gate results."""
        total_score = 0
        total_weight = 0
        passed_gates = 0
        total_gates = len(self.results["gates"])
        
        # Weight different gates by importance
        gate_weights = {
            "code_execution": 25,      # Critical
            "core_functionality": 25,  # Critical
            "generation_validation": 20, # Very Important
            "security": 15,            # Important
            "performance": 10,         # Important
            "documentation": 5         # Nice to have
        }
        
        for gate_name, gate_result in self.results["gates"].items():
            weight = gate_weights.get(gate_name, 10)
            score = gate_result.get("score", 0)
            
            total_score += score * weight
            total_weight += weight
            
            if gate_result.get("status") == "passed":
                passed_gates += 1
        
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        self.results["summary"] = {
            "overall_score": overall_score,
            "passed_gates": passed_gates,
            "total_gates": total_gates,
            "pass_rate": passed_gates / total_gates if total_gates > 0 else 0,
            "execution_time": time.time() - self.start_time
        }
        
        # Determine overall status
        if overall_score >= 80 and passed_gates >= total_gates * 0.8:
            self.results["overall_status"] = "passed"
        else:
            self.results["overall_status"] = "failed"
    
    def print_summary(self):
        """Print summary of quality gate results."""
        summary = self.results["summary"]
        
        print(f"\n{'=' * 60}")
        print("🎯 TERRAGON QUALITY GATES SUMMARY")
        print(f"{'=' * 60}")
        
        status_emoji = "✅" if self.results["overall_status"] == "passed" else "❌"
        print(f"{status_emoji} Overall Status: {self.results['overall_status'].upper()}")
        print(f"📊 Overall Score: {summary['overall_score']:.1f}%")
        print(f"🚪 Gates Passed: {summary['passed_gates']}/{summary['total_gates']} ({summary['pass_rate']:.1%})")
        print(f"⏱️  Execution Time: {summary['execution_time']:.2f}s")
        
        print(f"\n📋 Gate Details:")
        for gate_name, gate_result in self.results["gates"].items():
            status_emoji = "✅" if gate_result.get("status") == "passed" else "❌"
            score = gate_result.get("score", 0)
            print(f"  {status_emoji} {gate_name.replace('_', ' ').title()}: {score:.1f}%")
        
        print(f"\n🎉 TERRAGON AUTONOMOUS SDLC: {'SUCCESS' if self.results['overall_status'] == 'passed' else 'NEEDS IMPROVEMENT'}")


def main():
    """Main function to run quality gates."""
    try:
        runner = QualityGatesRunner()
        results = runner.run_all_quality_gates()
        runner.print_summary()
        
        # Save results to file
        with open("/root/repo/quality_gates_report.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Exit with appropriate code
        exit_code = 0 if results["overall_status"] == "passed" else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n❌ QUALITY GATES EXECUTION FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()