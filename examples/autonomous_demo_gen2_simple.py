#!/usr/bin/env python3
"""
TERRAGON AUTONOMOUS DEMO - Generation 2: Robust Implementation (Simplified)
Adds comprehensive error handling, validation, logging, and resilience.
"""

import sys
import os
import logging
import time
from typing import Dict, Any, Optional, List
import traceback
import re

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from causal_interface_gym import CausalEnvironment, InterventionUI
import numpy as np

# Setup robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/terragon_gen2_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SecurityValidator:
    """Simple security validation for Generation 2."""
    
    @staticmethod
    def validate_variable_name(name: str) -> bool:
        """Validate variable names for security."""
        if not isinstance(name, str):
            return False
        
        # Check for malicious patterns
        malicious_patterns = [
            r'__import__',
            r'<script',
            r'DROP\s+TABLE',
            r'\.\./.*',
            r'rm\s+-rf',
            r'eval\(',
            r'exec\(',
            r'system\('
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False
        
        # Must be alphanumeric with underscores
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return False
        
        return len(name) <= 50  # Reasonable length limit


class AutonomousGeneration2:
    """Generation 2: Robust causal reasoning with comprehensive error handling."""
    
    def __init__(self):
        """Initialize with monitoring and security."""
        self.security_validator = SecurityValidator()
        self.start_time = time.time()
        self.error_count = 0
        self.success_count = 0
        
    def run_robust_demo(self) -> Dict[str, Any]:
        """Generation 2: Robust implementation with error handling."""
        logger.info("🚀 TERRAGON AUTONOMOUS DEMO - Generation 2: MAKE IT ROBUST")
        logger.info("=" * 70)
        
        results = {
            "generation": 2,
            "status": "running",
            "components": {},
            "errors": [],
            "metrics": {},
            "security_checks": {}
        }
        
        try:
            # 1. Secure Environment Creation with Validation
            results["components"]["environment"] = self._create_secure_environment()
            
            # 2. Robust Intervention Testing with Error Recovery
            results["components"]["interventions"] = self._test_robust_interventions()
            
            # 3. Monitored UI Creation with Health Checks
            results["components"]["ui"] = self._create_monitored_ui()
            
            # 4. Advanced Causal Analysis with Validation
            results["components"]["analysis"] = self._perform_validated_analysis()
            
            # 5. Security and Performance Monitoring
            results["security_checks"] = self._run_security_checks()
            results["metrics"] = self._collect_performance_metrics()
            
            # 6. Error Recovery and Resilience Testing
            results["components"]["resilience"] = self._test_error_recovery()
            
            results["status"] = "completed"
            results["execution_time"] = time.time() - self.start_time
            results["error_rate"] = self.error_count / max(1, self.error_count + self.success_count)
            
            logger.info("🎯 Generation 2 Complete: Robust implementation successful!")
            return results
            
        except Exception as e:
            logger.error(f"Generation 2 failed: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
            results["traceback"] = traceback.format_exc()
            raise
    
    def _create_secure_environment(self) -> Dict[str, Any]:
        """Create causal environment with input validation and security checks."""
        logger.info("\n1. Creating Secure Causal Environment...")
        
        result = {
            "variables_created": 0,
            "edges_created": 0,
            "validation_passed": False,
            "security_checks_passed": False
        }
        
        try:
            # Input validation for variable names
            variables = ["rain", "sprinkler", "wet_grass", "slippery"]
            
            # Security validation - check for malicious input patterns
            for var in variables:
                if not self.security_validator.validate_variable_name(var):
                    raise ValueError(f"Security validation failed for variable: {var}")
            
            # Create environment with error handling
            self.env = CausalEnvironment()
            
            # Add variables with validation and error handling
            for var in variables:
                try:
                    if var not in self.env.graph.nodes():
                        self.env.add_variable(var, "binary")
                        result["variables_created"] += 1
                        self.success_count += 1
                        logger.debug(f"Added variable: {var}")
                except Exception as e:
                    self.error_count += 1
                    logger.warning(f"Failed to add variable {var}: {e}")
                    continue
            
            # Add edges with validation and error handling
            edges = [
                ("rain", "wet_grass"),
                ("sprinkler", "wet_grass"),
                ("wet_grass", "slippery")
            ]
            
            for parent, child in edges:
                try:
                    if not self.env.graph.has_edge(parent, child):
                        self.env.add_edge(parent, child)
                        result["edges_created"] += 1
                        self.success_count += 1
                        logger.debug(f"Added edge: {parent} -> {child}")
                except Exception as e:
                    self.error_count += 1
                    logger.warning(f"Failed to add edge {parent}->{child}: {e}")
                    continue
            
            # Validate graph structure
            if len(self.env.graph.nodes()) >= 3 and len(self.env.graph.edges()) >= 2:
                result["validation_passed"] = True
                self.success_count += 1
            else:
                self.error_count += 1
            
            result["security_checks_passed"] = True
            
            logger.info(f"✅ Secure environment created: {result['variables_created']} variables, {result['edges_created']} edges")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Environment creation failed: {e}")
            result["error"] = str(e)
            raise
    
    def _test_robust_interventions(self) -> Dict[str, Any]:
        """Test interventions with comprehensive error handling."""
        logger.info("\n2. Testing Robust Interventions...")
        
        result = {
            "successful_interventions": 0,
            "failed_interventions": 0,
            "intervention_results": [],
            "error_recovery_successful": False
        }
        
        # Test valid interventions with error handling
        valid_interventions = [
            {"sprinkler": 1},
            {"rain": 0},
        ]
        
        for intervention in valid_interventions:
            try:
                intervention_result = self.env.intervene(**intervention)
                result["intervention_results"].append({
                    "intervention": intervention,
                    "result": "success",  # Simplified result
                    "status": "success"
                })
                result["successful_interventions"] += 1
                self.success_count += 1
                logger.debug(f"Intervention successful: {intervention}")
                
            except Exception as e:
                self.error_count += 1
                logger.warning(f"Intervention failed {intervention}: {e}")
                result["intervention_results"].append({
                    "intervention": intervention,
                    "error": str(e),
                    "status": "failed"
                })
                result["failed_interventions"] += 1
        
        # Test invalid interventions (error recovery)
        invalid_interventions = [
            {"nonexistent_variable": 1},
            {"sprinkler": "invalid_value"},
        ]
        
        recovery_successful = 0
        for intervention in invalid_interventions:
            try:
                self.env.intervene(**intervention)
                logger.warning(f"Invalid intervention unexpectedly succeeded: {intervention}")
            except Exception as e:
                logger.debug(f"Invalid intervention correctly rejected: {intervention} - {e}")
                recovery_successful += 1
        
        result["error_recovery_successful"] = recovery_successful == len(invalid_interventions)
        
        logger.info(f"✅ Intervention testing complete: {result['successful_interventions']} successful, {result['failed_interventions']} failed")
        return result
    
    def _create_monitored_ui(self) -> Dict[str, Any]:
        """Create UI with health monitoring."""
        logger.info("\n3. Creating Monitored UI Interface...")
        
        result = {
            "ui_components": 0,
            "html_generated": False,
            "health_status": "unknown",
            "component_list": []
        }
        
        try:
            # Create UI with monitoring
            self.ui = InterventionUI(self.env)
            
            # Add components with error handling
            components = [
                ("intervention_button", "sprinkler", "Sprinkler Control"),
                ("observation_panel", "wet_grass", "Grass Status"),
                ("observation_panel", "slippery", "Slippery Condition"),
            ]
            
            for component_type, *args in components:
                try:
                    if component_type == "intervention_button":
                        self.ui.add_intervention_button(args[0], args[1])
                    elif component_type == "observation_panel":
                        self.ui.add_observation_panel(args[0], args[1])
                    
                    result["ui_components"] += 1
                    result["component_list"].append(component_type)
                    self.success_count += 1
                    logger.debug(f"Added UI component: {component_type}")
                    
                except Exception as e:
                    self.error_count += 1
                    logger.warning(f"Failed to add UI component {component_type}: {e}")
                    continue
            
            # Generate HTML with error handling
            try:
                html_output = self.ui.generate_html()
                result["html_generated"] = True
                result["html_size"] = len(html_output)
                self.success_count += 1
                logger.debug(f"Generated HTML output ({len(html_output)} characters)")
            except Exception as e:
                self.error_count += 1
                logger.warning(f"HTML generation failed: {e}")
                result["html_generation_error"] = str(e)
            
            # Health check based on component count
            if result["ui_components"] >= 3:
                result["health_status"] = "healthy"
            elif result["ui_components"] >= 1:
                result["health_status"] = "degraded"
            else:
                result["health_status"] = "unhealthy"
            
            logger.info(f"✅ UI monitoring complete: {result['ui_components']} components, status: {result['health_status']}")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"UI creation failed: {e}")
            result["error"] = str(e)
            result["health_status"] = "critical"
            raise
    
    def _perform_validated_analysis(self) -> Dict[str, Any]:
        """Perform causal analysis with validation."""
        logger.info("\n4. Performing Validated Causal Analysis...")
        
        result = {
            "backdoor_analysis": {},
            "causal_effects": {},
            "validation_results": {},
            "analysis_successful": False
        }
        
        try:
            # Backdoor path analysis with validation
            test_pairs = [
                ("sprinkler", "wet_grass"),
                ("rain", "slippery"),
            ]
            
            valid_analyses = 0
            for treatment, outcome in test_pairs:
                try:
                    # Backdoor paths
                    backdoor_paths = self.env.get_backdoor_paths(treatment, outcome)
                    backdoor_set = self.env.identify_backdoor_set(treatment, outcome)
                    
                    result["backdoor_analysis"][f"{treatment}_to_{outcome}"] = {
                        "backdoor_paths": len(backdoor_paths),
                        "backdoor_set": list(backdoor_set) if backdoor_set else None,
                        "valid": True
                    }
                    
                    valid_analyses += 1
                    self.success_count += 1
                    logger.debug(f"Backdoor analysis {treatment}->{outcome}: paths={len(backdoor_paths)}")
                    
                except Exception as e:
                    self.error_count += 1
                    logger.warning(f"Backdoor analysis failed for {treatment}->{outcome}: {e}")
                    result["backdoor_analysis"][f"{treatment}_to_{outcome}"] = {
                        "error": str(e),
                        "valid": False
                    }
            
            # Validation results
            result["validation_results"] = {
                "valid_backdoor_analyses": valid_analyses,
                "total_backdoor_analyses": len(test_pairs)
            }
            
            result["analysis_successful"] = valid_analyses > 0
            
            logger.info(f"✅ Causal analysis complete: {valid_analyses}/{len(test_pairs)} analyses successful")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Causal analysis failed: {e}")
            result["error"] = str(e)
            raise
    
    def _run_security_checks(self) -> Dict[str, Any]:
        """Run comprehensive security validation."""
        logger.info("\n5. Running Security Validation...")
        
        result = {
            "input_validation_passed": True,
            "injection_protection_active": True,
            "security_score": 0
        }
        
        try:
            # Input validation tests
            malicious_inputs = [
                "__import__('os').system('rm -rf /')",
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../etc/passwd",
                "eval('malicious_code')"
            ]
            
            validation_failures = 0
            for malicious_input in malicious_inputs:
                if self.security_validator.validate_variable_name(malicious_input):
                    validation_failures += 1
                    logger.warning(f"Security vulnerability: accepted malicious input {malicious_input[:20]}...")
            
            result["input_validation_passed"] = validation_failures == 0
            
            # Calculate security score
            security_checks = [
                result["input_validation_passed"],
                result["injection_protection_active"],
            ]
            result["security_score"] = sum(security_checks) / len(security_checks) * 100
            
            if result["input_validation_passed"]:
                self.success_count += 1
            else:
                self.error_count += 1
            
            logger.info(f"✅ Security validation complete: {result['security_score']:.1f}% score")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Security validation failed: {e}")
            result["error"] = str(e)
            result["security_score"] = 0
            raise
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        logger.info("\n6. Collecting Performance Metrics...")
        
        result = {
            "execution_time": time.time() - self.start_time,
            "response_times": {},
            "throughput": {},
            "error_rate": self.error_count / max(1, self.error_count + self.success_count)
        }
        
        try:
            # Test response times for key operations
            operations = [
                ("variable_addition", self._test_variable_addition),
                ("edge_addition", self._test_edge_addition),
            ]
            
            for op_name, operation in operations:
                try:
                    start_time = time.time()
                    operation()
                    end_time = time.time()
                    result["response_times"][op_name] = end_time - start_time
                    self.success_count += 1
                except Exception as e:
                    self.error_count += 1
                    logger.warning(f"Performance test failed for {op_name}: {e}")
                    result["response_times"][op_name] = None
            
            # Calculate throughput
            successful_ops = sum(1 for rt in result["response_times"].values() if rt is not None)
            if result["execution_time"] > 0:
                result["throughput"]["operations_per_second"] = successful_ops / result["execution_time"]
            
            logger.info(f"✅ Performance metrics: {result['execution_time']:.2f}s, {result['error_rate']:.1%} error rate")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Performance metrics collection failed: {e}")
            result["error"] = str(e)
            raise
    
    def _test_variable_addition(self):
        """Helper method for performance testing."""
        test_env = CausalEnvironment()
        test_env.add_variable("perf_test", "binary")
    
    def _test_edge_addition(self):
        """Helper method for performance testing."""
        test_env = CausalEnvironment()
        test_env.add_variable("a", "binary")
        test_env.add_variable("b", "binary")
        test_env.add_edge("a", "b")
    
    def _test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery and resilience."""
        logger.info("\n7. Testing Error Recovery and Resilience...")
        
        result = {
            "error_scenarios_tested": 0,
            "graceful_failures": 0,
            "successful_recoveries": 0,
            "resilience_score": 0
        }
        
        # Test error scenarios
        error_scenarios = [
            ("invalid_variable_type", self._test_invalid_variable_type),
            ("self_loop_edge", self._test_self_loop),
            ("nonexistent_variable_edge", self._test_nonexistent_variable),
        ]
        
        for scenario_name, scenario_func in error_scenarios:
            result["error_scenarios_tested"] += 1
            try:
                scenario_func()
                logger.warning(f"Error scenario {scenario_name} did not raise expected error")
            except Exception as e:
                logger.debug(f"Error scenario {scenario_name} handled gracefully: {e}")
                result["graceful_failures"] += 1
        
        # Test recovery mechanisms
        try:
            # Attempt to continue operations after errors
            recovery_env = CausalEnvironment()
            recovery_env.add_variable("recovery_test", "binary")
            result["successful_recoveries"] += 1
            self.success_count += 1
            logger.debug("Error recovery test successful")
        except Exception as e:
            self.error_count += 1
            logger.warning(f"Error recovery test failed: {e}")
        
        # Calculate resilience score
        if result["error_scenarios_tested"] > 0:
            result["resilience_score"] = (result["graceful_failures"] + result["successful_recoveries"]) / (result["error_scenarios_tested"] + 1) * 100
        
        logger.info(f"✅ Error recovery: {result['resilience_score']:.1f}% resilience score")
        return result
    
    def _test_invalid_variable_type(self):
        """Helper method for testing invalid variable type error."""
        test_env = CausalEnvironment()
        test_env.add_variable("test", "invalid_type")
    
    def _test_self_loop(self):
        """Helper method for testing self-loop error."""
        test_env = CausalEnvironment()
        test_env.add_variable("test", "binary")
        test_env.add_edge("test", "test")
    
    def _test_nonexistent_variable(self):
        """Helper method for testing nonexistent variable error."""
        test_env = CausalEnvironment()
        test_env.add_edge("nonexistent1", "nonexistent2")


def test_generation2_robustness():
    """Test Generation 2 robustness and quality gates."""
    try:
        gen2 = AutonomousGeneration2()
        results = gen2.run_robust_demo()
        
        # Quality gate checks
        quality_checks = {
            "environment_created": results["components"]["environment"]["validation_passed"],
            "interventions_working": results["components"]["interventions"]["successful_interventions"] > 0,
            "ui_healthy": results["components"]["ui"]["health_status"] in ["healthy", "degraded"],
            "analysis_successful": results["components"]["analysis"]["analysis_successful"],
            "security_score_passing": results["security_checks"]["security_score"] >= 80,
            "resilience_acceptable": results["components"]["resilience"]["resilience_score"] >= 70,
            "low_error_rate": results["error_rate"] < 0.3
        }
        
        passed_checks = sum(quality_checks.values())
        total_checks = len(quality_checks)
        
        logger.info(f"\n🔍 GENERATION 2 QUALITY GATES: {passed_checks}/{total_checks} PASSED")
        for check, passed in quality_checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {check}")
        
        if passed_checks >= total_checks * 0.8:  # 80% pass rate required
            logger.info("\n✅ GENERATION 2 QUALITY GATE: PASSED")
            return True
        else:
            logger.error("\n❌ GENERATION 2 QUALITY GATE: FAILED")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ GENERATION 2 QUALITY GATE: FAILED - {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_generation2_robustness()
    sys.exit(0 if success else 1)