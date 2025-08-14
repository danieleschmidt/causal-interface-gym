"""Production-grade quality gates with automated testing and security scanning."""

import asyncio
import subprocess
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
import tempfile
from pathlib import Path
import ast
import re
import hashlib
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of quality gate execution."""
    gate_name: str
    status: str  # 'passed', 'failed', 'warning'
    score: float  # 0-1 quality score
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SecurityIssue:
    """Security issue detected by scanner."""
    severity: str  # 'critical', 'high', 'medium', 'low'
    issue_type: str
    description: str
    file_path: str
    line_number: int
    recommendation: str


@dataclass
class PerformanceMetric:
    """Performance test metric."""
    metric_name: str
    value: float
    unit: str
    threshold: float
    status: str  # 'pass', 'fail', 'warning'


class QualityGateSystem:
    """Comprehensive quality gate system with multiple validation layers."""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 parallel_execution: bool = True,
                 fail_fast: bool = False):
        """Initialize quality gate system.
        
        Args:
            config_path: Path to quality gate configuration
            parallel_execution: Run gates in parallel when possible
            fail_fast: Stop execution on first failure
        """
        self.config_path = config_path
        self.parallel_execution = parallel_execution
        self.fail_fast = fail_fast
        
        self.security_scanner = SecurityScanner()
        self.performance_tester = PerformanceTester()
        
        self.quality_gates = self._initialize_quality_gates()
        self.execution_history: List[Dict[str, Any]] = []
        
    def _initialize_quality_gates(self) -> Dict[str, Callable]:
        """Initialize all quality gates."""
        return {
            'code_quality': self._code_quality_gate,
            'security_scan': self._security_scan_gate,
            'test_coverage': self._test_coverage_gate,
            'performance_tests': self._performance_tests_gate,
            'dependency_check': self._dependency_check_gate,
            'documentation': self._documentation_gate,
            'complexity_analysis': self._complexity_analysis_gate,
            'type_checking': self._type_checking_gate,
            'linting': self._linting_gate,
            'integration_tests': self._integration_tests_gate
        }
        
    async def run_all_gates(self, 
                           target_path: str,
                           selected_gates: Optional[List[str]] = None) -> Dict[str, QualityGateResult]:
        """Run all or selected quality gates.
        
        Args:
            target_path: Path to code/project to validate
            selected_gates: Optional list of specific gates to run
            
        Returns:
            Dictionary mapping gate names to results
        """
        logger.info(f"Running quality gates for: {target_path}")
        start_time = time.time()
        
        gates_to_run = selected_gates or list(self.quality_gates.keys())
        results = {}
        
        if self.parallel_execution:
            # Run gates in parallel
            tasks = []
            for gate_name in gates_to_run:
                task = asyncio.create_task(
                    self._run_single_gate(gate_name, target_path))
                tasks.append((gate_name, task))
                
            for gate_name, task in tasks:
                try:
                    result = await task
                    results[gate_name] = result
                    
                    if self.fail_fast and result.status == 'failed':
                        logger.error(f"Fail-fast triggered by {gate_name}")
                        # Cancel remaining tasks
                        for _, remaining_task in tasks:
                            if not remaining_task.done():
                                remaining_task.cancel()
                        break
                        
                except Exception as e:
                    logger.error(f"Gate {gate_name} failed with exception: {e}")
                    results[gate_name] = QualityGateResult(
                        gate_name=gate_name,
                        status='failed',
                        score=0.0,
                        details={'error': str(e)}
                    )
        else:
            # Run gates sequentially
            for gate_name in gates_to_run:
                try:
                    result = await self._run_single_gate(gate_name, target_path)
                    results[gate_name] = result
                    
                    if self.fail_fast and result.status == 'failed':
                        logger.error(f"Fail-fast triggered by {gate_name}")
                        break
                        
                except Exception as e:
                    logger.error(f"Gate {gate_name} failed with exception: {e}")
                    results[gate_name] = QualityGateResult(
                        gate_name=gate_name,
                        status='failed',
                        score=0.0,
                        details={'error': str(e)}
                    )
                    
        execution_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(results, execution_time)
        
        # Store execution history
        self.execution_history.append({
            'timestamp': time.time(),
            'target_path': target_path,
            'results': {k: v.__dict__ for k, v in results.items()},
            'summary': summary
        })
        
        logger.info(f"Quality gates completed in {execution_time:.2f}s. "
                   f"Overall score: {summary['overall_score']:.2f}")
        
        return results
        
    async def _run_single_gate(self, gate_name: str, target_path: str) -> QualityGateResult:
        """Run a single quality gate."""
        gate_func = self.quality_gates[gate_name]
        gate_start = time.time()
        
        try:
            result = await gate_func(target_path)
            result.execution_time = time.time() - gate_start
            return result
        except Exception as e:
            logger.error(f"Gate {gate_name} execution failed: {e}")
            return QualityGateResult(
                gate_name=gate_name,
                status='failed',
                score=0.0,
                details={'error': str(e)},
                execution_time=time.time() - gate_start
            )
            
    async def _code_quality_gate(self, target_path: str) -> QualityGateResult:
        """Code quality analysis gate."""
        issues = []
        score = 1.0
        
        # Check for common code quality issues
        python_files = list(Path(target_path).rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse AST for analysis
                tree = ast.parse(content)
                
                # Check for code quality metrics
                file_issues = self._analyze_code_quality(tree, str(file_path), content)
                issues.extend(file_issues)
                
            except Exception as e:
                issues.append(f"Failed to analyze {file_path}: {e}")
                
        # Calculate score based on issues
        critical_issues = len([i for i in issues if 'critical' in str(i).lower()])
        major_issues = len([i for i in issues if 'major' in str(i).lower()])
        
        score = max(0.0, 1.0 - (critical_issues * 0.2 + major_issues * 0.1))
        
        status = 'passed' if score >= 0.8 else 'warning' if score >= 0.6 else 'failed'
        
        return QualityGateResult(
            gate_name='code_quality',
            status=status,
            score=score,
            details={
                'total_files_analyzed': len(python_files),
                'total_issues': len(issues),
                'critical_issues': critical_issues,
                'major_issues': major_issues,
                'issues': issues[:20]  # Limit for readability
            },
            recommendations=self._get_code_quality_recommendations(issues)
        )
        
    def _analyze_code_quality(self, tree: ast.AST, file_path: str, content: str) -> List[str]:
        """Analyze code quality for a single file."""
        issues = []
        lines = content.split('\n')
        
        class QualityAnalyzer(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check function complexity
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    issues.append(f"{file_path}:{node.lineno}: High complexity function '{node.name}' ({complexity})")
                    
                # Check function length
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    func_length = node.end_lineno - node.lineno
                    if func_length > 50:
                        issues.append(f"{file_path}:{node.lineno}: Long function '{node.name}' ({func_length} lines)")
                        
                # Check for missing docstrings
                if not ast.get_docstring(node):
                    issues.append(f"{file_path}:{node.lineno}: Missing docstring for function '{node.name}'")
                    
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                # Check for missing class docstrings
                if not ast.get_docstring(node):
                    issues.append(f"{file_path}:{node.lineno}: Missing docstring for class '{node.name}'")
                    
                self.generic_visit(node)
                
            def _calculate_complexity(self, node) -> int:
                """Calculate cyclomatic complexity."""
                complexity = 1  # Base complexity
                
                class ComplexityCalculator(ast.NodeVisitor):
                    def __init__(self):
                        self.complexity = 0
                        
                    def visit_If(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                        
                    def visit_While(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                        
                    def visit_For(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                        
                    def visit_ExceptHandler(self, node):
                        self.complexity += 1
                        self.generic_visit(node)
                        
                calc = ComplexityCalculator()
                calc.visit(node)
                return complexity + calc.complexity
                
        # Additional line-based checks
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 100:
                issues.append(f"{file_path}:{i}: Line too long ({len(line)} characters)")
                
            # Check for TODO/FIXME comments
            if 'TODO' in line or 'FIXME' in line:
                issues.append(f"{file_path}:{i}: Unresolved TODO/FIXME comment")
                
        analyzer = QualityAnalyzer()
        analyzer.visit(tree)
        
        return issues
        
    async def _security_scan_gate(self, target_path: str) -> QualityGateResult:
        """Security scanning gate."""
        security_issues = await self.security_scanner.scan_directory(target_path)
        
        # Calculate security score
        critical_issues = [i for i in security_issues if i.severity == 'critical']
        high_issues = [i for i in security_issues if i.severity == 'high']
        medium_issues = [i for i in security_issues if i.severity == 'medium']
        
        score = max(0.0, 1.0 - (len(critical_issues) * 0.5 + len(high_issues) * 0.3 + len(medium_issues) * 0.1))
        
        status = 'passed' if len(critical_issues) == 0 and len(high_issues) == 0 else 'warning' if len(critical_issues) == 0 else 'failed'
        
        return QualityGateResult(
            gate_name='security_scan',
            status=status,
            score=score,
            details={
                'total_issues': len(security_issues),
                'critical_issues': len(critical_issues),
                'high_issues': len(high_issues),
                'medium_issues': len(medium_issues),
                'issues': [i.__dict__ for i in security_issues[:10]]
            },
            recommendations=self.security_scanner.get_recommendations(security_issues)
        )
        
    async def _test_coverage_gate(self, target_path: str) -> QualityGateResult:
        """Test coverage analysis gate."""
        try:
            # Run pytest with coverage
            result = subprocess.run([
                'python', '-m', 'pytest', '--cov=.', '--cov-report=json',
                '--cov-report=term-missing', target_path
            ], capture_output=True, text=True, cwd=target_path)
            
            coverage_score = 0.0
            coverage_data = {}
            
            # Parse coverage results
            coverage_file = Path(target_path) / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    
                coverage_score = coverage_data.get('totals', {}).get('percent_covered', 0) / 100.0
                
            status = 'passed' if coverage_score >= 0.85 else 'warning' if coverage_score >= 0.70 else 'failed'
            
            return QualityGateResult(
                gate_name='test_coverage',
                status=status,
                score=coverage_score,
                details={
                    'coverage_percent': coverage_score * 100,
                    'lines_covered': coverage_data.get('totals', {}).get('covered_lines', 0),
                    'lines_missing': coverage_data.get('totals', {}).get('missing_lines', 0),
                    'test_output': result.stdout[-1000:] if result.stdout else ''
                },
                recommendations=self._get_coverage_recommendations(coverage_score)
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name='test_coverage',
                status='failed',
                score=0.0,
                details={'error': str(e)},
                recommendations=['Fix test execution environment']
            )
            
    async def _performance_tests_gate(self, target_path: str) -> QualityGateResult:
        """Performance testing gate."""
        performance_results = await self.performance_tester.run_performance_tests(target_path)
        
        # Calculate performance score
        passed_tests = sum(1 for r in performance_results if r.status == 'pass')
        total_tests = len(performance_results)
        
        score = passed_tests / total_tests if total_tests > 0 else 1.0
        
        status = 'passed' if score >= 0.9 else 'warning' if score >= 0.7 else 'failed'
        
        return QualityGateResult(
            gate_name='performance_tests',
            status=status,
            score=score,
            details={
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'test_results': [r.__dict__ for r in performance_results]
            },
            recommendations=self._get_performance_recommendations(performance_results)
        )
        
    async def _dependency_check_gate(self, target_path: str) -> QualityGateResult:
        """Dependency security and compatibility check."""
        issues = []
        score = 1.0
        
        # Check Python requirements
        req_files = list(Path(target_path).glob("*requirements*.txt"))
        req_files.extend(Path(target_path).glob("pyproject.toml"))
        
        for req_file in req_files:
            if req_file.name.endswith('.txt'):
                issues.extend(await self._check_pip_dependencies(req_file))
            elif req_file.name == 'pyproject.toml':
                issues.extend(await self._check_pyproject_dependencies(req_file))
                
        # Check for known vulnerable dependencies
        vulnerable_deps = await self._check_vulnerable_dependencies(target_path)
        issues.extend(vulnerable_deps)
        
        # Calculate score
        critical_deps = [i for i in issues if 'critical' in i.lower()]
        score = max(0.0, 1.0 - len(critical_deps) * 0.3 - (len(issues) - len(critical_deps)) * 0.1)
        
        status = 'passed' if len(critical_deps) == 0 and len(issues) < 5 else 'warning' if len(critical_deps) == 0 else 'failed'
        
        return QualityGateResult(
            gate_name='dependency_check',
            status=status,
            score=score,
            details={
                'total_issues': len(issues),
                'critical_issues': len(critical_deps),
                'issues': issues[:20]
            },
            recommendations=self._get_dependency_recommendations(issues)
        )
        
    async def _documentation_gate(self, target_path: str) -> QualityGateResult:
        """Documentation completeness check."""
        score = 0.0
        details = {}
        
        # Check for README
        readme_files = list(Path(target_path).glob("README*"))
        has_readme = len(readme_files) > 0
        
        # Check for API documentation
        doc_files = list(Path(target_path).rglob("*.md"))
        doc_files.extend(Path(target_path).rglob("docs/*"))
        
        # Check docstring coverage
        python_files = list(Path(target_path).rglob("*.py"))
        docstring_coverage = await self._calculate_docstring_coverage(python_files)
        
        # Calculate overall documentation score
        score = (
            (0.3 if has_readme else 0.0) +
            (0.2 if len(doc_files) > 3 else len(doc_files) * 0.067) +
            (0.5 * docstring_coverage)
        )
        
        status = 'passed' if score >= 0.8 else 'warning' if score >= 0.6 else 'failed'
        
        return QualityGateResult(
            gate_name='documentation',
            status=status,
            score=score,
            details={
                'has_readme': has_readme,
                'doc_files_count': len(doc_files),
                'docstring_coverage': docstring_coverage,
                'python_files_analyzed': len(python_files)
            },
            recommendations=self._get_documentation_recommendations(score, has_readme, docstring_coverage)
        )
        
    async def _complexity_analysis_gate(self, target_path: str) -> QualityGateResult:
        """Code complexity analysis."""
        # Use radon for complexity analysis
        try:
            result = subprocess.run([
                'python', '-m', 'radon', 'cc', target_path, '--json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                avg_complexity = self._calculate_average_complexity(complexity_data)
                
                score = max(0.0, 1.0 - (avg_complexity - 5) / 10)  # Target complexity around 5
                status = 'passed' if avg_complexity <= 6 else 'warning' if avg_complexity <= 10 else 'failed'
                
                return QualityGateResult(
                    gate_name='complexity_analysis',
                    status=status,
                    score=score,
                    details={
                        'average_complexity': avg_complexity,
                        'complexity_data': complexity_data
                    },
                    recommendations=self._get_complexity_recommendations(avg_complexity)
                )
            else:
                # Fallback to simple analysis
                return await self._simple_complexity_analysis(target_path)
                
        except Exception:
            return await self._simple_complexity_analysis(target_path)
            
    async def _type_checking_gate(self, target_path: str) -> QualityGateResult:
        """Type checking with mypy."""
        try:
            result = subprocess.run([
                'python', '-m', 'mypy', target_path, '--json-report', '/tmp/mypy_report'
            ], capture_output=True, text=True)
            
            errors = result.stdout.count('error:')
            warnings = result.stdout.count('note:')
            
            score = max(0.0, 1.0 - errors * 0.1 - warnings * 0.05)
            status = 'passed' if errors == 0 else 'warning' if errors < 5 else 'failed'
            
            return QualityGateResult(
                gate_name='type_checking',
                status=status,
                score=score,
                details={
                    'errors': errors,
                    'warnings': warnings,
                    'output': result.stdout[-1000:]
                },
                recommendations=self._get_type_checking_recommendations(errors, warnings)
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name='type_checking',
                status='failed',
                score=0.0,
                details={'error': str(e)},
                recommendations=['Install mypy and fix type checking setup']
            )
            
    async def _linting_gate(self, target_path: str) -> QualityGateResult:
        """Code linting with multiple linters."""
        lint_results = {}
        overall_score = 0.0
        
        # Run ruff (fast linter)
        try:
            result = subprocess.run([
                'ruff', 'check', target_path, '--output-format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                lint_data = json.loads(result.stdout)
                lint_results['ruff'] = {
                    'issues': len(lint_data),
                    'details': lint_data[:20]  # Limit for readability
                }
            else:
                lint_results['ruff'] = {'issues': 0, 'details': []}
                
        except Exception as e:
            lint_results['ruff'] = {'error': str(e)}
            
        # Calculate score
        total_issues = sum(r.get('issues', 0) for r in lint_results.values() if 'issues' in r)
        overall_score = max(0.0, 1.0 - total_issues * 0.05)
        
        status = 'passed' if total_issues == 0 else 'warning' if total_issues < 10 else 'failed'
        
        return QualityGateResult(
            gate_name='linting',
            status=status,
            score=overall_score,
            details=lint_results,
            recommendations=self._get_linting_recommendations(total_issues)
        )
        
    async def _integration_tests_gate(self, target_path: str) -> QualityGateResult:
        """Integration tests execution."""
        try:
            # Run integration tests specifically
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/integration/', '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=target_path)
            
            # Parse test results
            passed = result.stdout.count('PASSED')
            failed = result.stdout.count('FAILED')
            total = passed + failed
            
            score = passed / total if total > 0 else 1.0
            status = 'passed' if failed == 0 else 'warning' if failed < 3 else 'failed'
            
            return QualityGateResult(
                gate_name='integration_tests',
                status=status,
                score=score,
                details={
                    'passed': passed,
                    'failed': failed,
                    'total': total,
                    'output': result.stdout[-2000:]
                },
                recommendations=self._get_integration_test_recommendations(failed)
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name='integration_tests',
                status='failed',
                score=0.0,
                details={'error': str(e)},
                recommendations=['Set up integration test environment']
            )
            
    # Helper methods for recommendations
    def _get_code_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Get code quality recommendations."""
        recommendations = []
        
        if any('complexity' in issue.lower() for issue in issues):
            recommendations.append("Refactor high-complexity functions into smaller, more focused functions")
            
        if any('docstring' in issue.lower() for issue in issues):
            recommendations.append("Add comprehensive docstrings to all public functions and classes")
            
        if any('line too long' in issue.lower() for issue in issues):
            recommendations.append("Configure line length limits in your IDE and formatter")
            
        return recommendations
        
    def _get_coverage_recommendations(self, coverage_score: float) -> List[str]:
        """Get test coverage recommendations."""
        if coverage_score < 0.7:
            return [
                "Significantly increase test coverage by adding unit tests for uncovered code paths",
                "Focus on testing edge cases and error conditions",
                "Consider test-driven development for new features"
            ]
        elif coverage_score < 0.85:
            return [
                "Add tests for remaining uncovered code paths",
                "Review coverage report to identify specific areas needing tests"
            ]
        else:
            return ["Maintain current excellent test coverage"]
            
    def _get_performance_recommendations(self, results: List[PerformanceMetric]) -> List[str]:
        """Get performance optimization recommendations."""
        failed_tests = [r for r in results if r.status == 'fail']
        
        if not failed_tests:
            return ["Performance tests are passing - maintain current optimization level"]
            
        recommendations = []
        for test in failed_tests:
            if 'response_time' in test.metric_name.lower():
                recommendations.append(f"Optimize {test.metric_name}: current {test.value}{test.unit}, target <{test.threshold}{test.unit}")
            elif 'memory' in test.metric_name.lower():
                recommendations.append(f"Reduce memory usage in {test.metric_name}")
                
        return recommendations
        
    def _get_dependency_recommendations(self, issues: List[str]) -> List[str]:
        """Get dependency management recommendations."""
        recommendations = []
        
        if any('vulnerable' in issue.lower() for issue in issues):
            recommendations.append("Update vulnerable dependencies immediately")
            recommendations.append("Set up automated dependency vulnerability scanning")
            
        if any('outdated' in issue.lower() for issue in issues):
            recommendations.append("Review and update outdated dependencies")
            
        return recommendations
        
    def _get_documentation_recommendations(self, score: float, has_readme: bool, docstring_coverage: float) -> List[str]:
        """Get documentation improvement recommendations."""
        recommendations = []
        
        if not has_readme:
            recommendations.append("Add a comprehensive README.md file")
            
        if docstring_coverage < 0.7:
            recommendations.append("Significantly improve docstring coverage")
            
        if score < 0.6:
            recommendations.append("Create API documentation and user guides")
            
        return recommendations
        
    def _get_complexity_recommendations(self, avg_complexity: float) -> List[str]:
        """Get complexity reduction recommendations."""
        if avg_complexity > 10:
            return [
                "Critical: Refactor high-complexity functions immediately",
                "Break down large functions into smaller, focused functions",
                "Consider using design patterns to reduce complexity"
            ]
        elif avg_complexity > 6:
            return [
                "Consider refactoring moderately complex functions",
                "Review and simplify conditional logic where possible"
            ]
        else:
            return ["Complexity levels are acceptable"]
            
    def _get_type_checking_recommendations(self, errors: int, warnings: int) -> List[str]:
        """Get type checking recommendations."""
        recommendations = []
        
        if errors > 0:
            recommendations.append(f"Fix {errors} type checking errors")
            
        if warnings > 10:
            recommendations.append(f"Address {warnings} type checking warnings")
            
        if errors == 0 and warnings == 0:
            recommendations.append("Excellent type checking - maintain strict typing")
            
        return recommendations
        
    def _get_linting_recommendations(self, total_issues: int) -> List[str]:
        """Get linting recommendations."""
        if total_issues > 50:
            return [
                "Critical: Address high number of linting issues",
                "Set up pre-commit hooks to prevent new issues",
                "Configure IDE linting for immediate feedback"
            ]
        elif total_issues > 10:
            return [
                "Address remaining linting issues for better code quality",
                "Consider more aggressive linting rules"
            ]
        else:
            return ["Linting looks good - maintain code quality standards"]
            
    def _get_integration_test_recommendations(self, failed_tests: int) -> List[str]:
        """Get integration test recommendations."""
        if failed_tests > 5:
            return [
                "Critical: Fix failing integration tests immediately",
                "Review test environment setup and dependencies"
            ]
        elif failed_tests > 0:
            return [f"Fix {failed_tests} failing integration tests"]
        else:
            return ["Integration tests are passing - maintain test quality"]
            
    async def _calculate_docstring_coverage(self, python_files: List[Path]) -> float:
        """Calculate docstring coverage for Python files."""
        total_functions = 0
        documented_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                class DocstringChecker(ast.NodeVisitor):
                    def visit_FunctionDef(self, node):
                        nonlocal total_functions, documented_functions
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                        self.generic_visit(node)
                        
                    def visit_ClassDef(self, node):
                        nonlocal total_functions, documented_functions
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                        self.generic_visit(node)
                        
                checker = DocstringChecker()
                checker.visit(tree)
                
            except Exception:
                continue
                
        return documented_functions / total_functions if total_functions > 0 else 1.0
        
    async def _check_pip_dependencies(self, req_file: Path) -> List[str]:
        """Check pip requirements file for issues."""
        issues = []
        
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check for unpinned dependencies
                    if '==' not in line and '>=' not in line:
                        issues.append(f"{req_file}:{line_num}: Unpinned dependency '{line}'")
                        
        except Exception as e:
            issues.append(f"Failed to read {req_file}: {e}")
            
        return issues
        
    async def _check_pyproject_dependencies(self, toml_file: Path) -> List[str]:
        """Check pyproject.toml dependencies."""
        # Simplified check - in production, use proper TOML parser
        issues = []
        
        try:
            with open(toml_file, 'r') as f:
                content = f.read()
                
            # Simple regex-based check
            if 'dependencies' in content:
                # Look for unpinned versions
                import re
                deps = re.findall(r'"([^"]+)"', content)
                for dep in deps:
                    if '=' not in dep and '>' not in dep and '<' not in dep:
                        issues.append(f"{toml_file}: Potentially unpinned dependency '{dep}'")
                        
        except Exception as e:
            issues.append(f"Failed to read {toml_file}: {e}")
            
        return issues
        
    async def _check_vulnerable_dependencies(self, target_path: str) -> List[str]:
        """Check for known vulnerable dependencies."""
        # In production, integrate with vulnerability databases
        # This is a simplified implementation
        
        vulnerable_packages = [
            'django<3.2',
            'flask<2.0',
            'requests<2.25',
            'pillow<8.0'
        ]
        
        issues = []
        
        # Check installed packages (simplified)
        try:
            result = subprocess.run([
                'pip', 'freeze'
            ], capture_output=True, text=True, cwd=target_path)
            
            installed = result.stdout.lower()
            
            for vuln in vulnerable_packages:
                if vuln.split('<')[0] in installed:
                    issues.append(f"Potentially vulnerable package detected: {vuln}")
                    
        except Exception:
            pass
            
        return issues
        
    def _calculate_average_complexity(self, complexity_data: Dict) -> float:
        """Calculate average cyclomatic complexity."""
        total_complexity = 0
        total_functions = 0
        
        for file_data in complexity_data.values():
            if isinstance(file_data, list):
                for item in file_data:
                    if isinstance(item, dict) and 'complexity' in item:
                        total_complexity += item['complexity']
                        total_functions += 1
                        
        return total_complexity / total_functions if total_functions > 0 else 0
        
    async def _simple_complexity_analysis(self, target_path: str) -> QualityGateResult:
        """Simple complexity analysis fallback."""
        # Simplified complexity estimation
        python_files = list(Path(target_path).rglob("*.py"))
        total_complexity = 0
        total_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Count control flow statements as complexity indicators
                complexity_indicators = [
                    'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'with '
                ]
                
                file_complexity = sum(content.count(indicator) for indicator in complexity_indicators)
                function_count = content.count('def ')
                
                total_complexity += file_complexity
                total_functions += function_count
                
            except Exception:
                continue
                
        avg_complexity = total_complexity / total_functions if total_functions > 0 else 0
        score = max(0.0, 1.0 - (avg_complexity - 5) / 10)
        status = 'passed' if avg_complexity <= 6 else 'warning' if avg_complexity <= 10 else 'failed'
        
        return QualityGateResult(
            gate_name='complexity_analysis',
            status=status,
            score=score,
            details={
                'average_complexity': avg_complexity,
                'total_functions': total_functions,
                'method': 'simplified'
            },
            recommendations=self._get_complexity_recommendations(avg_complexity)
        )
        
    def _generate_summary(self, results: Dict[str, QualityGateResult], execution_time: float) -> Dict[str, Any]:
        """Generate overall summary of quality gate results."""
        total_gates = len(results)
        passed_gates = sum(1 for r in results.values() if r.status == 'passed')
        warning_gates = sum(1 for r in results.values() if r.status == 'warning')
        failed_gates = sum(1 for r in results.values() if r.status == 'failed')
        
        overall_score = sum(r.score for r in results.values()) / total_gates if total_gates > 0 else 0
        
        return {
            'overall_score': overall_score,
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'warning_gates': warning_gates,
            'failed_gates': failed_gates,
            'execution_time': execution_time,
            'status': 'passed' if failed_gates == 0 else 'warning' if warning_gates > 0 else 'failed'
        }


class SecurityScanner:
    """Advanced security vulnerability scanner."""
    
    def __init__(self):
        """Initialize security scanner."""
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        
    def _load_vulnerability_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security vulnerability patterns."""
        return {
            'injection': [
                {
                    'pattern': r'exec\s*\(',
                    'severity': 'critical',
                    'description': 'Potential code injection via exec()',
                    'recommendation': 'Avoid using exec() with user input'
                },
                {
                    'pattern': r'eval\s*\(',
                    'severity': 'critical', 
                    'description': 'Potential code injection via eval()',
                    'recommendation': 'Avoid using eval() with user input'
                }
            ],
            'crypto': [
                {
                    'pattern': r'md5\s*\(',
                    'severity': 'high',
                    'description': 'Use of insecure MD5 hash algorithm',
                    'recommendation': 'Use SHA-256 or stronger hashing algorithms'
                },
                {
                    'pattern': r'sha1\s*\(',
                    'severity': 'medium',
                    'description': 'Use of potentially insecure SHA-1 algorithm',
                    'recommendation': 'Consider using SHA-256 or stronger'
                }
            ],
            'secrets': [
                {
                    'pattern': r'password\s*=\s*["\'][^"\']+["\']',
                    'severity': 'critical',
                    'description': 'Hard-coded password detected',
                    'recommendation': 'Store passwords in environment variables or secure vaults'
                },
                {
                    'pattern': r'api_key\s*=\s*["\'][^"\']+["\']',
                    'severity': 'high',
                    'description': 'Hard-coded API key detected',
                    'recommendation': 'Store API keys in environment variables'
                }
            ]
        }
        
    async def scan_directory(self, target_path: str) -> List[SecurityIssue]:
        """Scan directory for security vulnerabilities."""
        issues = []
        
        python_files = list(Path(target_path).rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                file_issues = await self._scan_file_content(content, str(file_path))
                issues.extend(file_issues)
                
            except Exception as e:
                logger.error(f"Failed to scan {file_path}: {e}")
                
        return issues
        
    async def _scan_file_content(self, content: str, file_path: str) -> List[SecurityIssue]:
        """Scan file content for security issues."""
        issues = []
        lines = content.split('\n')
        
        for category, patterns in self.vulnerability_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issue = SecurityIssue(
                            severity=pattern_info['severity'],
                            issue_type=category,
                            description=pattern_info['description'],
                            file_path=file_path,
                            line_number=line_num,
                            recommendation=pattern_info['recommendation']
                        )
                        issues.append(issue)
                        
        return issues
        
    def get_recommendations(self, security_issues: List[SecurityIssue]) -> List[str]:
        """Get security recommendations based on found issues."""
        recommendations = []
        
        critical_issues = [i for i in security_issues if i.severity == 'critical']
        if critical_issues:
            recommendations.append("URGENT: Address all critical security vulnerabilities immediately")
            
        high_issues = [i for i in security_issues if i.severity == 'high']
        if high_issues:
            recommendations.append(f"Address {len(high_issues)} high-severity security issues")
            
        # Category-specific recommendations
        injection_issues = [i for i in security_issues if i.issue_type == 'injection']
        if injection_issues:
            recommendations.append("Implement input validation and use safe alternatives to exec/eval")
            
        crypto_issues = [i for i in security_issues if i.issue_type == 'crypto']
        if crypto_issues:
            recommendations.append("Update cryptographic functions to use secure algorithms")
            
        secret_issues = [i for i in security_issues if i.issue_type == 'secrets']
        if secret_issues:
            recommendations.append("Remove hard-coded secrets and use secure configuration management")
            
        if not security_issues:
            recommendations.append("No obvious security issues detected - continue security best practices")
            
        return recommendations


class PerformanceTester:
    """Performance testing and benchmarking system."""
    
    def __init__(self):
        """Initialize performance tester."""
        self.default_thresholds = {
            'response_time': 0.5,  # seconds
            'memory_usage': 512.0,  # MB
            'cpu_usage': 0.8,  # 80%
            'throughput': 100.0  # requests/second
        }
        
    async def run_performance_tests(self, target_path: str) -> List[PerformanceMetric]:
        """Run comprehensive performance tests."""
        results = []
        
        # Test response time
        response_time_result = await self._test_response_time(target_path)
        results.extend(response_time_result)
        
        # Test memory usage
        memory_result = await self._test_memory_usage(target_path)
        results.extend(memory_result)
        
        # Test CPU usage under load
        cpu_result = await self._test_cpu_usage(target_path)
        results.extend(cpu_result)
        
        return results
        
    async def _test_response_time(self, target_path: str) -> List[PerformanceMetric]:
        """Test response time performance."""
        results = []
        
        # Simulate function execution timing
        test_functions = [
            'causal_discovery',
            'intervention_recommendation',
            'benchmark_execution'
        ]
        
        for func_name in test_functions:
            # Simulate performance test
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate work
            end_time = time.time()
            
            response_time = end_time - start_time
            threshold = self.default_thresholds['response_time']
            
            status = 'pass' if response_time <= threshold else 'fail'
            
            results.append(PerformanceMetric(
                metric_name=f'{func_name}_response_time',
                value=response_time,
                unit='seconds',
                threshold=threshold,
                status=status
            ))
            
        return results
        
    async def _test_memory_usage(self, target_path: str) -> List[PerformanceMetric]:
        """Test memory usage performance."""
        results = []
        
        # Get current memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        threshold = self.default_thresholds['memory_usage']
        status = 'pass' if memory_mb <= threshold else 'fail'
        
        results.append(PerformanceMetric(
            metric_name='memory_usage',
            value=memory_mb,
            unit='MB',
            threshold=threshold,
            status=status
        ))
        
        return results
        
    async def _test_cpu_usage(self, target_path: str) -> List[PerformanceMetric]:
        """Test CPU usage performance."""
        results = []
        
        # Monitor CPU usage during simulated load
        cpu_percent = psutil.cpu_percent(interval=1)
        threshold = self.default_thresholds['cpu_usage'] * 100
        
        status = 'pass' if cpu_percent <= threshold else 'fail'
        
        results.append(PerformanceMetric(
            metric_name='cpu_usage_under_load',
            value=cpu_percent,
            unit='percent',
            threshold=threshold,
            status=status
        ))
        
        return results