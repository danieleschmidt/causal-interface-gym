"""AI-driven code review system for autonomous quality gates."""

import ast
import os
import re
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging
from datetime import datetime
import subprocess
import hashlib

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class ReviewSeverity(Enum):
    """Severity levels for code review findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewCategory(Enum):
    """Categories of code review findings."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    CORRECTNESS = "correctness"
    STYLE = "style"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"


@dataclass
class CodeReviewFinding:
    """Represents a single code review finding."""
    id: str
    file_path: str
    line_number: int
    category: ReviewCategory
    severity: ReviewSeverity
    title: str
    description: str
    suggested_fix: Optional[str] = None
    code_snippet: Optional[str] = None
    confidence: float = 0.0
    auto_fixable: bool = False
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class CodeAnalysisResult:
    """Result of AI-driven code analysis."""
    file_path: str
    findings: List[CodeReviewFinding]
    overall_score: float
    complexity_metrics: Dict[str, float]
    test_coverage: float
    analyzed_at: datetime = field(default_factory=datetime.now)


class AICodeReviewer:
    """AI-driven code reviewer with advanced analysis capabilities."""
    
    def __init__(self, 
                 llm_client: LLMClient,
                 project_root: Path,
                 review_config: Optional[Dict[str, Any]] = None):
        """Initialize AI code reviewer.
        
        Args:
            llm_client: LLM client for AI-powered analysis
            project_root: Root directory of the project
            review_config: Configuration for code review rules and thresholds
        """
        self.llm_client = llm_client
        self.project_root = Path(project_root)
        self.review_config = review_config or self._default_config()
        
        self.review_history: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.pattern_cache: Dict[str, List[CodeReviewFinding]] = {}
        
        # Load project-specific patterns
        self._load_project_patterns()
        
        # Initialize static analysis tools
        self.static_analyzers = {
            'security': self._security_analysis,
            'complexity': self._complexity_analysis,
            'performance': self._performance_analysis,
            'style': self._style_analysis,
            'testing': self._testing_analysis
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration for code review."""
        return {
            "severity_thresholds": {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.5,
                "low": 0.3
            },
            "auto_fix_enabled": True,
            "ai_confidence_threshold": 0.6,
            "max_findings_per_file": 20,
            "supported_extensions": [".py", ".js", ".ts", ".jsx", ".tsx"],
            "excluded_paths": ["node_modules", "__pycache__", ".git", "dist", "build"],
            "review_categories": [
                "security", "performance", "maintainability", 
                "correctness", "style", "testing", "documentation"
            ]
        }
    
    def _load_project_patterns(self) -> None:
        """Load project-specific patterns and rules."""
        patterns_file = self.project_root / ".ai_review_patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    patterns = json.load(f)
                    self.knowledge_base.update(patterns)
                logger.info(f"Loaded {len(patterns)} project-specific patterns")
            except Exception as e:
                logger.warning(f"Failed to load project patterns: {e}")
    
    async def review_files(self, 
                          file_paths: List[str],
                          changed_lines: Optional[Dict[str, Set[int]]] = None) -> List[CodeAnalysisResult]:
        """Review multiple files with AI-powered analysis.
        
        Args:
            file_paths: List of file paths to review
            changed_lines: Optional dict mapping files to changed line numbers
            
        Returns:
            List of analysis results for each file
        """
        results = []
        
        # Process files in parallel batches
        batch_size = 5
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            # Create analysis tasks
            tasks = []
            for file_path in batch:
                if self._should_review_file(file_path):
                    task = asyncio.create_task(
                        self._analyze_file(file_path, changed_lines)
                    )
                    tasks.append(task)
            
            # Wait for batch completion
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, CodeAnalysisResult):
                        results.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"File analysis failed: {result}")
        
        # Update review history
        self.review_history.append({
            "timestamp": datetime.now(),
            "files_reviewed": len(results),
            "total_findings": sum(len(r.findings) for r in results),
            "avg_score": sum(r.overall_score for r in results) / max(len(results), 1)
        })
        
        return results
    
    def _should_review_file(self, file_path: str) -> bool:
        """Check if file should be reviewed."""
        path = Path(file_path)
        
        # Check extension
        if path.suffix not in self.review_config["supported_extensions"]:
            return False
        
        # Check excluded paths
        for excluded in self.review_config["excluded_paths"]:
            if excluded in str(path):
                return False
        
        # Check file size (skip very large files)
        try:
            if path.exists() and path.stat().st_size > 1000000:  # 1MB limit
                return False
        except Exception:
            return False
        
        return True
    
    async def _analyze_file(self, 
                           file_path: str, 
                           changed_lines: Optional[Dict[str, Set[int]]]) -> CodeAnalysisResult:
        """Analyze a single file with comprehensive AI review."""
        try:
            # Read file content
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Perform multiple analysis passes
            findings = []
            
            # 1. Static analysis
            static_findings = await self._run_static_analysis(file_path, content)
            findings.extend(static_findings)
            
            # 2. AI-powered semantic analysis
            ai_findings = await self._ai_semantic_analysis(file_path, content, changed_lines)
            findings.extend(ai_findings)
            
            # 3. Pattern-based analysis
            pattern_findings = await self._pattern_based_analysis(file_path, content)
            findings.extend(pattern_findings)
            
            # 4. Context-aware analysis (if part of larger codebase)
            context_findings = await self._context_aware_analysis(file_path, content)
            findings.extend(context_findings)
            
            # Remove duplicates and prioritize findings
            findings = self._deduplicate_findings(findings)
            findings = sorted(findings, key=lambda f: (f.severity.value, -f.confidence))
            
            # Limit findings per file
            max_findings = self.review_config["max_findings_per_file"]
            findings = findings[:max_findings]
            
            # Calculate overall score
            overall_score = self._calculate_file_score(findings)
            
            # Calculate complexity metrics
            complexity_metrics = await self._calculate_complexity_metrics(file_path, content)
            
            # Calculate test coverage (if possible)
            test_coverage = await self._calculate_test_coverage(file_path)
            
            return CodeAnalysisResult(
                file_path=file_path,
                findings=findings,
                overall_score=overall_score,
                complexity_metrics=complexity_metrics,
                test_coverage=test_coverage
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return CodeAnalysisResult(
                file_path=file_path,
                findings=[],
                overall_score=0.5,  # Neutral score for failed analysis
                complexity_metrics={},
                test_coverage=0.0
            )
    
    async def _run_static_analysis(self, file_path: str, content: str) -> List[CodeReviewFinding]:
        """Run static analysis using multiple analyzers."""
        findings = []
        
        # Run each static analyzer
        for analyzer_name, analyzer_func in self.static_analyzers.items():
            try:
                analyzer_findings = await analyzer_func(file_path, content)
                findings.extend(analyzer_findings)
            except Exception as e:
                logger.warning(f"Static analyzer {analyzer_name} failed for {file_path}: {e}")
        
        return findings
    
    async def _security_analysis(self, file_path: str, content: str) -> List[CodeReviewFinding]:
        """Security-focused static analysis."""
        findings = []
        lines = content.split('\n')
        
        # Common security patterns
        security_patterns = [
            (r'exec\s*\(', "Dangerous use of exec()", ReviewSeverity.CRITICAL),
            (r'eval\s*\(', "Dangerous use of eval()", ReviewSeverity.CRITICAL),
            (r'os\.system\s*\(', "Potential command injection", ReviewSeverity.HIGH),
            (r'subprocess\..*shell=True', "Shell injection risk", ReviewSeverity.HIGH),
            (r'pickle\.loads?\s*\(', "Unsafe pickle usage", ReviewSeverity.MEDIUM),
            (r'input\s*\([^)]*\)', "Unsafe input usage", ReviewSeverity.MEDIUM),
            (r'password\s*=\s*["\'][^"\']*["\']', "Hardcoded password", ReviewSeverity.HIGH),
            (r'api_key\s*=\s*["\'][^"\']*["\']', "Hardcoded API key", ReviewSeverity.HIGH),
            (r'secret\s*=\s*["\'][^"\']*["\']', "Hardcoded secret", ReviewSeverity.HIGH),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, message, severity in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    finding = CodeReviewFinding(
                        id=f"sec_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        file_path=file_path,
                        line_number=i,
                        category=ReviewCategory.SECURITY,
                        severity=severity,
                        title=message,
                        description=f"Security issue detected: {message}",
                        code_snippet=line.strip(),
                        confidence=0.8,
                        auto_fixable=False
                    )
                    findings.append(finding)
        
        return findings
    
    async def _complexity_analysis(self, file_path: str, content: str) -> List[CodeReviewFinding]:
        """Complexity-focused analysis."""
        findings = []
        
        try:
            # Parse AST for Python files
            if file_path.endswith('.py'):
                tree = ast.parse(content)
                
                # Analyze functions and methods
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        
                        if complexity > 15:  # High complexity threshold
                            finding = CodeReviewFinding(
                                id=f"comp_{hashlib.md5(f'{file_path}:{node.lineno}:complexity'.encode()).hexdigest()[:8]}",
                                file_path=file_path,
                                line_number=node.lineno,
                                category=ReviewCategory.MAINTAINABILITY,
                                severity=ReviewSeverity.HIGH if complexity > 20 else ReviewSeverity.MEDIUM,
                                title=f"High cyclomatic complexity ({complexity})",
                                description=f"Function '{node.name}' has complexity {complexity}. Consider refactoring.",
                                confidence=0.9,
                                auto_fixable=False
                            )
                            findings.append(finding)
        
        except Exception as e:
            logger.warning(f"Complexity analysis failed for {file_path}: {e}")
        
        return findings
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    async def _performance_analysis(self, file_path: str, content: str) -> List[CodeReviewFinding]:
        """Performance-focused analysis."""
        findings = []
        lines = content.split('\n')
        
        # Performance anti-patterns
        perf_patterns = [
            (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "Use enumerate() instead of range(len())", ReviewSeverity.LOW),
            (r'\.keys\s*\(\s*\)\s*:', "Iterating over dict.keys() unnecessarily", ReviewSeverity.LOW),
            (r'\+\s*=.*\[.*\]', "Inefficient list concatenation", ReviewSeverity.MEDIUM),
            (r'time\.sleep\s*\(\s*[0-9.]+\s*\)', "Blocking sleep in async context", ReviewSeverity.MEDIUM),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, message, severity in perf_patterns:
                if re.search(pattern, line):
                    finding = CodeReviewFinding(
                        id=f"perf_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                        file_path=file_path,
                        line_number=i,
                        category=ReviewCategory.PERFORMANCE,
                        severity=severity,
                        title=message,
                        description=f"Performance issue: {message}",
                        code_snippet=line.strip(),
                        confidence=0.7,
                        auto_fixable=True
                    )
                    findings.append(finding)
        
        return findings
    
    async def _style_analysis(self, file_path: str, content: str) -> List[CodeReviewFinding]:
        """Style and formatting analysis."""
        findings = []
        lines = content.split('\n')
        
        # Style issues
        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 120:
                finding = CodeReviewFinding(
                    id=f"style_{hashlib.md5(f'{file_path}:{i}:long_line'.encode()).hexdigest()[:8]}",
                    file_path=file_path,
                    line_number=i,
                    category=ReviewCategory.STYLE,
                    severity=ReviewSeverity.LOW,
                    title="Line too long",
                    description=f"Line length ({len(line)}) exceeds 120 characters",
                    confidence=1.0,
                    auto_fixable=True
                )
                findings.append(finding)
            
            # Trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                finding = CodeReviewFinding(
                    id=f"style_{hashlib.md5(f'{file_path}:{i}:trailing_ws'.encode()).hexdigest()[:8]}",
                    file_path=file_path,
                    line_number=i,
                    category=ReviewCategory.STYLE,
                    severity=ReviewSeverity.LOW,
                    title="Trailing whitespace",
                    description="Line has trailing whitespace",
                    confidence=1.0,
                    auto_fixable=True
                )
                findings.append(finding)
        
        return findings
    
    async def _testing_analysis(self, file_path: str, content: str) -> List[CodeReviewFinding]:
        """Testing-related analysis."""
        findings = []
        
        # Check for test files
        is_test_file = any(pattern in file_path.lower() for pattern in ['test_', '_test', 'spec_', '_spec'])
        
        if not is_test_file:
            # Look for testable functions without tests
            try:
                if file_path.endswith('.py'):
                    tree = ast.parse(content)
                    
                    public_functions = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                            public_functions.append((node.name, node.lineno))
                    
                    # Check if corresponding test file exists
                    test_file_patterns = [
                        f"test_{Path(file_path).stem}.py",
                        f"{Path(file_path).stem}_test.py",
                        f"tests/test_{Path(file_path).stem}.py"
                    ]
                    
                    test_file_exists = any(
                        (self.project_root / pattern).exists() 
                        for pattern in test_file_patterns
                    )
                    
                    if not test_file_exists and public_functions:
                        for func_name, line_no in public_functions[:3]:  # Limit findings
                            finding = CodeReviewFinding(
                                id=f"test_{hashlib.md5(f'{file_path}:{func_name}'.encode()).hexdigest()[:8]}",
                                file_path=file_path,
                                line_number=line_no,
                                category=ReviewCategory.TESTING,
                                severity=ReviewSeverity.MEDIUM,
                                title=f"Function '{func_name}' lacks tests",
                                description=f"Public function '{func_name}' should have unit tests",
                                confidence=0.6,
                                auto_fixable=False
                            )
                            findings.append(finding)
            
            except Exception as e:
                logger.warning(f"Testing analysis failed for {file_path}: {e}")
        
        return findings
    
    async def _ai_semantic_analysis(self, 
                                   file_path: str, 
                                   content: str, 
                                   changed_lines: Optional[Dict[str, Set[int]]]) -> List[CodeReviewFinding]:
        """AI-powered semantic analysis using LLM."""
        findings = []
        
        try:
            # Prepare context for AI analysis
            context = {
                "file_path": file_path,
                "content": content[:5000],  # Limit content size for LLM
                "changed_lines": list(changed_lines.get(file_path, [])) if changed_lines else [],
                "project_patterns": self.knowledge_base.get("common_patterns", []),
                "recent_issues": self.knowledge_base.get("recent_issues", [])
            }
            
            # AI analysis prompt
            prompt = f"""
            You are an expert code reviewer. Analyze the following code for potential issues:
            
            File: {file_path}
            Changed Lines: {context['changed_lines']}
            
            Code:
            ```
            {context['content']}
            ```
            
            Focus on:
            1. Logic errors and potential bugs
            2. Security vulnerabilities
            3. Performance issues
            4. Code maintainability
            5. Best practices violations
            
            Return findings as JSON array with format:
            {{
                "line_number": int,
                "category": "security|performance|maintainability|correctness|style",
                "severity": "critical|high|medium|low",
                "title": "Brief title",
                "description": "Detailed description",
                "suggested_fix": "Optional fix suggestion",
                "confidence": float (0-1)
            }}
            """
            
            # Get AI response
            response = await self.llm_client.generate_response(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.1
            )
            
            # Parse AI response
            ai_findings = self._parse_ai_response(response, file_path)
            findings.extend(ai_findings)
            
        except Exception as e:
            logger.warning(f"AI semantic analysis failed for {file_path}: {e}")
        
        return findings
    
    def _parse_ai_response(self, response: str, file_path: str) -> List[CodeReviewFinding]:
        """Parse AI response into CodeReviewFinding objects."""
        findings = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return findings
            
            ai_findings = json.loads(json_match.group())
            
            for finding_data in ai_findings:
                if isinstance(finding_data, dict):
                    # Validate required fields
                    required_fields = ['line_number', 'category', 'severity', 'title', 'description']
                    if all(field in finding_data for field in required_fields):
                        
                        finding = CodeReviewFinding(
                            id=f"ai_{hashlib.md5(f'{file_path}:{finding_data[\"line_number\"]}:{finding_data[\"title\"]}'.encode()).hexdigest()[:8]}",
                            file_path=file_path,
                            line_number=finding_data['line_number'],
                            category=ReviewCategory(finding_data['category']),
                            severity=ReviewSeverity(finding_data['severity']),
                            title=finding_data['title'],
                            description=finding_data['description'],
                            suggested_fix=finding_data.get('suggested_fix'),
                            confidence=finding_data.get('confidence', 0.5),
                            auto_fixable=finding_data.get('suggested_fix') is not None
                        )
                        
                        # Filter by confidence threshold
                        if finding.confidence >= self.review_config["ai_confidence_threshold"]:
                            findings.append(finding)
        
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
        
        return findings
    
    async def _pattern_based_analysis(self, file_path: str, content: str) -> List[CodeReviewFinding]:
        """Pattern-based analysis using project knowledge base."""
        findings = []
        
        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{file_path}:{content_hash}"
        
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        # Apply custom patterns from knowledge base
        custom_patterns = self.knowledge_base.get("custom_patterns", [])
        
        for pattern_config in custom_patterns:
            pattern = pattern_config.get("regex")
            message = pattern_config.get("message", "Custom pattern match")
            severity = ReviewSeverity(pattern_config.get("severity", "medium"))
            category = ReviewCategory(pattern_config.get("category", "maintainability"))
            
            if pattern:
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        finding = CodeReviewFinding(
                            id=f"pattern_{hashlib.md5(f'{file_path}:{i}:{pattern}'.encode()).hexdigest()[:8]}",
                            file_path=file_path,
                            line_number=i,
                            category=category,
                            severity=severity,
                            title=message,
                            description=f"Custom pattern match: {message}",
                            code_snippet=line.strip(),
                            confidence=0.8
                        )
                        findings.append(finding)
        
        # Cache results
        self.pattern_cache[cache_key] = findings
        return findings
    
    async def _context_aware_analysis(self, file_path: str, content: str) -> List[CodeReviewFinding]:
        """Context-aware analysis considering the broader codebase."""
        findings = []
        
        try:
            # Analyze imports and dependencies
            if file_path.endswith('.py'):
                tree = ast.parse(content)
                
                # Check for unused imports
                imports = set()
                used_names = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.Name):
                        used_names.add(node.id)
                
                # Find potentially unused imports
                unused_imports = imports - used_names
                for unused in list(unused_imports)[:5]:  # Limit findings
                    finding = CodeReviewFinding(
                        id=f"unused_{hashlib.md5(f'{file_path}:{unused}'.encode()).hexdigest()[:8]}",
                        file_path=file_path,
                        line_number=1,  # Simplified - would need better line tracking
                        category=ReviewCategory.MAINTAINABILITY,
                        severity=ReviewSeverity.LOW,
                        title=f"Potentially unused import: {unused}",
                        description=f"Import '{unused}' may not be used in this file",
                        confidence=0.5,  # Lower confidence for this heuristic
                        auto_fixable=True
                    )
                    findings.append(finding)
        
        except Exception as e:
            logger.warning(f"Context-aware analysis failed for {file_path}: {e}")
        
        return findings
    
    def _deduplicate_findings(self, findings: List[CodeReviewFinding]) -> List[CodeReviewFinding]:
        """Remove duplicate findings based on location and content similarity."""
        unique_findings = []
        seen_signatures = set()
        
        for finding in findings:
            # Create signature based on location and message
            signature = f"{finding.file_path}:{finding.line_number}:{finding.category.value}:{finding.title}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_findings.append(finding)
        
        return unique_findings
    
    def _calculate_file_score(self, findings: List[CodeReviewFinding]) -> float:
        """Calculate overall quality score for a file based on findings."""
        if not findings:
            return 1.0  # Perfect score if no issues
        
        # Weight findings by severity
        severity_weights = {
            ReviewSeverity.CRITICAL: -0.5,
            ReviewSeverity.HIGH: -0.3,
            ReviewSeverity.MEDIUM: -0.1,
            ReviewSeverity.LOW: -0.05,
            ReviewSeverity.INFO: -0.01
        }
        
        total_impact = 0
        for finding in findings:
            weight = severity_weights.get(finding.severity, -0.1)
            confidence_factor = finding.confidence
            total_impact += weight * confidence_factor
        
        # Calculate score (0-1 range)
        score = max(0.0, min(1.0, 1.0 + total_impact))
        return score
    
    async def _calculate_complexity_metrics(self, file_path: str, content: str) -> Dict[str, float]:
        """Calculate various complexity metrics for the file."""
        metrics = {}
        
        try:
            if file_path.endswith('.py'):
                tree = ast.parse(content)
                
                # Basic metrics
                metrics['lines_of_code'] = len([line for line in content.split('\n') if line.strip()])
                metrics['functions'] = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                metrics['classes'] = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                
                # Average complexity
                complexities = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        complexities.append(complexity)
                
                metrics['avg_complexity'] = sum(complexities) / len(complexities) if complexities else 0
                metrics['max_complexity'] = max(complexities) if complexities else 0
            
            # Language-agnostic metrics
            metrics['file_size_bytes'] = len(content.encode('utf-8'))
            metrics['comment_ratio'] = self._calculate_comment_ratio(content, file_path)
            
        except Exception as e:
            logger.warning(f"Failed to calculate complexity metrics for {file_path}: {e}")
        
        return metrics
    
    def _calculate_comment_ratio(self, content: str, file_path: str) -> float:
        """Calculate the ratio of comments to total lines."""
        lines = content.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines == 0:
            return 0.0
        
        comment_lines = 0
        if file_path.endswith('.py'):
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            comment_lines = len([line for line in lines if line.strip().startswith('//')])
        
        return comment_lines / total_lines if total_lines > 0 else 0.0
    
    async def _calculate_test_coverage(self, file_path: str) -> float:
        """Calculate test coverage for the file (if possible)."""
        try:
            # This would integrate with coverage tools in practice
            # For now, return a placeholder based on test file existence
            
            test_file_patterns = [
                f"test_{Path(file_path).stem}.py",
                f"{Path(file_path).stem}_test.py",
                f"tests/test_{Path(file_path).stem}.py"
            ]
            
            test_file_exists = any(
                (self.project_root / pattern).exists() 
                for pattern in test_file_patterns
            )
            
            # Simple heuristic - would integrate with actual coverage tools
            return 0.8 if test_file_exists else 0.0
            
        except Exception:
            return 0.0
    
    async def auto_fix_findings(self, findings: List[CodeReviewFinding]) -> Dict[str, Any]:
        """Automatically fix findings that are marked as auto-fixable."""
        fixed_count = 0
        failed_count = 0
        fixes_applied = []
        
        for finding in findings:
            if finding.auto_fixable and finding.suggested_fix:
                try:
                    # Apply the fix
                    success = await self._apply_fix(finding)
                    if success:
                        fixed_count += 1
                        fixes_applied.append({
                            "finding_id": finding.id,
                            "file_path": finding.file_path,
                            "line_number": finding.line_number,
                            "fix": finding.suggested_fix
                        })
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Failed to apply auto-fix for {finding.id}: {e}")
                    failed_count += 1
        
        return {
            "total_fixable": len([f for f in findings if f.auto_fixable]),
            "fixed_count": fixed_count,
            "failed_count": failed_count,
            "fixes_applied": fixes_applied
        }
    
    async def _apply_fix(self, finding: CodeReviewFinding) -> bool:
        """Apply an automatic fix for a finding."""
        try:
            # Read current file content
            with open(finding.file_path, 'r') as f:
                lines = f.readlines()
            
            # Apply fix based on finding type
            if finding.category == ReviewCategory.STYLE:
                if "trailing whitespace" in finding.description.lower():
                    # Remove trailing whitespace
                    if finding.line_number <= len(lines):
                        lines[finding.line_number - 1] = lines[finding.line_number - 1].rstrip() + '\n'
                        
                        # Write back to file
                        with open(finding.file_path, 'w') as f:
                            f.writelines(lines)
                        
                        return True
            
            # Add more auto-fix implementations as needed
            
        except Exception as e:
            logger.error(f"Auto-fix failed for {finding.id}: {e}")
        
        return False
    
    def get_review_summary(self) -> Dict[str, Any]:
        """Get summary of recent code reviews."""
        if not self.review_history:
            return {"message": "No reviews completed yet"}
        
        recent_reviews = self.review_history[-10:]  # Last 10 reviews
        
        return {
            "total_reviews": len(self.review_history),
            "recent_reviews": len(recent_reviews),
            "avg_files_per_review": sum(r["files_reviewed"] for r in recent_reviews) / len(recent_reviews),
            "avg_findings_per_review": sum(r["total_findings"] for r in recent_reviews) / len(recent_reviews),
            "avg_quality_score": sum(r["avg_score"] for r in recent_reviews) / len(recent_reviews),
            "pattern_cache_size": len(self.pattern_cache),
            "knowledge_base_entries": len(self.knowledge_base)
        }