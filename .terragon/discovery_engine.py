#!/usr/bin/env python3
"""Comprehensive signal harvesting and work item discovery engine."""

import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set
import yaml

from scoring_engine import ScoringEngine, WorkItem

@dataclass
class DiscoveredSignal:
    """Raw signal before processing into work item."""
    source: str
    type: str
    file_path: str
    line_number: Optional[int]
    content: str
    context: str
    severity: str
    discovered_at: str

class DiscoveryEngine:
    """Multi-source signal harvesting and work item generation."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.scoring_engine = ScoringEngine(config_path)
        self.signals: List[DiscoveredSignal] = []
        self.work_items: List[WorkItem] = []
    
    def discover_all_signals(self) -> List[DiscoveredSignal]:
        """Run comprehensive signal discovery from all sources."""
        self.signals = []
        
        # Git history analysis
        self.signals.extend(self._discover_git_signals())
        
        # Static analysis signals
        self.signals.extend(self._discover_static_analysis_signals())
        
        # Dependency analysis
        self.signals.extend(self._discover_dependency_signals())
        
        # Performance signals
        self.signals.extend(self._discover_performance_signals())
        
        # Documentation signals
        self.signals.extend(self._discover_documentation_signals())
        
        return self.signals
    
    def convert_signals_to_work_items(self, signals: List[DiscoveredSignal]) -> List[WorkItem]:
        """Convert discovered signals into scored work items."""
        work_items = []
        
        # Group related signals
        grouped_signals = self._group_related_signals(signals)
        
        for group_id, signal_group in grouped_signals.items():
            work_item = self._create_work_item_from_signals(group_id, signal_group)
            if work_item:
                work_items.append(work_item)
        
        # Sort by composite score
        work_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        return work_items
    
    def _discover_git_signals(self) -> List[DiscoveredSignal]:
        """Discover signals from Git history analysis."""
        signals = []
        
        try:
            # Find TODO/FIXME/HACK comments in recent commits
            result = subprocess.run([
                "git", "log", "--grep=TODO\\|FIXME\\|HACK\\|DEPRECATED", 
                "--oneline", "-n", "50"
            ], capture_output=True, text=True, cwd=".")
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash = line.split()[0]
                    message = ' '.join(line.split()[1:])
                    
                    signals.append(DiscoveredSignal(
                        source="git_history",
                        type="debt_marker",
                        file_path="",
                        line_number=None,
                        content=message,
                        context=f"Commit {commit_hash}",
                        severity="medium",
                        discovered_at=datetime.now().isoformat()
                    ))
            
            # Find quick fix patterns
            quick_fix_patterns = [
                "quick fix", "temporary", "hack", "workaround", 
                "hotfix", "band-aid", "kludge"
            ]
            
            for pattern in quick_fix_patterns:
                result = subprocess.run([
                    "git", "log", f"--grep={pattern}", 
                    "--oneline", "-n", "20"
                ], capture_output=True, text=True, cwd=".")
                
                for line in result.stdout.strip().split('\n'):
                    if line:
                        signals.append(DiscoveredSignal(
                            source="git_history",
                            type="quick_fix",
                            file_path="",
                            line_number=None,
                            content=line,
                            context=f"Quick fix pattern: {pattern}",
                            severity="high",
                            discovered_at=datetime.now().isoformat()
                        ))
        
        except subprocess.CalledProcessError:
            pass  # Git commands may fail in some environments
        
        return signals
    
    def _discover_static_analysis_signals(self) -> List[DiscoveredSignal]:
        """Discover signals from static code analysis."""
        signals = []
        
        # Find TODO/FIXME comments in source code
        todo_patterns = [
            r'TODO[:\s](.+)',
            r'FIXME[:\s](.+)', 
            r'HACK[:\s](.+)',
            r'XXX[:\s](.+)',
            r'DEPRECATED[:\s](.+)'
        ]
        
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.tsx')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                for pattern in todo_patterns:
                                    match = re.search(pattern, line, re.IGNORECASE)
                                    if match:
                                        signals.append(DiscoveredSignal(
                                            source="static_analysis",
                                            type="code_comment",
                                            file_path=file_path,
                                            line_number=line_num,
                                            content=match.group(1).strip(),
                                            context=line.strip(),
                                            severity="medium",
                                            discovered_at=datetime.now().isoformat()
                                        ))
                    except (UnicodeDecodeError, IOError):
                        continue
        
        # Complexity analysis using basic heuristics
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            lines = content.split('\n')
                            
                            # Detect high complexity functions (simple heuristic)
                            for i, line in enumerate(lines):
                                if line.strip().startswith('def '):
                                    # Count nested levels
                                    func_lines = []
                                    j = i + 1
                                    base_indent = len(line) - len(line.lstrip())
                                    
                                    while j < len(lines) and (
                                        not lines[j].strip() or 
                                        len(lines[j]) - len(lines[j].lstrip()) > base_indent
                                    ):
                                        func_lines.append(lines[j])
                                        j += 1
                                    
                                    # Simple complexity metric
                                    if len(func_lines) > 50:
                                        func_name = line.strip().split('(')[0].replace('def ', '')
                                        signals.append(DiscoveredSignal(
                                            source="static_analysis",
                                            type="high_complexity",
                                            file_path=file_path,
                                            line_number=i + 1,
                                            content=f"Function {func_name} has {len(func_lines)} lines",
                                            context="High complexity function",
                                            severity="high",
                                            discovered_at=datetime.now().isoformat()
                                        ))
                    except (IOError, UnicodeDecodeError):
                        continue
        
        return signals
    
    def _discover_dependency_signals(self) -> List[DiscoveredSignal]:
        """Discover dependency-related signals."""
        signals = []
        
        # Check for outdated Python dependencies
        if os.path.exists("requirements-dev.txt"):
            try:
                result = subprocess.run([
                    "pip", "list", "--outdated", "--format=json"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    outdated_packages = json.loads(result.stdout)
                    for package in outdated_packages[:10]:  # Limit to top 10
                        signals.append(DiscoveredSignal(
                            source="dependency_analysis",
                            type="outdated_dependency",
                            file_path="requirements-dev.txt",
                            line_number=None,
                            content=f"Update {package['name']} from {package['version']} to {package['latest_version']}",
                            context="Outdated dependency",
                            severity="medium",
                            discovered_at=datetime.now().isoformat()
                        ))
            except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Check for Node.js dependencies if package.json exists
        if os.path.exists("package.json"):
            try:
                result = subprocess.run([
                    "npm", "outdated", "--json"
                ], capture_output=True, text=True)
                
                if result.stdout:
                    outdated_npm = json.loads(result.stdout)
                    for package, info in list(outdated_npm.items())[:10]:
                        signals.append(DiscoveredSignal(
                            source="dependency_analysis",
                            type="outdated_npm_dependency",
                            file_path="package.json",
                            line_number=None,
                            content=f"Update {package} from {info.get('current', 'unknown')} to {info.get('latest', 'unknown')}",
                            context="Outdated npm dependency",
                            severity="medium",
                            discovered_at=datetime.now().isoformat()
                        ))
            except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
                pass
        
        return signals
    
    def _discover_performance_signals(self) -> List[DiscoveredSignal]:
        """Discover performance-related signals."""
        signals = []
        
        # Look for performance test files
        for root, dirs, files in os.walk("tests"):
            for file in files:
                if "performance" in file or "benchmark" in file:
                    file_path = os.path.join(root, file)
                    signals.append(DiscoveredSignal(
                        source="performance_analysis",
                        type="performance_test_exists",
                        file_path=file_path,
                        line_number=None,
                        content="Performance tests found - ensure they're run regularly",
                        context="Performance monitoring opportunity",
                        severity="low",
                        discovered_at=datetime.now().isoformat()
                    ))
        
        return signals
    
    def _discover_documentation_signals(self) -> List[DiscoveredSignal]:
        """Discover documentation-related signals."""
        signals = []
        
        # Check for missing docstrings in Python files
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            lines = content.split('\n')
                            
                            # Find functions without docstrings
                            for i, line in enumerate(lines):
                                if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                                    # Look for docstring in next few lines
                                    has_docstring = False
                                    for j in range(i + 1, min(i + 5, len(lines))):
                                        if '"""' in lines[j] or "'''" in lines[j]:
                                            has_docstring = True
                                            break
                                    
                                    if not has_docstring:
                                        func_name = line.strip().split('(')[0].replace('def ', '')
                                        signals.append(DiscoveredSignal(
                                            source="documentation_analysis",
                                            type="missing_docstring",
                                            file_path=file_path,
                                            line_number=i + 1,
                                            content=f"Function {func_name} missing docstring",
                                            context="Documentation improvement",
                                            severity="low",
                                            discovered_at=datetime.now().isoformat()
                                        ))
                    except (IOError, UnicodeDecodeError):
                        continue
        
        return signals
    
    def _group_related_signals(self, signals: List[DiscoveredSignal]) -> Dict[str, List[DiscoveredSignal]]:
        """Group related signals into work items."""
        groups = {}
        
        for signal in signals:
            # Group by type and file for related signals
            if signal.type == "outdated_dependency":
                group_key = "dependencies_update"
            elif signal.type == "high_complexity":
                group_key = f"refactor_{os.path.basename(signal.file_path)}"
            elif signal.type == "missing_docstring":
                group_key = f"document_{os.path.basename(signal.file_path)}"
            elif signal.type in ["debt_marker", "quick_fix"]:
                group_key = "technical_debt_cleanup"
            else:
                group_key = f"{signal.type}_{hash(signal.file_path) % 1000}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(signal)
        
        return groups
    
    def _create_work_item_from_signals(self, group_id: str, signals: List[DiscoveredSignal]) -> Optional[WorkItem]:
        """Create a work item from a group of related signals."""
        if not signals:
            return None
        
        # Determine category and effort based on signal patterns
        primary_signal = signals[0]
        
        if "dependencies" in group_id:
            category = "dependency_update"
            effort = len(signals) * 0.5  # 30 minutes per dependency
            title = f"Update {len(signals)} outdated dependencies"
        elif "refactor" in group_id:
            category = "technical_debt"
            effort = len(signals) * 2.0  # 2 hours per complex function
            title = f"Refactor high complexity functions in {os.path.basename(primary_signal.file_path)}"
        elif "document" in group_id:
            category = "documentation"
            effort = len(signals) * 0.25  # 15 minutes per missing docstring
            title = f"Add docstrings to {len(signals)} functions in {os.path.basename(primary_signal.file_path)}"
        elif "technical_debt" in group_id:
            category = "technical_debt"
            effort = len(signals) * 1.0  # 1 hour per debt item
            title = f"Address {len(signals)} technical debt items"
        else:
            category = "maintenance"
            effort = 2.0
            title = f"Address {primary_signal.type} issues"
        
        # Create item dictionary for scoring
        item_dict = {
            "id": f"auto_{hash(group_id) % 10000}",
            "title": title,
            "category": category,
            "description": f"Discovered {len(signals)} related signals: " + 
                          ", ".join([s.content[:50] + "..." if len(s.content) > 50 else s.content for s in signals[:3]]),
            "files_affected": list(set([s.file_path for s in signals if s.file_path])),
            "effort_estimate": effort
        }
        
        # Calculate scores
        wsjf_score = self.scoring_engine.calculate_wsjf(item_dict)
        ice_score = self.scoring_engine.calculate_ice(item_dict)
        debt_score = self.scoring_engine.calculate_technical_debt_score(item_dict)
        composite_score = self.scoring_engine.calculate_composite_score(item_dict)
        
        return WorkItem(
            id=item_dict["id"],
            title=title,
            description=item_dict["description"],
            category=category,
            files_affected=item_dict["files_affected"],
            effort_estimate=effort,
            wsjf_score=wsjf_score,
            ice_score=ice_score,
            technical_debt_score=debt_score,
            composite_score=composite_score,
            risk_level=self._calculate_risk_level(signals),
            discovered_at=datetime.now().isoformat(),
            source="autonomous_discovery"
        )
    
    def _calculate_risk_level(self, signals: List[DiscoveredSignal]) -> float:
        """Calculate risk level for a group of signals (0-1)."""
        high_severity_count = sum(1 for s in signals if s.severity == "high")
        total_count = len(signals)
        
        base_risk = 0.3
        severity_risk = (high_severity_count / total_count) * 0.4 if total_count > 0 else 0
        
        return min(1.0, base_risk + severity_risk)

if __name__ == "__main__":
    # Example usage
    engine = DiscoveryEngine()
    
    print("Discovering signals...")
    signals = engine.discover_all_signals()
    print(f"Found {len(signals)} signals")
    
    print("Converting to work items...")
    work_items = engine.convert_signals_to_work_items(signals)
    print(f"Generated {len(work_items)} work items")
    
    # Display top 5 work items
    for i, item in enumerate(work_items[:5]):
        print(f"\n{i+1}. {item.title}")
        print(f"   Score: {item.composite_score:.1f} | Category: {item.category}")
        print(f"   Effort: {item.effort_estimate:.1f}h | Risk: {item.risk_level:.2f}")
        print(f"   Files: {len(item.files_affected)}")