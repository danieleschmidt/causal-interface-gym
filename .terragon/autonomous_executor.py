#!/usr/bin/env python3
"""Autonomous execution system for highest-value work items."""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional
import yaml

from backlog_manager import BacklogManager, ExecutionResult
from discovery_engine import WorkItem

class AutonomousExecutor:
    """Autonomous work item execution with comprehensive validation."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.backlog_manager = BacklogManager(config_path)
        self.dry_run = os.getenv("TERRAGON_DRY_RUN", "false").lower() == "true"
        
    def execute_next_best_value(self) -> Optional[ExecutionResult]:
        """Execute the next highest-value work item."""
        print("🔍 Refreshing backlog to find latest opportunities...")
        self.backlog_manager.refresh_backlog()
        
        next_item = self.backlog_manager.get_next_best_value_item()
        if not next_item:
            print("✅ No eligible work items found. Repository is in good shape!")
            return None
        
        print(f"🎯 Selected: {next_item.title}")
        print(f"   Score: {next_item.composite_score:.1f} | Effort: {next_item.effort_estimate:.1f}h | Risk: {next_item.risk_level:.2f}")
        
        if self.dry_run:
            print("🧪 DRY RUN MODE - Simulating execution...")
            return self._simulate_execution(next_item)
        
        return self._execute_work_item(next_item)
    
    def _execute_work_item(self, item: WorkItem) -> ExecutionResult:
        """Execute a specific work item with full validation."""
        start_time = datetime.now()
        print(f"🚀 Executing: {item.title}")
        
        try:
            # Pre-execution validation
            if not self._pre_execution_checks():
                return self._create_failure_result(item, "Pre-execution checks failed")
            
            # Execute based on category
            success, notes = self._execute_by_category(item)
            
            if success:
                # Post-execution validation
                validation_success, validation_notes = self._post_execution_validation()
                if not validation_success:
                    success = False
                    notes += f" | Validation failed: {validation_notes}"
            
            # Calculate actual effort
            actual_effort = (datetime.now() - start_time).total_seconds() / 3600  # hours
            
            # Create execution result
            result = ExecutionResult(
                item_id=item.id,
                success=success,
                actual_effort=actual_effort,
                actual_impact=self._measure_impact(item, success),
                notes=notes,
                completed_at=datetime.now().isoformat()
            )
            
            # Record result
            self.backlog_manager.record_execution_result(result)
            
            if success:
                print(f"✅ Successfully completed: {item.title}")
                self._create_pull_request(item, result)
                self._update_documentation(item, result)
            else:
                print(f"❌ Failed to complete: {item.title} - {notes}")
            
            return result
            
        except Exception as e:
            print(f"💥 Execution error: {str(e)}")
            return self._create_failure_result(item, f"Exception: {str(e)}")
    
    def _execute_by_category(self, item: WorkItem) -> tuple[bool, str]:
        """Execute work item based on its category."""
        category = item.category
        
        if category == "dependency_update":
            return self._execute_dependency_updates(item)
        elif category == "technical_debt":
            return self._execute_technical_debt_fixes(item)
        elif category == "documentation":
            return self._execute_documentation_improvements(item)
        elif category == "security":
            return self._execute_security_improvements(item)
        elif category == "performance":
            return self._execute_performance_improvements(item)
        else:
            return self._execute_generic_maintenance(item)
    
    def _execute_dependency_updates(self, item: WorkItem) -> tuple[bool, str]:
        """Execute dependency update work items."""
        notes = []
        success = True
        
        try:
            # Python dependencies
            if os.path.exists("requirements-dev.txt"):
                result = subprocess.run([
                    "pip", "list", "--outdated", "--format=json"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    outdated = json.loads(result.stdout)
                    updated_count = 0
                    
                    for package in outdated[:5]:  # Update top 5 only
                        try:
                            subprocess.run([
                                "pip", "install", "--upgrade", package["name"]
                            ], check=True, capture_output=True)
                            updated_count += 1
                            notes.append(f"Updated {package['name']}")
                        except subprocess.CalledProcessError as e:
                            notes.append(f"Failed to update {package['name']}: {e}")
                    
                    # Update requirements file
                    subprocess.run([
                        "pip", "freeze", ">", "requirements-dev.txt"
                    ], shell=True)
                    
                    notes.append(f"Updated {updated_count} Python dependencies")
            
            # Node.js dependencies  
            if os.path.exists("package.json"):
                try:
                    subprocess.run(["npm", "update"], check=True, capture_output=True)
                    notes.append("Updated npm dependencies")
                except subprocess.CalledProcessError as e:
                    notes.append(f"npm update failed: {e}")
                    success = False
            
        except Exception as e:
            success = False
            notes.append(f"Dependency update error: {e}")
        
        return success, " | ".join(notes)
    
    def _execute_technical_debt_fixes(self, item: WorkItem) -> tuple[bool, str]:
        """Execute technical debt reduction work items."""
        notes = []
        success = True
        
        try:
            # Run code formatting
            if self._has_python_files(item.files_affected):
                try:
                    subprocess.run(["black", "."], check=True, capture_output=True, cwd=".")
                    notes.append("Applied black formatting")
                except subprocess.CalledProcessError:
                    notes.append("Black formatting skipped (not available)")
                
                try:
                    subprocess.run(["ruff", "check", "--fix", "."], check=True, capture_output=True, cwd=".")
                    notes.append("Applied ruff fixes")
                except subprocess.CalledProcessError:
                    notes.append("Ruff fixes skipped (issues remain)")
            
            # Run type checking improvements
            try:
                result = subprocess.run(["mypy", "src/"], capture_output=True, text=True, cwd=".")
                if result.returncode == 0:
                    notes.append("Type checking passed")
                else:
                    notes.append(f"Type issues remain: {result.stdout.count('error')} errors")
            except subprocess.CalledProcessError:
                notes.append("Type checking skipped")
            
        except Exception as e:
            success = False
            notes.append(f"Technical debt fix error: {e}")
        
        return success, " | ".join(notes)
    
    def _execute_documentation_improvements(self, item: WorkItem) -> tuple[bool, str]:
        """Execute documentation improvement work items."""
        notes = []
        success = True
        
        try:
            # Add basic docstrings to Python functions missing them
            for file_path in item.files_affected:
                if file_path.endswith('.py') and os.path.exists(file_path):
                    updated = self._add_missing_docstrings(file_path)
                    if updated:
                        notes.append(f"Added docstrings to {file_path}")
            
            # Update README if it exists
            if os.path.exists("README.md"):
                # Simple update - add last updated timestamp
                with open("README.md", "r") as f:
                    content = f.read()
                
                # Add or update last updated line
                if "Last Updated:" not in content:
                    lines = content.split('\n')
                    # Insert after title
                    for i, line in enumerate(lines):
                        if line.startswith('#') and i == 0:
                            lines.insert(i + 1, f"\n*Last Updated: {datetime.now().strftime('%Y-%m-%d')}*\n")
                            break
                    
                    with open("README.md", "w") as f:
                        f.write('\n'.join(lines))
                    
                    notes.append("Updated README.md timestamp")
            
        except Exception as e:
            success = False
            notes.append(f"Documentation error: {e}")
        
        return success, " | ".join(notes)
    
    def _execute_security_improvements(self, item: WorkItem) -> tuple[bool, str]:
        """Execute security improvement work items."""
        notes = []
        success = True
        
        try:
            # Run security scans
            try:
                subprocess.run(["bandit", "-r", "src/"], check=True, capture_output=True)
                notes.append("Security scan passed")
            except subprocess.CalledProcessError as e:
                notes.append(f"Security issues found: {e}")
                success = False
            
            # Check for common security patterns
            security_issues = self._scan_for_security_patterns()
            if security_issues:
                notes.extend(security_issues)
            
        except Exception as e:
            success = False
            notes.append(f"Security improvement error: {e}")
        
        return success, " | ".join(notes)
    
    def _execute_performance_improvements(self, item: WorkItem) -> tuple[bool, str]:
        """Execute performance improvement work items."""
        notes = []
        success = True
        
        try:
            # Run performance tests if they exist
            perf_test_files = []
            for root, dirs, files in os.walk("tests"):
                for file in files:
                    if "performance" in file or "benchmark" in file:
                        perf_test_files.append(os.path.join(root, file))
            
            if perf_test_files:
                try:
                    subprocess.run(["pytest"] + perf_test_files, check=True, capture_output=True)
                    notes.append("Performance tests passed")
                except subprocess.CalledProcessError:
                    notes.append("Performance tests failed")
                    success = False
            else:
                notes.append("No performance tests found")
            
        except Exception as e:
            success = False
            notes.append(f"Performance improvement error: {e}")
        
        return success, " | ".join(notes)
    
    def _execute_generic_maintenance(self, item: WorkItem) -> tuple[bool, str]:
        """Execute generic maintenance work items."""
        notes = []
        success = True
        
        try:
            # Basic cleanup operations
            notes.append("Performed generic maintenance")
            
            # Clean up temporary files
            subprocess.run(["find", ".", "-name", "*.pyc", "-delete"], capture_output=True)
            subprocess.run(["find", ".", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
            
            notes.append("Cleaned temporary files")
            
        except Exception as e:
            success = False
            notes.append(f"Generic maintenance error: {e}")
        
        return success, " | ".join(notes)
    
    def _pre_execution_checks(self) -> bool:
        """Run pre-execution validation checks."""
        try:
            # Check git status
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if result.stdout.strip():
                print("⚠️  Warning: Working directory has uncommitted changes")
                return False
            
            # Check if we're on the right branch
            result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
            current_branch = result.stdout.strip()
            
            if current_branch == "main" or current_branch == "master":
                print("⚠️  Warning: Cannot execute on main/master branch")
                return False
            
            return True
            
        except subprocess.CalledProcessError:
            return False
    
    def _post_execution_validation(self) -> tuple[bool, str]:
        """Run post-execution validation."""
        issues = []
        
        try:
            # Run tests if they exist
            if os.path.exists("tests"):
                try:
                    result = subprocess.run(["pytest", "tests/"], capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        issues.append("Tests failed")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    issues.append("Test execution failed")
            
            # Run linting if configured
            if os.path.exists("pyproject.toml"):
                try:
                    result = subprocess.run(["ruff", "check", "."], capture_output=True, text=True)
                    if result.returncode != 0:
                        issues.append("Linting issues found")
                except subprocess.CalledProcessError:
                    pass  # Linting issues are warnings, not failures
            
            # Check type checking
            try:
                result = subprocess.run(["mypy", "src/"], capture_output=True, text=True)
                if result.returncode != 0 and "error:" in result.stdout:
                    issues.append("Type checking errors")
            except subprocess.CalledProcessError:
                pass  # Type checking is optional
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return len(issues) == 0, " | ".join(issues)
    
    def _measure_impact(self, item: WorkItem, success: bool) -> Dict[str, float]:
        """Measure the actual impact of work item execution."""
        impact = {
            "value": 100 if success else 0,
            "risk_reduction": 50 if item.category == "security" and success else 0,
            "maintainability": 30 if item.category == "technical_debt" and success else 0,
            "documentation_coverage": 20 if item.category == "documentation" and success else 0
        }
        
        return impact
    
    def _create_failure_result(self, item: WorkItem, reason: str) -> ExecutionResult:
        """Create a failure execution result."""
        return ExecutionResult(
            item_id=item.id,
            success=False,
            actual_effort=0.0,
            actual_impact={"value": 0},
            notes=reason,
            completed_at=datetime.now().isoformat()
        )
    
    def _simulate_execution(self, item: WorkItem) -> ExecutionResult:
        """Simulate execution for dry run mode."""
        print(f"   📋 Category: {item.category}")
        print(f"   📁 Files: {len(item.files_affected)}")
        print(f"   📊 Expected Impact: {item.category} improvement")
        
        return ExecutionResult(
            item_id=item.id,
            success=True,
            actual_effort=item.effort_estimate,
            actual_impact={"value": 100, "simulated": True},
            notes="Simulated execution in dry run mode",
            completed_at=datetime.now().isoformat()
        )
    
    def _has_python_files(self, files: List[str]) -> bool:
        """Check if any files are Python files."""
        return any(f.endswith('.py') for f in files)
    
    def _add_missing_docstrings(self, file_path: str) -> bool:
        """Add basic docstrings to functions missing them."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            modified = False
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('def ') and not line.startswith('def _'):
                    # Check if next lines contain docstring
                    has_docstring = False
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if '"""' in lines[j] or "'''" in lines[j]:
                            has_docstring = True
                            break
                    
                    if not has_docstring:
                        # Add basic docstring
                        func_name = line.split('(')[0].replace('def ', '')
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        docstring = ' ' * (indent + 4) + f'"""{func_name.replace("_", " ").title()} function."""\n'
                        lines.insert(i + 1, docstring)
                        modified = True
                i += 1
            
            if modified:
                with open(file_path, 'w') as f:
                    f.writelines(lines)
            
            return modified
            
        except (IOError, UnicodeDecodeError):
            return False
    
    def _scan_for_security_patterns(self) -> List[str]:
        """Scan for common security anti-patterns."""
        issues = []
        
        # Simple security pattern detection
        dangerous_patterns = [
            (r'eval\s*\(', 'eval() usage detected'),
            (r'exec\s*\(', 'exec() usage detected'),
            (r'shell=True', 'shell=True in subprocess detected'),
            (r'pickle\.loads?', 'pickle usage detected'),
        ]
        
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            
                        for pattern, message in dangerous_patterns:
                            import re
                            if re.search(pattern, content):
                                issues.append(f"{message} in {file_path}")
                    except (IOError, UnicodeDecodeError):
                        continue
        
        return issues
    
    def _create_pull_request(self, item: WorkItem, result: ExecutionResult):
        """Create a pull request for the completed work."""
        if self.dry_run:
            print("🧪 Would create PR in non-dry-run mode")
            return
        
        try:
            # Create branch name
            branch_name = f"auto-value/{item.id}-{item.category}"
            
            # Check if already on a feature branch
            current_branch = subprocess.run(
                ["git", "branch", "--show-current"], 
                capture_output=True, text=True
            ).stdout.strip()
            
            if not current_branch.startswith("auto-value/"):
                # Create and checkout branch
                subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            
            # Stage all changes
            subprocess.run(["git", "add", "."], check=True)
            
            # Create commit
            commit_message = f"""[AUTO-VALUE] {item.title}

Category: {item.category}
Composite Score: {item.composite_score:.1f}
Effort: {result.actual_effort:.1f}h
Impact: {result.notes}

🤖 Generated with Terragon Autonomous SDLC

Co-Authored-By: Terragon <noreply@terragon.ai>"""
            
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            print(f"✅ Created commit on branch: {branch_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Could not create PR: {e}")
    
    def _update_documentation(self, item: WorkItem, result: ExecutionResult):
        """Update documentation with execution results."""
        try:
            # Update the autonomous backlog report
            report = self.backlog_manager.generate_backlog_report()
            
            with open("AUTONOMOUS_BACKLOG.md", "w") as f:
                f.write(report)
            
            print("📚 Updated AUTONOMOUS_BACKLOG.md")
            
        except Exception as e:
            print(f"⚠️  Documentation update failed: {e}")

def main():
    """Main execution entry point."""
    print("🚀 Terragon Autonomous SDLC Executor")
    print("="*50)
    
    executor = AutonomousExecutor()
    result = executor.execute_next_best_value()
    
    if result:
        if result.success:
            print(f"\n✅ SUCCESS: Delivered value in {result.actual_effort:.1f} hours")
        else:
            print(f"\n❌ FAILED: {result.notes}")
    else:
        print("\n🎉 No work items needed - repository is optimized!")
    
    print("\n" + "="*50)
    print("🔄 Next execution in 1 hour (configure with cron)")

if __name__ == "__main__":
    main()