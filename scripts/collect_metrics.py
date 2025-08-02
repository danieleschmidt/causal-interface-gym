#!/usr/bin/env python3
"""
Automated metrics collection script for Causal Interface Gym.

This script collects various project metrics and updates the project-metrics.json file.
Run this script weekly to maintain up-to-date project health metrics.

Usage:
    python scripts/collect_metrics.py
    python scripts/collect_metrics.py --update-github
    python scripts/collect_metrics.py --format json|yaml|csv
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import requests
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import causal_interface_gym
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    github_token: Optional[str] = None
    repo_owner: str = "danieleschmidt"
    repo_name: str = "causal-interface-gym"
    base_path: Path = Path(__file__).parent.parent
    metrics_file: Path = Path(__file__).parent.parent / ".github" / "project-metrics.json"


class MetricsCollector:
    """Collects various project metrics."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics = self._load_existing_metrics()
    
    def _load_existing_metrics(self) -> Dict[str, Any]:
        """Load existing metrics or create default structure."""
        if self.config.metrics_file.exists():
            with open(self.config.metrics_file) as f:
                return json.load(f)
        else:
            return self._default_metrics_structure()
    
    def _default_metrics_structure(self) -> Dict[str, Any]:
        """Create default metrics structure."""
        return {
            "project": {"name": "Causal Interface Gym"},
            "metrics": {
                "code": {},
                "repository": {},
                "dependencies": {},
                "performance": {},
                "security": {}
            },
            "benchmarks": {},
            "health": {"overall_score": 0}
        }
    
    def collect_code_metrics(self):
        """Collect code quality and complexity metrics."""
        print("📊 Collecting code metrics...")
        
        try:
            # Count lines of code
            python_files = list(self.config.base_path.rglob("src/**/*.py"))
            ts_files = list(self.config.base_path.rglob("*.ts")) + list(self.config.base_path.rglob("*.tsx"))
            js_files = list(self.config.base_path.rglob("*.js")) + list(self.config.base_path.rglob("*.jsx"))
            
            python_loc = sum(len(f.read_text().splitlines()) for f in python_files if f.exists())
            ts_loc = sum(len(f.read_text().splitlines()) for f in ts_files if f.exists())
            js_loc = sum(len(f.read_text().splitlines()) for f in js_files if f.exists())
            
            self.metrics["metrics"]["code"]["lines_of_code"] = {
                "python": python_loc,
                "typescript": ts_loc,
                "javascript": js_loc,
                "total": python_loc + ts_loc + js_loc
            }
            
            # Test coverage (if coverage report exists)
            coverage_file = self.config.base_path / "htmlcov" / "index.html"
            if coverage_file.exists():
                coverage_text = coverage_file.read_text()
                # Extract coverage percentage (simplified)
                import re
                coverage_match = re.search(r'(\d+)%', coverage_text)
                if coverage_match:
                    coverage = int(coverage_match.group(1))
                    self.metrics["metrics"]["code"]["test_coverage"]["current"] = coverage
            
            print(f"   Python LOC: {python_loc}")
            print(f"   TypeScript LOC: {ts_loc}")
            print(f"   JavaScript LOC: {js_loc}")
            
        except Exception as e:
            print(f"   ⚠️  Error collecting code metrics: {e}")
    
    def collect_repository_metrics(self):
        """Collect GitHub repository metrics."""
        print("📈 Collecting repository metrics...")
        
        if not self.config.github_token:
            print("   ⚠️  No GitHub token provided, skipping repository metrics")
            return
        
        try:
            headers = {"Authorization": f"token {self.config.github_token}"}
            repo_url = f"https://api.github.com/repos/{self.config.repo_owner}/{self.config.repo_name}"
            
            # Repository stats
            response = requests.get(repo_url, headers=headers)
            if response.status_code == 200:
                repo_data = response.json()
                self.metrics["metrics"]["repository"].update({
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "watchers": repo_data.get("watchers_count", 0),
                    "open_issues": repo_data.get("open_issues_count", 0)
                })
                
                print(f"   Stars: {repo_data.get('stargazers_count', 0)}")
                print(f"   Forks: {repo_data.get('forks_count', 0)}")
                print(f"   Issues: {repo_data.get('open_issues_count', 0)}")
            
            # Pull requests
            pr_url = f"{repo_url}/pulls"
            pr_response = requests.get(pr_url, headers=headers)
            if pr_response.status_code == 200:
                prs = pr_response.json()
                self.metrics["metrics"]["repository"]["open_prs"] = len(prs)
                print(f"   Open PRs: {len(prs)}")
            
            # Contributors
            contributors_url = f"{repo_url}/contributors"
            contributors_response = requests.get(contributors_url, headers=headers)
            if contributors_response.status_code == 200:
                contributors = contributors_response.json()
                self.metrics["metrics"]["repository"]["total_contributors"] = len(contributors)
                print(f"   Contributors: {len(contributors)}")
            
            # Commits (last 100)
            commits_url = f"{repo_url}/commits"
            commits_response = requests.get(commits_url, headers=headers, params={"per_page": 100})
            if commits_response.status_code == 200:
                commits = commits_response.json()
                self.metrics["metrics"]["repository"]["total_commits"] = len(commits)
                print(f"   Recent commits: {len(commits)}")
        
        except Exception as e:
            print(f"   ⚠️  Error collecting repository metrics: {e}")
    
    def collect_dependency_metrics(self):
        """Collect dependency information."""
        print("📦 Collecting dependency metrics...")
        
        try:
            # Python dependencies
            if (self.config.base_path / "pyproject.toml").exists():
                result = subprocess.run(
                    ["pip", "list", "--format=json"],
                    capture_output=True,
                    text=True,
                    cwd=self.config.base_path
                )
                if result.returncode == 0:
                    packages = json.loads(result.stdout)
                    self.metrics["metrics"]["dependencies"]["python"]["total"] = len(packages)
                    print(f"   Python packages: {len(packages)}")
                
                # Check for outdated packages
                outdated_result = subprocess.run(
                    ["pip", "list", "--outdated", "--format=json"],
                    capture_output=True,
                    text=True,
                    cwd=self.config.base_path
                )
                if outdated_result.returncode == 0:
                    try:
                        outdated = json.loads(outdated_result.stdout)
                        self.metrics["metrics"]["dependencies"]["python"]["outdated"] = len(outdated)
                        print(f"   Outdated Python packages: {len(outdated)}")
                    except json.JSONDecodeError:
                        self.metrics["metrics"]["dependencies"]["python"]["outdated"] = 0
            
            # JavaScript dependencies
            if (self.config.base_path / "package.json").exists():
                with open(self.config.base_path / "package.json") as f:
                    package_json = json.load(f)
                    deps = package_json.get("dependencies", {})
                    dev_deps = package_json.get("devDependencies", {})
                    total_js_deps = len(deps) + len(dev_deps)
                    self.metrics["metrics"]["dependencies"]["javascript"]["total"] = total_js_deps
                    print(f"   JavaScript packages: {total_js_deps}")
        
        except Exception as e:
            print(f"   ⚠️  Error collecting dependency metrics: {e}")
    
    def collect_performance_metrics(self):
        """Collect performance benchmarks."""
        print("⚡ Collecting performance metrics...")
        
        try:
            # Docker image size
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Size}}", "causal-interface-gym:latest"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                size_str = result.stdout.strip()
                # Convert size to MB (simplified)
                if "MB" in size_str:
                    size_mb = float(size_str.replace("MB", ""))
                elif "GB" in size_str:
                    size_mb = float(size_str.replace("GB", "")) * 1024
                else:
                    size_mb = 0
                
                self.metrics["metrics"]["performance"]["docker_image_size"] = {
                    "mb": size_mb,
                    "trend": "stable"
                }
                print(f"   Docker image size: {size_mb} MB")
            
            # Run basic performance tests if package available
            if PACKAGE_AVAILABLE:
                import time
                from causal_interface_gym.core import CausalEnvironment
                
                # Test small graph performance
                start_time = time.time()
                small_dag = {"A": [], "B": ["A"], "C": ["B"]}
                env = CausalEnvironment.from_dag(small_dag)
                result = env.intervene(A=1)
                small_graph_time = (time.time() - start_time) * 1000
                
                self.metrics["benchmarks"]["causal_inference"]["small_graph_ms"] = {
                    "target": 100,
                    "current": round(small_graph_time, 2),
                    "trend": "stable"
                }
                print(f"   Small graph inference: {small_graph_time:.2f}ms")
        
        except Exception as e:
            print(f"   ⚠️  Error collecting performance metrics: {e}")
    
    def collect_security_metrics(self):
        """Collect security scan results."""
        print("🔒 Collecting security metrics...")
        
        try:
            # Run bandit for security issues
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                cwd=self.config.base_path
            )
            
            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    bandit_data = json.loads(result.stdout)
                    results = bandit_data.get("results", [])
                    
                    # Count by severity
                    high_count = sum(1 for r in results if r.get("issue_severity") == "HIGH")
                    medium_count = sum(1 for r in results if r.get("issue_severity") == "MEDIUM")
                    low_count = sum(1 for r in results if r.get("issue_severity") == "LOW")
                    
                    self.metrics["metrics"]["security"]["vulnerabilities"] = {
                        "critical": 0,  # Bandit doesn't use critical
                        "high": high_count,
                        "medium": medium_count,
                        "low": low_count
                    }
                    
                    # Calculate security score (100 - weighted issues)
                    security_score = max(0, 100 - (high_count * 10 + medium_count * 5 + low_count * 1))
                    self.metrics["metrics"]["security"]["security_score"] = security_score
                    
                    print(f"   Security issues - High: {high_count}, Medium: {medium_count}, Low: {low_count}")
                    print(f"   Security score: {security_score}/100")
                    
                except json.JSONDecodeError:
                    print("   ⚠️  Could not parse bandit output")
            
            self.metrics["metrics"]["security"]["last_scan"] = datetime.now(timezone.utc).isoformat()
        
        except FileNotFoundError:
            print("   ⚠️  Bandit not available, skipping security scan")
        except Exception as e:
            print(f"   ⚠️  Error collecting security metrics: {e}")
    
    def calculate_health_score(self):
        """Calculate overall project health score."""
        print("💚 Calculating health score...")
        
        try:
            scores = {}
            
            # Code quality score
            code_metrics = self.metrics["metrics"]["code"]
            coverage = code_metrics.get("test_coverage", {}).get("current", 0)
            scores["code_quality"] = min(100, coverage + 20)  # Coverage + bonus for good practices
            
            # Security score
            security_score = self.metrics["metrics"]["security"].get("security_score", 100)
            scores["security"] = security_score
            
            # Performance score (based on benchmarks)
            performance_scores = []
            benchmarks = self.metrics.get("benchmarks", {})
            for category, tests in benchmarks.items():
                for test_name, test_data in tests.items():
                    if isinstance(test_data, dict) and "target" in test_data and "current" in test_data:
                        target = test_data["target"]
                        current = test_data["current"]
                        if current > 0 and target > 0:
                            # Score: 100 if at target, decreases as performance gets worse
                            perf_score = min(100, (target / current) * 100)
                            performance_scores.append(perf_score)
            
            scores["performance"] = sum(performance_scores) / len(performance_scores) if performance_scores else 80
            
            # Documentation score (based on file presence)
            doc_files = ["README.md", "CONTRIBUTING.md", "docs/"]
            doc_score = sum(1 for f in doc_files if (self.config.base_path / f).exists()) / len(doc_files) * 100
            scores["documentation"] = doc_score
            
            # Testing score (based on test file presence and coverage)
            test_files = list(self.config.base_path.rglob("test*.py")) + list(self.config.base_path.rglob("*test.py"))
            testing_score = min(100, len(test_files) * 10 + coverage)
            scores["testing"] = testing_score
            
            # Maintainability (based on code structure)
            scores["maintainability"] = 85  # Placeholder - could be improved with complexity metrics
            
            # Community (based on GitHub metrics)
            repo_metrics = self.metrics["metrics"]["repository"]
            stars = repo_metrics.get("stars", 0)
            forks = repo_metrics.get("forks", 0)
            contributors = repo_metrics.get("total_contributors", 0)
            community_score = min(100, stars * 2 + forks * 5 + contributors * 10)
            scores["community"] = community_score
            
            # Overall score (weighted average)
            weights = {
                "code_quality": 0.20,
                "security": 0.20,
                "performance": 0.15,
                "documentation": 0.15,
                "testing": 0.15,
                "maintainability": 0.10,
                "community": 0.05
            }
            
            overall_score = sum(scores[category] * weights[category] for category in scores)
            
            self.metrics["health"] = {
                "overall_score": round(overall_score, 1),
                "categories": {k: round(v, 1) for k, v in scores.items()},
                "trend": "improving",  # Would need historical data for real trend
                "last_assessment": datetime.now(timezone.utc).isoformat()
            }
            
            print(f"   Overall health score: {overall_score:.1f}/100")
            for category, score in scores.items():
                print(f"   {category}: {score:.1f}")
        
        except Exception as e:
            print(f"   ⚠️  Error calculating health score: {e}")
    
    def update_metadata(self):
        """Update metadata fields."""
        now = datetime.now(timezone.utc)
        self.metrics["metadata"] = {
            "schema_version": "1.0",
            "generated_by": "terragon-sdlc-automation",
            "generation_date": now.isoformat(),
            "next_update": (now.replace(day=now.day + 7)).isoformat(),
            "update_frequency": "weekly"
        }
    
    def save_metrics(self):
        """Save metrics to file."""
        print(f"💾 Saving metrics to {self.config.metrics_file}")
        
        self.config.metrics_file.parent.mkdir(exist_ok=True)
        with open(self.config.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, sort_keys=True)
    
    def collect_all(self):
        """Collect all metrics."""
        print("🚀 Starting metrics collection...")
        
        self.collect_code_metrics()
        self.collect_repository_metrics()
        self.collect_dependency_metrics()
        self.collect_performance_metrics()
        self.collect_security_metrics()
        self.calculate_health_score()
        self.update_metadata()
        
        print("✅ Metrics collection complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--github-token", help="GitHub personal access token")
    parser.add_argument("--update-github", action="store_true", help="Update GitHub repository metrics")
    parser.add_argument("--format", choices=["json", "yaml", "csv"], default="json", help="Output format")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Get GitHub token from environment or argument
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    
    config = MetricsConfig(github_token=github_token)
    collector = MetricsCollector(config)
    
    collector.collect_all()
    collector.save_metrics()
    
    if args.output:
        # Save in requested format
        output_path = Path(args.output)
        if args.format == "json":
            with open(output_path, 'w') as f:
                json.dump(collector.metrics, f, indent=2)
        elif args.format == "yaml":
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(collector.metrics, f, default_flow_style=False)
        elif args.format == "csv":
            # Flatten metrics for CSV
            import csv
            flattened = {}
            
            def flatten_dict(d, parent_key='', sep='_'):
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        flatten_dict(v, new_key, sep=sep)
                    else:
                        flattened[new_key] = v
            
            flatten_dict(collector.metrics)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for key, value in flattened.items():
                    writer.writerow([key, value])
        
        print(f"📄 Metrics saved to {output_path} in {args.format} format")


if __name__ == "__main__":
    main()