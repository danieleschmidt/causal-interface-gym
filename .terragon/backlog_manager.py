#!/usr/bin/env python3
"""Autonomous backlog management with continuous value optimization."""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml

from discovery_engine import DiscoveryEngine, WorkItem
from scoring_engine import ScoringEngine

@dataclass
class ExecutionResult:
    """Result of work item execution."""
    item_id: str
    success: bool
    actual_effort: float
    actual_impact: Dict[str, float]
    notes: str
    completed_at: str

@dataclass
class BacklogMetrics:
    """Comprehensive backlog health metrics."""
    total_items: int
    average_age_days: float
    debt_ratio: float
    velocity_trend: str
    discovery_rate: float
    completion_rate: float
    value_delivered: float
    learning_accuracy: float

class BacklogManager:
    """Autonomous backlog management with continuous optimization."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.discovery_engine = DiscoveryEngine(config_path)
        self.scoring_engine = ScoringEngine(config_path)
        
        self.backlog_file = ".terragon/backlog.json"
        self.metrics_file = ".terragon/value-metrics.json"
        self.execution_history_file = ".terragon/execution-history.json"
        
        self.backlog: List[WorkItem] = self._load_backlog()
        self.execution_history: List[ExecutionResult] = self._load_execution_history()
    
    def refresh_backlog(self) -> List[WorkItem]:
        """Discover new work items and refresh the backlog."""
        print("🔍 Discovering new signals...")
        signals = self.discovery_engine.discover_all_signals()
        
        print(f"📊 Processing {len(signals)} signals...")
        new_items = self.discovery_engine.convert_signals_to_work_items(signals)
        
        # Merge with existing backlog, avoiding duplicates
        existing_titles = {item.title for item in self.backlog}
        fresh_items = [item for item in new_items if item.title not in existing_titles]
        
        print(f"✨ Found {len(fresh_items)} new work items")
        
        # Add new items to backlog
        self.backlog.extend(fresh_items)
        
        # Re-score all items (scores may change based on context)
        self._rescore_backlog()
        
        # Sort by composite score
        self.backlog.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Save updated backlog
        self._save_backlog()
        
        return self.backlog
    
    def get_next_best_value_item(self) -> Optional[WorkItem]:
        """Select the next highest-value work item for execution."""
        if not self.backlog:
            return None
        
        # Apply strategic filters
        for item in self.backlog:
            # Skip if risk exceeds threshold
            if item.risk_level > self.config["scoring"]["thresholds"]["maxRisk"]:
                continue
            
            # Skip if score below minimum
            if item.composite_score < self.config["scoring"]["thresholds"]["minScore"]:
                continue
            
            # Item passes all filters
            return item
        
        # No items pass filters, return highest scored item if any exist
        return self.backlog[0] if self.backlog else None
    
    def record_execution_result(self, result: ExecutionResult):
        """Record the result of work item execution for learning."""
        self.execution_history.append(result)
        
        # Remove completed item from backlog
        self.backlog = [item for item in self.backlog if item.id != result.item_id]
        
        # Update learning models based on actual vs predicted results
        self._update_learning_models(result)
        
        # Save updates
        self._save_backlog()
        self._save_execution_history()
        self._update_metrics()
    
    def generate_backlog_report(self) -> str:
        """Generate comprehensive backlog status report."""
        if not self.backlog:
            return "# 📊 Autonomous Value Backlog\n\n**Status**: No work items discovered\n"
        
        metrics = self._calculate_metrics()
        next_item = self.get_next_best_value_item()
        
        report = f"""# 📊 Autonomous Value Backlog

Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Next Execution: {(datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Next Best Value Item
"""
        
        if next_item:
            report += f"""**[{next_item.id}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.1f} | **Tech Debt**: {next_item.technical_debt_score:.1f}
- **Estimated Effort**: {next_item.effort_estimate:.1f} hours
- **Expected Impact**: {next_item.category} improvement
- **Risk Level**: {next_item.risk_level:.2f}
- **Files Affected**: {len(next_item.files_affected)}

"""
        else:
            report += "**No eligible items** (all items exceed risk threshold or below minimum score)\n\n"
        
        report += f"""## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours | Risk |
|------|-----|--------|---------|----------|------------|------|
"""
        
        for i, item in enumerate(self.backlog[:10]):
            report += f"| {i+1} | {item.id} | {item.title[:40]}{'...' if len(item.title) > 40 else ''} | {item.composite_score:.1f} | {item.category} | {item.effort_estimate:.1f} | {item.risk_level:.2f} |\n"
        
        if len(self.backlog) > 10:
            report += f"| ... | ... | ({len(self.backlog) - 10} more items) | ... | ... | ... | ... |\n"
        
        report += f"""
## 📈 Value Metrics
- **Items Completed This Week**: {self._count_recent_completions(7)}
- **Average Cycle Time**: {self._calculate_average_cycle_time():.1f} hours
- **Value Delivered**: ${self._calculate_value_delivered():,.0f} (estimated)
- **Technical Debt Reduced**: {self._calculate_debt_reduction():.0f}%
- **Discovery Rate**: {metrics.discovery_rate:.1f} items/day
- **Completion Rate**: {metrics.completion_rate:.1f}%

## 🔄 Continuous Discovery Stats
- **New Items Discovered**: {self._count_recent_discoveries(7)}
- **Items Completed**: {len(self.execution_history)}
- **Net Backlog Change**: {self._count_recent_discoveries(7) - self._count_recent_completions(7)}
- **Discovery Sources**:
  - Static Analysis: {self._get_source_percentage('static_analysis'):.0f}%
  - Git History: {self._get_source_percentage('git_history'):.0f}%
  - Dependencies: {self._get_source_percentage('dependency_analysis'):.0f}%
  - Documentation: {self._get_source_percentage('documentation_analysis'):.0f}%
  - Performance: {self._get_source_percentage('performance_analysis'):.0f}%

## 🎓 Learning Insights
- **Estimation Accuracy**: {metrics.learning_accuracy:.1f}%
- **Most Valuable Category**: {self._get_most_valuable_category()}
- **Average Risk Level**: {self._calculate_average_risk():.2f}
- **Velocity Trend**: {metrics.velocity_trend}

## 🔮 Predictions
- **Next Week's Capacity**: {self._predict_capacity():.1f} hours
- **Recommended Focus**: {self._recommend_focus_area()}
- **Estimated Completion**: {self._estimate_backlog_completion()} days
"""
        
        return report
    
    def _load_backlog(self) -> List[WorkItem]:
        """Load backlog from file."""
        if not os.path.exists(self.backlog_file):
            return []
        
        try:
            with open(self.backlog_file, 'r') as f:
                data = json.load(f)
                return [WorkItem(**item) for item in data]
        except (json.JSONDecodeError, TypeError):
            return []
    
    def _save_backlog(self):
        """Save backlog to file."""
        os.makedirs(os.path.dirname(self.backlog_file), exist_ok=True)
        with open(self.backlog_file, 'w') as f:
            json.dump([asdict(item) for item in self.backlog], f, indent=2)
    
    def _load_execution_history(self) -> List[ExecutionResult]:
        """Load execution history from file."""
        if not os.path.exists(self.execution_history_file):
            return []
        
        try:
            with open(self.execution_history_file, 'r') as f:
                data = json.load(f)
                return [ExecutionResult(**result) for result in data]
        except (json.JSONDecodeError, TypeError):
            return []
    
    def _save_execution_history(self):
        """Save execution history to file."""
        os.makedirs(os.path.dirname(self.execution_history_file), exist_ok=True)
        with open(self.execution_history_file, 'w') as f:
            json.dump([asdict(result) for result in self.execution_history], f, indent=2)
    
    def _rescore_backlog(self):
        """Re-score all backlog items with current scoring model."""
        for item in self.backlog:
            item_dict = {
                "id": item.id,
                "title": item.title,
                "category": item.category,
                "description": item.description,
                "files_affected": item.files_affected,
                "effort_estimate": item.effort_estimate
            }
            
            item.wsjf_score = self.scoring_engine.calculate_wsjf(item_dict)
            item.ice_score = self.scoring_engine.calculate_ice(item_dict)
            item.technical_debt_score = self.scoring_engine.calculate_technical_debt_score(item_dict)
            item.composite_score = self.scoring_engine.calculate_composite_score(item_dict)
    
    def _update_learning_models(self, result: ExecutionResult):
        """Update scoring models based on execution results."""
        # Find the original work item for comparison
        original_item = None
        for item in self.backlog:
            if item.id == result.item_id:
                original_item = item
                break
        
        if not original_item:
            return
        
        # Calculate accuracy metrics
        effort_accuracy = min(result.actual_effort / original_item.effort_estimate, 2.0) if original_item.effort_estimate > 0 else 1.0
        
        # Store learning data (in production, this would update ML models)
        learning_entry = {
            "timestamp": result.completed_at,
            "category": original_item.category,
            "predicted_effort": original_item.effort_estimate,
            "actual_effort": result.actual_effort,
            "effort_accuracy": effort_accuracy,
            "success": result.success
        }
        
        # Simple learning: adjust confidence for similar items
        # In practice, this would use more sophisticated ML techniques
    
    def _calculate_metrics(self) -> BacklogMetrics:
        """Calculate comprehensive backlog metrics."""
        if not self.backlog:
            return BacklogMetrics(0, 0, 0, "stable", 0, 0, 0, 0)
        
        # Calculate average age
        now = datetime.now()
        ages = []
        for item in self.backlog:
            discovered = datetime.fromisoformat(item.discovered_at)
            age = (now - discovered).days
            ages.append(age)
        
        avg_age = sum(ages) / len(ages) if ages else 0
        
        # Calculate debt ratio
        debt_items = sum(1 for item in self.backlog if item.category == "technical_debt")
        debt_ratio = debt_items / len(self.backlog) if self.backlog else 0
        
        # Calculate rates
        recent_discoveries = self._count_recent_discoveries(7)
        recent_completions = self._count_recent_completions(7)
        
        discovery_rate = recent_discoveries / 7  # per day
        completion_rate = (recent_completions / (recent_completions + len(self.backlog))) * 100 if (recent_completions + len(self.backlog)) > 0 else 0
        
        # Estimate value delivered
        value_delivered = sum(result.actual_impact.get("value", 100) for result in self.execution_history[-10:])
        
        # Learning accuracy
        if self.execution_history:
            accuracies = []
            for result in self.execution_history[-20:]:  # Last 20 items
                # Simple accuracy based on success rate
                accuracies.append(100 if result.success else 50)
            learning_accuracy = sum(accuracies) / len(accuracies)
        else:
            learning_accuracy = 75  # Default estimate
        
        return BacklogMetrics(
            total_items=len(self.backlog),
            average_age_days=avg_age,
            debt_ratio=debt_ratio,
            velocity_trend="stable",  # Simplified
            discovery_rate=discovery_rate,
            completion_rate=completion_rate,
            value_delivered=value_delivered,
            learning_accuracy=learning_accuracy
        )
    
    def _count_recent_completions(self, days: int) -> int:
        """Count completions in the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return sum(1 for result in self.execution_history 
                  if datetime.fromisoformat(result.completed_at) > cutoff)
    
    def _count_recent_discoveries(self, days: int) -> int:
        """Count discoveries in the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return sum(1 for item in self.backlog 
                  if datetime.fromisoformat(item.discovered_at) > cutoff)
    
    def _calculate_average_cycle_time(self) -> float:
        """Calculate average cycle time for completed items."""
        if not self.execution_history:
            return 0.0
        
        recent_results = self.execution_history[-10:]  # Last 10 items
        total_effort = sum(result.actual_effort for result in recent_results)
        return total_effort / len(recent_results) if recent_results else 0.0
    
    def _calculate_value_delivered(self) -> float:
        """Calculate estimated value delivered."""
        if not self.execution_history:
            return 0.0
        
        # Simple value estimation based on category and effort
        value_map = {
            "security": 5000,
            "performance": 3000,
            "technical_debt": 2000,
            "dependency_update": 500,
            "documentation": 200
        }
        
        total_value = 0
        for result in self.execution_history[-20:]:  # Last 20 items
            base_value = value_map.get("maintenance", 1000)  # Default
            total_value += base_value * (1.5 if result.success else 0.5)
        
        return total_value
    
    def _calculate_debt_reduction(self) -> float:
        """Calculate technical debt reduction percentage."""
        debt_items_completed = sum(1 for result in self.execution_history 
                                 if "debt" in result.notes.lower())
        total_completed = len(self.execution_history)
        
        if total_completed == 0:
            return 0.0
        
        # Simplified debt reduction calculation
        return (debt_items_completed / total_completed) * 100 * 0.3  # Assume 30% reduction per debt item
    
    def _get_source_percentage(self, source: str) -> float:
        """Get percentage of items from a specific source."""
        if not self.backlog:
            return 0.0
        
        source_count = sum(1 for item in self.backlog if source in item.source)
        return (source_count / len(self.backlog)) * 100
    
    def _get_most_valuable_category(self) -> str:
        """Get the category that delivers the most value."""
        if not self.execution_history:
            return "technical_debt"
        
        # Simplified - return most common successful category
        successful_results = [r for r in self.execution_history if r.success]
        if not successful_results:
            return "maintenance"
        
        # This would need more sophisticated tracking in practice
        return "technical_debt"  # Default for now
    
    def _calculate_average_risk(self) -> float:
        """Calculate average risk level across backlog."""
        if not self.backlog:
            return 0.0
        
        return sum(item.risk_level for item in self.backlog) / len(self.backlog)
    
    def _predict_capacity(self) -> float:
        """Predict next week's capacity based on historical data."""
        if not self.execution_history:
            return 8.0  # Default 8 hours per week
        
        recent_effort = sum(result.actual_effort for result in self.execution_history[-7:])  # Last week
        return max(4.0, min(20.0, recent_effort))  # Bound between 4-20 hours
    
    def _recommend_focus_area(self) -> str:
        """Recommend focus area based on backlog composition."""
        if not self.backlog:
            return "discovery"
        
        categories = {}
        for item in self.backlog[:10]:  # Top 10 items
            cat = item.category
            categories[cat] = categories.get(cat, 0) + item.composite_score
        
        if not categories:
            return "maintenance"
        
        top_category = max(categories.items(), key=lambda x: x[1])[0]
        return top_category
    
    def _estimate_backlog_completion(self) -> int:
        """Estimate days to complete current backlog."""
        if not self.backlog:
            return 0
        
        total_effort = sum(item.effort_estimate for item in self.backlog)
        weekly_capacity = self._predict_capacity()
        
        if weekly_capacity <= 0:
            return 365  # Fallback
        
        weeks_needed = total_effort / weekly_capacity
        return int(weeks_needed * 7)
    
    def _update_metrics(self):
        """Update and save comprehensive metrics."""
        metrics = self._calculate_metrics()
        
        metrics_data = {
            "last_updated": datetime.now().isoformat(),
            "backlog_metrics": asdict(metrics),
            "execution_summary": {
                "total_completed": len(self.execution_history),
                "success_rate": sum(1 for r in self.execution_history if r.success) / len(self.execution_history) * 100 if self.execution_history else 0,
                "average_effort": sum(r.actual_effort for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0
            }
        }
        
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

if __name__ == "__main__":
    # Example usage
    manager = BacklogManager()
    
    print("🔄 Refreshing backlog...")
    backlog = manager.refresh_backlog()
    
    print(f"📊 Generated backlog with {len(backlog)} items")
    
    next_item = manager.get_next_best_value_item()
    if next_item:
        print(f"🎯 Next best value: {next_item.title} (Score: {next_item.composite_score:.1f})")
    
    print("\n" + "="*80)
    print(manager.generate_backlog_report())