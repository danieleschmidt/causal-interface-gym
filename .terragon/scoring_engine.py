#!/usr/bin/env python3
"""Advanced scoring engine for autonomous value discovery."""

import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml

@dataclass
class WorkItem:
    """Represents a work item with comprehensive scoring."""
    id: str
    title: str
    description: str
    category: str
    files_affected: List[str]
    effort_estimate: float  # hours
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    risk_level: float
    discovered_at: str
    source: str

class ScoringEngine:
    """WSJF + ICE + Technical Debt composite scoring system."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.weights = self.config["scoring"]["weights"]
        self.thresholds = self.config["scoring"]["thresholds"]
    
    def calculate_wsjf(self, item: Dict) -> float:
        """Calculate Weighted Shortest Job First score."""
        # Cost of Delay components
        business_value = self._score_business_value(item)
        time_criticality = self._score_time_criticality(item)  
        risk_reduction = self._score_risk_reduction(item)
        opportunity_enablement = self._score_opportunity_enablement(item)
        
        cost_of_delay = (
            business_value + time_criticality + 
            risk_reduction + opportunity_enablement
        )
        
        job_size = item.get("effort_estimate", 1.0)
        
        return cost_of_delay / max(job_size, 0.1)
    
    def calculate_ice(self, item: Dict) -> float:
        """Calculate Impact Confidence Ease score."""
        impact = self._score_impact(item)  # 1-10
        confidence = self._score_confidence(item)  # 1-10
        ease = self._score_ease(item)  # 1-10
        
        return impact * confidence * ease
    
    def calculate_technical_debt_score(self, item: Dict) -> float:
        """Calculate technical debt impact score."""
        debt_impact = self._calculate_debt_cost(item)
        debt_interest = self._calculate_debt_growth(item)
        hotspot_multiplier = self._get_churn_complexity_multiplier(item)
        
        return (debt_impact + debt_interest) * hotspot_multiplier
    
    def calculate_composite_score(self, item: Dict) -> float:
        """Calculate final composite score with adaptive weighting."""
        wsjf = self.calculate_wsjf(item)
        ice = self.calculate_ice(item)
        debt = self.calculate_technical_debt_score(item)
        
        # Normalize scores to 0-100 range
        normalized_wsjf = self._normalize_score(wsjf, 0, 50)
        normalized_ice = self._normalize_score(ice, 1, 1000)
        normalized_debt = self._normalize_score(debt, 0, 100)
        
        composite = (
            self.weights["wsjf"] * normalized_wsjf +
            self.weights["ice"] * normalized_ice +
            self.weights["technicalDebt"] * normalized_debt
        )
        
        # Apply boost factors
        if item.get("category") == "security":
            composite *= self.thresholds["securityBoost"]
        elif item.get("category") == "compliance":
            composite *= self.thresholds["complianceBoost"]
        elif item.get("category") == "modernization":
            composite *= self.thresholds["modernizationBoost"]
        
        return composite
    
    def _score_business_value(self, item: Dict) -> float:
        """Score business value impact (0-10)."""
        category = item.get("category", "")
        if category in ["security", "compliance"]:
            return 9.0
        elif category in ["performance", "user_experience"]:
            return 7.0
        elif category in ["maintainability", "technical_debt"]:
            return 6.0
        elif category in ["documentation", "testing"]:
            return 4.0
        return 3.0
    
    def _score_time_criticality(self, item: Dict) -> float:
        """Score time criticality (0-10)."""
        if "vulnerability" in item.get("description", "").lower():
            return 10.0
        elif "deprecated" in item.get("description", "").lower():
            return 8.0
        elif "outdated" in item.get("description", "").lower():
            return 6.0
        return 3.0
    
    def _score_risk_reduction(self, item: Dict) -> float:
        """Score risk reduction value (0-10)."""
        if item.get("category") == "security":
            return 9.0
        elif "test" in item.get("category", ""):
            return 7.0
        elif item.get("category") == "reliability":
            return 8.0
        return 2.0
    
    def _score_opportunity_enablement(self, item: Dict) -> float:
        """Score opportunity enablement (0-10)."""
        if item.get("category") == "modernization":
            return 8.0
        elif item.get("category") == "performance":
            return 6.0
        elif item.get("category") == "automation":
            return 7.0
        return 2.0
    
    def _score_impact(self, item: Dict) -> float:
        """Score implementation impact (1-10)."""
        affected_files = len(item.get("files_affected", []))
        if affected_files > 10:
            return 9.0
        elif affected_files > 5:
            return 7.0
        elif affected_files > 1:
            return 5.0
        return 3.0
    
    def _score_confidence(self, item: Dict) -> float:
        """Score execution confidence (1-10)."""
        effort = item.get("effort_estimate", 1.0)
        if effort <= 2:
            return 9.0
        elif effort <= 4:
            return 7.0
        elif effort <= 8:
            return 5.0
        return 3.0
    
    def _score_ease(self, item: Dict) -> float:
        """Score implementation ease (1-10)."""
        category = item.get("category", "")
        if category in ["documentation", "configuration"]:
            return 8.0
        elif category in ["dependency_update", "refactoring"]:
            return 6.0
        elif category in ["new_feature", "architecture"]:
            return 3.0
        return 5.0
    
    def _calculate_debt_cost(self, item: Dict) -> float:
        """Calculate technical debt maintenance cost."""
        category = item.get("category", "")
        if category == "technical_debt":
            return 50.0
        elif "complexity" in item.get("description", "").lower():
            return 30.0
        elif "duplicate" in item.get("description", "").lower():
            return 20.0
        return 5.0
    
    def _calculate_debt_growth(self, item: Dict) -> float:
        """Calculate debt interest - future cost if not addressed."""
        if "deprecated" in item.get("description", "").lower():
            return 40.0
        elif "outdated" in item.get("description", "").lower():
            return 25.0
        return 5.0
    
    def _get_churn_complexity_multiplier(self, item: Dict) -> float:
        """Get hotspot multiplier based on file churn and complexity."""
        files = item.get("files_affected", [])
        if not files:
            return 1.0
        
        # Simple heuristic - core files get higher multiplier
        for file in files:
            if "core" in file or "__init__" in file:
                return 2.5
            elif "test" in file:
                return 1.2
        
        return 1.5
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range."""
        if max_val == min_val:
            return 50.0
        normalized = ((score - min_val) / (max_val - min_val)) * 100
        return max(0, min(100, normalized))

if __name__ == "__main__":
    # Example usage
    engine = ScoringEngine()
    
    sample_item = {
        "id": "td-001",
        "title": "Refactor core authentication module",
        "category": "technical_debt",
        "description": "High complexity authentication code needs refactoring",
        "files_affected": ["src/causal_interface_gym/core.py"],
        "effort_estimate": 6.0
    }
    
    wsjf = engine.calculate_wsjf(sample_item)
    ice = engine.calculate_ice(sample_item)
    debt = engine.calculate_technical_debt_score(sample_item)
    composite = engine.calculate_composite_score(sample_item)
    
    print(f"WSJF: {wsjf:.2f}")
    print(f"ICE: {ice:.2f}")
    print(f"Technical Debt: {debt:.2f}")
    print(f"Composite Score: {composite:.2f}")