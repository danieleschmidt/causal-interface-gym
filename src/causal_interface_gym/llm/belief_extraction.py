"""Belief extraction utilities for LLM responses."""

import re
from typing import Optional, List, Dict, Any


class BeliefExtractor:
    """Extract beliefs and probabilities from LLM responses."""
    
    def __init__(self):
        """Initialize belief extractor."""
        pass
    
    def extract_probability(self, text: str, belief_statement: str) -> float:
        """Extract probability from text response."""
        text_clean = text.strip().lower()
        
        # Look for decimal probabilities
        decimal_matches = re.findall(r'0?\.\d+', text_clean)
        if decimal_matches:
            try:
                prob = float(decimal_matches[0])
                if 0 <= prob <= 1:
                    return prob
            except ValueError:
                pass
        
        # Look for percentages
        percent_matches = re.findall(r'(\d+)%', text_clean)
        if percent_matches:
            try:
                prob = float(percent_matches[0]) / 100.0
                if 0 <= prob <= 1:
                    return prob
            except ValueError:
                pass
        
        # Fallback
        return 0.5