"""Response parsing utilities for LLM outputs."""

from typing import List, Dict, Any, Optional
import re


class ResponseParser:
    """Parse structured information from LLM responses."""
    
    def __init__(self):
        """Initialize response parser."""
        pass
    
    def parse_variable_list(self, text: str, graph_variables: List[str]) -> List[str]:
        """Parse list of variables from response text."""
        # Split by common delimiters
        candidates = re.split(r'[,;\n]', text.lower())
        
        variables = []
        for candidate in candidates:
            candidate = candidate.strip()
            # Check if candidate matches any graph variable
            for var in graph_variables:
                if var.lower() in candidate or candidate in var.lower():
                    if var not in variables:
                        variables.append(var)
        
        return variables
    
    def parse_probability_statement(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse probability statement from text."""
        # Simple pattern matching for P(Y|X) format
        prob_pattern = r'P\(([^|)]+)\|([^)]+)\)'
        match = re.search(prob_pattern, text)
        
        if match:
            return {
                "outcome": match.group(1).strip(),
                "condition": match.group(2).strip(),
                "type": "conditional"
            }
        
        return None