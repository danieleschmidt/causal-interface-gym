"""Response parsing utilities for LLM outputs."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parse and extract structured information from LLM responses."""
    
    def __init__(self):
        """Initialize response parser."""
        pass
    
    def parse_variable_list(self, text: str, graph_variables: List[str]) -> List[str]:
        """Parse list of variables from text.
        
        Args:
            text: Text containing variable names
            graph_variables: Valid variable names in the graph
            
        Returns:
            List of extracted variable names
        """
        variables = []
        text_lower = text.lower()
        
        # Look for explicit list patterns
        list_patterns = [
            r'[-*]\s*([\w_]+)',  # Bullet points
            r'\d+\.\s*([\w_]+)',  # Numbered lists
            r'([\w_]+)(?:,|\n|$)',  # Comma or newline separated
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                clean_var = match.strip().lower()
                # Check if it's a valid graph variable
                for graph_var in graph_variables:
                    if clean_var == graph_var.lower() or clean_var in graph_var.lower():
                        if graph_var not in variables:
                            variables.append(graph_var)
        
        # If no structured list found, look for variable names mentioned in text
        if not variables:
            for var in graph_variables:
                if var.lower() in text_lower:
                    variables.append(var)
        
        # Handle "no confounders" or similar responses
        no_confounders_phrases = [
            'no confounders', 'no variables', 'none', 'no backdoor',
            'no common causes', 'no shared causes'
        ]
        
        for phrase in no_confounders_phrases:
            if phrase in text_lower:
                return []
        
        logger.debug(f"Extracted variables: {variables}")
        return variables
    
    def parse_causal_paths(self, text: str, graph_variables: List[str]) -> List[List[str]]:
        """Parse causal paths from text.
        
        Args:
            text: Text containing path descriptions
            graph_variables: Valid variable names
            
        Returns:
            List of causal paths (each path is a list of variables)
        """
        paths = []
        
        # Look for arrow notation: A → B → C
        arrow_patterns = [
            r'([\w_]+)\s*→\s*([\w_]+)(?:\s*→\s*([\w_]+))*',
            r'([\w_]+)\s*->\s*([\w_]+)(?:\s*->\s*([\w_]+))*',
            r'([\w_]+)\s*causes?\s*([\w_]+)(?:\s*causes?\s*([\w_]+))*',
        ]
        
        for pattern in arrow_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                path_vars = [var for var in match.groups() if var is not None]
                # Validate variables
                valid_path = []
                for var in path_vars:
                    clean_var = var.strip()
                    if clean_var in graph_variables:
                        valid_path.append(clean_var)
                
                if len(valid_path) >= 2:
                    paths.append(valid_path)
        
        # Look for path descriptions in natural language
        if not paths:
            paths = self._parse_natural_language_paths(text, graph_variables)
        
        logger.debug(f"Extracted paths: {paths}")
        return paths
    
    def _parse_natural_language_paths(self, text: str, variables: List[str]) -> List[List[str]]:
        """Parse paths from natural language descriptions.
        
        Args:
            text: Text with path descriptions
            variables: Valid variable names
            
        Returns:
            List of parsed paths
        """
        paths = []
        sentences = text.split('.')
        
        for sentence in sentences:
            # Look for mentions of multiple variables
            mentioned_vars = []
            for var in variables:
                if var.lower() in sentence.lower():
                    mentioned_vars.append(var)
            
            # If we found a sequence of variables, treat as potential path
            if len(mentioned_vars) >= 2:
                paths.append(mentioned_vars)
        
        return paths
    
    def parse_reasoning_quality(self, text: str) -> Dict[str, Any]:
        """Assess quality of causal reasoning in text.
        
        Args:
            text: Text containing reasoning
            
        Returns:
            Quality assessment
        """
        quality = {
            'causal_language_score': 0.0,
            'intervention_awareness': False,
            'confounder_awareness': False,
            'mechanism_description': False,
            'confidence_indicators': [],
            'reasoning_depth': 'shallow'
        }
        
        text_lower = text.lower()
        
        # Score causal language usage
        causal_terms = [
            'cause', 'effect', 'causal', 'mechanism', 'intervention',
            'do(', 'confound', 'backdoor', 'association', 'correlation'
        ]
        
        causal_score = 0
        for term in causal_terms:
            if term in text_lower:
                causal_score += 1
        
        quality['causal_language_score'] = min(1.0, causal_score / len(causal_terms))
        
        # Check intervention awareness
        intervention_terms = ['intervention', 'do(', 'manipulate', 'control']
        quality['intervention_awareness'] = any(term in text_lower for term in intervention_terms)
        
        # Check confounder awareness
        confounder_terms = ['confounder', 'confound', 'backdoor', 'spurious']
        quality['confounder_awareness'] = any(term in text_lower for term in confounder_terms)
        
        # Check mechanism description
        mechanism_terms = ['mechanism', 'process', 'pathway', 'how', 'why']
        quality['mechanism_description'] = any(term in text_lower for term in mechanism_terms)
        
        # Extract confidence indicators
        confidence_terms = {
            'high': ['certain', 'confident', 'clearly', 'definitely'],
            'medium': ['likely', 'probably', 'expect'],
            'low': ['uncertain', 'might', 'possibly', 'unclear']
        }
        
        for level, terms in confidence_terms.items():
            for term in terms:
                if term in text_lower:
                    quality['confidence_indicators'].append((term, level))
        
        # Assess reasoning depth
        if len(text.split('.')) > 5 and quality['causal_language_score'] > 0.3:
            quality['reasoning_depth'] = 'deep'
        elif len(text.split('.')) > 2:
            quality['reasoning_depth'] = 'medium'
        
        return quality
    
    def parse_comparison_response(self, text: str) -> Dict[str, Any]:
        """Parse response comparing intervention vs observation.
        
        Args:
            text: Text containing comparison
            
        Returns:
            Parsed comparison
        """
        comparison = {
            'observational_probability': None,
            'interventional_probability': None,
            'difference_explained': False,
            'understands_distinction': False
        }
        
        # Look for probability values in different contexts
        prob_pattern = r'([0-9]?\.[0-9]+)'
        probabilities = re.findall(prob_pattern, text)
        
        # Try to associate probabilities with contexts
        text_lower = text.lower()
        
        if 'observ' in text_lower or 'association' in text_lower:
            obs_match = re.search(r'observ.*?([0-9]?\.[0-9]+)', text_lower)
            if obs_match:
                comparison['observational_probability'] = float(obs_match.group(1))
        
        if 'interven' in text_lower or 'do(' in text_lower:
            int_match = re.search(r'interven.*?([0-9]?\.[0-9]+)', text_lower)
            if int_match:
                comparison['interventional_probability'] = float(int_match.group(1))
        
        # Check if difference is explained
        difference_indicators = [
            'different', 'differ', 'distinction', 'because', 'due to',
            'confound', 'causal', 'intervention breaks'
        ]
        
        comparison['difference_explained'] = any(
            indicator in text_lower for indicator in difference_indicators
        )
        
        # Check understanding of intervention/observation distinction
        understanding_indicators = [
            'intervention breaks', 'do() operator', 'causal effect',
            'removes confounding', 'backdoor', 'graph surgery'
        ]
        
        comparison['understands_distinction'] = any(
            indicator in text_lower for indicator in understanding_indicators
        )
        
        return comparison
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text.
        
        Args:
            text: Input text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of key phrases
        """
        # Simple extraction based on noun phrases and important terms
        phrases = []
        
        # Look for quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', text)
        phrases.extend(quoted_phrases)
        
        # Look for technical terms
        technical_terms = [
            'causal effect', 'intervention', 'do-calculus', 'backdoor path',
            'confounder', 'spurious correlation', 'causal mechanism',
            'observational study', 'randomized trial', 'confounding bias'
        ]
        
        for term in technical_terms:
            if term.lower() in text.lower():
                phrases.append(term)
        
        # Remove duplicates and limit
        unique_phrases = list(dict.fromkeys(phrases))  # Preserve order
        return unique_phrases[:max_phrases]
    
    def parse_structured_response(self, text: str, expected_sections: List[str]) -> Dict[str, str]:
        """Parse response with expected sections.
        
        Args:
            text: Response text
            expected_sections: List of expected section headers
            
        Returns:
            Dictionary mapping sections to content
        """
        sections = {}
        
        # Try to find numbered or bulleted sections
        for i, section in enumerate(expected_sections, 1):
            patterns = [
                f'{i}\.\s*{section}[:\.]?\s*([^\n]*(?:\n(?!\d+\.|[A-Za-z]+:)[^\n]*)*)',
                f'{section}[:\.]?\s*([^\n]*(?:\n(?!\d+\.|[A-Za-z]+:)[^\n]*)*)',
                f'\*\*{section}\*\*[:\.]?\s*([^\n]*(?:\n(?!\*\*)[^\n]*)*)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    sections[section] = match.group(1).strip()
                    break
        
        return sections