"""Belief extraction from LLM responses."""

import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class BeliefExtractor:
    """Extract belief probabilities and reasoning from LLM responses."""
    
    def __init__(self):
        """Initialize belief extractor."""
        # Patterns for extracting probabilities
        self.probability_patterns = [
            r'\b0\.[0-9]+\b',  # Decimal probabilities (0.75)
            r'\b1\.0+\b',      # 1.0
            r'\b0\.0+\b',      # 0.0
            r'\b[0-9]+%',      # Percentages (75%)
            r'\b[0-9]+/[0-9]+\b',  # Fractions (3/4)
        ]
        
        # Patterns for confidence indicators
        self.confidence_patterns = {
            'high': ['certain', 'confident', 'sure', 'definitely', 'clearly'],
            'medium': ['likely', 'probably', 'expect', 'believe'],
            'low': ['uncertain', 'unsure', 'maybe', 'possibly', 'might']
        }
    
    def extract_probability(self, text: str, target_belief: Optional[str] = None) -> float:
        """Extract probability value from text.
        
        Args:
            text: Text containing probability
            target_belief: Specific belief to look for (optional)
            
        Returns:
            Probability value between 0 and 1
        """
        # Clean text
        text = text.strip().lower()
        
        # Look for explicit probability statements
        prob_value = self._extract_numeric_probability(text)
        
        if prob_value is not None:
            return prob_value
        
        # Fall back to confidence-based estimation
        confidence_level = self._extract_confidence_level(text)
        return self._confidence_to_probability(confidence_level)
    
    def _extract_numeric_probability(self, text: str) -> Optional[float]:
        """Extract numeric probability from text.
        
        Args:
            text: Input text
            
        Returns:
            Probability value or None if not found
        """
        # Try each pattern
        for pattern in self.probability_patterns:
            matches = re.findall(pattern, text)
            
            for match in matches:
                try:
                    if '%' in match:
                        # Convert percentage to decimal
                        value = float(match.replace('%', '')) / 100.0
                    elif '/' in match:
                        # Convert fraction to decimal
                        numerator, denominator = match.split('/')
                        value = float(numerator) / float(denominator)
                    else:
                        # Direct decimal
                        value = float(match)
                    
                    # Validate range
                    if 0.0 <= value <= 1.0:
                        logger.debug(f"Extracted probability: {value} from '{match}'")
                        return value
                except ValueError:
                    continue
        
        # Look for special phrases
        special_phrases = {
            'impossible': 0.0,
            'never': 0.0,
            'no chance': 0.0,
            'unlikely': 0.2,
            'possible': 0.5,
            'maybe': 0.5,
            'likely': 0.7,
            'probably': 0.75,
            'almost certain': 0.9,
            'certain': 1.0,
            'always': 1.0,
            'definitely': 1.0
        }
        
        for phrase, prob in special_phrases.items():
            if phrase in text:
                logger.debug(f"Extracted probability: {prob} from phrase '{phrase}'")
                return prob
        
        return None
    
    def _extract_confidence_level(self, text: str) -> str:
        """Extract confidence level from text.
        
        Args:
            text: Input text
            
        Returns:
            Confidence level (high, medium, low)
        """
        for level, indicators in self.confidence_patterns.items():
            for indicator in indicators:
                if indicator in text:
                    return level
        
        return 'medium'  # Default
    
    def _confidence_to_probability(self, confidence: str) -> float:
        """Convert confidence level to probability.
        
        Args:
            confidence: Confidence level
            
        Returns:
            Probability value
        """
        mapping = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        return mapping.get(confidence, 0.5)
    
    def extract_belief_reasoning(self, text: str) -> Dict[str, Any]:
        """Extract reasoning behind belief.
        
        Args:
            text: Text containing reasoning
            
        Returns:
            Reasoning analysis
        """
        reasoning = {
            'causal_keywords': [],
            'intervention_indicators': [],
            'correlation_indicators': [],
            'uncertainty_indicators': []
        }
        
        text_lower = text.lower()
        
        # Causal keywords
        causal_keywords = [
            'cause', 'effect', 'intervention', 'do(', 'causal',
            'influence', 'impact', 'leads to', 'results in'
        ]
        
        for keyword in causal_keywords:
            if keyword in text_lower:
                reasoning['causal_keywords'].append(keyword)
        
        # Intervention indicators
        intervention_indicators = [
            'intervention', 'do(', 'manipulate', 'force', 'set to',
            'control', 'change', 'alter'
        ]
        
        for indicator in intervention_indicators:
            if indicator in text_lower:
                reasoning['intervention_indicators'].append(indicator)
        
        # Correlation indicators
        correlation_indicators = [
            'correlation', 'association', 'observe', 'see that',
            'related to', 'linked to', 'associated with'
        ]
        
        for indicator in correlation_indicators:
            if indicator in text_lower:
                reasoning['correlation_indicators'].append(indicator)
        
        # Uncertainty indicators
        uncertainty_indicators = [
            'uncertain', 'unsure', 'don\'t know', 'unclear',
            'difficult to say', 'hard to tell'
        ]
        
        for indicator in uncertainty_indicators:
            if indicator in text_lower:
                reasoning['uncertainty_indicators'].append(indicator)
        
        return reasoning
    
    def extract_multiple_beliefs(self, text: str, belief_statements: List[str]) -> Dict[str, float]:
        """Extract multiple belief probabilities from text.
        
        Args:
            text: Text containing multiple beliefs
            belief_statements: List of belief statements to extract
            
        Returns:
            Dictionary mapping belief statements to probabilities
        """
        beliefs = {}
        
        for belief in belief_statements:
            # Try to find belief-specific text
            belief_pattern = re.escape(belief)
            match = re.search(f'{belief_pattern}.*?([0-9.]+)', text, re.IGNORECASE)
            
            if match:
                try:
                    prob_value = float(match.group(1))
                    if prob_value > 1.0:  # Assume percentage
                        prob_value /= 100.0
                    beliefs[belief] = max(0.0, min(1.0, prob_value))
                except ValueError:
                    beliefs[belief] = self.extract_probability(text)
            else:
                # Fall back to general extraction
                beliefs[belief] = self.extract_probability(text)
        
        return beliefs
    
    def validate_belief_consistency(self, beliefs: Dict[str, float]) -> Dict[str, Any]:
        """Validate consistency of extracted beliefs.
        
        Args:
            beliefs: Dictionary of belief probabilities
            
        Returns:
            Validation results
        """
        validation = {
            'valid_probabilities': True,
            'consistency_score': 1.0,
            'issues': []
        }
        
        # Check probability ranges
        for belief, prob in beliefs.items():
            if not (0.0 <= prob <= 1.0):
                validation['valid_probabilities'] = False
                validation['issues'].append(f"Invalid probability for {belief}: {prob}")
        
        # Check for logical consistency (basic checks)
        belief_keys = list(beliefs.keys())
        
        for i, belief1 in enumerate(belief_keys):
            for belief2 in belief_keys[i+1:]:
                # Check for complementary beliefs
                if self._are_complementary(belief1, belief2):
                    prob_sum = beliefs[belief1] + beliefs[belief2]
                    if abs(prob_sum - 1.0) > 0.2:  # Allow some tolerance
                        validation['consistency_score'] *= 0.8
                        validation['issues'].append(
                            f"Complementary beliefs {belief1} and {belief2} "
                            f"don't sum to 1.0: {prob_sum}"
                        )
        
        return validation
    
    def _are_complementary(self, belief1: str, belief2: str) -> bool:
        """Check if two beliefs are complementary (should sum to 1).
        
        Args:
            belief1: First belief statement
            belief2: Second belief statement
            
        Returns:
            True if beliefs are complementary
        """
        # Simple heuristic - check for negation patterns
        negation_patterns = ['not ', '¬', '~']
        
        for pattern in negation_patterns:
            if pattern in belief1.lower() or pattern in belief2.lower():
                # Remove negation and compare
                clean1 = belief1.lower().replace(pattern, '').strip()
                clean2 = belief2.lower().replace(pattern, '').strip()
                if clean1 == clean2:
                    return True
        
        return False