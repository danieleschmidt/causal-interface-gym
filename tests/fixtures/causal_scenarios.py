"""
Curated causal scenarios for testing and benchmarking.

This module provides standardized causal inference scenarios used throughout
academic literature and real-world applications.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class CausalScenario:
    """Represents a complete causal scenario for testing."""
    name: str
    description: str
    dag: Dict[str, List[str]]
    ground_truth_effects: Dict[Tuple[str, str], float]
    confounders: List[List[str]]
    interventions: List[Dict[str, Any]]
    common_mistakes: List[str]
    difficulty: str  # "easy", "medium", "hard"


class CausalScenarios:
    """Collection of standard causal scenarios."""

    @staticmethod
    def rain_sprinkler_grass() -> CausalScenario:
        """The classic rain-sprinkler-grass scenario."""
        return CausalScenario(
            name="rain_sprinkler_grass",
            description="Classic example of confounding in causal reasoning",
            dag={
                "rain": [],
                "sprinkler": ["rain"],
                "wet_grass": ["rain", "sprinkler"],
                "slippery": ["wet_grass"]
            },
            ground_truth_effects={
                ("rain", "wet_grass"): 0.8,
                ("sprinkler", "wet_grass"): 0.9,
                ("wet_grass", "slippery"): 0.7,
                ("rain", "sprinkler"): -0.6  # Negative: rain makes sprinkler less likely
            },
            confounders=[["rain"]],  # Rain confounds sprinkler -> wet_grass
            interventions=[
                {"variable": "sprinkler", "value": True},
                {"variable": "sprinkler", "value": False},
                {"variable": "rain", "value": True}
            ],
            common_mistakes=[
                "Thinking P(wet_grass|sprinkler=on) = P(wet_grass|do(sprinkler=on))",
                "Ignoring that rain affects both sprinkler and wet_grass",
                "Confusing correlation between sprinkler and wet_grass with causation"
            ],
            difficulty="easy"
        )

    @staticmethod
    def smoking_cancer() -> CausalScenario:
        """Smoking causes cancer with genetic confounding."""
        return CausalScenario(
            name="smoking_cancer",
            description="Smoking-cancer relationship with genetic confounding",
            dag={
                "genetics": [],
                "smoking": ["genetics"],
                "tar_deposits": ["smoking"],
                "cancer": ["smoking", "genetics", "tar_deposits"],
                "coughing": ["cancer"],
                "yellow_fingers": ["smoking"]
            },
            ground_truth_effects={
                ("smoking", "cancer"): 0.4,
                ("genetics", "cancer"): 0.3,
                ("tar_deposits", "cancer"): 0.5,
                ("genetics", "smoking"): 0.2,
                ("smoking", "tar_deposits"): 0.9,
                ("cancer", "coughing"): 0.8,
                ("smoking", "yellow_fingers"): 0.95
            },
            confounders=[["genetics"]],  # Genetics confounds smoking -> cancer
            interventions=[
                {"variable": "smoking", "value": True},
                {"variable": "smoking", "value": False}
            ],
            common_mistakes=[
                "Ignoring genetic confounding when estimating smoking effect",
                "Assuming tar_deposits mediates all of smoking's effect on cancer",
                "Using yellow_fingers as instrumental variable incorrectly"
            ],
            difficulty="medium"
        )

    @staticmethod
    def simpsons_paradox_admissions() -> CausalScenario:
        """Simpson's Paradox in university admissions."""
        return CausalScenario(
            name="simpsons_paradox_admissions",
            description="Gender bias in admissions showing Simpson's Paradox",
            dag={
                "gender": [],
                "department_choice": ["gender"],
                "qualifications": ["gender"],
                "admission": ["gender", "department_choice", "qualifications"],
                "success_in_program": ["qualifications", "admission"]
            },
            ground_truth_effects={
                ("gender", "admission"): -0.1,  # Small bias against women
                ("department_choice", "admission"): 0.6,
                ("qualifications", "admission"): 0.8,
                ("gender", "department_choice"): 0.3,
                ("gender", "qualifications"): -0.1,
                ("qualifications", "success_in_program"): 0.9,
                ("admission", "success_in_program"): 0.2
            },
            confounders=[["department_choice", "qualifications"]],
            interventions=[
                {"variable": "gender", "value": "female"},
                {"variable": "gender", "value": "male"},
                {"variable": "department_choice", "value": "competitive"}
            ],
            common_mistakes=[
                "Looking only at overall admission rates by gender",
                "Ignoring that women apply to more competitive departments",
                "Not adjusting for qualifications when measuring bias",
                "Treating department choice as outcome rather than mediator"
            ],
            difficulty="hard"
        )

    @staticmethod
    def treatment_outcome_confounded() -> CausalScenario:
        """Medical treatment with confounding by indication."""
        return CausalScenario(
            name="treatment_outcome_confounded",
            description="Medical treatment confounded by patient severity",
            dag={
                "severity": [],
                "comorbidities": ["severity"],
                "treatment": ["severity", "comorbidities"],
                "side_effects": ["treatment"],
                "recovery": ["treatment", "severity", "comorbidities", "side_effects"],
                "hospital_stay": ["recovery", "side_effects"]
            },
            ground_truth_effects={
                ("treatment", "recovery"): 0.3,
                ("severity", "recovery"): -0.7,
                ("comorbidities", "recovery"): -0.4,
                ("side_effects", "recovery"): -0.2,
                ("severity", "treatment"): 0.8,  # Sicker patients get treatment
                ("comorbidities", "treatment"): 0.5,
                ("treatment", "side_effects"): 0.4,
                ("recovery", "hospital_stay"): -0.6,
                ("side_effects", "hospital_stay"): 0.3
            },
            confounders=[["severity", "comorbidities"]],
            interventions=[
                {"variable": "treatment", "value": True},
                {"variable": "treatment", "value": False}
            ],
            common_mistakes=[
                "Comparing treated vs untreated without adjusting for severity",
                "Using side_effects as evidence against treatment effectiveness",
                "Ignoring that sicker patients are more likely to get treatment",
                "Not distinguishing correlation from causation in observational data"
            ],
            difficulty="medium"
        )

    @staticmethod
    def instrumental_variable_education() -> CausalScenario:
        """Education returns using compulsory schooling as instrument."""
        return CausalScenario(
            name="instrumental_variable_education",
            description="Using compulsory schooling laws as instrument for education",
            dag={
                "ability": [],
                "compulsory_schooling": [],
                "education": ["ability", "compulsory_schooling"],
                "earnings": ["education", "ability"],
                "job_satisfaction": ["earnings", "education"]
            },
            ground_truth_effects={
                ("education", "earnings"): 0.4,
                ("ability", "earnings"): 0.5,
                ("ability", "education"): 0.6,
                ("compulsory_schooling", "education"): 0.3,
                ("earnings", "job_satisfaction"): 0.4,
                ("education", "job_satisfaction"): 0.3
            },
            confounders=[["ability"]],  # Ability confounds education -> earnings
            interventions=[
                {"variable": "compulsory_schooling", "value": 12},
                {"variable": "compulsory_schooling", "value": 16},
                {"variable": "education", "value": 16}  # Direct intervention
            ],
            common_mistakes=[
                "Using OLS to estimate education returns without considering ability bias",
                "Thinking compulsory_schooling affects earnings directly",
                "Not checking if compulsory_schooling is truly exogenous",
                "Confusing local average treatment effect with average treatment effect"
            ],
            difficulty="hard"
        )

    @staticmethod
    def mediation_analysis() -> CausalScenario:
        """Mediation analysis scenario."""
        return CausalScenario(
            name="mediation_analysis",
            description="Direct vs indirect effects through mediator",
            dag={
                "treatment": [],
                "mediator": ["treatment"],
                "outcome": ["treatment", "mediator"],
                "confounder": [],
                "mediator_confounder": ["confounder", "treatment"],
                "outcome_confounder": ["confounder", "treatment"]
            },
            ground_truth_effects={
                ("treatment", "outcome"): 0.5,  # Total effect
                ("treatment", "mediator"): 0.6,
                ("mediator", "outcome"): 0.4,
                # Direct effect: 0.5 - 0.6 * 0.4 = 0.26
                # Indirect effect: 0.6 * 0.4 = 0.24
            },
            confounders=[["mediator_confounder", "outcome_confounder"]],
            interventions=[
                {"variable": "treatment", "value": True},
                {"variable": "mediator", "value": 1.0},
                {"variable": "treatment", "value": True, "mediator": "fixed"}
            ],
            common_mistakes=[
                "Using product-of-coefficients without proper identification",
                "Ignoring treatment-mediator confounding",
                "Assuming no interaction between treatment and mediator",
                "Not distinguishing direct from total effect"
            ],
            difficulty="hard"
        )

    @classmethod
    def get_all_scenarios(cls) -> List[CausalScenario]:
        """Get all available causal scenarios."""
        return [
            cls.rain_sprinkler_grass(),
            cls.smoking_cancer(),
            cls.simpsons_paradox_admissions(),
            cls.treatment_outcome_confounded(),
            cls.instrumental_variable_education(),
            cls.mediation_analysis()
        ]

    @classmethod
    def get_by_difficulty(cls, difficulty: str) -> List[CausalScenario]:
        """Get scenarios by difficulty level."""
        all_scenarios = cls.get_all_scenarios()
        return [s for s in all_scenarios if s.difficulty == difficulty]

    @classmethod
    def get_by_name(cls, name: str) -> CausalScenario:
        """Get scenario by name."""
        method_name = name.replace("_", "_")
        if hasattr(cls, method_name):
            return getattr(cls, method_name)()
        else:
            raise ValueError(f"Scenario '{name}' not found")


# Utility functions for test data generation
def generate_observational_data(scenario: CausalScenario, n_samples: int = 1000) -> Dict[str, np.ndarray]:
    """Generate synthetic observational data for a causal scenario."""
    np.random.seed(42)  # Reproducible for testing
    
    data = {}
    variables = list(scenario.dag.keys())
    
    # Topological sort to generate data in causal order
    ordered_vars = topological_sort(scenario.dag)
    
    for var in ordered_vars:
        parents = scenario.dag[var]
        if not parents:
            # Root variable - generate from prior
            data[var] = np.random.binomial(1, 0.5, n_samples)
        else:
            # Generate based on parents
            parent_effects = np.zeros(n_samples)
            for parent in parents:
                effect_key = (parent, var)
                if effect_key in scenario.ground_truth_effects:
                    effect = scenario.ground_truth_effects[effect_key]
                    parent_effects += effect * data[parent]
            
            # Convert to probability and sample
            probs = 1 / (1 + np.exp(-parent_effects))  # Sigmoid
            data[var] = np.random.binomial(1, probs)
    
    return data


def topological_sort(dag: Dict[str, List[str]]) -> List[str]:
    """Topologically sort DAG variables."""
    in_degree = {var: 0 for var in dag.keys()}
    for var, parents in dag.items():
        in_degree[var] = len(parents)
    
    queue = [var for var, degree in in_degree.items() if degree == 0]
    result = []
    
    while queue:
        current = queue.pop(0)
        result.append(current)
        
        # Update in-degrees of children
        for var, parents in dag.items():
            if current in parents:
                in_degree[var] -= 1
                if in_degree[var] == 0:
                    queue.append(var)
    
    return result