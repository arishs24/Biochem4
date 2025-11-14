"""
Neuroblastoma tumor subtype definitions with biological parameters.
"""

from typing import Dict
from src.utils.parameters import (
    DEPENDENCY_HIGH_MYCN,
    DEPENDENCY_ALK_MUTATED,
    DEPENDENCY_ATRX_ALTERED,
    DEPENDENCY_LOW_RISK,
    BASELINE_GROWTH_RATE
)


class TumorSubtype:
    """Represents a neuroblastoma tumor subtype with specific characteristics."""
    
    def __init__(
        self,
        name: str,
        dependency: float,
        growth_rate: float,
        pathway_weights: Dict[str, float]
    ):
        """
        Initialize tumor subtype.
        
        Args:
            name: Subtype name
            dependency: HSP90 dependency multiplier (0-1)
            growth_rate: Baseline growth rate per day
            pathway_weights: Relative importance of each protein pathway
        """
        self.name = name
        self.dependency = dependency
        self.growth_rate = growth_rate
        self.pathway_weights = pathway_weights
    
    def __repr__(self):
        return f"TumorSubtype(name='{self.name}', dependency={self.dependency:.2f})"


# Define neuroblastoma subtypes
MYCN_AMPLIFIED = TumorSubtype(
    name="MYCN Amplified (High Risk)",
    dependency=DEPENDENCY_HIGH_MYCN,
    growth_rate=BASELINE_GROWTH_RATE * 1.2,  # More aggressive
    pathway_weights={
        'MYCN': 0.5,
        'ALK': 0.2,
        'AKT': 0.2,
        'HIF1A': 0.1
    }
)

ALK_MUTATED = TumorSubtype(
    name="ALK Mutated",
    dependency=DEPENDENCY_ALK_MUTATED,
    growth_rate=BASELINE_GROWTH_RATE * 1.1,
    pathway_weights={
        'MYCN': 0.2,
        'ALK': 0.5,
        'AKT': 0.2,
        'HIF1A': 0.1
    }
)

ATRX_ALTERED = TumorSubtype(
    name="ATRX Altered",
    dependency=DEPENDENCY_ATRX_ALTERED,
    growth_rate=BASELINE_GROWTH_RATE * 0.9,
    pathway_weights={
        'MYCN': 0.2,
        'ALK': 0.2,
        'AKT': 0.3,
        'HIF1A': 0.3
    }
)

LOW_RISK = TumorSubtype(
    name="Low Risk Subtype",
    dependency=DEPENDENCY_LOW_RISK,
    growth_rate=BASELINE_GROWTH_RATE * 0.7,
    pathway_weights={
        'MYCN': 0.15,
        'ALK': 0.15,
        'AKT': 0.35,
        'HIF1A': 0.35
    }
)

# Registry of all subtypes
SUBTYPE_REGISTRY = {
    'MYCN Amplified': MYCN_AMPLIFIED,
    'ALK Mutated': ALK_MUTATED,
    'ATRX Altered': ATRX_ALTERED,
    'Low Risk': LOW_RISK
}


def get_subtype(name: str) -> TumorSubtype:
    """Get tumor subtype by name."""
    return SUBTYPE_REGISTRY.get(name, MYCN_AMPLIFIED)

