"""
Tumor growth and apoptosis dynamics model.
"""

import numpy as np
from typing import List, Optional
from src.utils.parameters import (
    CARRYING_CAPACITY,
    BASE_APOPTOSIS_COEFFICIENT,
    APOPTOSIS_GAIN_COEFFICIENT
)
from src.models.subtypes import TumorSubtype


class TumorModel:
    """Models tumor growth and response to HSP90 inhibition."""
    
    def __init__(self, subtype: TumorSubtype, initial_volume: float = 1e9):
        """
        Initialize tumor model.
        
        Args:
            subtype: Tumor subtype with growth parameters
            initial_volume: Initial tumor volume in cells
        """
        self.subtype = subtype
        self.volume = initial_volume
        self.initial_volume = initial_volume
        self.growth_rate = subtype.growth_rate
        self.carrying_capacity = CARRYING_CAPACITY
        self.dependency = subtype.dependency
        self.base_apoptosis_coeff = BASE_APOPTOSIS_COEFFICIENT
        self.apoptosis_gain_coeff = APOPTOSIS_GAIN_COEFFICIENT
    
    def calculate_growth_rate(
        self,
        drug_effect: float,
        mycn_level: Optional[float] = None
    ) -> float:
        """
        Calculate effective growth rate under drug effect.
        
        Growth depends on MYCN level:
        growth = r * V * (1 - V/K) * MYCN_factor
        
        When MYCN drops below 0.3, growth becomes near zero.
        
        Args:
            drug_effect: Current drug effect (0-1)
            mycn_level: Current MYCN protein level (0-1), if None uses 1.0
            
        Returns:
            Growth rate per day
        """
        # Logistic growth factor
        volume_factor = 1.0 - (self.volume / self.carrying_capacity)
        
        # MYCN-dependent growth factor
        if mycn_level is not None:
            # Clamp MYCN level and use as growth factor
            # When MYCN < 0.3, growth is severely reduced
            mycn_factor = max(0.0, min(1.0, mycn_level))
        else:
            # Default to full growth if MYCN not provided
            mycn_factor = 1.0
        
        growth = self.growth_rate * self.volume * volume_factor * mycn_factor
        return max(0.0, growth)
    
    def calculate_apoptosis_rate(
        self,
        drug_effect: float,
        time_hours: float
    ) -> float:
        """
        Calculate apoptosis rate with strong drug-dependent response.
        
        Apoptosis formula:
        apoptosis = base_apoptosis + apoptosis_gain * (effect^2)
        
        where:
        base_apoptosis = 0.001 * V
        apoptosis_gain = 0.05 * dependency
        
        Args:
            drug_effect: Current drug effect (0-1)
            time_hours: Current time in hours (not used but kept for compatibility)
            
        Returns:
            Apoptosis rate in cells per day
        """
        # Base apoptosis
        base_apoptosis = self.base_apoptosis_coeff * self.volume
        
        # Apoptosis gain (quadratic in effect for strong response)
        apoptosis_gain = self.apoptosis_gain_coeff * self.dependency
        apoptosis_increase = apoptosis_gain * (drug_effect ** 2) * self.volume
        
        apoptosis = base_apoptosis + apoptosis_increase
        return apoptosis
    
    def update(
        self,
        drug_effect: float,
        time_hours: float,
        time_step_days: float,
        mycn_level: Optional[float] = None
    ) -> float:
        """
        Update tumor volume for one time step.
        
        dV/dt = growth - apoptosis
        V = max(V + dVdt * dt, 0)
        
        Args:
            drug_effect: Current drug effect (0-1)
            time_hours: Current time in hours
            time_step_days: Time step size in days
            mycn_level: Current MYCN protein level (0-1), if None uses 1.0
            
        Returns:
            New tumor volume in cells
        """
        growth = self.calculate_growth_rate(drug_effect, mycn_level)
        apoptosis = self.calculate_apoptosis_rate(drug_effect, time_hours)
        
        # Update volume: dV/dt = growth - apoptosis
        delta_volume = (growth - apoptosis) * time_step_days
        self.volume = max(0.0, self.volume + delta_volume)
        
        return self.volume
    
    def reset(self):
        """Reset tumor to initial volume."""
        self.volume = self.initial_volume
    
    def get_volume_mm3(self) -> float:
        """
        Convert cell count to approximate volume in mm³.
        
        Assumes ~1e6 cells per mm³ (approximate for tumor tissue).
        """
        return self.volume / 1e6

