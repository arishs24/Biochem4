"""
Tumor growth and apoptosis dynamics model.
"""

import numpy as np
from typing import List
from src.utils.parameters import (
    CARRYING_CAPACITY,
    BASE_APOPTOSIS_RATE,
    APOPTOSIS_DELAY,
    APOPTOSIS_MULTIPLIER
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
        self.base_apoptosis = BASE_APOPTOSIS_RATE
    
    def calculate_growth_rate(self, drug_effect: float) -> float:
        """
        Calculate effective growth rate under drug effect.
        
        Growth is reduced proportionally to drug effect:
        growth = r * V * (1 - V/K) * (1 - E(t))
        
        Args:
            drug_effect: Current drug effect (0-1)
            
        Returns:
            Growth rate per day
        """
        # Logistic growth with drug inhibition
        volume_factor = 1.0 - (self.volume / self.carrying_capacity)
        drug_inhibition = 1.0 - drug_effect
        
        growth = self.growth_rate * self.volume * volume_factor * drug_inhibition
        return max(0.0, growth)
    
    def calculate_apoptosis_rate(
        self,
        drug_effect: float,
        time_hours: float
    ) -> float:
        """
        Calculate apoptosis rate with delayed drug effect.
        
        Apoptosis increases after delay:
        apoptosis = base + dependency * E(t) * multiplier (if delay passed)
        
        Args:
            drug_effect: Current drug effect (0-1)
            time_hours: Current time in hours
            
        Returns:
            Apoptosis rate per day
        """
        base = self.base_apoptosis
        
        # Apoptosis increase only after delay
        if time_hours >= APOPTOSIS_DELAY:
            apoptosis_increase = (
                self.dependency *
                drug_effect *
                APOPTOSIS_MULTIPLIER *
                self.base_apoptosis
            )
        else:
            # Linear ramp-up during delay period
            delay_progress = time_hours / APOPTOSIS_DELAY
            apoptosis_increase = (
                self.dependency *
                drug_effect *
                APOPTOSIS_MULTIPLIER *
                self.base_apoptosis *
                delay_progress
            )
        
        apoptosis = base + apoptosis_increase
        return apoptosis * self.volume
    
    def update(
        self,
        drug_effect: float,
        time_hours: float,
        time_step_days: float
    ) -> float:
        """
        Update tumor volume for one time step.
        
        dV/dt = growth - apoptosis
        
        Args:
            drug_effect: Current drug effect (0-1)
            time_hours: Current time in hours
            time_step_days: Time step size in days
            
        Returns:
            New tumor volume in cells
        """
        growth = self.calculate_growth_rate(drug_effect)
        apoptosis = self.calculate_apoptosis_rate(drug_effect, time_hours)
        
        # Convert apoptosis rate to per-day if needed
        # (apoptosis is already in cells/day)
        
        # Update volume
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

