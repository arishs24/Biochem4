"""
Oncogenic client protein stability models under HSP90 inhibition.
"""

import numpy as np
from typing import Dict, List
from src.utils.parameters import (
    PROTEIN_HALF_LIVES_BASELINE,
    PROTEIN_HALF_LIVES_INHIBITED
)


class ProteinStabilityModel:
    """Models protein stability and degradation under HSP90 inhibition."""
    
    def __init__(self):
        """Initialize with baseline half-lives."""
        self.baseline_half_lives = PROTEIN_HALF_LIVES_BASELINE.copy()
        self.inhibited_half_lives = PROTEIN_HALF_LIVES_INHIBITED.copy()
        self.proteins = list(self.baseline_half_lives.keys())
    
    def calculate_stability(
        self,
        time_minutes: float,
        half_life_minutes: float,
        initial_stability: float = 1.0
    ) -> float:
        """
        Calculate protein stability using exponential decay.
        
        Formula: stability(t) = initial * exp(-t * ln(2) / half_life)
        
        Args:
            time_minutes: Time elapsed in minutes
            half_life_minutes: Half-life in minutes
            initial_stability: Initial stability level (default 1.0)
            
        Returns:
            Current stability level (0-1)
        """
        if half_life_minutes <= 0:
            return 0.0
        
        decay_constant = np.log(2) / half_life_minutes
        stability = initial_stability * np.exp(-time_minutes * decay_constant)
        return max(0.0, stability)
    
    def get_effective_half_life(
        self,
        protein_name: str,
        drug_effect: float
    ) -> float:
        """
        Calculate effective half-life based on drug effect.
        
        Args:
            protein_name: Name of protein
            drug_effect: Current drug effect (0-1)
            
        Returns:
            Effective half-life in minutes
        """
        baseline = self.baseline_half_lives[protein_name]
        inhibited = self.inhibited_half_lives[protein_name]
        
        # Interpolate between baseline and inhibited based on drug effect
        effective = baseline - (baseline - inhibited) * drug_effect
        return max(inhibited, effective)
    
    def calculate_protein_levels(
        self,
        time_hours: float,
        drug_effects: List[float],
        time_points: List[float]
    ) -> Dict[str, List[float]]:
        """
        Calculate protein stability over time with dynamic drug effects.
        
        Models proteins with ongoing synthesis and degradation.
        Protein levels represent relative stability/degradation rate.
        
        Args:
            time_hours: Current simulation time in hours
            drug_effects: List of drug effects at each time point
            time_points: List of time points in hours
            
        Returns:
            Dictionary mapping protein names to stability levels over time
        """
        protein_levels = {protein: [] for protein in self.proteins}
        
        # Initialize protein levels at steady state (synthesis = degradation)
        current_levels = {protein: 1.0 for protein in self.proteins}
        
        for i, t in enumerate(time_points):
            if t > time_hours:
                break
            
            # Get current drug effect
            if i < len(drug_effects):
                effect = drug_effects[i]
            else:
                effect = drug_effects[-1] if drug_effects else 0.0
            
            # Calculate new levels for each protein
            for protein in self.proteins:
                # Get baseline and effective half-lives
                baseline_half_life = self.baseline_half_lives[protein]
                half_life = self.get_effective_half_life(protein, effect)
                
                # Calculate time step in minutes
                if i > 0:
                    dt_minutes = (t - time_points[i-1]) * 60
                else:
                    dt_minutes = 0.0
                
                if dt_minutes > 0:
                    # Decay constant
                    decay_constant = np.log(2) / half_life
                    
                    # Synthesis rate (assumed constant, balances baseline degradation)
                    baseline_decay = np.log(2) / baseline_half_life
                    synthesis_rate = baseline_decay  # Maintains steady state at baseline
                    
                    # Update protein level: synthesis - degradation
                    # dP/dt = synthesis - decay * P
                    # Simplified Euler step
                    current_level = current_levels[protein]
                    dP = (synthesis_rate - decay_constant * current_level) * dt_minutes
                    current_level = max(0.0, current_level + dP)
                    current_levels[protein] = current_level
                
                # Store relative stability (normalized to show degradation rate effect)
                # Higher value = more stable = slower degradation
                stability_ratio = half_life / baseline_half_life
                relative_stability = current_levels[protein] * stability_ratio
                
                protein_levels[protein].append(relative_stability)
        
        return protein_levels

