"""
HSP90 inhibitor pharmacokinetic and pharmacodynamic models.
"""

import numpy as np
from typing import Dict, List
from src.utils.parameters import HILL_COEFFICIENT, E_MAX


class DrugModel:
    """Models HSP90 inhibitor pharmacokinetics and pharmacodynamics."""
    
    def __init__(self, pk_params: Dict):
        """
        Initialize drug model.
        
        Args:
            pk_params: Dictionary with 'peak_time', 'half_life', 'ic50', 'name'
        """
        self.pk_params = pk_params
        self.name = pk_params['name']
        self.peak_time = pk_params['peak_time']  # hours
        self.half_life = pk_params['half_life']  # hours
        self.ic50 = pk_params['ic50']  # nM
        self.elimination_rate = np.log(2) / self.half_life  # per hour
    
    def calculate_concentration(
        self,
        time_hours: float,
        dose: float,
        dosing_times: List[float]
    ) -> float:
        """
        Calculate drug concentration at given time using one-compartment PK model.
        
        Args:
            time_hours: Current time in hours
            dose: Dose amount in nM
            dosing_times: List of dosing times in hours
            
        Returns:
            Total concentration from all doses (nM)
        """
        total_concentration = 0.0
        
        for dose_time in dosing_times:
            if time_hours >= dose_time:
                # Time since this dose
                time_since_dose = time_hours - dose_time
                
                # Peak concentration occurs at peak_time after dose
                if time_since_dose <= self.peak_time:
                    # Rising phase (linear approximation to peak)
                    c_max = dose * (time_since_dose / self.peak_time)
                else:
                    # Elimination phase
                    time_in_elimination = time_since_dose - self.peak_time
                    c_max = dose * np.exp(-self.elimination_rate * time_in_elimination)
                
                total_concentration += c_max
        
        return total_concentration
    
    def calculate_effect(
        self,
        concentration: float,
        dependency: float
    ) -> float:
        """
        Calculate drug effect using Hill equation.
        
        Args:
            concentration: Current drug concentration (nM)
            dependency: Tumor HSP90 dependency (0-1)
            
        Returns:
            Effect magnitude (0-1), scaled by dependency
        """
        if concentration <= 0:
            return 0.0
        
        # Hill equation: E = Emax * C^h / (C^h + IC50^h)
        h = HILL_COEFFICIENT
        c_h = concentration ** h
        ic50_h = self.ic50 ** h
        
        effect = E_MAX * c_h / (c_h + ic50_h)
        
        # Scale by dependency
        return effect * dependency
    
    def generate_dosing_schedule(
        self,
        start_time: float,
        end_time: float,
        interval_hours: float
    ) -> List[float]:
        """
        Generate dosing schedule.
        
        Args:
            start_time: Start time in hours
            end_time: End time in hours
            interval_hours: Dosing interval in hours
            
        Returns:
            List of dosing times in hours
        """
        dosing_times = []
        current_time = start_time
        
        while current_time <= end_time:
            dosing_times.append(current_time)
            current_time += interval_hours
        
        return dosing_times

