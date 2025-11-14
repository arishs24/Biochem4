"""
Biological parameters for neuroblastoma HSP90 inhibitor simulation.
All values are based on published literature.
"""

# Tumor Growth Parameters
BASELINE_GROWTH_RATE = 0.03  # per day (range: 0.015 to 0.045)
CARRYING_CAPACITY = 1e11  # cells
BASE_APOPTOSIS_RATE = 0.001  # per day

# HSP90 Dependency Multipliers (tumor sensitivity)
DEPENDENCY_HIGH_MYCN = 1.0
DEPENDENCY_ALK_MUTATED = 0.8
DEPENDENCY_ATRX_ALTERED = 0.5
DEPENDENCY_LOW_RISK = 0.2

# Protein Half-Lives (minutes) - Baseline (without HSP90 inhibition)
PROTEIN_HALF_LIVES_BASELINE = {
    'MYCN': 60,      # minutes
    'ALK': 240,      # 4 hours
    'AKT': 360,      # 6 hours
    'HIF1A': 30      # minutes
}

# Protein Half-Lives (minutes) - Under HSP90 Inhibition
PROTEIN_HALF_LIVES_INHIBITED = {
    'MYCN': 15,      # minutes
    'ALK': 60,       # 1 hour
    'AKT': 120,      # 2 hours
    'HIF1A': 10      # minutes
}

# HSP90 Inhibitor Pharmacokinetic Parameters
# 17-AAG
PK_17AAG = {
    'peak_time': 1.0,      # hours
    'half_life': 4.0,     # hours
    'ic50': 100.0,        # nM
    'name': '17-AAG'
}

# XL-888
PK_XL888 = {
    'peak_time': 1.0,      # hours
    'half_life': 4.0,     # hours
    'ic50': 60.0,         # nM (average of 40-80 range)
    'name': 'XL-888'
}

# Debio-0932
PK_DEBIO0932 = {
    'peak_time': 1.0,      # hours
    'half_life': 4.0,     # hours
    'ic50': 50.0,         # nM
    'name': 'Debio-0932'
}

# Dose-Response Parameters
HILL_COEFFICIENT = 1.2
E_MAX = 1.0  # Maximum effect (scaled by dependency)

# Apoptosis Parameters
APOPTOSIS_DELAY = 12.0  # hours before apoptosis increase
APOPTOSIS_MULTIPLIER = 1.5  # multiplier based on dependency

# Simulation Defaults
DEFAULT_SIMULATION_DURATION = 30  # days
DEFAULT_TIME_STEP = 0.1  # days
DEFAULT_DOSE = 100.0  # nM
DEFAULT_DOSING_INTERVAL = 24.0  # hours

