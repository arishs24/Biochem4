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
    'MYCN': 5,       # minutes (increased collapse)
    'ALK': 20,       # minutes (increased collapse)
    'AKT': 40,       # minutes (increased collapse)
    'HIF1A': 5       # minutes (increased collapse)
}

# HSP90 Inhibitor Pharmacokinetic Parameters
# 17-AAG
PK_17AAG = {
    'peak_time': 1.0,      # hours
    'half_life': 4.0,     # hours
    'ic50': 40.0,         # nM (clinically realistic: 30-50 nM)
    'name': '17-AAG'
}

# XL-888
PK_XL888 = {
    'peak_time': 1.0,      # hours
    'half_life': 4.0,     # hours
    'ic50': 25.0,         # nM (clinically realistic: 20-30 nM)
    'name': 'XL-888'
}

# Debio-0932
PK_DEBIO0932 = {
    'peak_time': 1.0,      # hours
    'half_life': 4.0,     # hours
    'ic50': 7.5,          # nM (clinically realistic: 5-10 nM)
    'name': 'Debio-0932'
}

# Dose-Response Parameters
HILL_COEFFICIENT = 2.5  # Steep dose-response curve

# Apoptosis Parameters
BASE_APOPTOSIS_COEFFICIENT = 0.001  # Base apoptosis rate per cell
APOPTOSIS_GAIN_COEFFICIENT = 0.05  # Apoptosis gain multiplier

# Simulation Defaults
DEFAULT_SIMULATION_DURATION = 30  # days
DEFAULT_TIME_STEP = 0.1  # days
DEFAULT_DOSE = 300.0  # nM (higher for therapeutic effect)
DEFAULT_DOSING_INTERVAL = 12.0  # hours (more frequent dosing)

