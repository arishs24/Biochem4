# Neuroblastoma HSP90 Inhibitor Therapy Simulation

A comprehensive Python simulation tool that models neuroblastoma tumor response to HSP90 inhibitor therapy using real biological parameters from published literature. This project creates a medically realistic digital twin that predicts tumor growth, HSP90 dependency, protein stability, and drug effects over time.

## Overview

### What is HSP90?

Heat Shock Protein 90 (HSP90) is a molecular chaperone that stabilizes numerous client proteins, many of which are oncogenic drivers in cancer. HSP90 helps maintain the proper folding and stability of proteins like MYCN, ALK, AKT, and HIF1A, which are critical for tumor cell survival and proliferation.

### Why Does Neuroblastoma Depend on HSP90?

Neuroblastoma, especially high-risk subtypes, exhibits high dependency on HSP90 because:

1. **Oncogene Addiction**: High-risk neuroblastomas often have amplified MYCN or mutated ALK, which produce unstable oncoproteins that require HSP90 for stability.
2. **Protein Instability**: These oncoproteins have short half-lives and depend on HSP90 chaperone function to remain functional.
3. **Tumor Microenvironment**: Neuroblastoma cells in hypoxic conditions rely on HIF1A, which is also stabilized by HSP90.

When HSP90 is inhibited, these client proteins become unstable and are rapidly degraded, leading to:
- Reduced proliferation
- Increased apoptosis
- Tumor regression

## Biological Models

### Tumor Growth Model

The simulation uses a **logistic growth model** with drug inhibition:

```
dV/dt = growth - apoptosis

growth = r * V * (1 - V/K) * (1 - E(t))
apoptosis = base_apoptosis + dependency * E(t) * multiplier
```

**Parameters:**
- Baseline growth rate (r): 0.03 per day (range: 0.015-0.045)
- Carrying capacity (K): 1×10¹¹ cells
- Growth is inhibited proportionally to drug effect E(t)
- Apoptosis increases after a 12-hour delay

### HSP90 Dependency

Different neuroblastoma subtypes have varying sensitivity to HSP90 inhibition:

| Subtype | Dependency | Characteristics |
|---------|------------|-----------------|
| MYCN Amplified | 1.0 | Highest dependency, most aggressive |
| ALK Mutated | 0.8 | Moderate-high dependency |
| ATRX Altered | 0.5 | Moderate dependency |
| Low Risk | 0.2 | Low dependency, less aggressive |

### Protein Stability Model

Oncogenic client proteins have different half-lives that change under HSP90 inhibition:

**Baseline Half-Lives (without inhibition):**
- MYCN: 60 minutes
- ALK: 4 hours (240 minutes)
- AKT: 6 hours (360 minutes)
- HIF1A: 30 minutes

**Inhibited Half-Lives (with HSP90 inhibition):**
- MYCN: 15 minutes (4× faster degradation)
- ALK: 1 hour (4× faster degradation)
- AKT: 2 hours (3× faster degradation)
- HIF1A: 10 minutes (3× faster degradation)

The stability is calculated using exponential decay:
```
stability(t) = exp(-t * ln(2) / half_life)
```

### Drug Pharmacokinetics

The model uses a **one-compartment pharmacokinetic model**:

```
C(t) = Cmax * exp(-k * t)
```

where `k = ln(2) / half_life`

**HSP90 Inhibitor Properties:**

| Drug | IC50 (nM) | Half-Life (hours) | Peak Time (hours) |
|------|-----------|-------------------|-------------------|
| 17-AAG | 100 | 4 | 1 |
| XL-888 | 60 | 4 | 1 |
| Debio-0932 | 50 | 4 | 1 |

### Dose-Response Model

Drug effect is calculated using the **Hill equation**:

```
E(t) = Emax * C(t)^h / (C(t)^h + IC50^h)
```

- **h (Hill coefficient)**: 1.2
- **Emax**: Scaled by tumor dependency (0-1)
- Effect is proportional to concentration and tumor sensitivity

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── src/
    ├── models/
    │   ├── tumor_model.py      # Tumor growth and apoptosis
    │   ├── drug_model.py       # PK/PD models
    │   ├── pathways.py         # Protein stability
    │   └── subtypes.py         # Tumor subtype definitions
    ├── ui/
    │   └── dashboard.py        # Streamlit UI components
    └── utils/
        ├── parameters.py       # Biological constants
        └── plotting.py         # Visualization functions
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

### Running a Simulation

1. **Select Tumor Subtype**: Choose from MYCN Amplified, ALK Mutated, ATRX Altered, or Low Risk
2. **Select Drug**: Choose 17-AAG, XL-888, or Debio-0932
3. **Adjust Dosing Parameters**:
   - Dose amount (nM)
   - Dosing interval (hours)
4. **Set Simulation Duration**: 7-90 days
5. **Click "Run Simulation"**

### Understanding the Results

The dashboard displays:

- **Tumor Volume**: Shows tumor growth/regression over time
- **Drug Concentration**: Pharmacokinetic profile showing drug levels
- **Drug Effect**: Calculated therapeutic effect (0-1 scale)
- **Protein Stability**: How oncogenic proteins degrade under treatment
- **Growth vs Apoptosis**: Dynamic balance between cell division and death

### Example Simulations

#### High-Risk MYCN Amplified Tumor with 17-AAG
- **Subtype**: MYCN Amplified
- **Drug**: 17-AAG
- **Dose**: 100 nM
- **Interval**: 24 hours
- **Expected**: Strong initial response due to high dependency (1.0), rapid MYCN degradation

#### Low-Risk Subtype with XL-888
- **Subtype**: Low Risk
- **Drug**: XL-888
- **Dose**: 100 nM
- **Interval**: 24 hours
- **Expected**: Minimal response due to low dependency (0.2), continued growth

#### ALK Mutated with Debio-0932
- **Subtype**: ALK Mutated
- **Drug**: Debio-0932
- **Dose**: 150 nM
- **Interval**: 12 hours (more frequent dosing)
- **Expected**: Moderate response, ALK protein degradation, delayed apoptosis increase

## Model Limitations

This simulation is a **simplified model** for educational and research purposes:

1. **Simplified PK Model**: Uses one-compartment model; real drugs may have more complex kinetics
2. **Protein Synthesis**: Assumes constant synthesis rates; doesn't model feedback loops
3. **Tumor Heterogeneity**: Models homogeneous tumors; real tumors are heterogeneous
4. **Resistance Mechanisms**: Doesn't account for acquired resistance
5. **Toxicity**: Doesn't model drug toxicity or side effects
6. **Immune System**: Doesn't include immune response effects

## Biological Data Sources

All parameters are based on published literature:

- Neuroblastoma growth rates: Multiple studies (average 0.015-0.045/day)
- HSP90 dependency: Studies on MYCN-amplified neuroblastoma cell lines
- Protein half-lives: Published degradation kinetics
- Drug IC50 values: Cell line screening studies
- Pharmacokinetics: Clinical trial data for HSP90 inhibitors

## Future Enhancements

Potential improvements:

- [ ] Add resistance mechanisms
- [ ] Include combination therapy (multiple drugs)
- [ ] Model tumor heterogeneity
- [ ] Add patient-specific parameters
- [ ] Include toxicity modeling
- [ ] Export simulation data
- [ ] Batch simulation mode

## License

This project is for educational and research purposes. Please cite relevant biological literature when using this model.

## Contact

For questions or suggestions about the simulation model, please refer to the biological literature on neuroblastoma and HSP90 inhibitors.

---

**Disclaimer**: This simulation is a research tool and should not be used for clinical decision-making. Always consult with medical professionals for actual treatment decisions.

