# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## First Simulation

1. Open the app in your browser (usually `http://localhost:8501`)

2. In the sidebar:
   - Select **"MYCN Amplified (High Risk)"** as the tumor subtype
   - Select **"17-AAG"** as the drug
   - Keep default dose (100 nM) and interval (24 hours)
   - Set duration to 30 days

3. Click **"Run Simulation"**

4. Observe:
   - Tumor volume decreases (high dependency = strong response)
   - Drug concentration peaks after each dose
   - Protein stability (MYCN, ALK, AKT, HIF1A) decreases under treatment
   - Apoptosis increases after 12-hour delay

## Key Features to Explore

- **Different Subtypes**: Compare MYCN Amplified vs Low Risk to see dependency differences
- **Different Drugs**: Try XL-888 or Debio-0932 with different IC50 values
- **Dosing Frequency**: Change interval to 12 hours for more frequent dosing
- **Dose Escalation**: Increase dose to 200-300 nM to see stronger effects
- **Advanced Parameters**: Override dependency to test sensitivity scenarios

## Understanding the Output

- **Tumor Volume**: Shows growth/regression in mm³
- **Drug Concentration**: PK profile showing drug levels over time
- **Drug Effect**: Therapeutic effect (0-1) based on Hill equation
- **Protein Stability**: How oncogenic proteins degrade (lower = faster degradation)
- **Growth vs Apoptosis**: Balance between cell division and death

## Troubleshooting

- If imports fail, ensure you're in the project root directory
- If Streamlit doesn't start, check that all dependencies are installed
- For faster simulations, reduce duration or increase time step

