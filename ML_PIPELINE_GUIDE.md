# ML Pipeline Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your data files in the following structure:

```
data/
├── gdsc/
│   ├── GDSC1_fitted_dose_response.xlsx
│   ├── GDSC2_fitted_dose_response.xlsx
│   ├── screened_compounds_rel.xlsx
│   └── Cell_Lines_Details.xlsx
└── depmap/
    ├── CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz
    ├── CCLE_ABSOLUTE_combined_20181227.xlsx
    └── Cell_lines_annotations_20181226.txt
```

### 3. Run ML Pipeline

#### Step 1: Preprocess DepMap Data

```bash
python -m src.ml.preprocess
```

This will:
- Load RNA expression and copy number data
- Normalize gene names
- Filter low-expression genes
- Create unified feature matrix

#### Step 2: Merge GDSC and DepMap

```bash
python -m src.ml.merge_datasets
```

This will:
- Load GDSC drug response data
- Identify HSP90 inhibitors
- Extract IC50 values
- Match cell line IDs
- Create ML-ready datasets

#### Step 3: Train Models

```bash
python -m src.ml.train_model
```

This will:
- Train XGBoost models for each HSP90 inhibitor
- Evaluate performance
- Save models to `models/` directory
- Save feature importance and metrics

### 4. Use Trained Models

#### In Python:

```python
from src.ml.predict_patient import predict_all_inhibitors, convert_to_digital_twin_params
import pandas as pd

# Load patient data
patient_data = pd.read_csv("patient_omics.csv", index_col=0)

# Predict
predictions = predict_all_inhibitors(patient_data, model_dir="models")

# Convert to digital twin parameters
dt_params = convert_to_digital_twin_params(predictions)

print(f"Recommended drug: {dt_params['recommended_drug']}")
print(f"Predicted IC50: {dt_params['predicted_ic50_nm']:.1f} nM")
print(f"Dependency: {dt_params['dependency']:.3f}")
```

#### In Streamlit:

1. Open the app: `streamlit run app.py`
2. Navigate to "Personalized Therapy via ML" tab
3. Upload patient omics data
4. Click "Predict Sensitivity"
5. Review predictions and apply to simulation

## Data Format Requirements

### Patient Omics Data

CSV/TSV/Excel file with:
- **Rows**: Patient samples (typically 1 row)
- **Columns**: Gene names (e.g., MYCN, ALK, AKT, HIF1A)
- **Values**: 
  - Gene expression: TPM values (will be log2 transformed)
  - Copy number: Absolute copy number values

Example:
```csv
,MYCN,ALK,AKT,HIF1A
Patient_1,5.2,4.8,6.1,3.9
```

## Troubleshooting

### No models found
- Ensure you've run `train_model.py` first
- Check that models are saved in `models/` directory

### Cell line ID mismatch
- The pipeline attempts automatic ID matching
- Check debug output for matched vs. unmatched cell lines
- May need manual ID mapping for some cell lines

### Missing features in patient data
- Missing features are filled with 0.0
- This may reduce prediction accuracy
- Ensure patient data includes key genes (MYCN, ALK, AKT, etc.)

### Large file loading
- RNA expression files are large (~2GB compressed)
- Ensure sufficient RAM (8GB+ recommended)
- Processing may take several minutes

## Model Performance

Models are evaluated using:
- **RMSE**: Root mean squared error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)
- **MAE**: Mean absolute error (lower is better)

Typical performance:
- R²: 0.3-0.6 (moderate predictive power)
- RMSE: 0.5-1.5 log units

## Integration with Digital Twin

ML predictions modify:
1. **HSP90 Dependency**: `dependency = baseline * sensitivity_score`
2. **Drug Selection**: Most sensitive inhibitor recommended
3. **IC50**: Can be used to adjust drug model (future)

The digital twin then simulates tumor response using these personalized parameters.

## Advanced Usage

### Custom Feature Selection

Modify `preprocess.py` to:
- Select specific gene sets
- Adjust TPM thresholds
- Include/exclude copy number features

### Pan-Cancer vs. Neuroblastoma-Only

In `preprocess.py` and `merge_datasets.py`:
- Set `filter_neuroblastoma=False` for pan-cancer training
- Set `filter_neuroblastoma=True` for neuroblastoma-only

### Model Hyperparameters

In `train_model.py`, adjust:
- `n_estimators`: Number of boosting rounds
- `max_depth`: Tree depth
- `learning_rate`: Learning rate

## File Structure

After running the pipeline:

```
models/
├── 17-AAG_model.pkl
├── 17-AAG_metrics.json
├── 17-AAG_importance.csv
├── AUY922_model.pkl
├── AUY922_metrics.json
└── ...
```

## References

- GDSC: https://www.cancerrxgene.org/
- DepMap: https://depmap.org/portal/
- XGBoost: https://xgboost.readthedocs.io/


