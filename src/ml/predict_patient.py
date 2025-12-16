"""
Predict HSP90 inhibitor sensitivity for individual patients.

Takes patient omics data and predicts IC50 values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_trained_model(
    inhibitor_name: str,
    model_dir: str = "models"
) -> Tuple[object, list]:
    """
    Load trained model and feature names.
    
    Args:
        inhibitor_name: Name of HSP90 inhibitor
        model_dir: Directory containing saved models
        
    Returns:
        Tuple of (model, feature_names)
    """
    model_path = Path(model_dir)
    
    # Try different filename formats
    model_file = model_path / f"{inhibitor_name.replace('-', '_')}_model.pkl"
    if not model_file.exists():
        # Try alternative
        model_file = model_path / f"{inhibitor_name}_model.pkl"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found for {inhibitor_name}")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names from metrics file
    metrics_file = model_path / f"{inhibitor_name.replace('-', '_')}_metrics.json"
    feature_names = None
    
    if metrics_file.exists():
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            feature_names = metrics.get('feature_names')
    
    # If not in metrics, try to get from model
    if feature_names is None and hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    
    return model, feature_names


def preprocess_patient_data(
    patient_data: pd.DataFrame,
    feature_names: list,
    log_transform: bool = True
) -> pd.DataFrame:
    """
    Preprocess patient data to match training data format.
    
    Args:
        patient_data: Patient omics data (genes as columns)
        feature_names: Required feature names from training
        log_transform: Whether to apply log2 transform
        
    Returns:
        Preprocessed patient data with matching features
    """
    # Ensure genes are columns
    if patient_data.index.name in ['gene', 'Gene', 'GENE']:
        patient_data = patient_data.T
    
    # Normalize gene names (remove version numbers)
    patient_data.columns = [str(col).split('.')[0] for col in patient_data.columns]
    
    # Log transform if needed
    if log_transform:
        patient_data = np.log2(patient_data + 1)
    
    # Create feature matrix matching training data
    # Training features are like: GENE_RNA, GENE_CNV
    patient_features = pd.DataFrame(index=patient_data.index)
    
    for feature in feature_names:
        if '_RNA' in feature:
            gene = feature.replace('_RNA', '')
            if gene in patient_data.columns:
                patient_features[feature] = patient_data[gene]
            else:
                patient_features[feature] = 0.0  # Missing gene
        elif '_CNV' in feature:
            gene = feature.replace('_CNV', '')
            if gene in patient_data.columns:
                patient_features[feature] = patient_data[gene]
            else:
                patient_features[feature] = 0.0  # Missing gene
        else:
            # Direct feature match
            if feature in patient_data.columns:
                patient_features[feature] = patient_data[feature]
            else:
                patient_features[feature] = 0.0
    
    return patient_features


def predict_ic50(
    patient_features: pd.DataFrame,
    inhibitor_name: str,
    model_dir: str = "models"
) -> Tuple[float, Dict]:
    """
    Predict IC50 for a patient.
    
    Args:
        patient_features: Preprocessed patient feature matrix
        inhibitor_name: Name of HSP90 inhibitor
        model_dir: Directory containing saved models
        
    Returns:
        Tuple of (predicted_IC50, prediction_info)
    """
    # Load model
    model, feature_names = load_trained_model(inhibitor_name, model_dir)
    
    # Ensure features match
    if feature_names:
        # Reorder/select features
        missing_features = set(feature_names) - set(patient_features.columns)
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features, filling with 0")
            for feat in missing_features:
                patient_features[feat] = 0.0
        
        patient_features = patient_features[feature_names]
    
    # Predict
    predicted_ic50 = model.predict(patient_features)[0]
    
    # Calculate sensitivity score (lower IC50 = more sensitive)
    # Normalize to 0-1 scale (assuming IC50 range from training)
    # This is a simplified score - can be improved with actual training data stats
    sensitivity_score = 1.0 / (1.0 + np.exp(predicted_ic50 - 5.0))  # Sigmoid transform
    
    info = {
        'predicted_ic50': float(predicted_ic50),
        'sensitivity_score': float(sensitivity_score),
        'inhibitor': inhibitor_name
    }
    
    return predicted_ic50, info


def predict_all_inhibitors(
    patient_data: pd.DataFrame,
    model_dir: str = "models",
    available_inhibitors: Optional[list] = None
) -> pd.DataFrame:
    """
    Predict IC50 for all available HSP90 inhibitors.
    
    Args:
        patient_data: Patient omics data
        model_dir: Directory containing saved models
        available_inhibitors: List of inhibitors to predict (None = all available)
        
    Returns:
        DataFrame with predictions for each inhibitor
    """
    model_path = Path(model_dir)
    
    # Find available models
    if available_inhibitors is None:
        model_files = list(model_path.glob("*_model.pkl"))
        available_inhibitors = [
            f.name.replace('_model.pkl', '').replace('_', '-')
            for f in model_files
        ]
    
    predictions = []
    
    for inhibitor_name in available_inhibitors:
        try:
            # Load model to get feature names
            model, feature_names = load_trained_model(inhibitor_name, model_dir)
            
            # Preprocess patient data
            patient_features = preprocess_patient_data(
                patient_data,
                feature_names,
                log_transform=True
            )
            
            # Predict
            ic50, info = predict_ic50(
                patient_features,
                inhibitor_name,
                model_dir
            )
            
            predictions.append(info)
            print(f"{inhibitor_name}: IC50 = {ic50:.3f}, Sensitivity = {info['sensitivity_score']:.3f}")
            
        except Exception as e:
            print(f"Error predicting for {inhibitor_name}: {e}")
            continue
    
    if predictions:
        return pd.DataFrame(predictions)
    else:
        return pd.DataFrame()


def convert_to_digital_twin_params(
    predictions_df: pd.DataFrame,
    dependency_baseline: float = 1.0
) -> Dict:
    """
    Convert ML predictions to digital twin parameters.
    
    Args:
        predictions_df: DataFrame with IC50 predictions
        dependency_baseline: Baseline dependency value
        
    Returns:
        Dictionary with digital twin parameters
    """
    if predictions_df.empty:
        return {}
    
    # Find most sensitive inhibitor (lowest IC50)
    best_inhibitor = predictions_df.loc[predictions_df['predicted_ic50'].idxmin()]
    
    # Convert IC50 to drug model parameters
    # Lower IC50 = higher sensitivity = higher dependency
    predicted_ic50 = best_inhibitor['predicted_ic50']
    sensitivity = best_inhibitor['sensitivity_score']
    
    # Map IC50 to dependency (inverse relationship)
    # IC50 in log uM, typical range: 0-10
    # Lower IC50 -> higher dependency
    dependency = dependency_baseline * sensitivity
    
    # Map IC50 to drug IC50 parameter (convert from log uM to nM)
    # Assuming IC50 is in log(uM), convert to nM
    ic50_nm = np.exp(predicted_ic50) * 1000  # Convert uM to nM
    
    params = {
        'recommended_drug': best_inhibitor['inhibitor'],
        'predicted_ic50_nm': float(ic50_nm),
        'predicted_ic50_log_um': float(predicted_ic50),
        'dependency': float(dependency),
        'sensitivity_score': float(sensitivity),
        'all_predictions': predictions_df.to_dict('records')
    }
    
    return params


if __name__ == "__main__":
    print("=" * 60)
    print("Patient Sensitivity Prediction")
    print("=" * 60)
    
    # Example: Create dummy patient data
    print("\nExample with dummy patient data...")
    
    # Load a sample to get feature structure
    try:
        from src.ml.merge_datasets import merge_gdsc_depmap
        ml_datasets = merge_gdsc_depmap(filter_neuroblastoma=False)
        
        if ml_datasets:
            # Get feature names from first dataset
            X_sample, _ = list(ml_datasets.values())[0]
            
            # Create dummy patient data
            patient_data = pd.DataFrame(
                np.random.randn(1, len(X_sample.columns)) * 2 + 5,
                columns=X_sample.columns
            )
            
            # Predict
            predictions = predict_all_inhibitors(patient_data, model_dir="models")
            
            if not predictions.empty:
                print("\nPredictions:")
                print(predictions)
                
                # Convert to digital twin params
                dt_params = convert_to_digital_twin_params(predictions)
                print("\nDigital Twin Parameters:")
                for key, value in dt_params.items():
                    if key != 'all_predictions':
                        print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


