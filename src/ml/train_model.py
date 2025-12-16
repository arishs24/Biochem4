"""
Train XGBoost models to predict HSP90 inhibitor sensitivity.

Trains separate models for each HSP90 inhibitor.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import pickle
import json
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = RANDOM_SEED
) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train XGBoost model for IC50 prediction.
    
    Args:
        X: Feature matrix
        y: Target IC50 values
        test_size: Fraction for test set
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f"Training XGBoost model...")
    print(f"  Training samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    print(f"  Train RMSE: {train_rmse:.3f}, R²: {train_r2:.3f}")
    print(f"  Test RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}, MAE: {test_mae:.3f}")
    
    return model, metrics


def get_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: pd.Index,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and rank feature importance.
    
    Args:
        model: Trained XGBoost model
        feature_names: Feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def train_all_models(
    ml_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    model_dir: str = "models",
    save_models: bool = True
) -> Dict[str, Dict]:
    """
    Train models for all HSP90 inhibitors.
    
    Args:
        ml_datasets: Dictionary mapping inhibitor name to (X, y) tuple
        model_dir: Directory to save models
        save_models: Whether to save models to disk
        
    Returns:
        Dictionary mapping inhibitor name to model info
    """
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    
    results = {}
    
    for inhibitor_name, (X, y) in ml_datasets.items():
        print(f"\n{'='*60}")
        print(f"Training model for {inhibitor_name}")
        print(f"{'='*60}")
        
        if len(X) < 20:
            print(f"  Skipping {inhibitor_name}: insufficient samples ({len(X)})")
            continue
        
        # Train model
        model, metrics = train_xgboost_model(X, y)
        
        # Get feature importance
        importance_df = get_feature_importance(model, X.columns)
        print(f"\n  Top 10 features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        if save_models:
            model_file = model_path / f"{inhibitor_name.replace('-', '_')}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved model to {model_file}")
            
            # Save feature importance
            importance_file = model_path / f"{inhibitor_name.replace('-', '_')}_importance.csv"
            importance_df.to_csv(importance_file, index=False)
            
            # Save metrics
            metrics_file = model_path / f"{inhibitor_name.replace('-', '_')}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        results[inhibitor_name] = {
            'model': model,
            'metrics': metrics,
            'importance': importance_df,
            'feature_names': X.columns.tolist()
        }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("HSP90 Inhibitor Sensitivity Model Training")
    print("=" * 60)
    
    # Load merged datasets
    from src.ml.merge_datasets import merge_gdsc_depmap
    
    try:
        ml_datasets = merge_gdsc_depmap(
            gdsc_dir="data/gdsc",
            filter_neuroblastoma=False
        )
        
        if not ml_datasets:
            print("No datasets available for training")
        else:
            # Train models
            results = train_all_models(ml_datasets, model_dir="models")
            
            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            for inhibitor_name, result in results.items():
                print(f"\n{inhibitor_name}:")
                print(f"  Test R²: {result['metrics']['test_r2']:.3f}")
                print(f"  Test RMSE: {result['metrics']['test_rmse']:.3f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


