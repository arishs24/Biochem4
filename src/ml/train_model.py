"""
Train XGBoost models to predict HSP90 inhibitor sensitivity.

Trains separate models for each HSP90 inhibitor.
Includes multi-model benchmarking with 5-fold cross-validation.
"""

import pandas as pd
import numpy as np
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
) -> Tuple[object, Dict]:
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
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Install libomp (macOS: brew install libomp) then reinstall xgboost.")

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
    model: object,
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


def train_benchmark_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = RANDOM_SEED
) -> Dict:
    """
    Train and cross-validate ElasticNet, RandomForest, MLP, and XGBoost
    on the same dataset using K-Fold cross-validation.

    For each model and fold:
        - Compute R², RMSE, MAE on the validation set.
    Aggregate across folds as mean ± std.

    Args:
        X: Feature matrix (pd.DataFrame)
        y: Target log(IC50) values (pd.Series)
        cv_splits: Number of K-Fold splits (default 5)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping model name → {
            r2_mean, r2_std,
            rmse_mean, rmse_std,
            mae_mean, mae_std
        }
    """
    print(f"\nRunning {cv_splits}-fold cross-validation benchmark...")
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

    model_definitions = {
        'ElasticNet': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(
                alpha=0.1, l1_ratio=0.5,
                max_iter=10000, random_state=random_state
            ))
        ]),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=8,
            random_state=random_state, n_jobs=-1
        ),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state
            ))
        ]),
    }

    if XGBOOST_AVAILABLE:
        model_definitions['XGBoost'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            objective='reg:squarederror',
            verbosity=0
        )
    else:
        print("  [warning] XGBoost unavailable (missing libomp) — skipping XGBoost model")

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_results = {}

    for model_name, base_model in model_definitions.items():
        print(f"\n  [{model_name}]")
        r2_folds, rmse_folds, mae_folds = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_f = X.iloc[train_idx]
            X_val_f   = X.iloc[val_idx]
            y_train_f = y.iloc[train_idx]
            y_val_f   = y.iloc[val_idx]

            model_fold = clone(base_model)
            model_fold.fit(X_train_f, y_train_f)
            y_pred_f = model_fold.predict(X_val_f)

            fold_r2   = r2_score(y_val_f, y_pred_f)
            fold_rmse = float(np.sqrt(mean_squared_error(y_val_f, y_pred_f)))
            fold_mae  = float(mean_absolute_error(y_val_f, y_pred_f))

            r2_folds.append(fold_r2)
            rmse_folds.append(fold_rmse)
            mae_folds.append(fold_mae)

            print(f"    Fold {fold_idx + 1}: R²={fold_r2:.3f}  "
                  f"RMSE={fold_rmse:.3f}  MAE={fold_mae:.3f}")

        cv_results[model_name] = {
            'r2_mean':   float(np.mean(r2_folds)),
            'r2_std':    float(np.std(r2_folds)),
            'rmse_mean': float(np.mean(rmse_folds)),
            'rmse_std':  float(np.std(rmse_folds)),
            'mae_mean':  float(np.mean(mae_folds)),
            'mae_std':   float(np.std(mae_folds)),
        }
        print(f"    → R²={np.mean(r2_folds):.3f}±{np.std(r2_folds):.3f}  "
              f"RMSE={np.mean(rmse_folds):.3f}±{np.std(rmse_folds):.3f}  "
              f"MAE={np.mean(mae_folds):.3f}±{np.std(mae_folds):.3f}")

    return cv_results


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


