"""
Merge GDSC drug response data with DepMap omics features.

Identifies HSP90 inhibitors and creates ML-ready dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# HSP90 inhibitor identifiers in GDSC
HSP90_INHIBITORS = {
    '17-AAG': ['17-AAG', 'Tanespimycin', '17AAG', '17AAG1'],
    'AUY922': ['AUY922', 'Luminespib', 'NVP-AUY922'],
    'IPI504': ['IPI504', 'Retaspimycin'],
    'Ganetespib': ['Ganetespib', 'STA-9090']
}


def load_gdsc_dose_response(
    gdsc1_file: str,
    gdsc2_file: str
) -> pd.DataFrame:
    """
    Load and combine GDSC1 and GDSC2 dose response data.
    
    Args:
        gdsc1_file: Path to GDSC1_fitted_dose_response file
        gdsc2_file: Path to GDSC2_fitted_dose_response file
        
    Returns:
        Combined DataFrame with drug response data
    """
    print("Loading GDSC dose response data...")
    
    # Load GDSC1
    print(f"  Loading GDSC1 from {gdsc1_file}...")
    try:
        gdsc1 = pd.read_excel(gdsc1_file, engine='openpyxl')
        gdsc1['Dataset'] = 'GDSC1'
        print(f"    GDSC1 shape: {gdsc1.shape}")
    except Exception as e:
        print(f"    Error loading GDSC1: {e}")
        gdsc1 = pd.DataFrame()
    
    # Load GDSC2
    print(f"  Loading GDSC2 from {gdsc2_file}...")
    try:
        gdsc2 = pd.read_excel(gdsc2_file, engine='openpyxl')
        gdsc2['Dataset'] = 'GDSC2'
        print(f"    GDSC2 shape: {gdsc2.shape}")
    except Exception as e:
        print(f"    Error loading GDSC2: {e}")
        gdsc2 = pd.DataFrame()
    
    # Combine
    if not gdsc1.empty and not gdsc2.empty:
        # Ensure same columns
        common_cols = gdsc1.columns.intersection(gdsc2.columns)
        combined = pd.concat([
            gdsc1[common_cols],
            gdsc2[common_cols]
        ], ignore_index=True)
    elif not gdsc1.empty:
        combined = gdsc1
    elif not gdsc2.empty:
        combined = gdsc2
    else:
        raise ValueError("Both GDSC files failed to load")
    
    print(f"  Combined shape: {combined.shape}")
    print(f"  Columns: {list(combined.columns)[:10]}...")
    
    return combined


def identify_hsp90_inhibitors(
    drug_response_df: pd.DataFrame,
    drug_name_col: str = 'DRUG_NAME'
) -> Dict[str, pd.DataFrame]:
    """
    Identify HSP90 inhibitor entries in GDSC data.
    
    Args:
        drug_response_df: GDSC drug response DataFrame
        drug_name_col: Column name for drug names
        
    Returns:
        Dictionary mapping inhibitor name to filtered DataFrame
    """
    print("Identifying HSP90 inhibitors...")
    
    if drug_name_col not in drug_response_df.columns:
        # Try common alternatives
        for alt_col in ['DRUG_NAME', 'Drug Name', 'drug_name', 'compound']:
            if alt_col in drug_response_df.columns:
                drug_name_col = alt_col
                break
        else:
            raise ValueError(f"Drug name column not found. Available: {drug_response_df.columns.tolist()}")
    
    hsp90_data = {}
    
    for inhibitor_name, aliases in HSP90_INHIBITORS.items():
        # Find rows matching any alias
        mask = drug_response_df[drug_name_col].str.contains(
            '|'.join(aliases), case=False, na=False, regex=True
        )
        
        if mask.any():
            hsp90_data[inhibitor_name] = drug_response_df[mask].copy()
            print(f"  Found {mask.sum()} entries for {inhibitor_name}")
        else:
            print(f"  No entries found for {inhibitor_name}")
    
    return hsp90_data


def extract_ic50_values(
    hsp90_data: Dict[str, pd.DataFrame],
    ic50_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract IC50 values for each HSP90 inhibitor.
    
    Args:
        hsp90_data: Dictionary of HSP90 inhibitor DataFrames
        ic50_col: Column name for IC50 (auto-detect if None)
        
    Returns:
        DataFrame with cell lines as rows, inhibitors as columns (IC50 values)
    """
    print("Extracting IC50 values...")
    
    ic50_dict = {}
    
    for inhibitor_name, df in hsp90_data.items():
        if df.empty:
            continue
        
        # Auto-detect IC50 column
        if ic50_col is None:
            possible_cols = ['IC50', 'LN_IC50', 'IC50 (uM)', 'IC50_uM', 'IC50_rescaled']
            ic50_col = None
            for col in possible_cols:
                if col in df.columns:
                    ic50_col = col
                    break
            
            if ic50_col is None:
                print(f"  Warning: IC50 column not found for {inhibitor_name}")
                print(f"    Available columns: {df.columns.tolist()}")
                continue
        
        # Get cell line column
        cell_line_cols = ['CELL_LINE_NAME', 'Cell Line Name', 'cell_line_name', 'SANGER_MODEL_ID']
        cell_line_col = None
        for col in cell_line_cols:
            if col in df.columns:
                cell_line_col = col
                break
        
        if cell_line_col is None:
            print(f"  Warning: Cell line column not found for {inhibitor_name}")
            continue
        
        # Extract IC50 and cell line
        ic50_series = df.groupby(cell_line_col)[ic50_col].first()  # Take first if duplicates
        
        # Convert to numeric, handling any non-numeric values
        ic50_series = pd.to_numeric(ic50_series, errors='coerce')
        
        # Remove NaN
        ic50_series = ic50_series.dropna()
        
        ic50_dict[inhibitor_name] = ic50_series
        print(f"  {inhibitor_name}: {len(ic50_series)} cell lines with IC50")
    
    # Combine into DataFrame
    if ic50_dict:
        ic50_df = pd.DataFrame(ic50_dict)
        print(f"  Combined IC50 DataFrame shape: {ic50_df.shape}")
        return ic50_df
    else:
        return pd.DataFrame()


def match_cell_line_ids(
    gdsc_ids: pd.Index,
    depmap_ids: pd.Index,
    annotation_df: Optional[pd.DataFrame] = None,
    gdsc_metadata: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Match GDSC cell line IDs to DepMap IDs using annotations.
    
    Args:
        gdsc_ids: GDSC cell line IDs
        depmap_ids: DepMap cell line IDs
        annotation_df: DepMap annotations DataFrame with ID mappings
        gdsc_metadata: Optional GDSC metadata with ID mappings
        
    Returns:
        Series mapping GDSC IDs to DepMap IDs
    """
    print("Matching cell line IDs between GDSC and DepMap...")
    
    id_mapping = {}
    
    # Use annotations file if available
    if annotation_df is not None:
        print(f"  Using annotations file with {len(annotation_df)} entries")
        
        # Try different column combinations for matching
        # GDSC uses CELL_LINE_NAME which often matches CCLE_ID or Name in annotations
        # DepMap features use IDs that match depMapID or CCLE_ID
        
        # Create lookup dictionaries
        # Map from various ID formats to DepMap feature IDs
        name_to_depmap = {}
        ccle_to_depmap = {}
        
        # Build mappings from annotations
        # The annotation index should be CCLE_ID (matching feature matrix)
        for ccle_id_idx, row in annotation_df.iterrows():
            # The index is the CCLE_ID (matches feature matrix)
            depmap_id = str(ccle_id_idx) if ccle_id_idx in depmap_ids else None
            
            # Also check if depMapID column exists and matches
            if depmap_id is None and 'depMapID' in row and pd.notna(row['depMapID']):
                if row['depMapID'] in depmap_ids:
                    depmap_id = row['depMapID']
            
            # If we found a matching DepMap ID
            if depmap_id and depmap_id in depmap_ids:
                # Map Name to DepMap ID (for GDSC matching)
                if 'Name' in row and pd.notna(row['Name']):
                    name = str(row['Name']).strip().upper()
                    name_to_depmap[name] = depmap_id
                
                # Map CCLE_ID to DepMap ID (index is CCLE_ID)
                ccle_id = str(ccle_id_idx).strip().upper()
                ccle_to_depmap[ccle_id] = depmap_id
                
                # Also map the CCLE_ID column if it exists and is different
                if 'CCLE_ID' in row and pd.notna(row['CCLE_ID']):
                    ccle_col = str(row['CCLE_ID']).strip().upper()
                    if ccle_col != ccle_id:
                        ccle_to_depmap[ccle_col] = depmap_id
        
        print(f"  Built {len(name_to_depmap)} name mappings, {len(ccle_to_depmap)} CCLE mappings")
        
        # Match GDSC IDs to DepMap IDs
        for gdsc_id in gdsc_ids:
            gdsc_upper = str(gdsc_id).strip().upper()
            
            # Try direct name match
            if gdsc_upper in name_to_depmap:
                id_mapping[gdsc_id] = name_to_depmap[gdsc_upper]
                continue
            
            # Try CCLE ID match
            if gdsc_upper in ccle_to_depmap:
                id_mapping[gdsc_id] = ccle_to_depmap[gdsc_upper]
                continue
            
            # Try normalized matching (remove special chars)
            gdsc_norm = gdsc_upper.replace('_', '').replace('-', '').replace(' ', '')
            for name, depmap_id in name_to_depmap.items():
                name_norm = name.replace('_', '').replace('-', '').replace(' ', '')
                if gdsc_norm == name_norm:
                    id_mapping[gdsc_id] = depmap_id
                    break
    
    # Fallback: try direct normalized matching
    if len(id_mapping) == 0:
        print("  Trying direct normalized matching...")
        gdsc_normalized = pd.Series(gdsc_ids).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
        depmap_normalized = pd.Series(depmap_ids).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
        
        for gdsc_id, gdsc_norm in zip(gdsc_ids, gdsc_normalized):
            matches = depmap_normalized[depmap_normalized == gdsc_norm].index
            if len(matches) > 0:
                id_mapping[gdsc_id] = depmap_ids[matches[0]]
    
    print(f"  Matched {len(id_mapping)} cell lines")
    if len(id_mapping) > 0:
        print(f"  Sample matches: {list(id_mapping.items())[:5]}")
    
    return pd.Series(id_mapping)


def create_ml_dataset(
    omics_features: pd.DataFrame,
    ic50_data: pd.DataFrame,
    id_mapping: Optional[pd.Series] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create ML-ready dataset for each HSP90 inhibitor.
    
    Args:
        omics_features: DepMap omics features (rows=cell lines, cols=features)
        ic50_data: IC50 values (rows=cell lines, cols=inhibitors)
        id_mapping: Optional mapping from IC50 cell line IDs to omics IDs
        
    Returns:
        Dictionary mapping inhibitor name to (X, y) tuple
    """
    print("Creating ML-ready datasets...")
    
    ml_datasets = {}
    
    for inhibitor_name in ic50_data.columns:
        print(f"\n  Processing {inhibitor_name}...")
        
        # Get IC50 values for this inhibitor
        ic50_series = ic50_data[inhibitor_name].dropna()
        
        if len(ic50_series) == 0:
            print(f"    No IC50 data for {inhibitor_name}")
            continue
        
        # Handle duplicate indices in IC50 data (take mean if duplicates)
        if ic50_series.index.duplicated().any():
            print(f"    Warning: {inhibitor_name} has {ic50_series.index.duplicated().sum()} duplicate indices, taking mean")
            ic50_series = ic50_series.groupby(ic50_series.index).mean()
        
        # Match cell line IDs
        if id_mapping is not None and len(id_mapping) > 0:
            # Map IC50 cell line IDs to omics IDs
            # id_mapping is a Series: index=GDSC IDs, values=DepMap IDs
            ic50_mapped = ic50_series.index.map(id_mapping)
            
            # Get only the ones that were successfully mapped (not NaN)
            valid_mask = ic50_mapped.notna()
            if valid_mask.any():
                # Keep only mapped cell lines
                ic50_series = ic50_series[valid_mask].copy()
                # Set new index to DepMap IDs
                ic50_series.index = ic50_mapped[valid_mask].values
                
                # Handle any duplicates after mapping (multiple GDSC IDs -> same DepMap ID)
                if ic50_series.index.duplicated().any():
                    print(f"    Warning: {inhibitor_name} has duplicates after mapping, taking mean")
                    ic50_series = ic50_series.groupby(ic50_series.index).mean()
            else:
                print(f"    Warning: No cell lines mapped for {inhibitor_name}")
                continue
        
        # Find common cell lines
        common_cell_lines = omics_features.index.intersection(ic50_series.index)
        
        if len(common_cell_lines) == 0:
            print(f"    No common cell lines for {inhibitor_name}")
            continue
        
        # Create X, y - ensure perfect alignment
        # Re-index both to common cell lines to ensure alignment
        common_cell_lines = common_cell_lines.unique()  # Remove duplicates if any
        
        X = omics_features.loc[common_cell_lines].copy()
        y = ic50_series.loc[common_cell_lines].copy()
        
        # Ensure same length and alignment
        if len(X) != len(y):
            # Find intersection of indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
        
        # Remove any remaining NaN
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Final alignment check
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch after cleaning: {len(X)} vs {len(y)}")
        
        # Ensure indices match
        if not X.index.equals(y.index):
            common_final = X.index.intersection(y.index)
            X = X.loc[common_final]
            y = y.loc[common_final]
        
        if len(X) > 0 and len(y) > 0:
            ml_datasets[inhibitor_name] = (X, y)
            print(f"    Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        else:
            print(f"    Warning: Empty dataset for {inhibitor_name}")
    
    return ml_datasets


def merge_gdsc_depmap(
    gdsc_dir: str = "data/gdsc",
    depmap_features: Optional[pd.DataFrame] = None,
    filter_neuroblastoma: bool = False
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Main function to merge GDSC and DepMap data.
    
    Args:
        gdsc_dir: Directory containing GDSC files
        depmap_features: Preprocessed DepMap features (if None, will load)
        filter_neuroblastoma: Filter to neuroblastoma only
        
    Returns:
        Dictionary mapping inhibitor name to (X, y) tuple
    """
    gdsc_path = Path(gdsc_dir)
    
    # Load GDSC dose response
    gdsc1_file = gdsc_path / "GDSC1_fitted_dose_response.xlsx"
    gdsc2_file = gdsc_path / "GDSC2_fitted_dose_response.xlsx"
    
    # Try alternative filenames
    if not gdsc1_file.exists():
        alt_files = list(gdsc_path.glob("*GDSC1*.xlsx"))
        if alt_files:
            gdsc1_file = alt_files[0]
    
    if not gdsc2_file.exists():
        alt_files = list(gdsc_path.glob("*GDSC2*.xlsx"))
        if alt_files:
            gdsc2_file = alt_files[0]
    
    drug_response_df = load_gdsc_dose_response(str(gdsc1_file), str(gdsc2_file))
    
    # Identify HSP90 inhibitors
    hsp90_data = identify_hsp90_inhibitors(drug_response_df)
    
    if not hsp90_data:
        raise ValueError("No HSP90 inhibitors found in GDSC data")
    
    # Extract IC50 values
    ic50_df = extract_ic50_values(hsp90_data)
    
    if ic50_df.empty:
        raise ValueError("No IC50 values extracted")
    
    # Load DepMap features if not provided
    if depmap_features is None:
        from src.ml.preprocess import preprocess_depmap_data
        depmap_features, annotation_df = preprocess_depmap_data(
            filter_neuroblastoma=filter_neuroblastoma
        )
    else:
        # Try to load annotations separately
        from src.ml.preprocess import load_cell_line_annotations
        annot_file = Path("data/depmap") / "Cell_lines_annotations_20181226.txt"
        if annot_file.exists():
            annotation_df = load_cell_line_annotations(str(annot_file))
            # Set index to CCLE_ID to match feature matrix indices
            if 'CCLE_ID' in annotation_df.columns:
                annotation_df = annotation_df.set_index('CCLE_ID')
            elif 'depMapID' in annotation_df.columns:
                annotation_df = annotation_df.set_index('depMapID')
        else:
            annotation_df = None
    
    # Match cell line IDs
    id_mapping = match_cell_line_ids(
        ic50_df.index,
        depmap_features.index,
        annotation_df=annotation_df
    )
    
    # Create ML datasets
    ml_datasets = create_ml_dataset(
        depmap_features,
        ic50_df,
        id_mapping if len(id_mapping) > 0 else None
    )
    
    return ml_datasets


if __name__ == "__main__":
    print("=" * 60)
    print("GDSC-DepMap Data Merging")
    print("=" * 60)
    
    try:
        ml_datasets = merge_gdsc_depmap(
            gdsc_dir="data/gdsc",
            filter_neuroblastoma=False
        )
        
        print("\nMerging complete!")
        for inhibitor_name, (X, y) in ml_datasets.items():
            print(f"\n{inhibitor_name}:")
            print(f"  Samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
            print(f"  IC50 range: {y.min():.2f} - {y.max():.2f} (log uM)")
            print(f"  IC50 mean: {y.mean():.2f}")
        
    except Exception as e:
        print(f"Error during merging: {e}")
        import traceback
        traceback.print_exc()

