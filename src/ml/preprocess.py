"""
Preprocess DepMap (CCLE) omics data for ML model training.

Loads and processes:
- RNA expression (TPM)
- Copy number variation (ABSOLUTE)
- Cell line annotations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_rna_expression(
    file_path: str,
    min_tpm: float = 0.1,
    top_n_genes: Optional[int] = None
) -> pd.DataFrame:
    """
    Load and normalize CCLE RNA expression data.
    
    Args:
        file_path: Path to CCLE_RNAseq_rsem_genes_tpm file
        min_tpm: Minimum TPM threshold for filtering
        top_n_genes: If specified, select top N most variable genes
        
    Returns:
        DataFrame with cell lines as rows, genes as columns
    """
    print(f"Loading RNA expression from {file_path}...")
    
    # Load compressed file
    df = pd.read_csv(file_path, sep='\t', compression='gzip', low_memory=False)
    
    # First column is gene name/ID, rest are cell lines
    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    
    # Transpose: rows = cell lines, columns = genes
    df = df.T
    
    # Normalize gene names (remove version numbers if present)
    df.columns = [str(col).split('.')[0] for col in df.columns]
    
    # Convert all columns to numeric, coercing errors to NaN
    print(f"  Converting {df.shape[1]} columns to numeric...")
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns that are all NaN (couldn't be converted)
    initial_cols = df.shape[1]
    df = df.dropna(axis=1, how='all')
    if df.shape[1] < initial_cols:
        print(f"  Dropped {initial_cols - df.shape[1]} non-numeric columns")
    
    # Drop rows (cell lines) that are all NaN
    initial_rows = df.shape[0]
    df = df.dropna(axis=0, how='all')
    if df.shape[0] < initial_rows:
        print(f"  Dropped {initial_rows - df.shape[0]} cell lines with all NaN")
    
    # Fill remaining NaN with 0 (for genes with missing values in some cell lines)
    df = df.fillna(0)
    
    print(f"  Data shape after conversion: {df.shape}")
    
    # Filter low expression genes
    if min_tpm > 0:
        mean_expression = df.mean(axis=0)
        df = df.loc[:, mean_expression >= min_tpm]
        print(f"  Filtered to {df.shape[1]} genes with mean TPM >= {min_tpm}")
    
    # Select top variable genes if specified
    if top_n_genes and top_n_genes < df.shape[1]:
        gene_variance = df.var(axis=0)
        top_genes = gene_variance.nlargest(top_n_genes).index
        df = df[top_genes]
        print(f"  Selected top {top_n_genes} most variable genes")
    
    # Log2 transform (add pseudocount)
    df = np.log2(df + 1)
    
    print(f"  Final RNA expression shape: {df.shape}")
    return df


def load_copy_number(
    file_path: str,
    gene_mapping: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Load CCLE copy number data (ABSOLUTE).
    
    Args:
        file_path: Path to CCLE_ABSOLUTE_combined file
        gene_mapping: Optional gene name mapping DataFrame
        
    Returns:
        DataFrame with cell lines as rows, genes as columns (CNV values)
    """
    print(f"Loading copy number from {file_path}...")
    
    # Load Excel file
    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
    
    print(f"  Raw data shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:10])}")
    
    # Check if data is in long format (needs pivoting)
    # Look for cell line and gene columns
    cell_line_cols = ['CCLE_ID', 'DepMap_ID', 'Cell Line', 'cell_line', 'CellLine']
    gene_cols = ['Gene', 'gene', 'Gene Symbol', 'Hugo_Symbol', 'SYMBOL']
    value_cols = ['Copy Number', 'CNV', 'Absolute CN', 'copy_number', 'CN']
    
    cell_line_col = None
    gene_col = None
    value_col = None
    
    # Find cell line column
    for col in cell_line_cols:
        if col in df.columns:
            cell_line_col = col
            break
    if cell_line_col is None:
        # Try first column
        cell_line_col = df.columns[0]
    
    # Find gene column
    for col in gene_cols:
        if col in df.columns:
            gene_col = col
            break
    if gene_col is None:
        # Look for column with 'gene' in name (case insensitive)
        for col in df.columns:
            if 'gene' in str(col).lower():
                gene_col = col
                break
    
    # Find value column
    for col in value_cols:
        if col in df.columns:
            value_col = col
            break
    if value_col is None:
        # Look for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Exclude cell line column if it's numeric
            value_col = [c for c in numeric_cols if c != cell_line_col][0] if len(numeric_cols) > 1 else numeric_cols[0]
    
    # If we have cell_line, gene, and value columns, pivot
    if cell_line_col and gene_col and value_col:
        print(f"  Detected long format: pivoting using {cell_line_col} x {gene_col}")
        # Pivot to wide format
        df_pivot = df.pivot_table(
            index=cell_line_col,
            columns=gene_col,
            values=value_col,
            aggfunc='first'  # Take first value if duplicates
        )
        df = df_pivot
        print(f"  Pivoted shape: {df.shape}")
    else:
        # Assume wide format: first column is cell line, rest are genes
        print(f"  Assuming wide format: first column is cell line")
        cell_line_col = df.columns[0]
        df = df.set_index(cell_line_col)
    
    # Normalize gene names (remove version numbers if present)
    df.columns = [str(col).split('.')[0] for col in df.columns]
    
    # Convert all columns to numeric, coercing errors to NaN
    print(f"  Converting {df.shape[1]} columns to numeric...")
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns that are all NaN
    initial_cols = df.shape[1]
    df = df.dropna(axis=1, how='all')
    if df.shape[1] < initial_cols:
        print(f"  Dropped {initial_cols - df.shape[1]} non-numeric columns")
    
    # Ensure index is unique (handle duplicates)
    if df.index.duplicated().any():
        print(f"  Warning: {df.index.duplicated().sum()} duplicate indices found, taking mean")
        df = df.groupby(df.index).mean()
    
    # Fill NaN with 0 (neutral copy number)
    df = df.fillna(0)
    
    print(f"  Final copy number shape: {df.shape}")
    return df


def load_cell_line_annotations(file_path: str) -> pd.DataFrame:
    """
    Load CCLE cell line annotations.
    
    Args:
        file_path: Path to Cell_lines_annotations file
        
    Returns:
        DataFrame with cell line metadata
    """
    print(f"Loading cell line annotations from {file_path}...")
    
    # Try different separators
    try:
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
    except:
        try:
            df = pd.read_csv(file_path, sep=',', low_memory=False)
        except:
            df = pd.read_excel(file_path, engine='openpyxl')
    
    print(f"  Annotations shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}...")
    
    return df


def normalize_cell_line_ids(
    depmap_ids: pd.Series,
    annotation_df: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Normalize DepMap cell line IDs to match GDSC format.
    
    Args:
        depmap_ids: Series of DepMap cell line IDs
        annotation_df: Optional annotations DataFrame with ID mappings
        
    Returns:
        Normalized cell line IDs
    """
    normalized = depmap_ids.copy()
    
    # Common transformations:
    # - Remove suffixes like "_SKIN", "_LUNG", etc.
    # - Convert to uppercase
    # - Handle special characters
    
    normalized = normalized.str.upper()
    normalized = normalized.str.replace(r'_[A-Z]+$', '', regex=True)
    normalized = normalized.str.replace(r'[^A-Z0-9]', '', regex=True)
    
    return normalized


def filter_neuroblastoma(
    df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    primary_disease_col: str = 'primary_disease'
) -> pd.DataFrame:
    """
    Filter cell lines to only neuroblastoma.
    
    Args:
        df: DataFrame with cell lines as index
        annotation_df: Annotations DataFrame
        primary_disease_col: Column name for primary disease
        
    Returns:
        Filtered DataFrame
    """
    if primary_disease_col not in annotation_df.columns:
        print(f"  Warning: {primary_disease_col} not found, returning all cell lines")
        return df
    
    # Find neuroblastoma cell lines
    nb_mask = annotation_df[primary_disease_col].str.contains(
        'neuroblastoma', case=False, na=False
    )
    nb_cell_lines = annotation_df[nb_mask].index
    
    # Filter main dataframe
    common_lines = df.index.intersection(nb_cell_lines)
    df_filtered = df.loc[common_lines]
    
    print(f"  Filtered to {len(common_lines)} neuroblastoma cell lines")
    return df_filtered


def merge_omics_features(
    rna_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    annotation_df: Optional[pd.DataFrame] = None,
    filter_nb: bool = False
) -> pd.DataFrame:
    """
    Merge RNA expression and copy number into unified feature matrix.
    
    Args:
        rna_df: RNA expression DataFrame
        cnv_df: Copy number DataFrame
        annotation_df: Optional annotations DataFrame
        filter_nb: If True, filter to neuroblastoma only
        
    Returns:
        Unified DataFrame with combined features
    """
    print("Merging omics features...")
    
    # Find common cell lines
    common_cell_lines = rna_df.index.intersection(cnv_df.index)
    print(f"  Common cell lines: {len(common_cell_lines)}")
    
    if len(common_cell_lines) == 0:
        raise ValueError("No common cell lines between RNA and CNV data")
    
    # Subset to common cell lines
    rna_subset = rna_df.loc[common_cell_lines].copy()
    cnv_subset = cnv_df.loc[common_cell_lines].copy()
    
    # Ensure indices are unique
    if rna_subset.index.duplicated().any():
        print(f"  Warning: {rna_subset.index.duplicated().sum()} duplicate RNA indices, taking mean")
        rna_subset = rna_subset.groupby(rna_subset.index).mean()
    if cnv_subset.index.duplicated().any():
        print(f"  Warning: {cnv_subset.index.duplicated().sum()} duplicate CNV indices, taking mean")
        cnv_subset = cnv_subset.groupby(cnv_subset.index).mean()
    
    # Re-align indices
    common_cell_lines = rna_subset.index.intersection(cnv_subset.index)
    rna_subset = rna_subset.loc[common_cell_lines]
    cnv_subset = cnv_subset.loc[common_cell_lines]
    
    # Find common genes
    common_genes = rna_subset.columns.intersection(cnv_subset.columns)
    print(f"  Common genes: {len(common_genes)}")
    
    if len(common_genes) == 0:
        print("  Warning: No common genes found. Using all genes separately.")
        # Use all genes from both, with different suffixes
        rna_features = rna_subset.add_suffix('_RNA')
        cnv_features = cnv_subset.add_suffix('_CNV')
    else:
        # Merge: RNA + CNV for common genes
        rna_features = rna_subset[common_genes].add_suffix('_RNA')
        cnv_features = cnv_subset[common_genes].add_suffix('_CNV')
    
    # Combine - ensure indices match
    merged_df = pd.concat([rna_features, cnv_features], axis=1, join='inner')
    
    # Remove any remaining duplicates
    if merged_df.index.duplicated().any():
        merged_df = merged_df.groupby(merged_df.index).mean()
    
    # Filter to neuroblastoma if requested
    if filter_nb and annotation_df is not None:
        merged_df = filter_neuroblastoma(merged_df, annotation_df)
    
    print(f"  Final merged features shape: {merged_df.shape}")
    return merged_df


def preprocess_depmap_data(
    data_dir: str = "data/depmap",
    filter_neuroblastoma: bool = False,
    top_n_genes: Optional[int] = 5000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing function for DepMap data.
    
    Args:
        data_dir: Directory containing DepMap files
        filter_neuroblastoma: If True, filter to neuroblastoma only
        top_n_genes: Number of top variable genes to select
        
    Returns:
        Tuple of (feature_matrix, annotations)
    """
    data_path = Path(data_dir)
    
    # Load RNA expression
    rna_file = data_path / "CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz"
    if not rna_file.exists():
        raise FileNotFoundError(f"RNA expression file not found: {rna_file}")
    
    rna_df = load_rna_expression(rna_file, top_n_genes=top_n_genes)
    
    # Load copy number
    cnv_file = data_path / "CCLE_ABSOLUTE_combined_20181227.xlsx"
    if not cnv_file.exists():
        raise FileNotFoundError(f"Copy number file not found: {cnv_file}")
    
    cnv_df = load_copy_number(cnv_file)
    
    # Load annotations
    annot_file = data_path / "Cell_lines_annotations_20181226.txt"
    if not annot_file.exists():
        print(f"Warning: Annotations file not found: {annot_file}")
        annotation_df = None
    else:
        annotation_df = load_cell_line_annotations(annot_file)
        # Try to set appropriate index - use CCLE_ID to match RNA/CNV data
        # The RNA/CNV data uses CCLE names as indices (like "22RV1_PROSTATE")
        if 'CCLE_ID' in annotation_df.columns:
            annotation_df = annotation_df.set_index('CCLE_ID')
        elif 'depMapID' in annotation_df.columns:
            annotation_df = annotation_df.set_index('depMapID')
        elif 'Name' in annotation_df.columns:
            annotation_df = annotation_df.set_index('Name')
    
    # Merge features
    feature_df = merge_omics_features(
        rna_df, cnv_df, annotation_df, filter_nb=filter_neuroblastoma
    )
    
    return feature_df, annotation_df


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("DepMap Data Preprocessing")
    print("=" * 60)
    
    try:
        features, annotations = preprocess_depmap_data(
            data_dir="data/depmap",
            filter_neuroblastoma=False,  # Use all cancer types for training
            top_n_genes=5000
        )
        
        print("\nPreprocessing complete!")
        print(f"Feature matrix shape: {features.shape}")
        print(f"Sample cell lines: {list(features.index[:5])}")
        print(f"Sample features: {list(features.columns[:10])}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

