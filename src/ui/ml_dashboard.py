"""
Streamlit UI for ML-powered personalized therapy.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import io


def render_ml_upload_section() -> Optional[pd.DataFrame]:
    """
    Render file upload section for patient omics data.
    
    Returns:
        DataFrame with patient data or None
    """
    st.subheader("Upload Patient Omics Data")
    
    st.markdown("""
    Upload patient gene expression and copy number data.
    Supported formats: CSV, TSV, Excel
    
    **Required format:**
    - Rows: Samples/patients (typically 1 row)
    - Columns: Gene names (e.g., MYCN, ALK, AKT)
    - Values: Expression (TPM) or copy number
    """)
    
    uploaded_file = st.file_uploader(
        "Choose patient omics file",
        type=['csv', 'tsv', 'xlsx', 'txt'],
        help="Upload patient gene expression and/or copy number data"
    )
    
    if uploaded_file is not None:
        try:
            # Detect file type
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            if file_ext == '.xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_ext == '.tsv' or (file_ext == '.txt' and 'tsv' in uploaded_file.name):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display preview
            with st.expander("Preview uploaded data"):
                st.dataframe(df.head(10))
            
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    return None


def render_prediction_results(
    predictions_df: pd.DataFrame,
    dt_params: Dict
):
    """
    Render ML prediction results.
    
    Args:
        predictions_df: DataFrame with predictions for all inhibitors
        dt_params: Digital twin parameters derived from predictions
    """
    st.subheader("Predicted Sensitivity")
    
    if predictions_df.empty:
        st.warning("No predictions available")
        return
    
    # Display predictions table
    st.markdown("### IC50 Predictions for All HSP90 Inhibitors")
    
    display_df = predictions_df.copy()
    display_df['IC50 (nM)'] = np.exp(display_df['predicted_ic50']) * 1000
    display_df = display_df[['inhibitor', 'IC50 (nM)', 'sensitivity_score']].copy()
    display_df.columns = ['Inhibitor', 'Predicted IC50 (nM)', 'Sensitivity Score']
    display_df = display_df.sort_values('Sensitivity Score', ascending=False)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Highlight best inhibitor
    if dt_params:
        st.markdown("### Recommended Therapy")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Recommended Drug",
                dt_params.get('recommended_drug', 'N/A')
            )
        
        with col2:
            st.metric(
                "Predicted IC50",
                f"{dt_params.get('predicted_ic50_nm', 0):.1f} nM"
            )
        
        with col3:
            st.metric(
                "Sensitivity Score",
                f"{dt_params.get('sensitivity_score', 0):.3f}"
            )
        
        # Display dependency
        st.info(f"""
        **Predicted HSP90 Dependency:** {dt_params.get('dependency', 0):.3f}
        
        This value will be used to adjust the tumor subtype sensitivity in the simulation.
        Higher dependency indicates greater sensitivity to HSP90 inhibition.
        """)


def render_ml_integration_controls(
    dt_params: Dict,
    default_parameters: Dict
) -> Dict:
    """
    Render controls for integrating ML predictions into simulation.
    
    Args:
        dt_params: Digital twin parameters from ML predictions
        default_parameters: Default simulation parameters
        
    Returns:
        Updated parameters dictionary
    """
    st.subheader("Apply ML Predictions to Simulation")
    
    use_ml_predictions = st.checkbox(
        "Use ML-predicted parameters",
        value=True,
        help="Apply ML predictions to adjust simulation parameters"
    )
    
    if use_ml_predictions and dt_params:
        # Override drug selection
        recommended_drug = dt_params.get('recommended_drug')
        if recommended_drug:
            st.info(f"**Recommended Drug:** {recommended_drug}")
            
            # Map to available drugs
            drug_mapping = {
                '17-AAG': '17-AAG',
                'AUY922': '17-AAG',  # Use 17-AAG as proxy
                'IPI504': '17-AAG',  # Use 17-AAG as proxy
                'Ganetespib': '17-AAG'  # Use 17-AAG as proxy
            }
            
            mapped_drug = drug_mapping.get(recommended_drug, '17-AAG')
            default_parameters['drug_name'] = mapped_drug
        
        # Override dependency
        predicted_dependency = dt_params.get('dependency', default_parameters.get('dependency', 1.0))
        default_parameters['dependency'] = predicted_dependency
        
        # Override IC50 if available
        predicted_ic50_nm = dt_params.get('predicted_ic50_nm')
        if predicted_ic50_nm:
            st.info(f"**Predicted IC50:** {predicted_ic50_nm:.1f} nM")
            # Note: IC50 modification would require updating drug_model.py
            # For now, we adjust dependency which affects effect calculation
        
        # Show adjusted parameters
        with st.expander("View adjusted parameters"):
            st.json({
                'dependency': predicted_dependency,
                'recommended_drug': recommended_drug,
                'predicted_ic50_nm': predicted_ic50_nm
            })
    
    return default_parameters


def render_ml_tab():
    """
    Main function to render ML-powered personalized therapy tab.
    """
    st.header("Personalized Therapy via ML")
    
    st.markdown("""
    This module uses machine learning models trained on GDSC drug response data
    and DepMap omics data to predict patient-specific HSP90 inhibitor sensitivity.
    
    **Workflow:**
    1. Upload patient omics data (gene expression + copy number)
    2. ML models predict IC50 for each HSP90 inhibitor
    3. Predictions are converted to digital twin parameters
    4. Simulation is automatically adjusted for personalized therapy
    """)
    
    # Upload section
    patient_data = render_ml_upload_section()
    
    # Prediction section
    if patient_data is not None:
        st.divider()
        
        if st.button("Predict Sensitivity", type="primary"):
            with st.spinner("Running ML predictions..."):
                try:
                    from src.ml.predict_patient import (
                        predict_all_inhibitors,
                        convert_to_digital_twin_params
                    )
                    
                    # Predict for all inhibitors
                    predictions_df = predict_all_inhibitors(
                        patient_data,
                        model_dir="models"
                    )
                    
                    if not predictions_df.empty:
                        # Convert to digital twin parameters
                        dt_params = convert_to_digital_twin_params(predictions_df)
                        
                        # Store in session state
                        st.session_state['ml_predictions'] = predictions_df
                        st.session_state['ml_dt_params'] = dt_params
                        
                        st.success("Predictions complete!")
                        
                        # Display results
                        render_prediction_results(predictions_df, dt_params)
                        
                    else:
                        st.warning("No predictions generated. Ensure models are trained.")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display stored predictions if available
        if 'ml_predictions' in st.session_state:
            st.divider()
            render_prediction_results(
                st.session_state['ml_predictions'],
                st.session_state.get('ml_dt_params', {})
            )
            
            # Integration controls
            st.divider()
            from src.ui.dashboard import render_sidebar_controls
            
            # Get default parameters
            default_params = render_sidebar_controls()
            
            # Apply ML predictions
            updated_params = render_ml_integration_controls(
                st.session_state.get('ml_dt_params', {}),
                default_params
            )
            
            # Store updated parameters
            st.session_state['ml_adjusted_params'] = updated_params
    
    else:
        # Show example data format
        st.info("""
        **Example Data Format:**
        
        Upload a CSV/TSV file with:
        - First row: Gene names (MYCN, ALK, AKT, HIF1A, etc.)
        - Subsequent rows: Patient samples
        - Values: Gene expression (TPM) or copy number
        
        The ML models will automatically match features and predict sensitivity.
        """)
        
        # Show sample data structure
        with st.expander("View example data structure"):
            example_data = pd.DataFrame({
                'MYCN': [5.2, 3.1],
                'ALK': [4.8, 2.9],
                'AKT': [6.1, 5.4],
                'HIF1A': [3.9, 2.7]
            }, index=['Patient_1', 'Patient_2'])
            st.dataframe(example_data)


