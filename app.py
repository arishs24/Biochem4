"""
Main Streamlit application for neuroblastoma HSP90 inhibitor simulation.
"""

import streamlit as st
import numpy as np
from src.models.tumor_model import TumorModel
from src.models.drug_model import DrugModel
from src.models.pathways import ProteinStabilityModel
from src.ui.dashboard import render_sidebar_controls, render_main_dashboard
from src.utils.parameters import DEFAULT_TIME_STEP


def run_simulation(parameters: dict) -> dict:
    """
    Run the complete simulation.
    
    Args:
        parameters: Dictionary with simulation parameters
        
    Returns:
        Dictionary with simulation results
    """
    # Initialize models
    tumor = TumorModel(
        subtype=parameters['subtype'],
        initial_volume=parameters['initial_volume']
    )
    
    # Override dependency if specified
    if 'dependency' in parameters:
        tumor.dependency = parameters['dependency']
    
    drug = DrugModel(parameters['drug_pk'])
    protein_model = ProteinStabilityModel()
    
    # Simulation parameters
    duration_days = parameters['duration_days']
    time_step_hours = parameters['time_step_hours']
    time_step_days = time_step_hours / 24.0
    
    # Generate time points
    num_steps = int(duration_days / time_step_days)
    time_days = np.linspace(0, duration_days, num_steps)
    time_hours = time_days * 24.0
    
    # Generate dosing schedule
    dosing_times = drug.generate_dosing_schedule(
        start_time=0.0,
        end_time=time_hours[-1],
        interval_hours=parameters['dosing_interval']
    )
    
    # Initialize result arrays
    volumes = []
    concentrations = []
    drug_effects = []
    growth_rates = []
    apoptosis_rates = []
    
    # Run simulation
    for i, t_hours in enumerate(time_hours):
        t_days = time_days[i]
        
        # Calculate drug concentration
        concentration = drug.calculate_concentration(
            time_hours=t_hours,
            dose=parameters['dose'],
            dosing_times=dosing_times
        )
        concentrations.append(concentration)
        
        # Calculate drug effect
        effect = drug.calculate_effect(
            concentration=concentration,
            dependency=tumor.dependency
        )
        drug_effects.append(effect)
        
        # Update tumor
        current_volume = tumor.update(
            drug_effect=effect,
            time_hours=t_hours,
            time_step_days=time_step_days
        )
        volumes.append(current_volume)
        
        # Record rates
        growth = tumor.calculate_growth_rate(effect)
        apoptosis = tumor.calculate_apoptosis_rate(effect, t_hours)
        growth_rates.append(growth)
        apoptosis_rates.append(apoptosis)
    
    # Calculate protein stability levels
    protein_levels = protein_model.calculate_protein_levels(
        time_hours=time_hours[-1],
        drug_effects=drug_effects,
        time_points=time_hours.tolist()
    )
    
    return {
        'time_days': time_days.tolist(),
        'time_hours': time_hours.tolist(),
        'volumes': volumes,
        'concentrations': concentrations,
        'drug_effects': drug_effects,
        'protein_levels': protein_levels,
        'growth_rates': growth_rates,
        'apoptosis_rates': apoptosis_rates
    }


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Neuroblastoma HSP90 Inhibitor Simulation",
        page_icon="🧬",
        layout="wide"
    )
    
    # Render sidebar and get parameters
    parameters = render_sidebar_controls()
    
    # Run simulation button
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            results = run_simulation(parameters)
            
            # Store results in session state
            st.session_state['results'] = results
            st.session_state['parameters'] = parameters
    
    # Display results if available
    if 'results' in st.session_state:
        # Toggle for protein stability
        show_proteins = st.sidebar.checkbox(
            "Show Protein Stability",
            value=True,
            help="Display oncogenic protein stability curves"
        )
        
        render_main_dashboard(
            **st.session_state['results'],
            parameters=st.session_state['parameters'],
            show_protein_stability=show_proteins
        )
    else:
        # Welcome screen
        st.title("Neuroblastoma HSP90 Inhibitor Therapy Simulation")
        st.markdown("""
        ### Welcome to the Neuroblastoma Digital Twin Simulation
        
        This tool simulates how different neuroblastoma tumor subtypes respond to 
        HSP90 inhibitor therapy using real biological parameters from published literature.
        
        **To get started:**
        1. Adjust simulation parameters in the sidebar
        2. Select tumor subtype and drug
        3. Click "Run Simulation" to see results
        
        **Features:**
        - Real biological parameters from literature
        - Multiple tumor subtypes (MYCN amplified, ALK mutated, ATRX altered, Low risk)
        - Three HSP90 inhibitors (17-AAG, XL-888, Debio-0932)
        - Protein stability modeling (MYCN, ALK, AKT, HIF1A)
        - Interactive visualizations
        
        See the README for detailed information about the biological models.
        """)


if __name__ == "__main__":
    main()

