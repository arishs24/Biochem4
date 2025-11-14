"""
Streamlit dashboard UI components.
"""

import streamlit as st
from typing import Dict, List, Tuple
from src.models.subtypes import SUBTYPE_REGISTRY
from src.utils.parameters import (
    PK_17AAG,
    PK_XL888,
    PK_DEBIO0932
)


def render_sidebar_controls() -> Dict:
    """
    Render sidebar controls and return parameter dictionary.
    
    Returns:
        Dictionary with simulation parameters
    """
    st.sidebar.header("Simulation Parameters")
    
    # Tumor subtype selection
    subtype_names = list(SUBTYPE_REGISTRY.keys())
    selected_subtype_name = st.sidebar.selectbox(
        "Tumor Subtype",
        subtype_names,
        index=0,
        help="Select the neuroblastoma subtype to simulate"
    )
    selected_subtype = SUBTYPE_REGISTRY[selected_subtype_name]
    
    # Display subtype info
    st.sidebar.info(
        f"**{selected_subtype_name}**\n\n"
        f"Dependency: {selected_subtype.dependency:.2f}\n"
        f"Growth Rate: {selected_subtype.growth_rate:.4f} per day"
    )
    
    # Drug selection
    st.sidebar.subheader("Drug Selection")
    drug_options = {
        '17-AAG': PK_17AAG,
        'XL-888': PK_XL888,
        'Debio-0932': PK_DEBIO0932
    }
    selected_drug_name = st.sidebar.selectbox(
        "HSP90 Inhibitor",
        list(drug_options.keys()),
        index=0
    )
    selected_drug_pk = drug_options[selected_drug_name]
    
    # Dosing parameters
    st.sidebar.subheader("Dosing Parameters")
    dose = st.sidebar.slider(
        "Dose (nM)",
        min_value=10.0,
        max_value=500.0,
        value=100.0,
        step=10.0,
        help="Drug concentration per dose"
    )
    
    dosing_interval = st.sidebar.slider(
        "Dosing Interval (hours)",
        min_value=6.0,
        max_value=48.0,
        value=24.0,
        step=6.0,
        help="Time between doses"
    )
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Settings")
    duration_days = st.sidebar.slider(
        "Simulation Duration (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )
    
    time_step = st.sidebar.slider(
        "Time Step (hours)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Simulation resolution"
    )
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        dependency_override = st.slider(
            "HSP90 Dependency Override",
            min_value=0.0,
            max_value=1.0,
            value=selected_subtype.dependency,
            step=0.05,
            help="Override subtype dependency (0-1)"
        )
        
        initial_volume = st.number_input(
            "Initial Tumor Volume (cells)",
            min_value=1e6,
            max_value=1e10,
            value=1e9,
            step=1e8,
            format="%.0e"
        )
    
    # Display parameters
    st.sidebar.subheader("Drug Properties")
    st.sidebar.write(f"**{selected_drug_name}**")
    st.sidebar.write(f"IC50: {selected_drug_pk['ic50']:.1f} nM")
    st.sidebar.write(f"Half-life: {selected_drug_pk['half_life']:.1f} hours")
    
    return {
        'subtype': selected_subtype,
        'subtype_name': selected_subtype_name,
        'drug_name': selected_drug_name,
        'drug_pk': selected_drug_pk,
        'dose': dose,
        'dosing_interval': dosing_interval,
        'duration_days': duration_days,
        'time_step_hours': time_step,
        'dependency': dependency_override,
        'initial_volume': initial_volume
    }


def render_main_dashboard(
    time_days: List[float],
    time_hours: List[float],
    volumes: List[float],
    concentrations: List[float],
    drug_effects: List[float],
    protein_levels: Dict[str, List[float]],
    growth_rates: List[float],
    apoptosis_rates: List[float],
    parameters: Dict,
    show_protein_stability: bool = True
):
    """
    Render main dashboard with all visualizations.
    
    Args:
        time_days: Time points in days
        time_hours: Time points in hours
        volumes: Tumor volumes
        concentrations: Drug concentrations
        drug_effects: Drug effects
        protein_levels: Protein stability levels
        growth_rates: Growth rates
        apoptosis_rates: Apoptosis rates
        parameters: Simulation parameters
        show_protein_stability: Whether to show protein stability plot
    """
    from src.utils.plotting import (
        plot_tumor_volume,
        plot_drug_concentration,
        plot_protein_stability,
        plot_dynamics,
        plot_comprehensive_dashboard
    )
    
    st.title("Neuroblastoma HSP90 Inhibitor Therapy Simulation")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    final_volume = volumes[-1] / 1e6  # mm³
    initial_volume = volumes[0] / 1e6
    volume_change = ((final_volume - initial_volume) / initial_volume) * 100
    
    with col1:
        st.metric("Initial Volume", f"{initial_volume:.2f} mm³")
    with col2:
        st.metric("Final Volume", f"{final_volume:.2f} mm³")
    with col3:
        st.metric("Volume Change", f"{volume_change:+.1f}%")
    with col4:
        max_effect = max(drug_effects) if drug_effects else 0.0
        st.metric("Peak Drug Effect", f"{max_effect:.2%}")
    
    # Main plots
    st.subheader("Tumor Response")
    fig_volume = plot_tumor_volume(
        time_days,
        volumes,
        title=f"Tumor Volume - {parameters['subtype_name']}"
    )
    st.plotly_chart(fig_volume, use_container_width=True)
    
    st.subheader("Drug Pharmacokinetics")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_conc = plot_drug_concentration(
            time_hours,
            concentrations,
            drug_name=parameters['drug_name'],
            title=f"{parameters['drug_name']} Concentration"
        )
        st.plotly_chart(fig_conc, use_container_width=True)
    
    with col2:
        from plotly import graph_objects as go
        fig_effect = go.Figure()
        fig_effect.add_trace(go.Scatter(
            x=time_hours,
            y=drug_effects,
            mode='lines',
            name='Drug Effect',
            line=dict(color='#9b59b6', width=2),
            fill='tozeroy',
            hovertemplate='Effect: %{y:.3f}<extra></extra>'
        ))
        fig_effect.update_layout(
            title="Drug Effect Over Time",
            xaxis_title='Time (hours)',
            yaxis_title='Effect (0-1)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_effect, use_container_width=True)
    
    if show_protein_stability:
        st.subheader("Oncogenic Protein Stability")
        fig_protein = plot_protein_stability(
            time_hours,
            protein_levels,
            title="Protein Stability Under HSP90 Inhibition"
        )
        st.plotly_chart(fig_protein, use_container_width=True)
    
    st.subheader("Tumor Dynamics")
    fig_dynamics = plot_dynamics(
        time_days,
        growth_rates,
        apoptosis_rates,
        title="Growth vs Apoptosis Rates"
    )
    st.plotly_chart(fig_dynamics, use_container_width=True)
    
    # Comprehensive dashboard option
    with st.expander("View Comprehensive Dashboard"):
        fig_comprehensive = plot_comprehensive_dashboard(
            time_days,
            time_hours,
            volumes,
            concentrations,
            drug_effects,
            protein_levels,
            growth_rates,
            apoptosis_rates,
            drug_name=parameters['drug_name']
        )
        st.plotly_chart(fig_comprehensive, use_container_width=True)

