"""
Plotting utilities for simulation visualization.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional


def plot_tumor_volume(
    time_days: List[float],
    volumes: List[float],
    title: str = "Tumor Volume Over Time"
) -> go.Figure:
    """
    Plot tumor volume over time.
    
    Args:
        time_days: Time points in days
        volumes: Tumor volumes in cells
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Convert to mm³ for display
    volumes_mm3 = [v / 1e6 for v in volumes]
    
    fig.add_trace(go.Scatter(
        x=time_days,
        y=volumes_mm3,
        mode='lines',
        name='Tumor Volume',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='Day: %{x:.1f}<br>Volume: %{y:.2f} mm³<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (days)',
        yaxis_title='Tumor Volume (mm³)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_drug_concentration(
    time_hours: List[float],
    concentrations: List[float],
    drug_name: str = "Drug",
    title: str = "Drug Concentration Over Time"
) -> go.Figure:
    """
    Plot drug concentration over time.
    
    Args:
        time_hours: Time points in hours
        concentrations: Drug concentrations in nM
        drug_name: Name of drug
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_hours,
        y=concentrations,
        mode='lines',
        name=f'{drug_name} Concentration',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        hovertemplate='Time: %{x:.1f} h<br>Concentration: %{y:.2f} nM<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (hours)',
        yaxis_title='Concentration (nM)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_protein_stability(
    time_hours: List[float],
    protein_levels: Dict[str, List[float]],
    title: str = "Oncogenic Protein Stability"
) -> go.Figure:
    """
    Plot protein stability levels over time.
    
    Args:
        time_hours: Time points in hours
        protein_levels: Dictionary mapping protein names to stability levels
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = ['#9b59b6', '#e67e22', '#16a085', '#c0392b']
    
    for i, (protein, levels) in enumerate(protein_levels.items()):
        fig.add_trace(go.Scatter(
            x=time_hours[:len(levels)],
            y=levels,
            mode='lines',
            name=protein,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'{protein}: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (hours)',
        yaxis_title='Relative Stability',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_dynamics(
    time_days: List[float],
    growth_rates: List[float],
    apoptosis_rates: List[float],
    title: str = "Growth vs Apoptosis Dynamics"
) -> go.Figure:
    """
    Plot growth and apoptosis rates over time.
    
    Args:
        time_days: Time points in days
        growth_rates: Growth rates (cells/day)
        apoptosis_rates: Apoptosis rates (cells/day)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_days,
        y=[g / 1e6 for g in growth_rates],  # Convert to millions
        mode='lines',
        name='Growth Rate',
        line=dict(color='#27ae60', width=2),
        hovertemplate='Growth: %{y:.2f} M cells/day<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_days,
        y=[a / 1e6 for a in apoptosis_rates],  # Convert to millions
        mode='lines',
        name='Apoptosis Rate',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='Apoptosis: %{y:.2f} M cells/day<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (days)',
        yaxis_title='Rate (M cells/day)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_comprehensive_dashboard(
    time_days: List[float],
    time_hours: List[float],
    volumes: List[float],
    concentrations: List[float],
    drug_effects: List[float],
    protein_levels: Dict[str, List[float]],
    growth_rates: List[float],
    apoptosis_rates: List[float],
    drug_name: str = "Drug"
) -> go.Figure:
    """
    Create comprehensive dashboard with all metrics.
    
    Args:
        time_days: Time points in days
        time_hours: Time points in hours
        volumes: Tumor volumes in cells
        concentrations: Drug concentrations in nM
        drug_effects: Drug effects (0-1)
        protein_levels: Protein stability levels
        growth_rates: Growth rates
        apoptosis_rates: Apoptosis rates
        drug_name: Name of drug
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Tumor Volume',
            'Drug Concentration & Effect',
            'Protein Stability',
            'Growth vs Apoptosis',
            'Drug Effect Over Time',
            'Volume Change Rate'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Tumor Volume
    volumes_mm3 = [v / 1e6 for v in volumes]
    fig.add_trace(
        go.Scatter(x=time_days, y=volumes_mm3, name='Volume (mm³)',
                  line=dict(color='#e74c3c')),
        row=1, col=1
    )
    
    # Drug Concentration & Effect
    fig.add_trace(
        go.Scatter(x=time_hours, y=concentrations, name='Concentration (nM)',
                  line=dict(color='#3498db')),
        row=1, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=time_hours, y=drug_effects, name='Effect',
                  line=dict(color='#9b59b6', dash='dash')),
        row=1, col=2, secondary_y=True
    )
    
    # Protein Stability
    colors = ['#9b59b6', '#e67e22', '#16a085', '#c0392b']
    for i, (protein, levels) in enumerate(protein_levels.items()):
        fig.add_trace(
            go.Scatter(x=time_hours[:len(levels)], y=levels, name=protein,
                      line=dict(color=colors[i % len(colors)])),
            row=2, col=1
        )
    
    # Growth vs Apoptosis
    fig.add_trace(
        go.Scatter(x=time_days, y=[g / 1e6 for g in growth_rates],
                  name='Growth (M cells/day)', line=dict(color='#27ae60')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=time_days, y=[a / 1e6 for a in apoptosis_rates],
                  name='Apoptosis (M cells/day)', line=dict(color='#e74c3c')),
        row=2, col=2
    )
    
    # Drug Effect
    fig.add_trace(
        go.Scatter(x=time_hours, y=drug_effects, name='Drug Effect',
                  line=dict(color='#9b59b6'), fill='tozeroy'),
        row=3, col=1
    )
    
    # Volume Change Rate
    volume_changes = []
    for i in range(1, len(volumes)):
        change = (volumes[i] - volumes[i-1]) / (time_days[i] - time_days[i-1])
        volume_changes.append(change / 1e6)
    fig.add_trace(
        go.Scatter(x=time_days[1:], y=volume_changes,
                  name='Volume Change (M cells/day)', line=dict(color='#f39c12')),
        row=3, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (days)", row=1, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
    fig.update_xaxes(title_text="Time (days)", row=3, col=2)
    
    fig.update_yaxes(title_text="Volume (mm³)", row=1, col=1)
    fig.update_yaxes(title_text="Concentration (nM)", row=1, col=2)
    fig.update_yaxes(title_text="Effect", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Stability", row=2, col=1)
    fig.update_yaxes(title_text="Rate (M cells/day)", row=2, col=2)
    fig.update_yaxes(title_text="Effect", row=3, col=1)
    fig.update_yaxes(title_text="Change (M cells/day)", row=3, col=2)
    
    fig.update_layout(
        height=1200,
        title_text=f"Neuroblastoma HSP90 Inhibitor Simulation - {drug_name}",
        template='plotly_white',
        showlegend=True
    )
    
    return fig

