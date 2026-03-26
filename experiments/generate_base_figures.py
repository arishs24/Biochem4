#!/usr/bin/env python3
"""
Generate Figure 3 and Figure 4 directly from the mechanistic simulation.

Figure 3 — Tumor Volume Dynamics Under HSP90 Inhibition
Figure 4 — Pharmacokinetics and Pharmacodynamic Response of HSP90 Inhibition

Settings mirror the Streamlit app defaults:
  subtype  : MYCN Amplified (High Risk)
  drug     : Debio-0932
  duration : 30 days
  dose     : 300 nM
  interval : 12-hour dosing

Run from project root:
    python experiments/generate_base_figures.py

Outputs
-------
    results/figures/figure3_tumor_volume.png
    results/figures/figure4_pk_pd.png
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tumor_model import TumorModel
from src.models.drug_model import DrugModel
from src.models.pathways import ProteinStabilityModel
from src.models.subtypes import MYCN_AMPLIFIED
from src.utils.parameters import (
    PK_DEBIO0932,
    DEFAULT_DOSE,
    DEFAULT_DOSING_INTERVAL,
    DEFAULT_SIMULATION_DURATION,
    DEFAULT_TIME_STEP,
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1-4  RUN SIMULATION — collect all raw time-series arrays
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation() -> dict:
    """
    Execute the full mechanistic simulation using the same logic as app.py
    but without any Streamlit or Plotly dependency.

    Data sources
    ────────────
    • TumorModel.update()            → tumor_volume
    • DrugModel.calculate_concentration() → drug_concentration
    • DrugModel.calculate_effect()        → drug_effect
    • ProteinStabilityModel              → MYCN stability trajectory

    Returns a dict with all raw simulation arrays.
    """
    # ── Parameters (identical to Streamlit app defaults) ────────────────────
    subtype          = MYCN_AMPLIFIED
    pk_params        = dict(PK_DEBIO0932)      # Debio-0932
    initial_volume   = 1e9                     # cells
    duration_days    = DEFAULT_SIMULATION_DURATION   # 30 days
    time_step_days   = DEFAULT_TIME_STEP             # 0.1 days
    dose_nM          = DEFAULT_DOSE                  # 300 nM
    dosing_interval  = DEFAULT_DOSING_INTERVAL       # 12 hours

    # ── Initialise models ────────────────────────────────────────────────────
    tumor         = TumorModel(subtype=subtype, initial_volume=initial_volume)
    drug          = DrugModel(pk_params)
    protein_model = ProteinStabilityModel()

    # ── Build time grid ──────────────────────────────────────────────────────
    num_steps  = int(duration_days / time_step_days)
    time_days  = np.linspace(0.0, duration_days, num_steps)
    time_hours = time_days * 24.0

    # ── Dosing schedule ──────────────────────────────────────────────────────
    dosing_times = drug.generate_dosing_schedule(
        start_time=0.0,
        end_time=time_hours[-1],
        interval_hours=dosing_interval,
    )

    # ── Output arrays ────────────────────────────────────────────────────────
    tumor_volume       = []
    drug_concentration = []
    drug_effect        = []
    apoptosis_rates    = []
    growth_rates       = []

    # Track per-protein levels across time steps (needed for MYCN-driven growth)
    current_protein = {p: 1.0 for p in protein_model.proteins}

    # ── Simulation loop (mirrors app.py:run_simulation) ─────────────────────
    for i, t_h in enumerate(time_hours):
        # PK: concentration at this time point
        conc = drug.calculate_concentration(
            time_hours=t_h,
            dose=dose_nM,
            dosing_times=dosing_times,
        )
        drug_concentration.append(conc)

        # PD: Hill-equation effect
        effect = drug.calculate_effect(
            concentration=conc,
            dependency=tumor.dependency,
        )
        drug_effect.append(effect)

        # Update protein levels (ODE Euler step, dt in minutes)
        dt_min = (t_h - time_hours[i - 1]) * 60.0 if i > 0 else 0.0
        if dt_min > 0:
            for prot in protein_model.proteins:
                hl_base = protein_model.baseline_half_lives[prot]
                hl_eff  = protein_model.get_effective_half_life(prot, effect)
                k_deg   = np.log(2) / hl_eff
                k_syn   = np.log(2) / hl_base          # constant synthesis
                lvl     = current_protein[prot]
                current_protein[prot] = max(0.0, lvl + (k_syn - k_deg * lvl) * dt_min)

        mycn_level = current_protein.get('MYCN', 1.0)

        # Tumour dynamics
        vol = tumor.update(
            drug_effect=effect,
            time_hours=t_h,
            time_step_days=time_step_days,
            mycn_level=mycn_level,
        )
        tumor_volume.append(vol)
        growth_rates.append(tumor.calculate_growth_rate(effect, mycn_level))
        apoptosis_rates.append(tumor.calculate_apoptosis_rate(effect, t_h))

    return {
        'time_days':         time_days,
        'time_hours':        time_hours,
        'tumor_volume':      np.array(tumor_volume),
        'drug_concentration': np.array(drug_concentration),
        'drug_effect':       np.array(drug_effect),
        'apoptosis_rates':   np.array(apoptosis_rates),
        'growth_rates':      np.array(growth_rates),
        'drug_name':         pk_params['name'],
        'subtype_name':      subtype.name,
        'dose_nM':           dose_nM,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STYLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_base_style() -> None:
    """Apply publication-quality matplotlib style with white background."""
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.size':        12,
        'axes.labelsize':   14,
        'axes.titlesize':   15,
        'axes.titleweight': 'bold',
        'xtick.labelsize':  12,
        'ytick.labelsize':  12,
        'legend.fontsize':  11,
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
        'axes.edgecolor':   '#333333',
        'axes.grid':        True,
        'grid.color':       '#dddddd',
        'grid.linestyle':   '-',
        'grid.linewidth':   0.6,
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5  FIGURE 3 — Tumour Volume Dynamics
# ══════════════════════════════════════════════════════════════════════════════

def generate_figure3(sim: dict, output_path: str) -> None:
    """
    Figure 3: Tumour volume over 30 days under Debio-0932 treatment.

    Single panel, black line, volume in normalised 10⁹ cells.
    """
    time_days    = sim['time_days']
    tumor_volume = sim['tumor_volume']

    # Extra top margin so the title never clips annotations
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.12, right=0.97)

    ax.plot(
        time_days,
        tumor_volume / 1e9,
        color='black',
        linewidth=2.5,
    )

    ax.set_title('Tumor Volume Dynamics Under HSP90 Inhibition', pad=14)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Tumor Volume (×10⁹ cells)')

    v0  = tumor_volume[0]  / 1e9
    vf  = tumor_volume[-1] / 1e9
    t_f = float(time_days[-1])

    # V₀ annotation — text to the right of the point, inside the plot
    ax.annotate(
        f'V₀ = {v0:.2f} ×10⁹',
        xy=(time_days[0], v0),
        xytext=(3.5, v0 - 0.07),          # below-right, well inside axes
        fontsize=10, color='#333333',
        arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2),
    )

    # V₃₀ annotation — text to the left of the final point
    ax.annotate(
        f'V₃₀ = {vf:.2f} ×10⁹',
        xy=(t_f, vf),
        xytext=(t_f - 10, vf + 0.12),     # above-left of the endpoint
        fontsize=10, color='#333333',
        arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2),
    )

    # Info box — lower-left corner, away from both annotations
    info = (
        f"Drug: {sim['drug_name']}\n"
        f"Dose: {sim['dose_nM']:.0f} nM · q12 h\n"
        f"Subtype: {sim['subtype_name']}"
    )
    ax.text(
        0.02, 0.05, info,
        transform=ax.transAxes,
        fontsize=9.5, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=0.95),
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure 3 saved  →  {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6  FIGURE 4 — PK / PD Response
# ══════════════════════════════════════════════════════════════════════════════

def generate_figure4(sim: dict, output_path: str) -> None:
    """
    Figure 4: Dual-axis PK/PD plot.

    Left  axis (blue)  — Drug concentration (nM).
    Right axis (red)   — Fractional HSP90 inhibition (0–1).

    First 5 days (120 h) shown so the q12 h dosing oscillations are visible.
    """
    time_hours = sim['time_hours']
    conc       = sim['drug_concentration']
    effect     = sim['drug_effect']

    mask = time_hours <= 120.0
    t_h  = time_hours[mask]
    c    = conc[mask]
    e    = effect[mask]

    color_conc   = '#1565C0'   # deep blue
    color_effect = '#C62828'   # deep red
    dosing_interval = 12.0     # hours

    # constrained_layout handles all margins automatically — no manual coords
    fig, ax1 = plt.subplots(figsize=(11, 6), constrained_layout=True)

    # ── Left axis: concentration ──────────────────────────────────────────────
    line1, = ax1.plot(t_h, c, color=color_conc, linewidth=2.5,
                      label='Drug Concentration (nM)')
    ax1.set_xlabel('Time (hours)', labelpad=8)
    ax1.set_ylabel('Drug Concentration (nM)', color=color_conc, labelpad=10)
    ax1.tick_params(axis='y', labelcolor=color_conc)
    ax1.set_xlim(left=0, right=float(t_h[-1]))
    ax1.set_ylim(bottom=0)

    # ── Right axis: HSP90 inhibition ─────────────────────────────────────────
    ax2 = ax1.twinx()
    line2, = ax2.plot(t_h, e, color=color_effect, linewidth=2.0,
                      linestyle='--', label='Fractional HSP90 Inhibition')
    ax2.set_ylabel('Fractional HSP90 Inhibition', color=color_effect, labelpad=10)
    ax2.tick_params(axis='y', labelcolor=color_effect)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)

    # ── Dose markers (subtle vertical lines) ─────────────────────────────────
    for dm in np.arange(0, float(t_h[-1]) + 1, dosing_interval):
        ax1.axvline(dm, color='#bbbbbb', linewidth=0.8, linestyle=':',
                    alpha=0.7, zorder=0)

    # ── Combined legend — placed in lower-centre to avoid both y-axes ─────────
    lines  = [line1, line2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='lower center', framealpha=0.92,
               bbox_to_anchor=(0.5, 0.03), ncol=2)

    # ── Two-line title inside set_title so matplotlib owns the spacing ────────
    ax1.set_title(
        'Pharmacokinetics and Pharmacodynamic Response of HSP90 Inhibition\n'
        f'{sim["drug_name"]}  ·  {sim["dose_nM"]:.0f} nM  ·  '
        f'q{dosing_interval:.0f} h dosing  ·  first 120 h shown',
        pad=14,
        fontsize=13,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure 4 saved  →  {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _apply_base_style()

    print("Running simulation (Debio-0932, MYCN Amplified, 30 days)...")
    sim = run_simulation()

    print(f"  Time steps   : {len(sim['time_days'])}")
    print(f"  V₀           : {sim['tumor_volume'][0]:.3e} cells")
    print(f"  V₃₀          : {sim['tumor_volume'][-1]:.3e} cells")
    print(f"  Peak conc.   : {sim['drug_concentration'].max():.1f} nM")
    print(f"  Peak effect  : {sim['drug_effect'].max():.4f}")

    # Store in named variables (Step 4)
    time_days          = sim['time_days']         # noqa: F841
    tumor_volume       = sim['tumor_volume']       # noqa: F841
    drug_concentration = sim['drug_concentration'] # noqa: F841
    drug_effect        = sim['drug_effect']        # noqa: F841

    print("\nGenerating figures...")
    generate_figure3(sim, 'results/figures/figure3_tumor_volume.png')
    generate_figure4(sim, 'results/figures/figure4_pk_pd.png')

    print("\nBase simulation figures generated successfully.")


if __name__ == '__main__':
    main()
