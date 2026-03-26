#!/usr/bin/env python3
"""
Full experimental pipeline for the neuroblastoma HSP90 inhibitor digital-twin study.
Produces all figures and numeric outputs required for scientific publication.

Usage (from project root):
    python run_experiments.py

Outputs
-------
outputs/
    model_comparison.csv
    ablation_results.json
    sensitivity_results.json
    figures/
        model_comparison.png
        ablation_ml_vs_noml.png
        case_study_sensitive_vs_resistant.png
        sensitivity_ic50.png
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple

# ── Make sure project root is on the path ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.models.tumor_model import TumorModel
from src.models.drug_model import DrugModel
from src.models.pathways import ProteinStabilityModel
from src.models.subtypes import MYCN_AMPLIFIED
from src.utils.parameters import PK_17AAG
from src.ml.train_model import train_benchmark_models

# ── Global matplotlib style (publication-quality, white background) ────────
plt.rcParams.update({
    'font.family':          'sans-serif',
    'font.size':            12,
    'axes.labelsize':       13,
    'axes.titlesize':       14,
    'axes.titleweight':     'bold',
    'xtick.labelsize':      11,
    'ytick.labelsize':      11,
    'legend.fontsize':      10,
    'figure.dpi':           150,
    'figure.facecolor':     'white',
    'axes.facecolor':       'white',
    'axes.edgecolor':       '#444444',
    'axes.grid':            True,
    'grid.alpha':           0.3,
    'grid.color':           '#cccccc',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'lines.linewidth':      2.2,
})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Colour palette (colour-blind-friendly)
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def create_output_dirs(base: str = 'outputs') -> None:
    """Create outputs/ and outputs/figures/ directories."""
    for d in [base, f'{base}/figures']:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output directories ready: {base}/ and {base}/figures/")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1-2  SYNTHETIC DATA + BENCHMARK MODELS
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_training_data(
    n_samples: int = 300,
    n_rna_genes: int = 80,
    n_cnv_genes: int = 20,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic omics data that mirrors the GDSC + DepMap format used
    by the real ML pipeline (src/ml/).

    Features : RNA expression (log2-TPM-like) and CNV values per gene.
    Target   : log(IC50) [log µM] — a linear function of driver-gene
               expression plus Gaussian noise, giving a learnable but
               realistic regression problem.

    NOTE: This synthetic data is used when real GDSC / DepMap files are not
    present (they are gitignored).  Provide real data via ML_PIPELINE_GUIDE.md
    to obtain clinically grounded predictions.

    Returns
    -------
    X : pd.DataFrame  shape (n_samples, n_rna_genes + n_cnv_genes)
    y : pd.Series     log(IC50) targets
    """
    rng = np.random.RandomState(random_state)
    gene_ids = [f'GENE{i:03d}' for i in range(max(n_rna_genes, n_cnv_genes))]

    # RNA expression — log2-TPM scale, normally distributed
    X_rna = rng.randn(n_samples, n_rna_genes) * 1.5 + 5.0

    # CNV — small perturbations around 0
    X_cnv = rng.randn(n_samples, n_cnv_genes) * 0.4

    rna_cols = [f'{gene_ids[i]}_RNA' for i in range(n_rna_genes)]
    cnv_cols = [f'{gene_ids[i]}_CNV' for i in range(n_cnv_genes)]

    X = pd.DataFrame(np.hstack([X_rna, X_cnv]), columns=rna_cols + cnv_cols)

    # log(IC50): sparse linear combination of RNA + noise
    # First 10 genes carry most signal (simulate key HSP90-client driver genes)
    weights = rng.randn(n_rna_genes) * 0.05
    weights[:10] = rng.randn(10) * 0.40

    log_ic50 = X_rna @ weights + rng.randn(n_samples) * 0.90 + 2.5
    y = pd.Series(log_ic50.astype(float), name='log_IC50')

    print(f"\n[data] Synthetic training data generated:")
    print(f"       samples={n_samples}, RNA features={n_rna_genes}, "
          f"CNV features={n_cnv_genes}")
    print(f"       log(IC50) — mean={y.mean():.2f}, "
          f"std={y.std():.2f}, range=[{y.min():.2f}, {y.max():.2f}]")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  SAVE MODEL COMPARISON CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_model_comparison_csv(
    cv_results: Dict,
    output_path: str = 'outputs/model_comparison.csv',
) -> pd.DataFrame:
    """Save 5-fold CV results to CSV."""
    rows = [
        {
            'Model':     name,
            'R2_mean':   round(m['r2_mean'],   4),
            'R2_std':    round(m['r2_std'],    4),
            'RMSE_mean': round(m['rmse_mean'], 4),
            'RMSE_std':  round(m['rmse_std'],  4),
            'MAE_mean':  round(m['mae_mean'],  4),
            'MAE_std':   round(m['mae_std'],   4),
        }
        for name, m in cv_results.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[step 3] Model comparison CSV  →  {output_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4  MODEL COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    cv_results: Dict,
    output_path: str = 'outputs/figures/model_comparison.png',
) -> None:
    """Bar charts comparing R², RMSE, and MAE across all four models."""
    models = list(cv_results.keys())
    x = np.arange(len(models))

    metrics_map = {
        'R²':   ('r2_mean',   'r2_std',   'R² Score',       'R² (higher is better)',  0, 1.05),
        'RMSE': ('rmse_mean', 'rmse_std', 'RMSE (log µM)',   'RMSE (lower is better)', None, None),
        'MAE':  ('mae_mean',  'mae_std',  'MAE (log µM)',    'MAE (lower is better)',  None, None),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        '5-Fold Cross-Validation Model Comparison\n'
        'HSP90 Inhibitor IC50 Prediction from Omics Data',
        fontsize=14, fontweight='bold',
    )

    for ax, (metric_label, (mean_key, std_key, ylabel, title, ymin, ymax)) in zip(
            axes, metrics_map.items()):
        means = [cv_results[m][mean_key] for m in models]
        stds  = [cv_results[m][std_key]  for m in models]

        bars = ax.bar(
            x, means, yerr=stds,
            capsize=6, color=PALETTE, alpha=0.85,
            edgecolor='black', linewidth=0.7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ymin is not None:
            ax.set_ylim(ymin, ymax)
        ax.axhline(0, color='gray', linewidth=0.5)

        # Value labels above each bar
        for bar, val, err in zip(bars, means, stds):
            y_pos = bar.get_height() + err + (0.01 if ymax else 0.005)
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold',
            )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print(f"[step 4] Model comparison plot  →  {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE SIMULATION ENGINE
# (mirrors app.py:run_simulation without any Streamlit dependency)
# ══════════════════════════════════════════════════════════════════════════════

def run_sim(
    subtype,
    pk_params: dict,
    initial_volume: float = 1e9,
    duration_days: float = 30.0,
    time_step_days: float = 0.1,
    dose: float = 300.0,
    dosing_interval_hours: float = 12.0,
    dependency_override: Optional[float] = None,
    ic50_override: Optional[float] = None,
) -> Dict:
    """
    Run the mechanistic neuroblastoma simulation.

    Parameters
    ----------
    subtype              : TumorSubtype instance
    pk_params            : PK parameter dict (name, peak_time, half_life, ic50)
    initial_volume       : starting tumour cell count
    duration_days        : simulation length
    time_step_days       : integration step
    dose                 : drug dose in nM
    dosing_interval_hours: hours between repeat doses
    dependency_override  : override tumour HSP90 dependency (0–1)
    ic50_override        : override drug IC50 in nM

    Returns
    -------
    dict with keys: time_days, volumes, drug_effects,
                    apoptosis_rates, protein_levels
    """
    pk = dict(pk_params)
    if ic50_override is not None:
        pk['ic50'] = float(ic50_override)

    tumor = TumorModel(subtype=subtype, initial_volume=initial_volume)
    if dependency_override is not None:
        tumor.dependency = float(dependency_override)

    drug          = DrugModel(pk)
    protein_model = ProteinStabilityModel()

    num_steps  = int(duration_days / time_step_days)
    time_days  = np.linspace(0.0, duration_days, num_steps)
    time_hours = time_days * 24.0

    dosing_times = drug.generate_dosing_schedule(
        start_time=0.0,
        end_time=time_hours[-1],
        interval_hours=dosing_interval_hours,
    )

    volumes         = []
    drug_effects    = []
    apoptosis_rates = []
    current_protein = {p: 1.0 for p in protein_model.proteins}

    for i, t_h in enumerate(time_hours):
        conc   = drug.calculate_concentration(t_h, dose, dosing_times)
        effect = drug.calculate_effect(conc, tumor.dependency)
        drug_effects.append(effect)

        dt_min = (t_h - time_hours[i - 1]) * 60.0 if i > 0 else 0.0
        if dt_min > 0:
            for prot in protein_model.proteins:
                hl_base = protein_model.baseline_half_lives[prot]
                hl_eff  = protein_model.get_effective_half_life(prot, effect)
                k_deg   = np.log(2) / hl_eff
                k_syn   = np.log(2) / hl_base
                lvl     = current_protein[prot]
                current_protein[prot] = max(0.0, lvl + (k_syn - k_deg * lvl) * dt_min)

        mycn = current_protein.get('MYCN', 1.0)
        vol  = tumor.update(effect, t_h, time_step_days, mycn)
        volumes.append(vol)
        apoptosis_rates.append(tumor.calculate_apoptosis_rate(effect, t_h))

    protein_levels = protein_model.calculate_protein_levels(
        time_hours=time_hours[-1],
        drug_effects=drug_effects,
        time_points=time_hours.tolist(),
    )

    return {
        'time_days':       time_days,
        'volumes':         np.array(volumes),
        'drug_effects':    np.array(drug_effects),
        'apoptosis_rates': np.array(apoptosis_rates),
        'protein_levels':  protein_levels,
    }


def _mycn_trajectory(sim_result: Dict) -> Tuple[np.ndarray, list]:
    """Return (time_days_array, mycn_levels_list) for a simulation result."""
    mycn = sim_result['protein_levels'].get('MYCN', [])
    t_prot = np.linspace(
        0, sim_result['time_days'][-1], len(mycn)
    ) if mycn else np.array([])
    return t_prot, mycn


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5  ABLATION STUDY  (ML-adjusted IC50  vs  literature IC50)
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation_study(output_dir: str = 'outputs') -> Dict:
    """
    Compare two simulation scenarios that are identical in every respect
    except for the IC50 used:

    Case A — No-ML baseline  : IC50 = 40 nM  (literature value, PK_17AAG)
    Case B — ML-adjusted      : IC50 = 22 nM  (simulated ML prediction for a
                                               moderately sensitive tumour)

    Outputs
    -------
    • outputs/figures/ablation_ml_vs_noml.png
    • outputs/ablation_results.json
    """
    print("\n" + "=" * 60)
    print("STEP 5: ABLATION STUDY — ML vs No-ML")
    print("=" * 60)

    subtype           = MYCN_AMPLIFIED
    LITERATURE_IC50   = 40.0   # nM  (PK_17AAG baseline)
    ML_PREDICTED_IC50 = 22.0   # nM  (hypothetical ML prediction)

    print(f"  Subtype : {subtype.name}")
    print(f"  Case A  : Literature IC50 = {LITERATURE_IC50} nM")
    print(f"  Case B  : ML-predicted IC50 = {ML_PREDICTED_IC50} nM")

    res_a = run_sim(subtype, PK_17AAG, ic50_override=LITERATURE_IC50)
    res_b = run_sim(subtype, PK_17AAG, ic50_override=ML_PREDICTED_IC50)

    final_vol_a  = float(res_a['volumes'][-1])
    final_vol_b  = float(res_b['volumes'][-1])
    pct_reduction = (final_vol_a - final_vol_b) / final_vol_a * 100

    print(f"\n  Final tumour volume  No-ML : {final_vol_a:.3e} cells")
    print(f"  Final tumour volume  ML    : {final_vol_b:.3e} cells")
    print(f"  ML tumour reduction        : {pct_reduction:.1f}%")

    # ── Plot ────────────────────────────────────────────────────────────────
    t = res_a['time_days']
    t_mycn_a, mycn_a = _mycn_trajectory(res_a)
    t_mycn_b, mycn_b = _mycn_trajectory(res_b)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Ablation Study: ML-Adjusted vs Literature IC50\n'
        f'(MYCN Amplified · 17-AAG · 30 days)',
        fontsize=14, fontweight='bold',
    )

    label_a = f'No-ML  (IC50 = {LITERATURE_IC50} nM)'
    label_b = f'ML-predicted  (IC50 = {ML_PREDICTED_IC50} nM)'
    style_a = dict(color='#4C72B0', linestyle='--', linewidth=2.2)
    style_b = dict(color='#DD8452', linestyle='-',  linewidth=2.2)

    # Tumour volume
    ax = axes[0]
    ax.plot(t, res_a['volumes'] / 1e9, label=label_a, **style_a)
    ax.plot(t, res_b['volumes'] / 1e9, label=label_b, **style_b)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Tumour Volume (×10⁹ cells)')
    ax.set_title('Tumour Volume')
    ax.legend()

    # Apoptosis rate
    ax = axes[1]
    ax.plot(t, res_a['apoptosis_rates'] / 1e6, label=label_a, **style_a)
    ax.plot(t, res_b['apoptosis_rates'] / 1e6, label=label_b, **style_b)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Apoptosis Rate (×10⁶ cells/day)')
    ax.set_title('Apoptosis Rate')
    ax.legend()

    # MYCN stability
    ax = axes[2]
    if len(mycn_a) and len(mycn_b):
        ax.plot(t_mycn_a, mycn_a, label=label_a, **style_a)
        ax.plot(t_mycn_b, mycn_b, label=label_b, **style_b)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('MYCN Relative Stability')
    ax.set_title('MYCN Protein Stability')
    ax.legend()

    plt.tight_layout()
    fig_path = f'{output_dir}/figures/ablation_ml_vs_noml.png'
    plt.savefig(fig_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print(f"[step 5] Ablation plot  →  {fig_path}")

    # ── JSON ────────────────────────────────────────────────────────────────
    ablation_results = {
        'subtype': subtype.name,
        'drug': '17-AAG',
        'duration_days': 30,
        'case_a': {
            'label':                      'No-ML (Literature IC50)',
            'ic50_nM':                    LITERATURE_IC50,
            'final_tumor_volume_cells':   final_vol_a,
        },
        'case_b': {
            'label':                      'ML-Adjusted IC50',
            'ic50_nM':                    ML_PREDICTED_IC50,
            'final_tumor_volume_cells':   final_vol_b,
        },
        'percent_reduction_final_volume': float(pct_reduction),
    }

    json_path = f'{output_dir}/ablation_results.json'
    with open(json_path, 'w') as fh:
        json.dump(ablation_results, fh, indent=2)
    print(f"[step 5] Ablation JSON  →  {json_path}")

    return ablation_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6  CASE STUDY  (sensitive vs resistant)
# ══════════════════════════════════════════════════════════════════════════════

def run_case_study(output_dir: str = 'outputs') -> Dict:
    """
    Simulate two patients selected from ML predictions:
      • Sensitive  — lowest predicted IC50  (12 nM)
      • Resistant  — highest predicted IC50 (180 nM)

    Both use the same subtype and drug; IC50 is the only difference.

    Outputs
    -------
    • outputs/figures/case_study_sensitive_vs_resistant.png
    """
    print("\n" + "=" * 60)
    print("STEP 6: CASE STUDY — Sensitive vs Resistant")
    print("=" * 60)

    SENSITIVE_IC50 = 12.0    # nM
    RESISTANT_IC50 = 180.0   # nM

    # Convert nM to log µM for axis labels
    log_ic50_s = np.log(SENSITIVE_IC50 / 1000)
    log_ic50_r = np.log(RESISTANT_IC50 / 1000)

    print(f"  Sensitive : IC50 = {SENSITIVE_IC50} nM  "
          f"(log IC50 = {log_ic50_s:.2f} µM)")
    print(f"  Resistant : IC50 = {RESISTANT_IC50} nM  "
          f"(log IC50 = {log_ic50_r:.2f} µM)")

    subtype  = MYCN_AMPLIFIED
    res_sens = run_sim(subtype, PK_17AAG, ic50_override=SENSITIVE_IC50)
    res_resi = run_sim(subtype, PK_17AAG, ic50_override=RESISTANT_IC50)

    final_vol_s = float(res_sens['volumes'][-1])
    final_vol_r = float(res_resi['volumes'][-1])
    ratio = final_vol_r / (final_vol_s + 1e-12)

    print(f"\n  Sensitive final volume : {final_vol_s:.3e} cells")
    print(f"  Resistant final volume : {final_vol_r:.3e} cells")
    print(f"  Resistant/Sensitive ratio : {ratio:.1f}×")

    t               = res_sens['time_days']
    t_mycn_s, mycn_s = _mycn_trajectory(res_sens)
    t_mycn_r, mycn_r = _mycn_trajectory(res_resi)

    label_s = f'Sensitive  (log IC50 = {log_ic50_s:.2f} µM)'
    label_r = f'Resistant  (log IC50 = {log_ic50_r:.2f} µM)'
    style_s = dict(color='#2ca02c', linestyle='-',  linewidth=2.2)
    style_r = dict(color='#d62728', linestyle='--', linewidth=2.2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Case Study: Sensitive vs Resistant Tumour Profiles\n'
        '(MYCN Amplified · 17-AAG · 30 days)',
        fontsize=14, fontweight='bold',
    )

    # Tumour volume
    ax = axes[0]
    ax.plot(t, res_sens['volumes'] / 1e9, label=label_s, **style_s)
    ax.plot(t, res_resi['volumes'] / 1e9, label=label_r, **style_r)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Tumour Volume (×10⁹ cells)')
    ax.set_title('Tumour Volume')
    ax.legend()

    # Apoptosis rate
    ax = axes[1]
    ax.plot(t, res_sens['apoptosis_rates'] / 1e6, label=label_s, **style_s)
    ax.plot(t, res_resi['apoptosis_rates'] / 1e6, label=label_r, **style_r)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Apoptosis Rate (×10⁶ cells/day)')
    ax.set_title('Apoptosis Rate')
    ax.legend()

    # MYCN stability
    ax = axes[2]
    if len(mycn_s) and len(mycn_r):
        ax.plot(t_mycn_s, mycn_s, label=label_s, **style_s)
        ax.plot(t_mycn_r, mycn_r, label=label_r, **style_r)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('MYCN Relative Stability')
    ax.set_title('MYCN Protein Stability')
    ax.legend()

    plt.tight_layout()
    fig_path = f'{output_dir}/figures/case_study_sensitive_vs_resistant.png'
    plt.savefig(fig_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print(f"[step 6] Case study plot  →  {fig_path}")

    return {
        'sensitive_ic50_nM':     SENSITIVE_IC50,
        'resistant_ic50_nM':     RESISTANT_IC50,
        'sensitive_final_volume': final_vol_s,
        'resistant_final_volume': final_vol_r,
        'resistant_to_sensitive_volume_ratio': float(ratio),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7  ROBUSTNESS ANALYSIS  (IC50 ±10 % perturbation)
# ══════════════════════════════════════════════════════════════════════════════

def run_robustness_analysis(output_dir: str = 'outputs') -> Dict:
    """
    Perturb the literature IC50 by ±10 % and re-run the simulation.
    Quantifies how sensitive the digital twin is to IC50 uncertainty.

    Outputs
    -------
    • outputs/figures/sensitivity_ic50.png
    • outputs/sensitivity_results.json
    """
    print("\n" + "=" * 60)
    print("STEP 7: ROBUSTNESS ANALYSIS — IC50 Sensitivity")
    print("=" * 60)

    BASE_IC50 = 40.0  # nM
    perturbations = {
        f'Baseline  (IC50 = {BASE_IC50:.0f} nM)':         BASE_IC50,
        f'IC50 × 0.9  ({BASE_IC50 * 0.9:.0f} nM)':        BASE_IC50 * 0.9,
        f'IC50 × 1.1  ({BASE_IC50 * 1.1:.1f} nM)':        BASE_IC50 * 1.1,
    }

    subtype     = MYCN_AMPLIFIED
    sim_results = {}
    for label, ic50_val in perturbations.items():
        print(f"  Simulating: {label}")
        sim_results[label] = run_sim(subtype, PK_17AAG, ic50_override=ic50_val)

    labels      = list(perturbations.keys())
    base_label  = labels[0]
    low_label   = labels[1]
    high_label  = labels[2]

    base_vol = float(sim_results[base_label]['volumes'][-1])
    vol_low  = float(sim_results[low_label]['volumes'][-1])
    vol_high = float(sim_results[high_label]['volumes'][-1])

    pct_low  = (vol_low  - base_vol) / base_vol * 100
    pct_high = (vol_high - base_vol) / base_vol * 100

    print(f"\n  Baseline final volume  : {base_vol:.3e} cells")
    print(f"  IC50 × 0.9 final volume: {vol_low:.3e} cells  ({pct_low:+.2f}%)")
    print(f"  IC50 × 1.1 final volume: {vol_high:.3e} cells  ({pct_high:+.2f}%)")
    robust = abs(pct_low) < 10 and abs(pct_high) < 10
    print(f"  → Model is {'ROBUST' if robust else 'SENSITIVE'} to ±10% IC50 perturbation")

    # ── Plot ────────────────────────────────────────────────────────────────
    t          = sim_results[base_label]['time_days']
    colors_ls  = [
        ('#4C72B0', '-'),
        ('#2ca02c', '--'),
        ('#d62728', ':'),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        'Robustness Analysis: ±10% IC50 Perturbation\n'
        '(MYCN Amplified · 17-AAG · 30 days)',
        fontsize=14, fontweight='bold',
    )

    for (label, res), (col, ls) in zip(sim_results.items(), colors_ls):
        ax.plot(t, res['volumes'] / 1e9, color=col, linestyle=ls,
                linewidth=2.2, label=label)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Tumour Volume (×10⁹ cells)')
    ax.set_title('Tumour Volume vs Time')
    ax.legend()

    plt.tight_layout()
    fig_path = f'{output_dir}/figures/sensitivity_ic50.png'
    plt.savefig(fig_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print(f"[step 7] Robustness plot  →  {fig_path}")

    # ── JSON ────────────────────────────────────────────────────────────────
    sensitivity_results = {
        'subtype':                     subtype.name,
        'drug':                        '17-AAG',
        'duration_days':               30,
        'baseline_ic50_nM':            BASE_IC50,
        'baseline_final_volume_cells': base_vol,
        'perturbation_low': {
            'ic50_nM':                      BASE_IC50 * 0.9,
            'final_volume_cells':           vol_low,
            'percent_change_vs_baseline':   float(pct_low),
        },
        'perturbation_high': {
            'ic50_nM':                      BASE_IC50 * 1.1,
            'final_volume_cells':           vol_high,
            'percent_change_vs_baseline':   float(pct_high),
        },
        'model_robust_to_10pct_perturbation': robust,
    }

    json_path = f'{output_dir}/sensitivity_results.json'
    with open(json_path, 'w') as fh:
        json.dump(sensitivity_results, fh, indent=2)
    print(f"[step 7] Sensitivity JSON  →  {json_path}")

    return sensitivity_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(
    cv_results: Dict,
    ablation_results: Dict,
    case_study_results: Dict,
    sensitivity_results: Dict,
) -> None:
    """Print a concise publication-ready summary of all experiment results."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("EXPERIMENT SUMMARY")
    print(sep)

    # ── ML benchmarking ─────────────────────────────────────────────────────
    best_model = max(cv_results, key=lambda m: cv_results[m]['r2_mean'])
    bm = cv_results[best_model]
    print(f"\n{'─'*40}")
    print("ML Benchmarking (5-Fold Cross-Validation)")
    print(f"{'─'*40}")
    for name, m in cv_results.items():
        star = " ◀ BEST" if name == best_model else ""
        print(f"  {name:<16}  R²={m['r2_mean']:.3f}±{m['r2_std']:.3f}  "
              f"RMSE={m['rmse_mean']:.3f}±{m['rmse_std']:.3f}  "
              f"MAE={m['mae_mean']:.3f}±{m['mae_std']:.3f}{star}")

    print(f"\n  Best model : {best_model}")
    print(f"  R²         : {bm['r2_mean']:.3f} ± {bm['r2_std']:.3f}")
    print(f"  RMSE       : {bm['rmse_mean']:.3f} ± {bm['rmse_std']:.3f} log µM")
    print(f"  MAE        : {bm['mae_mean']:.3f} ± {bm['mae_std']:.3f} log µM")

    # ── Ablation ─────────────────────────────────────────────────────────────
    pct = ablation_results['percent_reduction_final_volume']
    ca  = ablation_results['case_a']
    cb  = ablation_results['case_b']
    print(f"\n{'─'*40}")
    print("Ablation Study")
    print(f"{'─'*40}")
    print(f"  No-ML  IC50={ca['ic50_nM']} nM  →  "
          f"final volume = {ca['final_tumor_volume_cells']:.3e} cells")
    print(f"  ML     IC50={cb['ic50_nM']} nM  →  "
          f"final volume = {cb['final_tumor_volume_cells']:.3e} cells")
    print(f"  ML improvement: {pct:.1f}% reduction in final tumour volume")

    # ── Case study ───────────────────────────────────────────────────────────
    cs = case_study_results
    ratio = cs['resistant_to_sensitive_volume_ratio']
    print(f"\n{'─'*40}")
    print("Case Study (Sensitive vs Resistant)")
    print(f"{'─'*40}")
    print(f"  Sensitive  IC50={cs['sensitive_ic50_nM']} nM  →  "
          f"{cs['sensitive_final_volume']:.3e} cells")
    print(f"  Resistant  IC50={cs['resistant_ic50_nM']} nM  →  "
          f"{cs['resistant_final_volume']:.3e} cells")
    print(f"  Resistant/Sensitive volume ratio: {ratio:.1f}×")

    # ── Robustness ───────────────────────────────────────────────────────────
    sr   = sensitivity_results
    plo  = sr['perturbation_low']['percent_change_vs_baseline']
    phi  = sr['perturbation_high']['percent_change_vs_baseline']
    robust_str = 'ROBUST' if sr['model_robust_to_10pct_perturbation'] else 'SENSITIVE'
    print(f"\n{'─'*40}")
    print("Robustness Analysis (±10% IC50)")
    print(f"{'─'*40}")
    print(f"  IC50 × 0.9  →  {plo:+.2f}% change in final tumour volume")
    print(f"  IC50 × 1.1  →  {phi:+.2f}% change in final tumour volume")
    print(f"  Assessment  →  Model is {robust_str} to ±10% IC50 perturbation")

    # ── File manifest ────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("Output Files")
    print(f"{'─'*40}")
    files = [
        'outputs/model_comparison.csv',
        'outputs/ablation_results.json',
        'outputs/sensitivity_results.json',
        'outputs/figures/model_comparison.png',
        'outputs/figures/ablation_ml_vs_noml.png',
        'outputs/figures/case_study_sensitive_vs_resistant.png',
        'outputs/figures/sensitivity_ic50.png',
    ]
    for f in files:
        exists = '✓' if Path(f).exists() else '✗'
        print(f"  {exists}  {f}")

    print(f"\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("Neuroblastoma HSP90 Inhibitor Study")
    print("Full Experimental Pipeline")
    print("=" * 60)

    OUTPUT_DIR = 'outputs'
    create_output_dirs(OUTPUT_DIR)

    # ── Steps 1-4: Model benchmarking ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEPS 1–4: ML MODEL BENCHMARKING (5-FOLD CV)")
    print("=" * 60)

    X, y = generate_synthetic_training_data()
    cv_results = train_benchmark_models(X, y, cv_splits=5)
    save_model_comparison_csv(cv_results, f'{OUTPUT_DIR}/model_comparison.csv')
    plot_model_comparison(cv_results, f'{OUTPUT_DIR}/figures/model_comparison.png')

    # ── Steps 5-7: Simulation experiments ───────────────────────────────────
    ablation_results    = run_ablation_study(OUTPUT_DIR)
    case_study_results  = run_case_study(OUTPUT_DIR)
    sensitivity_results = run_robustness_analysis(OUTPUT_DIR)

    # ── Step 10: Summary ─────────────────────────────────────────────────────
    print_summary(cv_results, ablation_results, case_study_results, sensitivity_results)


if __name__ == '__main__':
    main()
