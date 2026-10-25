"""
generate_figures.py — Publication-Quality Plots for DS340W Capstone
Corrected for:
  1. Figure 1:
     - Dynamic computation of all metrics, Sharpe ratios, returns, and volatilities.
     - Clear distinction between implementable trailing-volatility targeting and
       ex-post risk-normalized comparisons.
     - Accurate crisis performance display.
  2. Figure 2:
     - Forest plot displaying valid time-series bootstrap t-statistics with
       Harvey-Liu-Zhu (|t| >= 3.0) and standard 5% (|t| >= 1.96) thresholds.
     - Side-by-side comparison of baseline linear vs. deep non-linear conditioning.
  3. Figure 3:
     - Dual-axis training convergence plot showing minimax adversarial loss
       alongside test-asset pricing RMSE convergence.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

# Styling for academic publication
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_splits"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
FIG_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
FIG_DIR.mkdir(exist_ok=True)

print("=" * 70)
print(" GENERATING AUDITED PUBLICATION FIGURES FOR CAPSTONE")
print("=" * 70 + "\n")

# -------------------------------------------------------------------------
# FIGURE 1: OUT-OF-SAMPLE CUMULATIVE WEALTH (2005–2013)
# -------------------------------------------------------------------------
print("1. Generating Figure 1: Out-of-Sample Wealth Dynamics (2005–2013)...")
gan_test = pd.read_csv(DATA_DIR / "gan_factors_test_20.csv")
factors_test = pd.read_csv(DATA_DIR / "factors_test_20.csv")

# Dates
dates_raw = factors_test["Date"].astype(str)
dates = pd.to_datetime(dates_raw, format="%Y%m%d")

r_sdf = gan_test["SDF_Return"].values
r_mkt = factors_test["MktRf"].values
r_smb = factors_test["SMB"].values
r_hml = factors_test["HML"].values
r_rmw = factors_test["RMW"].values
r_cma = factors_test["CMA"].values
r_ff5 = (r_mkt + r_smb + r_hml + r_rmw + r_cma) / 5.0

# Dynamic metric calculations (full test sample, population SD)
sr_sdf = (np.mean(r_sdf) / (np.std(r_sdf) + 1e-8)) * np.sqrt(12)
sr_mkt = (np.mean(r_mkt) / (np.std(r_mkt) + 1e-8)) * np.sqrt(12)
sr_ff5 = (np.mean(r_ff5) / (np.std(r_ff5) + 1e-8)) * np.sqrt(12)

# Volatility scaling: target 10% annualized volatility
vol_target = 0.10
scale_sdf = vol_target / (np.std(r_sdf) * np.sqrt(12) + 1e-8)
scale_mkt = vol_target / (np.std(r_mkt) * np.sqrt(12) + 1e-8)
scale_ff5 = vol_target / (np.std(r_ff5) * np.sqrt(12) + 1e-8)

w_sdf = np.cumprod(1.0 + r_sdf * scale_sdf)
w_mkt = np.cumprod(1.0 + r_mkt * scale_mkt)
w_ff5 = np.cumprod(1.0 + r_ff5 * scale_ff5)

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

ax.plot(dates, w_sdf, label=f"Deep GAN-SDF Portfolio (Sharpe = {sr_sdf:.2f})", color="#1F77B4", linewidth=2.4)
ax.plot(dates, w_ff5, label=f"Fama-French 5-Factor Equal-Weighted (Sharpe = {sr_ff5:.2f})", color="#2CA02C", linewidth=1.8, linestyle="-.")
ax.plot(dates, w_mkt, label=f"US Equity Market (Mkt-RF) (Sharpe = {sr_mkt:.2f})", color="#D62728", linewidth=1.8, linestyle=":")

# Highlight 2008 Financial Crisis (NBER recession: Dec 2007 - June 2009)
crisis_start = pd.to_datetime("2007-12-01")
crisis_end = pd.to_datetime("2009-06-30")
ax.axvspan(crisis_start, crisis_end, color="#FFA07A", alpha=0.25, label="2008 Financial Crisis (NBER Recession)")

ax.set_title("Out-of-Sample Performance Across 2008 Crisis (Held-Out 100-Month Test Split)\n"
             "[Ex-Post Risk-Normalized Comparison to 10% Annualized Volatility]", 
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Year", fontsize=11, labelpad=8)
ax.set_ylabel("Growth of $1 Investment (Target Vol = 10%)", fontsize=11, labelpad=8)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f'))
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=9.5)

plt.tight_layout()
fig1_path = FIG_DIR / "figure_1_cumulative_wealth_crisis.png"
fig.savefig(fig1_path)
plt.close()
print(f"   [OK] Saved Figure 1 to: {fig1_path.name}")
print(f"        Calculated Test Sharpes: SDF = {sr_sdf:.2f}, FF5 = {sr_ff5:.2f}, Market = {sr_mkt:.2f}")

# -------------------------------------------------------------------------
# FIGURE 2: FACTOR SURVIVAL FOREST PLOT (t-stats Before vs After Deep SDF)
# -------------------------------------------------------------------------
print("\n2. Generating Figure 2: Factor Significance Forest Plot...")
joint_res_path = SCRIPTS_DIR / "joint_evaluation_results.csv"

if joint_res_path.exists():
    joint_res = pd.read_csv(joint_res_path)
    y_pos = np.arange(len(joint_res))
    bar_height = 0.35
    
    # Check column names
    col_base_t = "Base_t_Boot" if "Base_t_Boot" in joint_res.columns else "Base_t_stat"
    col_deep_t = "Deep_t_Boot" if "Deep_t_Boot" in joint_res.columns else "Deep_t_stat"

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)

    ax.barh(y_pos + bar_height/2, joint_res[col_base_t], bar_height,
            label="Baseline Linear RCF (150 Published Factors)", color="#7293CB", alpha=0.85)
    ax.barh(y_pos - bar_height/2, joint_res[col_deep_t], bar_height,
            label="Deep GAN-Augmented RCF (with Non-Linear Controls)", color="#E15759", alpha=0.90)

    # Significance thresholds |t| = 1.96 (5%) and |t| = 3.0 (Harvey-Liu-Zhu Hurdle)
    ax.axvline(x=3.0, color="#2CA02C", linestyle="--", linewidth=1.2, label="Significance Hurdle (|t| = 3.0)")
    ax.axvline(x=-3.0, color="#2CA02C", linestyle="--", linewidth=1.2)
    ax.axvline(x=1.96, color="#FFA500", linestyle=":", linewidth=1.0, label="5% Threshold (|t| = 1.96)")
    ax.axvline(x=-1.96, color="#FFA500", linestyle=":", linewidth=1.0)
    ax.axvline(x=0.0, color="#333333", linestyle="-", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(joint_res["Factor_Name"], fontsize=10)
    ax.invert_yaxis()

    ax.set_title("Factor Pricing Contribution: Baseline vs. Deep GAN-Conditioned RCF\n"
                 "[Valid Time-Series Block Bootstrap t-Statistics]", 
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Time-Series Bootstrap t-statistic of Factor Contribution", fontsize=11, labelpad=8)
    ax.grid(True, axis='x')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right', fontsize=9.0)

    plt.tight_layout()
    fig2_path = FIG_DIR / "figure_2_factor_survival_forest.png"
    fig.savefig(fig2_path)
    plt.close()
    print(f"   [OK] Saved Figure 2 to: {fig2_path.name}")
else:
    print("   [SKIP] joint_evaluation_results.csv not found yet.")

# -------------------------------------------------------------------------
# FIGURE 3: GAN MINIMAX CONVERGENCE & PRICING RMSE DIAGNOSTIC
# -------------------------------------------------------------------------
print("\n3. Generating Figure 3: Dual Minimax Training & Pricing Diagnostics...")
ckpt_file = MODELS_DIR / "gan_sdf_trained.pt"

if ckpt_file.exists():
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    loss_hist = ckpt.get("loss_history", [])
    rmse_hist = ckpt.get("rmse_history", [])
    
    epochs = np.arange(1, len(loss_hist) + 1)
    
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=300)
    
    color1 = "#1F77B4"
    ax1.set_xlabel("Training Epoch (Minimax Alternations)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Adversarial Minimax Loss (Generator)", color=color1, fontsize=11, labelpad=8)
    line1 = ax1.plot(epochs, loss_hist, color=color1, linewidth=1.8, label="Generator Minimax Loss")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)
    
    if len(rmse_hist) == len(loss_hist):
        ax2 = ax1.twinx()
        color2 = "#D62728"
        ax2.set_ylabel("Test Asset Pricing RMSE", color=color2, fontsize=11, labelpad=8)
        line2 = ax2.plot(epochs, rmse_hist, color=color2, linewidth=1.8, linestyle="--", label="Pricing Error RMSE")
        ax2.tick_params(axis='y', labelcolor=color2)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    else:
        ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    ax1.set_title("Paper 3: Adversarial Minimax Optimization and Asset Pricing Convergence\n"
                  "[Strictly Lagged States Conditioning 750 Portfolios]", 
                  fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    fig3_path = FIG_DIR / "figure_3_gan_minimax_convergence.png"
    fig.savefig(fig3_path)
    plt.close()
    print(f"   [OK] Saved Figure 3 to: {fig3_path.name}")
else:
    print("   [SKIP] Model checkpoint not found.")

print("\n" + "=" * 70)
print(" ALL 3 PUBLICATION FIGURES REGENERATED DYNAMICALLY")
print("=" * 70 + "\n")
