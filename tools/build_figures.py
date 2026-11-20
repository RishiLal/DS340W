"""
Build publication-quality figures and underlying data for DS340W JKP methodology.

Generates:
  Figure 1: Decomposition of Out-of-Sample Squared Pricing Loss (Ordinary vs Managed)
  Figure 2: Loss Differences and 95% Moving-Block Bootstrap Intervals (Validation 2021-2025)
  Figure 3: Training Forward-Fold Tuning Curves and Penalty Selection (U, C1, C2)
  Artifacts:
    - PDF (vector) and PNG (300 DPI high-resolution) for each figure
    - figures_data.json: Complete machine-readable data for all plotted values
    - figure_index.json: Metadata index mapping files, artifacts, captions, placement
    - FIGURES.md: Formatted index and captions for manuscript review
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def build_figures(run_dir, dest_dir=None):
    """
    Generate all manuscript figures and machine-readable source artifacts from a completed run.

    Args:
        run_dir: Path to completed pipeline run directory.
        dest_dir: Destination directory for figures (defaults to run_dir / 'figures').

    Returns:
        manifest_entries: dict mapping figure names to saved file paths.
    """
    run_dir = Path(run_dir)
    dest = Path(dest_dir) if dest_dir else run_dir / "figures"
    dest.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. Load Numerical Artifacts and Metadata Dynamically
    # -------------------------------------------------------------------------
    # Primary tuning, fits, bootstrap, evaluations
    tuning_prim = json.loads((run_dir / "primary/tuning_curves.json").read_text(encoding="utf-8"))
    fits_prim = json.loads((run_dir / "primary/frozen_models.json").read_text(encoding="utf-8"))
    boot_prim = json.loads((run_dir / "primary/bootstrap_results.json").read_text(encoding="utf-8"))
    test_eval = json.loads((run_dir / "primary/test_evaluation.json").read_text(encoding="utf-8"))
    val_eval = json.loads((run_dir / "primary/validation_evaluation.json").read_text(encoding="utf-8"))

    # Diagnostic tuning, fits, contrasts
    tuning_diag = json.loads((run_dir / "diagnostic/tuning_and_checks.json").read_text(encoding="utf-8"))
    fits_diag = json.loads((run_dir / "diagnostic/fits.json").read_text(encoding="utf-8"))

    # Contrast tables (from tables/ or synthesized from primary/diagnostic artifacts)
    prim_contrasts_csv = run_dir / "tables/primary_contrasts.csv"
    diag_contrasts_csv = run_dir / "tables/diagnostic_contrasts.csv"
    if prim_contrasts_csv.exists() and diag_contrasts_csv.exists():
        prim_contrasts_df = pd.read_csv(prim_contrasts_csv)
        diag_contrasts_df = pd.read_csv(diag_contrasts_csv)
    else:
        # Fallback reading directly from diagnostic contrasts and bootstrap results
        diag_contrasts_df = pd.read_csv(run_dir / "diagnostic/contrasts.csv")
        # Synthesize primary contrasts table for validation
        c1_u_12 = diag_contrasts_df[
            (diag_contrasts_df.period == "Validation") & (diag_contrasts_df.model == "U") &
            (diag_contrasts_df.comparator == "C1") & (diag_contrasts_df.block == 12)
        ].iloc[0]
        c1_u_6 = diag_contrasts_df[
            (diag_contrasts_df.period == "Validation") & (diag_contrasts_df.model == "U") &
            (diag_contrasts_df.comparator == "C1") & (diag_contrasts_df.block == 6)
        ].iloc[0]
        prim_contrasts_df = pd.DataFrame([
            {"period": "Validation", "comparison": "C_G minus U_G", "block": 12,
             "difference": val_eval["delta_goldilocks"], "lower": -c1_u_12.upper, "upper": -c1_u_12.lower},
            {"period": "Validation", "comparison": "C_G minus U_G", "block": 6,
             "difference": val_eval["delta_goldilocks"], "lower": -c1_u_6.upper, "upper": -c1_u_6.lower},
        ])

    # Dynamic extraction of Figure 3 selection curves and parameters
    lambdas = np.array(tuning_prim["curves_data"]["lambdas"])
    u_rmse = np.array(tuning_prim["curves_data"]["unconditional_rmse"])
    c_rmse = np.array(tuning_prim["curves_data"]["conditional_rmse"])
    c2_rmse = np.array(tuning_diag["pooled_rmse"])

    u_sel_summary = tuning_prim["selection_summary"]
    u_min_idx = int(u_sel_summary["unconditional_minimum"]["lambda_idx"])
    u_min_lam = float(u_sel_summary["unconditional_minimum"]["lambda_val"])
    u_gold_idx = int(u_sel_summary["unconditional_goldilocks"]["lambda_idx"])
    u_gold_lam = float(u_sel_summary["unconditional_goldilocks"]["lambda_val"])
    u_win_idx = int(u_sel_summary["unconditional_goldilocks"]["goldilocks_details"]["best_window_idx"])
    u_agree = (u_min_idx == u_gold_idx)
    u_boundary = (u_gold_idx == 0 or u_gold_idx == len(lambdas) - 1)

    c_min_idx = int(u_sel_summary["conditional_minimum"]["lambda_idx"])
    c_min_lam = float(u_sel_summary["conditional_minimum"]["lambda_val"])
    c_gold_idx = int(u_sel_summary["conditional_goldilocks"]["lambda_idx"])
    c_gold_lam = float(u_sel_summary["conditional_goldilocks"]["lambda_val"])
    c_win_idx = int(u_sel_summary["conditional_goldilocks"]["goldilocks_details"]["best_window_idx"])
    c_agree = (c_min_idx == c_gold_idx)
    c_boundary = (c_gold_idx == 0 or c_gold_idx == len(lambdas) - 1)

    c2_min_idx = int(tuning_diag["minimum"])
    c2_gold_idx = int(tuning_diag["goldilocks"]["best_penalty_idx"])
    c2_min_lam = float(lambdas[c2_min_idx])
    c2_gold_lam = float(lambdas[c2_gold_idx])
    c2_win_idx = int(tuning_diag["goldilocks"]["best_window_idx"])
    c2_agree = (c2_min_idx == c2_gold_idx)
    c2_boundary = (c2_gold_idx == 0 or c2_gold_idx == len(lambdas) - 1)
    c2_boundary_position = ("lower" if c2_gold_idx == 0 else
                            "upper" if c2_gold_idx == len(lambdas) - 1 else "interior")

    # Diagnostic model penalties
    c2_matched_penalty = float(fits_diag["lambdas"]["C2_matched"])
    c2_g_penalty = float(fits_diag["lambdas"]["C2_G"])
    c2_m_penalty = float(fits_diag["lambdas"]["C2_M"])

    # Bootstrap interaction intervals
    boot_12 = boot_prim["primary_block_12"]["ci_intervals"]
    boot_6 = boot_prim["sensitivity_block_6"]["ci_intervals"]
    interaction_diff = float(val_eval["interaction"])
    interaction_12_ci = [float(boot_12["interaction"]["p025"]), float(boot_12["interaction"]["p975"])]
    interaction_6_ci = [float(boot_6["interaction"]["p025"]), float(boot_6["interaction"]["p975"])]

    # -------------------------------------------------------------------------
    # 1. FIGURE 1: Pricing-loss decomposition (Ordinary vs Managed)
    # -------------------------------------------------------------------------
    test_prim_csv = run_dir / "tables/test_primary.csv"
    val_prim_csv = run_dir / "tables/validation_primary.csv"
    if test_prim_csv.exists() and val_prim_csv.exists():
        test_df = pd.read_csv(test_prim_csv)
        val_df = pd.read_csv(val_prim_csv)
        u_test = test_df[test_df.model.eq("U_G")].iloc[0]
        c_test = test_df[test_df.model.eq("C_G")].iloc[0]
        u_val = val_df[val_df.model.eq("U_G")].iloc[0]
        c_val = val_df[val_df.model.eq("C_G")].iloc[0]
    else:
        def extract_decomp(scores, model_key):
            m = scores["models"][model_key]
            e = np.asarray(m["e_bar"])
            ord_loss = float(np.sum(e[:154] ** 2) / 308)
            mng_loss = float(np.sum(e[154:] ** 2) / 308)
            return pd.Series({"loss": m["loss"], "ordinary": ord_loss, "managed": mng_loss})

        u_test = extract_decomp(test_eval, "U_G")
        c_test = extract_decomp(test_eval, "C_G")
        u_val = extract_decomp(val_eval, "U_G")
        c_val = extract_decomp(val_eval, "C_G")

    # Numerical verification: ordinary and managed must sum to total loss
    for name, row in [("U_Test", u_test), ("C1_Test", c_test), ("U_Val", u_val), ("C1_Val", c_val)]:
        sum_components = float(row.ordinary + row.managed)
        np.testing.assert_allclose(sum_components, float(row.loss), atol=1e-14,
                                   err_msg=f"Figure 1 component sum mismatch in {name}")

    pct_increase = 100.0 * (float(c_val.loss) - float(u_val.loss)) / float(u_val.loss)
    ord_change = float(c_val.ordinary) - float(u_val.ordinary)
    mng_change = float(c_val.managed) - float(u_val.managed)

    fig1, ax1 = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    categories = [
        "Dev Test: U\n(2010–2020)", "Dev Test: C1\n(2010–2020)",
        "Validation: U\n(2021–2025)", "Validation: C1\n(2021–2025)"
    ]
    ordinary_vals = [float(u_test.ordinary), float(c_test.ordinary), float(u_val.ordinary), float(c_val.ordinary)]
    managed_vals = [float(u_test.managed), float(c_test.managed), float(u_val.managed), float(c_val.managed)]
    total_vals = [float(u_test.loss), float(c_test.loss), float(u_val.loss), float(c_val.loss)]

    x = np.array([0.0, 1.0, 2.6, 3.6])
    width = 0.65

    ax1.bar(x, ordinary_vals, width, label="Ordinary Positions (N=154)", color="#2b5c8f", edgecolor="#1a365d")
    ax1.bar(x, managed_vals, width, bottom=ordinary_vals, label="Volatility-Managed Positions (N=154)", color="#e28743", edgecolor="#b85d19")

    ax1.set_ylabel(r"Scaled Squared Pricing Loss ($L = \frac{1}{308} \sum_{m=1}^{308} \bar{e}_m^2$)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Figure 1: Decomposition of Out-of-Sample Squared Pricing Loss", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9.5)
    ax1.set_ylim(0, 0.0240)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.0, loc="upper left")

    # Value annotations on bars
    for xi, ord_v, mng_v, tot_v in zip(x, ordinary_vals, managed_vals, total_vals):
        ax1.text(xi, tot_v + 0.00045, f"{tot_v:.6f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax1.text(xi, ord_v / 2.0, f"Ord:\n{ord_v:.4f}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        ax1.text(xi, ord_v + mng_v / 2.0, f"Mng:\n{mng_v:.4f}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

    # Concise annotation positioned safely away from the upper-left legend and bar values
    ax1.annotate(
        f"Validation (2021–2025):\nTotal loss change {pct_increase:+.2f}%\n(Ord: {ord_change:+.6f}, Mng: {mng_change:+.6f})",
        xy=(3.35, float(c_val.ordinary) + float(c_val.managed) * 0.45), xytext=(2.2, 0.0205),
        arrowprops=dict(arrowstyle="->", color="#9b1c1c", lw=1.3, connectionstyle="arc3,rad=-0.12"),
        fontsize=8, color="#9b1c1c", bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f5", edgecolor="#feb2b2", lw=1.2)
    )

    fig1.text(0.5, -0.02,
              "Note: Metric is mean squared discounted payoff error across 308 moments; components divide by 308 and sum to total loss.\n"
              "Values represent pricing accuracy on discounted payoffs, not realized portfolio returns.",
              ha="center", fontsize=8, color="#4a5568", style="italic")

    fig1.tight_layout()
    fig1_pdf = dest / "figure1_loss_decomposition.pdf"
    fig1_png = dest / "figure1_loss_decomposition.png"
    fig1.savefig(fig1_pdf, bbox_inches="tight", metadata={"CreationDate": None})
    fig1.savefig(fig1_png, bbox_inches="tight", dpi=300)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # 2. FIGURE 2: Loss differences and uncertainty (Validation 2021-2025)
    # -------------------------------------------------------------------------
    c_minus_u_12 = prim_contrasts_df[(prim_contrasts_df.period == "Validation") & (prim_contrasts_df.comparison == "C_G minus U_G") & (prim_contrasts_df.block == 12)].iloc[0]
    c_minus_u_6 = prim_contrasts_df[(prim_contrasts_df.period == "Validation") & (prim_contrasts_df.comparison == "C_G minus U_G") & (prim_contrasts_df.block == 6)].iloc[0]

    c2_matched_12 = diag_contrasts_df[(diag_contrasts_df.period == "Validation") & (diag_contrasts_df.model == "C2_matched") & (diag_contrasts_df.comparator == "C1") & (diag_contrasts_df.block == 12)].iloc[0]
    c2_matched_6 = diag_contrasts_df[(diag_contrasts_df.period == "Validation") & (diag_contrasts_df.model == "C2_matched") & (diag_contrasts_df.comparator == "C1") & (diag_contrasts_df.block == 6)].iloc[0]

    c2_g_12 = diag_contrasts_df[(diag_contrasts_df.period == "Validation") & (diag_contrasts_df.model == "C2_G") & (diag_contrasts_df.comparator == "C1") & (diag_contrasts_df.block == 12)].iloc[0]
    c2_g_6 = diag_contrasts_df[(diag_contrasts_df.period == "Validation") & (diag_contrasts_df.model == "C2_G") & (diag_contrasts_df.comparator == "C1") & (diag_contrasts_df.block == 6)].iloc[0]

    # Verify interval endpoint ordering: lower <= upper
    for row, name in [
        (c_minus_u_12, "C_minus_U_12"), (c_minus_u_6, "C_minus_U_6"),
        (c2_matched_12, "C2_matched_12"), (c2_matched_6, "C2_matched_6"),
        (c2_g_12, "C2_G_12"), (c2_g_6, "C2_G_6")
    ]:
        assert row.lower <= row.upper, f"Endpoint ordering violation in {name}: {row.lower} <= {row.upper}"

    fig2, ax2 = plt.subplots(figsize=(9.6, 5.2), dpi=300)

    y_positions_12 = [2.12, 1.12, 0.12]
    y_positions_6  = [1.88, 0.88, -0.12]
    labels = [
        "Primary: C1 minus U\n(Ordinary-only vs Ordinary-plus-managed fitting)",
        f"Exploratory: C2-matched minus C1\n(State interacted, matched $\\lambda$={c2_matched_penalty:.3g})",
        f"Exploratory: C2_G minus C1\n(State interacted, re-tuned $\\lambda$={c2_g_penalty:.3g})"
    ]

    diffs_12 = [float(c_minus_u_12.difference), float(c2_matched_12.contrast), float(c2_g_12.contrast)]
    lowers_12 = [float(c_minus_u_12.lower), float(c2_matched_12.lower), float(c2_g_12.lower)]
    uppers_12 = [float(c_minus_u_12.upper), float(c2_matched_12.upper), float(c2_g_12.upper)]

    diffs_6 = [float(c_minus_u_6.difference), float(c2_matched_6.contrast), float(c2_g_6.contrast)]
    lowers_6 = [float(c_minus_u_6.lower), float(c2_matched_6.lower), float(c2_g_6.lower)]
    uppers_6 = [float(c_minus_u_6.upper), float(c2_matched_6.upper), float(c2_g_6.upper)]

    ax2.axvline(0, color="#718096", linestyle="--", lw=1.2, zorder=1)

    # Subtle horizontal line separating Primary from Exploratory contrasts
    ax2.axhline(1.5, color="#cbd5e0", linestyle=":", lw=1.0, zorder=1)
    ax2.text(-0.0032, 2.38, "PRIMARY HYPOTHESIS", fontsize=7.5, fontweight="bold", color="#2b5c8f")
    ax2.text(-0.0032, 1.38, "EXPLORATORY DIAGNOSTICS", fontsize=7.5, fontweight="bold", color="#4a5568")

    for y, d, l, u in zip(y_positions_12, diffs_12, lowers_12, uppers_12):
        ax2.plot([l, u], [y, y], color="#1a365d", lw=2.4, solid_capstyle="round", zorder=3)
        ax2.scatter(d, y, color="#1a365d", s=55, zorder=4)
        ax2.text(u + 0.0004, y, f"{d:+.6f} [{l:+.6f}, {u:+.6f}] (12m)", va="center", ha="left", fontsize=7.6, color="#1a365d", fontweight="bold")

    for y, d, l, u in zip(y_positions_6, diffs_6, lowers_6, uppers_6):
        ax2.plot([l, u], [y, y], color="#4a5568", lw=1.6, linestyle=":", solid_capstyle="round", zorder=2)
        ax2.scatter(d, y, color="#4a5568", s=38, marker="s", zorder=4)
        ax2.text(u + 0.0004, y, f"{d:+.6f} [{l:+.6f}, {u:+.6f}] (6m)", va="center", ha="left", fontsize=7.1, color="#4a5568")

    legend_elements = [
        Line2D([0], [0], color="#1a365d", lw=2.4, marker="o", markersize=6, label="12-month Block (Primary)"),
        Line2D([0], [0], color="#4a5568", lw=1.6, linestyle=":", marker="s", markersize=5, label="6-month Block (Sensitivity)")
    ]
    ax2.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

    ax2.set_yticks([2.0, 1.0, 0.0])
    ax2.set_yticklabels(labels, fontsize=8.8, fontweight="bold")
    ax2.set_xlabel(r"Pairwise Difference in Scaled Squared Pricing Loss ($\Delta L = L_{\mathrm{first}} - L_{\mathrm{second}}$)", fontsize=10, fontweight="bold")
    ax2.set_title("Figure 2: Validation Loss Differences and 95% Bootstrap Intervals (2021–2025)\n"
                 r"$\longleftarrow$ Favors First-Named Model (Lower Loss)  |  Favors Benchmark (Higher Loss) $\longrightarrow$",
                 fontsize=10.5, fontweight="bold", pad=12)
    ax2.set_xlim(-0.0035, 0.0210)
    ax2.set_ylim(-0.55, 2.55)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    agree_text = (f"M and G selectors agree (interaction = {interaction_diff:.6f})"
                  if (u_agree and c_agree) else f"Tuning interaction = {interaction_diff:.6f}")
    intervals_cross_zero = all(lo <= 0 <= hi for lo, hi in
                               zip(lowers_12 + lowers_6, uppers_12 + uppers_6))
    uncertainty_note = ("All displayed intervals include zero, leaving population direction uncertain."
                        if intervals_cross_zero else
                        "Intervals condition on frozen fits; exploratory comparisons were chosen after evaluation.")
    fig2.text(0.5, -0.03,
              f"Note: 999 paired moving-block draws (seed 340) condition on frozen fits. {agree_text}.\n"
              f"{uncertainty_note}",
              ha="center", fontsize=8, color="#4a5568", style="italic")

    fig2.tight_layout()
    fig2_pdf = dest / "figure2_contrasts_uncertainty.pdf"
    fig2_png = dest / "figure2_contrasts_uncertainty.png"
    fig2.savefig(fig2_pdf, bbox_inches="tight", metadata={"CreationDate": None})
    fig2.savefig(fig2_png, bbox_inches="tight", dpi=300)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # 3. FIGURE 3: Penalty-selection curves (U, C1, C2)
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=300, sharex=True)

    panels_cfg = [
        {
            "ax": axes3[0],
            "title": "(A) Unconditional SDF (U, p=14)",
            "rmse": u_rmse,
            "sel_idx_min": u_min_idx,
            "sel_idx_gold": u_gold_idx,
            "sel_lam_min": u_min_lam,
            "sel_lam_gold": u_gold_lam,
            "win_idx": u_win_idx,
            "agree": u_agree,
            "is_boundary": u_boundary
        },
        {
            "ax": axes3[1],
            "title": "(B) Static Conditional SDF (C1, p=14)",
            "rmse": c_rmse,
            "sel_idx_min": c_min_idx,
            "sel_idx_gold": c_gold_idx,
            "sel_lam_min": c_min_lam,
            "sel_lam_gold": c_gold_lam,
            "win_idx": c_win_idx,
            "agree": c_agree,
            "is_boundary": c_boundary
        },
        {
            "ax": axes3[2],
            "title": "(C) Exploratory SDF (C2, p=28)",
            "rmse": c2_rmse,
            "sel_idx_min": c2_min_idx,
            "sel_idx_gold": c2_gold_idx,
            "sel_lam_min": c2_min_lam,
            "sel_lam_gold": c2_gold_lam,
            "win_idx": c2_win_idx,
            "agree": c2_agree,
            "is_boundary": c2_boundary
        }
    ]

    for p in panels_cfg:
        ax = p["ax"]
        rmse = p["rmse"]
        ax.plot(lambdas, rmse, color="#2b5c8f", lw=1.8, marker="o", markersize=3.5, label="Tuning RMSE")

        win_idx = p["win_idx"]
        win_start = float(lambdas[win_idx])
        win_end = float(lambdas[min(win_idx + 2, len(lambdas) - 1)])
        ax.axvspan(win_start, win_end, color="#48bb78", alpha=0.22,
                   label=f"Goldilocks Window\n[{win_start:.2g}, {win_end:.2g}]")

        if p["agree"]:
            sel_lam = p["sel_lam_gold"]
            sel_rmse = float(rmse[p["sel_idx_gold"]])
            ax.scatter(sel_lam, sel_rmse, color="#e53e3e", s=65, zorder=5, edgecolor="black",
                       label=rf"Selected $\lambda$={sel_lam:.3g}")
            annot = rf"$\lambda_M = \lambda_G = {sel_lam:.3g}$"
            if p["is_boundary"]:
                annot += "\n(Grid Boundary)"
            else:
                annot += "\n(Selectors Agree)"

            if p["sel_idx_gold"] >= len(lambdas) - 3:
                # Near right edge (Panel C): right-align text well within panel boundary
                ax.annotate(annot, xy=(sel_lam, sel_rmse), xytext=(2.2, sel_rmse + 0.35),
                            ha="right", arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.3),
                            fontsize=7.8, fontweight="bold", color="#c53030",
                            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#feb2b2"))
            else:
                ax.annotate(annot, xy=(sel_lam, sel_rmse), xytext=(sel_lam * 0.15, sel_rmse + 0.35),
                            ha="center", arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.3),
                            fontsize=7.8, fontweight="bold", color="#c53030",
                            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#feb2b2"))
        else:
            # When selectors differ, plot both markers and distinct annotations
            min_lam = p["sel_lam_min"]
            gold_lam = p["sel_lam_gold"]
            min_rmse = float(rmse[p["sel_idx_min"]])
            gold_rmse = float(rmse[p["sel_idx_gold"]])
            ax.scatter(min_lam, min_rmse, color="#2b6cb0", marker="s", s=60, zorder=5, edgecolor="black",
                       label=rf"Min Error $\lambda_M$={min_lam:.3g}")
            ax.scatter(gold_lam, gold_rmse, color="#e53e3e", marker="o", s=65, zorder=5, edgecolor="black",
                       label=rf"Goldilocks $\lambda_G$={gold_lam:.3g}")
            ax.annotate(rf"$\lambda_M = {min_lam:.3g}$", xy=(min_lam, min_rmse),
                        xytext=(min_lam * 0.05, min_rmse + 0.45),
                        arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=1.2),
                        fontsize=7.5, fontweight="bold", color="#2b6cb0",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ebf8ff", edgecolor="#bee3f8"))
            ax.annotate(rf"$\lambda_G = {gold_lam:.3g}$", xy=(gold_lam, gold_rmse),
                        xytext=(gold_lam * 0.05, gold_rmse + 0.25),
                        arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.2),
                        fontsize=7.5, fontweight="bold", color="#c53030",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", edgecolor="#feb2b2"))

        ax.set_xscale("log")
        ax.set_title(p["title"], fontsize=10.2, fontweight="bold", pad=8)
        ax.set_xlabel(r"Ridge Penalty $\lambda$ (log scale)", fontsize=9.2)
        ax.grid(True, linestyle="--", alpha=0.35, which="both")
        ax.legend(loc="lower left", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#cccccc")

    axes3[0].set_ylabel("Pooled Tuning RMSE (Folds 1–3)", fontsize=10, fontweight="bold")
    fig3.suptitle("Figure 3: Training Forward-Fold Tuning Curves and Penalty Selection", fontsize=12, fontweight="bold", y=1.03)

    boundary_note = (f" In Panel C, selected λ={c2_gold_lam:.3g} is at the {c2_boundary_position} grid boundary."
                     if c2_boundary else f" In Panel C, selected λ={c2_gold_lam:.3g} is interior to the grid.")
    fig3.text(0.5, -0.06,
              "Note: Average squared pricing loss pooled across expanding forward folds (1975–2009 Training), then square-rooted to yield RMSE.\n"
              f"Shading marks the winning Goldilocks window; markers show the selected penalties.{boundary_note}",
              ha="center", fontsize=8, color="#4a5568", style="italic")

    fig3.tight_layout()
    fig3_pdf = dest / "figure3_tuning_curves.pdf"
    fig3_png = dest / "figure3_tuning_curves.png"
    fig3.savefig(fig3_pdf, bbox_inches="tight", metadata={"CreationDate": None})
    fig3.savefig(fig3_png, bbox_inches="tight", dpi=300)
    plt.close(fig3)

    # -------------------------------------------------------------------------
    # 4. Underlying machine-readable data artifact (All Dynamic)
    # -------------------------------------------------------------------------
    figures_data = {
        "figure1": {
            "metric": "scaled_squared_pricing_loss",
            "loss_definition": "squared_period_mean_pricing_errors",
            "denominator": 308,
            "development_test_132m": {
                "dates": "2010-01-31 to 2020-12-31",
                "U": {"loss": float(u_test.loss), "ordinary": float(u_test.ordinary), "managed": float(u_test.managed)},
                "C1": {"loss": float(c_test.loss), "ordinary": float(c_test.ordinary), "managed": float(c_test.managed)}
            },
            "final_validation_60m": {
                "dates": "2021-01-31 to 2025-12-31",
                "U": {"loss": float(u_val.loss), "ordinary": float(u_val.ordinary), "managed": float(u_val.managed)},
                "C1": {"loss": float(c_val.loss), "ordinary": float(c_val.ordinary), "managed": float(c_val.managed)},
                "headline_delta": float(c_val.loss) - float(u_val.loss),
                "headline_relative_increase_pct": pct_increase
            }
        },
        "figure2": {
            "period": "Final Validation (2021-01-31 to 2025-12-31, 60 months)",
            "loss_definition": "squared_period_mean_pricing_errors",
            "bootstrap_settings": {"num_draws": 999, "seed": 340},
            "contrasts": {
                "C1_minus_U": {
                    "role": "primary",
                    "difference": float(c_minus_u_12.difference),
                    "block_12m": {"lower": float(c_minus_u_12.lower), "upper": float(c_minus_u_12.upper)},
                    "block_6m": {"lower": float(c_minus_u_6.lower), "upper": float(c_minus_u_6.upper)}
                },
                "C2_matched_minus_C1": {
                    "role": "exploratory_diagnostic",
                    "penalty": c2_matched_penalty,
                    "contrast": float(c2_matched_12.contrast),
                    "block_12m": {"lower": float(c2_matched_12.lower), "upper": float(c2_matched_12.upper)},
                    "block_6m": {"lower": float(c2_matched_6.lower), "upper": float(c2_matched_6.upper)}
                },
                "C2_G_minus_C1": {
                    "role": "exploratory_diagnostic",
                    "penalty": c2_g_penalty,
                    "contrast": float(c2_g_12.contrast),
                    "block_12m": {"lower": float(c2_g_12.lower), "upper": float(c2_g_12.upper)},
                    "block_6m": {"lower": float(c2_g_6.lower), "upper": float(c2_g_6.upper)}
                },
                "tuning_interaction": {
                    "difference": interaction_diff,
                    "ci_12m": interaction_12_ci,
                    "ci_6m": interaction_6_ci
                }
            }
        },
        "figure3": {
            "sample": "Training (1975-01-31 to 2009-12-31, 420 months)",
            "lambdas": lambdas.tolist(),
            "unconditional": {
                "rmse_curve": u_rmse.tolist(),
                "selected_lambda": u_gold_lam,
                "selected_idx": u_gold_idx,
                "selected_idx_min": u_min_idx,
                "selected_idx_gold": u_gold_idx,
                "selectors_agree": u_agree,
                "is_grid_boundary": u_boundary,
                "winning_window_idx": u_win_idx,
                "winning_window_lambdas": lambdas[u_win_idx:min(u_win_idx + 3, len(lambdas))].tolist()
            },
            "conditional_static": {
                "rmse_curve": c_rmse.tolist(),
                "selected_lambda": c_gold_lam,
                "selected_idx": c_gold_idx,
                "selected_idx_min": c_min_idx,
                "selected_idx_gold": c_gold_idx,
                "selectors_agree": c_agree,
                "is_grid_boundary": c_boundary,
                "winning_window_idx": c_win_idx,
                "winning_window_lambdas": lambdas[c_win_idx:min(c_win_idx + 3, len(lambdas))].tolist()
            },
            "conditional_interacted": {
                "rmse_curve": c2_rmse.tolist(),
                "selected_lambda": c2_gold_lam,
                "selected_idx": c2_gold_idx,
                "selected_idx_min": c2_min_idx,
                "selected_idx_gold": c2_gold_idx,
                "selectors_agree": c2_agree,
                "is_grid_boundary": c2_boundary,
                "boundary_position": c2_boundary_position,
                "winning_window_idx": c2_win_idx,
                "winning_window_lambdas": lambdas[c2_win_idx:min(c2_win_idx + 3, len(lambdas))].tolist()
            }
        }
    }
    data_path = dest / "figures_data.json"
    data_path.write_text(json.dumps(figures_data, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # 5. Captions and Figure Index (Dynamically Interpolated from figures_data)
    # -------------------------------------------------------------------------
    f1_dat = figures_data["figure1"]
    f2_dat = figures_data["figure2"]
    f3_dat = figures_data["figure3"]

    caption_1 = (
        "Figure 1: Decomposition of out-of-sample squared pricing loss across evaluation periods. "
        "Bars display total scaled squared pricing loss (L = (1/308) * sum e_m^2) for the unconditional SDF (U, 14 coefficients) "
        "and static conditional SDF (C1, 14 coefficients) across Development Test (January 2010–December 2020, 132 months) "
        "and Final Validation (January 2021–December 2025, 60 months). Each stacked bar decomposes total loss into contributions "
        "from the 154 ordinary asset restrictions (bottom) and the 154 volatility-managed asset restrictions (top), both scaled "
        "by common denominator 308 so components sum to reported total loss. In Development Test, conditioning yielded a "
        f"loss change from {f1_dat['development_test_132m']['U']['loss']:.6f} to {f1_dat['development_test_132m']['C1']['loss']:.6f} "
        f"(Delta = {f1_dat['development_test_132m']['C1']['loss'] - f1_dat['development_test_132m']['U']['loss']:+.6f}). "
        f"In Final Validation, the managed contribution changed from {f1_dat['final_validation_60m']['U']['managed']:.6f} "
        f"to {f1_dat['final_validation_60m']['C1']['managed']:.6f} ({mng_change:+.6f}), and the ordinary contribution changed "
        f"from {f1_dat['final_validation_60m']['U']['ordinary']:.6f} to {f1_dat['final_validation_60m']['C1']['ordinary']:.6f} ({ord_change:+.6f}). "
        f"Total loss changed from {f1_dat['final_validation_60m']['U']['loss']:.6f} to {f1_dat['final_validation_60m']['C1']['loss']:.6f} ({pct_increase:+.2f}%). "
        "Plotted values represent squared period-mean pricing errors on discounted payoffs (not average monthly squared errors or investment returns)."
    )

    agree_desc = (
        f"Minimum-error and Goldilocks selectors make identical choices (lambda_M = lambda_G = {c_gold_lam:.2f} for C1, {u_gold_lam:.2f} for U), "
        f"resulting in an exact zero tuning interaction (I = {interaction_diff:.6f} [{interaction_12_ci[0]:.6f}, {interaction_12_ci[1]:.6f}]). "
        if (u_agree and c_agree)
        else f"Selectors choose lambda_M = {c_min_lam:.2f}, lambda_G = {c_gold_lam:.2f} for C1, and lambda_M = {u_min_lam:.2f}, lambda_G = {u_gold_lam:.2f} for U (tuning interaction I = {interaction_diff:+.6f}). "
    )

    caption_2 = (
        "Figure 2: Out-of-sample loss contrasts and 95% moving-block bootstrap confidence intervals (Final Validation 2021–2025). "
        "Point estimates and 95% paired bootstrap intervals for pairwise differences in scaled squared pricing loss (Delta L) over the "
        "60-month final validation period (January 2021–December 2025). Resampling uses 999 paired moving-block draws (seed 340) with primary "
        "12-month blocks (solid whiskers, circles) and 6-month sensitivity blocks (dashed whiskers, squares) conditioning on frozen model coefficients. "
        "Negative differences (Delta L < 0) indicate superior pricing accuracy for the first-named model. The primary comparison (C1 - U, top) evaluates "
        f"whether adding 154 volatility-managed restrictions reduces pricing loss; the observed difference is {f2_dat['contrasts']['C1_minus_U']['difference']:+.6f} "
        f"(12-mo 95% CI: [{f2_dat['contrasts']['C1_minus_U']['block_12m']['lower']:+.6f}, {f2_dat['contrasts']['C1_minus_U']['block_12m']['upper']:+.6f}]; "
        f"6-mo CI: [{f2_dat['contrasts']['C1_minus_U']['block_6m']['lower']:+.6f}, {f2_dat['contrasts']['C1_minus_U']['block_6m']['upper']:+.6f}]), "
        f"with negative differences favoring conditional fitting. {agree_desc}"
        "The exploratory diagnostic contrasts (bottom) examine an expanded 28-coefficient SDF (C2); both the matched-penalty contrast "
        f"(C2-matched - C1, lambda={c2_matched_penalty:.2f}, Delta = {f2_dat['contrasts']['C2_matched_minus_C1']['contrast']:+.6f}, "
        f"12-mo CI: [{f2_dat['contrasts']['C2_matched_minus_C1']['block_12m']['lower']:+.6f}, {f2_dat['contrasts']['C2_matched_minus_C1']['block_12m']['upper']:+.6f}]) "
        f"and separately tuned contrast (C2_G - C1, lambda={c2_g_penalty:.2f}, Delta = {f2_dat['contrasts']['C2_G_minus_C1']['contrast']:+.6f}, "
        f"12-mo CI: [{f2_dat['contrasts']['C2_G_minus_C1']['block_12m']['lower']:+.6f}, {f2_dat['contrasts']['C2_G_minus_C1']['block_12m']['upper']:+.6f}]) "
        f"are exploratory. {uncertainty_note} Loss is defined as squared period-mean pricing errors (not average monthly squared errors)."
    )

    sel_summary_text = (
        f"In all three models, minimum-error and Goldilocks tuning choose the exact same penalty: lambda = {u_gold_lam:.4f} for U, "
        f"lambda = {c_gold_lam:.4g} for C1, and lambda = {c2_gold_lam:.4g} for C2. "
        if (u_agree and c_agree and c2_agree)
        else f"Selected penalties: U (lambda_M={u_min_lam:.4g}, lambda_G={u_gold_lam:.4g}), C1 (lambda_M={c_min_lam:.4g}, lambda_G={c_gold_lam:.4g}), C2 (lambda_M={c2_min_lam:.4g}, lambda_G={c2_gold_lam:.4g}). "
    )
    boundary_text = (
        f"In Panel C, the selected penalty of {c2_gold_lam:.3g} sits at the {c2_boundary_position} boundary of the prespecified grid. "
        "The penalty applies to all 28 coefficients; values beyond the tested grid were not evaluated."
        if c2_boundary
        else f"In Panel C, the selected penalty is lambda={c2_gold_lam:.3g} (interior to grid)."
    )

    caption_3 = (
        "Figure 3: Penalty-selection curves and Goldilocks window evaluation. Panels show pooled root mean squared pricing error (Tuning RMSE) "
        "across expanding forward validation folds within the Training sample (January 1975–December 2009, 420 months) plotted against the "
        f"{len(lambdas)} log-spaced ridge penalties (lambda in [{lambdas[0]:.0e}, {lambdas[-1]:.0e}]) for: (A) Unconditional model (U, 14 coefficients), "
        "(B) Static conditional model (C1, 14 coefficients), and (C) Exploratory state-interacted model (C2, 28 coefficients). "
        "The winning 3-penalty Goldilocks window (minimizing normalized window mean RMSE plus normalized variance) is indicated by the shaded region. "
        f"Markers identify the selected penalties under minimum-error (lambda_M) and Goldilocks (lambda_G) rules. {sel_summary_text}"
        f"{boundary_text} Loss is defined as squared period-mean pricing errors (not average monthly squared errors)."
    )

    figure_index = [
        {
            "filename": "figure1_loss_decomposition.pdf",
            "png_filename": "figure1_loss_decomposition.png",
            "title": "Figure 1: Decomposition of Out-of-Sample Squared Pricing Loss",
            "input_artifacts": ["tables/test_primary.csv", "tables/validation_primary.csv"],
            "fields": ["ordinary", "managed", "loss"],
            "suggested_manuscript_placement": "Section 4: Primary Empirical Findings (Loss Decomposition and Validation Results)",
            "caption": caption_1
        },
        {
            "filename": "figure2_contrasts_uncertainty.pdf",
            "png_filename": "figure2_contrasts_uncertainty.png",
            "title": "Figure 2: Validation Loss Differences and 95% Bootstrap Intervals (2021–2025)",
            "input_artifacts": ["tables/primary_contrasts.csv", "tables/diagnostic_contrasts.csv"],
            "fields": ["difference", "contrast", "lower", "upper", "block"],
            "suggested_manuscript_placement": "Section 4: Primary Hypothesis Testing & Statistical Uncertainty",
            "caption": caption_2
        },
        {
            "filename": "figure3_tuning_curves.pdf",
            "png_filename": "figure3_tuning_curves.png",
            "title": "Figure 3: Training Forward-Fold Tuning Curves and Penalty Selection",
            "input_artifacts": ["primary/tuning_curves.json", "diagnostic/tuning_and_checks.json"],
            "fields": ["curves_data.lambdas", "curves_data.unconditional_rmse", "curves_data.conditional_rmse", "pooled_rmse", "selection_summary", "goldilocks"],
            "suggested_manuscript_placement": "Section 3 / Section 4: Model Tuning, Regularization, and Parameter Selection",
            "caption": caption_3
        }
    ]
    index_path = dest / "figure_index.json"
    index_path.write_text(json.dumps(figure_index, indent=2), encoding="utf-8")

    # FIGURES.md formatted report
    md_lines = [
        "# Publication Figures Index & Manuscript Placements",
        "",
        "This directory contains vector PDF and 300 DPI PNG figures generated directly from current run numerical artifacts.",
        "",
        "| Figure | Vector PDF | High-Res PNG | Suggested Manuscript Placement |",
        "|---|---|---|---|",
        "| **Figure 1: Loss Decomposition** | [figure1_loss_decomposition.pdf](figure1_loss_decomposition.pdf) | [figure1_loss_decomposition.png](figure1_loss_decomposition.png) | Section 4: Primary Empirical Findings (Loss Decomposition) |",
        "| **Figure 2: Loss Differences & CI** | [figure2_contrasts_uncertainty.pdf](figure2_contrasts_uncertainty.pdf) | [figure2_contrasts_uncertainty.png](figure2_contrasts_uncertainty.png) | Section 4: Primary Hypothesis Testing & Statistical Uncertainty |",
        "| **Figure 3: Tuning Curves** | [figure3_tuning_curves.pdf](figure3_tuning_curves.pdf) | [figure3_tuning_curves.png](figure3_tuning_curves.png) | Section 3 / 4: Model Tuning, Regularization & Selection |",
        "",
        "## Detailed Captions",
        ""
    ]
    for item in figure_index:
        md_lines.extend([
            f"### {item['title']}",
            "",
            f"- **PDF:** `{item['filename']}`",
            f"- **PNG:** `{item['png_filename']}`",
            f"- **Source Artifacts:** {', '.join(f'`{a}`' for a in item['input_artifacts'])}",
            f"- **Plotted Fields:** {', '.join(f'`{f}`' for f in item['fields'])}",
            f"- **Placement:** {item['suggested_manuscript_placement']}",
            "",
            f"> **Caption:** {item['caption']}",
            ""
        ])
    (dest / "FIGURES.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Final Validation Self-Check
    required_outputs = [
        "figure1_loss_decomposition.pdf", "figure1_loss_decomposition.png",
        "figure2_contrasts_uncertainty.pdf", "figure2_contrasts_uncertainty.png",
        "figure3_tuning_curves.pdf", "figure3_tuning_curves.png",
        "figures_data.json", "figure_index.json", "FIGURES.md"
    ]
    for fname in required_outputs:
        fpath = dest / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            raise RuntimeError(f"Figure generation failed to produce valid artifact: {fname}")

    return {
        item["filename"]: (dest / item["filename"]).as_posix()
        for item in figure_index
    }


if __name__ == "__main__":
    import sys
    run_target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/primary"
    build_figures(run_target)
