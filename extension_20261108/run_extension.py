"""
Main orchestrator for the extension methodology:
Runs the complete pipeline end-to-end and exports machine-readable results.
"""
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import numpy as np
import pandas as pd

from extension_20261108.data import load_screened_universe, compute_volatility_state
from extension_20261108.tuning import run_forward_inner_tuning, fit_and_freeze_full_models
from extension_20261108.evaluate import evaluate_period_models, run_paired_moving_block_bootstrap

OUTPUT_DIR = ROOT_DIR / "runs" / "primary"

def main(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    print("==================================================================")
    print("STARTING EXTENSION PIPELINE: CONDITIONAL THEME PRICING & STABILITY")
    print("==================================================================")
    
    # 1. Load and screen universe
    print("\n[Step 1/5] Loading data and screening universe on first 210 Training months...")
    factors_panel, market_series, full_market, eligible_factors, target_dates = load_screened_universe()
    print(f"Loaded {len(eligible_factors)} eligible factors + excess market return across {len(target_dates)} months.")
    
    # Assert date partitions
    train_dates = target_dates[:420]
    test_dates = target_dates[420:552]
    val_dates = target_dates[552:612]
    
    assert train_dates[0] == "1975-01-31" and train_dates[-1] == "2009-12-31"
    assert test_dates[0] == "2010-01-31" and test_dates[-1] == "2020-12-31"
    assert val_dates[0] == "2021-01-31" and val_dates[-1] == "2025-12-31"
    print(f"Date boundaries confirmed:")
    print(f"  Training:         {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} months, 68.6%)")
    print(f"  Development Test: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} months, 21.6%)")
    print(f"  Final Validation: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} months, 9.8%)")
    
    # 2. Forward inner validation tuning
    print("\n[Step 2/5] Running forward expanding 3-fold inner validation tuning...")
    tuning_res = run_forward_inner_tuning(factors_panel, market_series, full_market, train_dates)
    sel = tuning_res["selection_summary"]
    
    print("\nTuning selections across 31 log-spaced penalties (10^-8 to 10^1):")
    print(f"  Unconditional Minimum (M):    lambda = {sel['unconditional_minimum']['lambda_val']:.6e} (idx {sel['unconditional_minimum']['lambda_idx']}), RMSE = {sel['unconditional_minimum']['rmse']:.6f}")
    print(f"  Unconditional Goldilocks (G): lambda = {sel['unconditional_goldilocks']['lambda_val']:.6e} (idx {sel['unconditional_goldilocks']['lambda_idx']}), RMSE = {sel['unconditional_goldilocks']['rmse']:.6f}")
    print(f"  Conditional Minimum (M):      lambda = {sel['conditional_minimum']['lambda_val']:.6e} (idx {sel['conditional_minimum']['lambda_idx']}), RMSE = {sel['conditional_minimum']['rmse']:.6f}")
    print(f"  Conditional Goldilocks (G):   lambda = {sel['conditional_goldilocks']['lambda_val']:.6e} (idx {sel['conditional_goldilocks']['lambda_idx']}), RMSE = {sel['conditional_goldilocks']['rmse']:.6f}")
    
    # Save tuning curves
    (output_dir / "tuning_curves.json").write_text(json.dumps(tuning_res, indent=2))
    print("Saved tuning_curves.json")
    
    # 3. Refit and freeze full training models
    print("\n[Step 3/5] Refitting the four prespecified models on full 420 Training months...")
    frozen_models, transforms = fit_and_freeze_full_models(factors_panel, market_series, full_market, train_dates, sel)
    (output_dir / "frozen_models.json").write_text(json.dumps(frozen_models, indent=2))
    print("Saved frozen_models.json")
    saved_transforms = {key: value for key, value in transforms.items()
                        if key not in ("F_all", "X_all", "h_all")}
    (output_dir / "training_transforms.json").write_text(
        json.dumps(saved_transforms, indent=2, allow_nan=False))

    
    # Prepare matrices for Test and Validation
    F_all = transforms["F_all"]
    X_all = transforms["X_all"]
    h_all = transforms["h_all"]
    f_scales = np.array(transforms["factor_scales"])
    m_scale = transforms["market_scale"]
    
    # Full conditional dictionary: Q = [X, h * X] (612 x 308)
    Q_all = np.hstack([X_all, h_all[:, np.newaxis] * X_all])
    assert Q_all.shape == (612, 308)
    
    # Test partition (months 420 to 552)
    F_test = F_all[420:552]
    Q_test = Q_all[420:552]
    
    # Validation partition (months 552 to 612)
    F_val = F_all[552:612]
    Q_val = Q_all[552:612]
    
    # 4. Evaluate on Development Test
    print("\n[Step 4/5] Evaluating on Development Test (Jan 2010–Dec 2020, 132 months)...")
    test_summary, test_MQ = evaluate_period_models(frozen_models, F_test, Q_test, f_scales, m_scale)
    (output_dir / "test_evaluation.json").write_text(json.dumps(test_summary, indent=2))
    print(f"Development Test Results:")
    for k, m_res in test_summary["models"].items():
        print(f"  {k} ({m_res['name']}): Loss = {m_res['loss']:.6f}, RMSE = {m_res['rmse']:.6f}, MAE = {m_res['mae_bps']:.2f} bps, RMSE = {m_res['rmse_bps']:.2f} bps")
    print(f"  Main Contrast Delta(G) = {test_summary['delta_goldilocks']:.6f}")
    print(f"  Contrast Delta(M)      = {test_summary['delta_minimum']:.6f}")
    print(f"  Interaction I          = {test_summary['interaction']:.6f}")
    
    # 5. Evaluate on Final Validation & Bootstrap
    print("\n[Step 5/5] Evaluating on Final Primary Validation (Jan 2021–Dec 2025, 60 months)...")
    val_summary, val_MQ = evaluate_period_models(frozen_models, F_val, Q_val, f_scales, m_scale)
    (output_dir / "validation_evaluation.json").write_text(json.dumps(val_summary, indent=2))
    
    print(f"Final Validation Results:")
    for k, m_res in val_summary["models"].items():
        print(f"  {k} ({m_res['name']}): Loss = {m_res['loss']:.6f}, RMSE = {m_res['rmse']:.6f}, MAE = {m_res['mae_bps']:.2f} bps, RMSE = {m_res['rmse_bps']:.2f} bps")
    print(f"  Main Contrast Delta(G) = {val_summary['delta_goldilocks']:.6f}")
    print(f"  Contrast Delta(M)      = {val_summary['delta_minimum']:.6f}")
    print(f"  Interaction I          = {val_summary['interaction']:.6f}")
    
    print("\nRunning 999 paired moving-block bootstrap resamples (block length = 12)...")
    boot_12 = run_paired_moving_block_bootstrap(val_MQ, block_length=12, num_resamples=999, seed=340)
    print("Running 999 paired moving-block bootstrap resamples (sensitivity block length = 6)...")
    boot_6 = run_paired_moving_block_bootstrap(val_MQ, block_length=6, num_resamples=999, seed=340)
    
    boot_res = {
        "primary_block_12": boot_12,
        "sensitivity_block_6": boot_6
    }
    (output_dir / "bootstrap_results.json").write_text(json.dumps(boot_res, indent=2))
    
    # Print bootstrap intervals
    ci12 = boot_12["ci_intervals"]
    print(f"\nBootstrap 95% Confidence Intervals (Block Length 12):")
    print(f"  Delta(Goldilocks): {val_summary['delta_goldilocks']:.6f} [95% CI: {ci12['delta_goldilocks']['p025']:.6f}, {ci12['delta_goldilocks']['p975']:.6f}]")
    print(f"  Delta(Minimum):    {val_summary['delta_minimum']:.6f} [95% CI: {ci12['delta_minimum']['p025']:.6f}, {ci12['delta_minimum']['p975']:.6f}]")
    print(f"  Interaction I:     {val_summary['interaction']:.6f} [95% CI: {ci12['interaction']['p025']:.6f}, {ci12['interaction']['p975']:.6f}]")
    
    # Build markdown summary table
    md_table = f"""# Extension Four-Cell Evaluation Results

## Primary Endpoint: Final Validation (Jan 2021–Dec 2025, 60 months)

| Tuning Rule | Unconditional SDF (U) | Conditional SDF (C) | Contrast (C − U) |
|---|---|---|---|
| **Minimum Error (M)** | Loss: {val_summary['models']['U_M']['loss']:.6f}<br>RMSE: {val_summary['models']['U_M']['rmse']:.6f}<br>Unc MAE: {val_summary['models']['U_M']['mae_bps']:.2f} bps<br>$\\lambda = {val_summary['models']['U_M']['lambda']:.2e}$ | Loss: {val_summary['models']['C_M']['loss']:.6f}<br>RMSE: {val_summary['models']['C_M']['rmse']:.6f}<br>Unc MAE: {val_summary['models']['C_M']['mae_bps']:.2f} bps<br>$\\lambda = {val_summary['models']['C_M']['lambda']:.2e}$ | $\\Delta_M = {val_summary['delta_minimum']:.6f}$<br>[95% CI: {ci12['delta_minimum']['p025']:.6f}, {ci12['delta_minimum']['p975']:.6f}] |
| **Goldilocks (G)** | Loss: {val_summary['models']['U_G']['loss']:.6f}<br>RMSE: {val_summary['models']['U_G']['rmse']:.6f}<br>Unc MAE: {val_summary['models']['U_G']['mae_bps']:.2f} bps<br>$\\lambda = {val_summary['models']['U_G']['lambda']:.2e}$ | Loss: {val_summary['models']['C_G']['loss']:.6f}<br>RMSE: {val_summary['models']['C_G']['rmse']:.6f}<br>Unc MAE: {val_summary['models']['C_G']['mae_bps']:.2f} bps<br>$\\lambda = {val_summary['models']['C_G']['lambda']:.2e}$ | $\\Delta_G = {val_summary['delta_goldilocks']:.6f}$<br>[95% CI: {ci12['delta_goldilocks']['p025']:.6f}, {ci12['delta_goldilocks']['p975']:.6f}] |
| **Tuning Contrast (G − M)** | Loss Diff: {val_summary['models']['U_G']['loss'] - val_summary['models']['U_M']['loss']:.6f} | Loss Diff: {val_summary['models']['C_G']['loss'] - val_summary['models']['C_M']['loss']:.6f} | **Interaction $I = {val_summary['interaction']:.6f}$**<br>[95% CI: {ci12['interaction']['p025']:.6f}, {ci12['interaction']['p975']:.6f}] |

### Sensitivity: Block Length 6 Bootstrap 95% Intervals
- Main Contrast $\\Delta_G$: [{boot_6['ci_intervals']['delta_goldilocks']['p025']:.6f}, {boot_6['ci_intervals']['delta_goldilocks']['p975']:.6f}]
- Interaction $I$: [{boot_6['ci_intervals']['interaction']['p025']:.6f}, {boot_6['ci_intervals']['interaction']['p975']:.6f}]

## Development Evidence: Test (Jan 2010–Dec 2020, 132 months)

| Tuning Rule | Unconditional SDF (U) | Conditional SDF (C) | Contrast (C − U) |
|---|---|---|---|
| **Minimum Error (M)** | Loss: {test_summary['models']['U_M']['loss']:.6f}<br>RMSE: {test_summary['models']['U_M']['rmse']:.6f}<br>Unc MAE: {test_summary['models']['U_M']['mae_bps']:.2f} bps | Loss: {test_summary['models']['C_M']['loss']:.6f}<br>RMSE: {test_summary['models']['C_M']['rmse']:.6f}<br>Unc MAE: {test_summary['models']['C_M']['mae_bps']:.2f} bps | $\\Delta_M = {test_summary['delta_minimum']:.6f}$ |
| **Goldilocks (G)** | Loss: {test_summary['models']['U_G']['loss']:.6f}<br>RMSE: {test_summary['models']['U_G']['rmse']:.6f}<br>Unc MAE: {test_summary['models']['U_G']['mae_bps']:.2f} bps | Loss: {test_summary['models']['C_G']['loss']:.6f}<br>RMSE: {test_summary['models']['C_G']['rmse']:.6f}<br>Unc MAE: {test_summary['models']['C_G']['mae_bps']:.2f} bps | $\\Delta_G = {test_summary['delta_goldilocks']:.6f}$ |
| **Tuning Contrast (G − M)** | Loss Diff: {test_summary['models']['U_G']['loss'] - test_summary['models']['U_M']['loss']:.6f} | Loss Diff: {test_summary['models']['C_G']['loss'] - test_summary['models']['C_M']['loss']:.6f} | **Interaction $I = {test_summary['interaction']:.6f}$** |
"""
    (output_dir / "summary_table.md").write_text(md_table, encoding="utf-8")
    print("Saved summary_table.md")
    
    elapsed = time.time() - start_time
    print(f"\nExtension pipeline completed successfully in {elapsed:.2f} seconds.")
    print("==================================================================")

if __name__ == "__main__":
    main()
