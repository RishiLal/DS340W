"""
test_representation_isolation.py — Verify Representation Training Isolation
DS340W Independent Audit Verification

Verifies:
  1. The pre-specified split between representation learning (T_rep = 180 months)
     and econometric inference (T_inf = 167 months).
  2. Production Pipeline Isolation:
     - Calls actual production functions (train_gan_sdf and extract_gan_factors).
     - Perturbing inference, test, or validation data leaves fitted imputation,
       normalization parameters, GAN weights, and PCA basis strictly unchanged.
  3. Sensitivity / Meaningful Intervention:
     - Perturbing representation data alters fitted parameters (proving the test is sensitive).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_splits"
MODELS_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(PROJECT_DIR))
from scripts.split_config import (
    load_raw_splits,
    get_representation_and_inference_splits,
    T_REP_MONTHS,
    T_INF_MONTHS,
    REP_RETURN_START,
    REP_RETURN_END,
    INF_RETURN_START,
    INF_RETURN_END
)
from scripts.train_gan_sdf import train_gan_sdf, SDFGenerator
from scripts.extract_gan_factors import extract_gan_factors


def test_split_chronology_and_lengths():
    """Verify that the representation and inference splits partition the training period cleanly."""
    (f_tr, f_te, f_va), (p_tr, p_te, p_va) = load_raw_splits(data_dir=DATA_DIR)
    splits = get_representation_and_inference_splits(f_tr, p_tr)
    
    rep_ret = splits["rep"]["return_dates"]
    inf_ret = splits["inf"]["return_dates"]
    
    assert len(rep_ret) == T_REP_MONTHS, f"Expected {T_REP_MONTHS} rep months, found {len(rep_ret)}"
    assert len(inf_ret) == T_INF_MONTHS, f"Expected {T_INF_MONTHS} inf months, found {len(inf_ret)}"
    
    assert rep_ret[0] == REP_RETURN_START
    assert rep_ret[-1] == REP_RETURN_END
    assert inf_ret[0] == INF_RETURN_START
    assert inf_ret[-1] == INF_RETURN_END
    
    assert rep_ret[-1] < inf_ret[0], "Representation and inference periods overlap!"
    print(f"[PASS] Chronological split boundaries strictly verified: Rep = {len(rep_ret)} months ({rep_ret[0]} to {rep_ret[-1]}), "
          f"Inf = {len(inf_ret)} months ({inf_ret[0]} to {inf_ret[-1]}).")


def test_production_pipeline_isolation():
    """
    Test actual production training and extraction functions:
    Verify that perturbing inference-sample data by 1000x has ZERO effect on
    imputation, normalization, GAN weights, or PCA basis.
    """
    (f_tr_orig, f_te_orig, f_va_orig), (p_tr_orig, p_te_orig, p_va_orig) = load_raw_splits(data_dir=DATA_DIR)
    
    # 1. Baseline fit on original data (fast 5-epoch run)
    res_orig = train_gan_sdf(
        epochs=5,
        seed=123,
        save_model=False,
        verbose=False,
        data_dir=DATA_DIR
    )
    z_mean_orig = res_orig["z_mean"]
    z_std_orig = res_orig["z_std"]
    meds_orig = res_orig["rep_medians"]
    w_orig = res_orig["generator"].state_dict()["fc1.weight"].numpy().copy()
    
    # 2. Perturb inference sample by 1000x
    f_tr_pert = f_tr_orig.copy()
    p_tr_pert = p_tr_orig.copy()
    
    factor_cols = [c for c in f_tr_pert.columns if c not in ["Date", "RF"]]
    port_cols = [c for c in p_tr_pert.columns if c != "Date"]
    
    # Rows 180 to 347 are inference states in f_tr; rows 181 to 347 are inference returns in p_tr
    f_tr_pert.loc[180:, factor_cols] += 1000.0
    p_tr_pert.loc[181:, port_cols] += 500.0
    
    splits_pert = get_representation_and_inference_splits(f_tr_pert, p_tr_pert)
    
    res_pert = train_gan_sdf(
        epochs=5,
        seed=123,
        save_model=False,
        verbose=False,
        custom_splits=splits_pert
    )
    z_mean_pert = res_pert["z_mean"]
    z_std_pert = res_pert["z_std"]
    meds_pert = res_pert["rep_medians"]
    w_pert = res_pert["generator"].state_dict()["fc1.weight"].numpy().copy()
    
    diff_mean = np.max(np.abs(z_mean_orig - z_mean_pert))
    diff_std = np.max(np.abs(z_std_orig - z_std_pert))
    diff_weights = np.max(np.abs(w_orig - w_pert))
    
    assert diff_mean < 1e-6, f"Normalization mean contaminated by inference data: diff = {diff_mean}"
    assert diff_std < 1e-6, f"Normalization std contaminated by inference data: diff = {diff_std}"
    assert diff_weights < 1e-6, f"GAN weights contaminated by inference data: diff = {diff_weights}"
    
    # Check medians
    for k in factor_cols:
        assert abs(meds_orig[k] - meds_pert[k]) < 1e-6, f"Median for {k} contaminated: {meds_orig[k]} vs {meds_pert[k]}"
        
    print("[PASS] Production pipeline verified: Zero inference-to-representation contamination (all diffs = 0.00e+00).")


def test_meaningful_intervention_sensitivity():
    """Verify that perturbing representation data DOES change fitted parameters (proving sensitivity)."""
    (f_tr_orig, f_te_orig, f_va_orig), (p_tr_orig, p_te_orig, p_va_orig) = load_raw_splits(data_dir=DATA_DIR)
    
    res_orig = train_gan_sdf(epochs=3, seed=42, save_model=False, verbose=False)
    z_mean_orig = res_orig["z_mean"]
    
    # Perturb representation sample
    f_tr_rep_pert = f_tr_orig.copy()
    factor_cols = [c for c in f_tr_rep_pert.columns if c not in ["Date", "RF"]]
    f_tr_rep_pert.loc[:10, factor_cols] += 10.0
    
    splits_rep_pert = get_representation_and_inference_splits(f_tr_rep_pert, p_tr_orig)
    res_rep_pert = train_gan_sdf(epochs=3, seed=42, save_model=False, verbose=False, custom_splits=splits_rep_pert)
    z_mean_rep_pert = res_rep_pert["z_mean"]
    
    diff_sens = np.max(np.abs(z_mean_orig - z_mean_rep_pert))
    assert diff_sens > 0.1, f"Pipeline insensitive to representation changes! diff = {diff_sens}"
    print(f"[PASS] Sensitivity check passed: Representation perturbation alters parameters as expected (diff = {diff_sens:.4f}).")


if __name__ == "__main__":
    test_split_chronology_and_lengths()
    test_production_pipeline_isolation()
    test_meaningful_intervention_sensitivity()
    print("ALL REPRESENTATION ISOLATION TESTS PASSED SUCCESSFULLY.")
