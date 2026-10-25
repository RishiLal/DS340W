"""
extract_gan_factors.py — Extract Non-Linear Latent Factors with Frozen PCA
Based on: Chen, Pelger, and Zhu (2024), "Deep Learning in Asset Pricing"

Audited & Corrected for:
  1. Strict Representation Isolation:
     - PCA centering and SVD projection basis fitted STRICTLY on representation activations (T_rep = 180).
     - Deterministic column sign orientation enforced.
     - Out-of-sample inference, test, and validation activations projected onto frozen basis.
  2. Strictly Lagged Information Timing:
     - Representation: t-1 state (0..179) -> return t (1..180) [180 months]
     - Inference: t-1 state (180..346) -> return t (181..347) [167 months]
     - Test: t-1 state begins with last training month (index 347) [100 months]
     - Validation: t-1 state begins with last test month [50 months]
  3. Import-safe, callable architecture.
"""

import os
import sys
from pathlib import Path

NUM_CORES = "4"
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["MKL_NUM_THREADS"] = NUM_CORES

import torch
import numpy as np
import pandas as pd

torch.set_num_threads(int(NUM_CORES))

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_splits"
MODELS_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(PROJECT_DIR))
from scripts.train_gan_sdf import SDFGenerator
from scripts.split_config import (
    load_raw_splits,
    get_representation_and_inference_splits,
    T_REP_MONTHS,
    T_INF_MONTHS,
    T_TEST_MONTHS,
    T_VAL_MONTHS
)


def extract_gan_factors(
    ckpt_path=None,
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    save_outputs=True,
    verbose=True
):
    """
    Extracts latent factors from trained GAN SDF using frozen representation PCA basis.
    Saves and returns split-specific factor files and performance diagnostics.
    """
    if ckpt_path is None:
        ckpt_path = models_dir / "gan_sdf_trained.pt"

    if verbose:
        print("=" * 72)
        print(" PAPER 3: EXTRACTING NON-LINEAR LATENT FACTORS (FROZEN PCA)")
        print(f" Loading weights from: {ckpt_path.name}")
        print("=" * 72 + "\n")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    z_mean = ckpt["z_mean"]
    z_std = ckpt["z_std"]
    rep_medians = ckpt.get("rep_medians", {})
    state_dict = ckpt["generator_state"]
    factor_cols = ckpt["factor_cols"]
    port_cols = ckpt["port_cols"]

    P_states = z_mean.shape[1]
    N_ports = state_dict["fc3.weight"].shape[0]

    generator = SDFGenerator(p_in=P_states, n_out=N_ports, hidden_dim=64)
    generator.load_state_dict(state_dict)
    generator.eval()

    # Load raw inputs
    (f_tr, f_te, f_va), (p_tr, p_te, p_va) = load_raw_splits(data_dir=data_dir)

    # Impute missing values across all splits strictly using frozen representation medians
    def clean_states(df):
        clean_df = df[factor_cols].copy()
        for col in factor_cols:
            med_val = rep_medians.get(col, 0.0)
            clean_df[col] = clean_df[col].fillna(med_val)
        return clean_df.values.astype(np.float32)

    Z_train_raw = clean_states(f_tr)
    Z_test_raw  = clean_states(f_te)
    Z_val_raw   = clean_states(f_va)

    R_train_raw = p_tr[port_cols].fillna(0.0).values.astype(np.float32)
    R_test_raw  = p_te[port_cols].fillna(0.0).values.astype(np.float32)
    R_val_raw   = p_va[port_cols].fillna(0.0).values.astype(np.float32)

    # --- STRICTLY LAGGED STATE AND RETURN PAIRING ---
    # 1. Representation Split (180 months)
    Z_rep_states = Z_train_raw[:T_REP_MONTHS]
    R_rep_eval   = R_train_raw[1:T_REP_MONTHS + 1]
    dates_rep    = f_tr["Date"].values[1:T_REP_MONTHS + 1]

    # 2. Inference Split (167 months)
    Z_inf_states = Z_train_raw[T_REP_MONTHS:-1]
    R_inf_eval   = R_train_raw[T_REP_MONTHS + 1:]
    dates_inf    = f_tr["Date"].values[T_REP_MONTHS + 1:]

    # 3. Full Training Reference (347 months)
    Z_tr_states  = Z_train_raw[:-1]
    R_tr_eval    = R_train_raw[1:]
    dates_tr     = f_tr["Date"].values[1:]

    # 4. Out-of-Sample Test Split (100 months)
    # Month 0 test return uses month 347 of training (last training month)
    Z_test_states = np.vstack([Z_train_raw[[-1]], Z_test_raw[:-1]])
    R_test_eval   = R_test_raw
    dates_test    = f_te["Date"].values

    # 5. Out-of-Sample Validation Split (50 months)
    # Month 0 val return uses month 99 of test (last test month)
    Z_val_states  = np.vstack([Z_test_raw[[-1]], Z_val_raw[:-1]])
    R_val_eval    = R_val_raw
    dates_val     = f_va["Date"].values

    # --- STAGE 1: COMPUTE & FREEZE PCA ON REPRESENTATION ACTIVATIONS ONLY ---
    if verbose:
        print("Computing latent activations strictly on representation sample (T_rep = 180)...")

    Z_rep_norm = np.clip((Z_rep_states - z_mean) / z_std, -5.0, 5.0)
    with torch.no_grad():
        _, h_rep_latent = generator(torch.tensor(Z_rep_norm, dtype=torch.float32))

    h_rep_np = h_rep_latent.numpy()
    pca_center = h_rep_np.mean(axis=0, keepdims=True)
    h_rep_centered = h_rep_np - pca_center

    U_rep, S_rep, Vt_rep = np.linalg.svd(h_rep_centered, full_matrices=False)
    pca_basis = Vt_rep[:3, :].T  # (hidden_dim // 2, 3)

    # Deterministic sign orientation: largest absolute element in each column is positive
    for j in range(3):
        max_idx = np.argmax(np.abs(pca_basis[:, j]))
        if pca_basis[max_idx, j] < 0:
            pca_basis[:, j] *= -1

    if verbose:
        print(f"  Frozen PCA Centering Mean Shape: {pca_center.shape}")
        print(f"  Frozen PCA Basis Matrix V Shape: {pca_basis.shape}")
        print(f"  Top 3 Singular Values: {S_rep[:3].round(4)}")

    if save_outputs:
        np.savez(
            models_dir / "frozen_pca_basis.npz",
            pca_center=pca_center,
            pca_basis=pca_basis,
            singular_values=S_rep[:3],
            t_rep_months=T_REP_MONTHS,
            rep_start_date=int(dates_rep[0]),
            rep_end_date=int(dates_rep[-1])
        )

    # --- STAGE 2: PROCESS ALL SPLITS WITH FROZEN PCA ---
    splits_dict = {
        "rep_180": (Z_rep_states, R_rep_eval, dates_rep),
        "inf_167": (Z_inf_states, R_inf_eval, dates_inf),
        "train_70": (Z_tr_states, R_tr_eval, dates_tr),
        "test_20": (Z_test_states, R_test_eval, dates_test),
        "val_10": (Z_val_states, R_val_eval, dates_val)
    }

    extracted_summaries = {}
    extracted_dfs = {}

    with torch.no_grad():
        for sp_name, (Z_s, R_e, d_e) in splits_dict.items():
            Z_norm = np.clip((Z_s - z_mean) / z_std, -5.0, 5.0)
            Z_t = torch.tensor(Z_norm, dtype=torch.float32)
            R_t = torch.tensor(R_e, dtype=torch.float32)

            w_t, h_latent = generator(Z_t)
            M_t = 1.0 - torch.sum(w_t * R_t, dim=1, keepdim=True)
            R_sdf = (1.0 - M_t).numpy().flatten()

            h_np = h_latent.numpy()
            h_proj = (h_np - pca_center) @ pca_basis

            out_df = pd.DataFrame({
                "Date": d_e,
                "SDF_Return": R_sdf,
                "GAN_F1": h_proj[:, 0],
                "GAN_F2": h_proj[:, 1],
                "GAN_F3": h_proj[:, 2]
            })

            extracted_dfs[sp_name] = out_df

            if save_outputs:
                out_path = data_dir / f"gan_factors_{sp_name}.csv"
                out_df.to_csv(out_path, index=False)

            pe = (M_t.numpy() * R_e)
            sdf_rmse = float(np.sqrt(np.mean(np.mean(pe, axis=0) ** 2)))
            bm_rmse = float(np.sqrt(np.mean(np.mean(R_e, axis=0) ** 2)))

            gross_exp = float(np.mean(np.sum(np.abs(w_t.numpy()), axis=1)))
            neg_m = float(np.mean(M_t.numpy().flatten() < 0) * 100)

            ann_sr = float((np.mean(R_sdf) / (np.std(R_sdf) + 1e-8)) * np.sqrt(12))
            ann_ret = float(np.mean(R_sdf) * 12 * 100)
            ann_vol = float(np.std(R_sdf) * np.sqrt(12) * 100)

            extracted_summaries[sp_name] = {
                "Months": len(R_sdf),
                "Start_Date": int(d_e[0]),
                "End_Date": int(d_e[-1]),
                "Annual_Return_%": round(ann_ret, 2),
                "Annual_Vol_%": round(ann_vol, 2),
                "Sharpe_Ratio": round(ann_sr, 2),
                "Pricing_RMSE": round(sdf_rmse, 6),
                "Benchmark_RMSE": round(bm_rmse, 6),
                "Gross_Exposure": round(gross_exp, 2),
                "Negative_M_%": round(neg_m, 1)
            }

    if verbose:
        print("\n" + "=" * 72)
        print(" PERFORMANCE SUMMARY ACROSS CHRONOLOGICAL SPLITS")
        print("=" * 72)
        sum_df = pd.DataFrame(extracted_summaries).T
        print(sum_df.to_string())
        print("=" * 72 + "\n")

    return {
        "summaries": extracted_summaries,
        "dfs": extracted_dfs,
        "pca_center": pca_center,
        "pca_basis": pca_basis
    }


def main():
    extract_gan_factors(save_outputs=True, verbose=True)


if __name__ == "__main__":
    main()
