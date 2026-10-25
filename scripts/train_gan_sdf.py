"""
train_gan_sdf.py — Deep Learning in Asset Pricing (Paper 3 Implementation)
Based on: Chen, Pelger, and Zhu (2024), "Deep Learning in Asset Pricing"

Audited & Corrected for:
  1. Strict Representation Isolation:
     - Imputation parameters fitted strictly on representation-training states (T_rep = 180).
     - State normalization (z_mean, z_std) fitted strictly on representation-training states.
     - GAN trained exclusively on designated representation-training pairs (1976-08 to 1991-07).
  2. Strictly Lagged Information Timing:
     - w_{t-1} = G(z_{t-1}) prices return R_t.
     - Zero look-ahead bias across all intervals.
  3. Import-safe, deterministic callable architecture.
"""

import os
import sys
from pathlib import Path

# Thread limits to prevent CPU starvation
NUM_CORES = "4"
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["MKL_NUM_THREADS"] = NUM_CORES
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CORES
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_CORES
os.environ["NUMEXPR_NUM_THREADS"] = NUM_CORES

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

torch.set_num_threads(int(NUM_CORES))
torch.set_num_interop_threads(int(NUM_CORES))

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_splits"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

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


class SDFGenerator(nn.Module):
    """
    SDF Generator:
    Input: Lagged state vector z_{t-1} in R^P.
    Output: Portfolio weights w_{t-1} in R^N.
    Candidate SDF: M_t = 1 - w_{t-1}^T R_t.
    """
    def __init__(self, p_in, n_out, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(p_in, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.act2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(hidden_dim // 2, n_out)
        
        with torch.no_grad():
            self.fc3.weight.data.mul_(0.001)
            self.fc3.bias.data.zero_()

    def forward(self, z):
        h1 = self.act1(self.norm1(self.fc1(z)))
        h2 = self.act2(self.norm2(self.fc2(h1)))
        raw_w = self.fc3(h2)
        w = torch.tanh(raw_w) * 0.1  # Constrain coordinatewise weights
        return w, h2


class MomentDiscriminator(nn.Module):
    """
    Moment Discriminator:
    Input: Lagged state vector z_{t-1} in R^P.
    Output: Test asset instruments h_phi(z_{t-1}) in R^{N x n_moments}.
    Conditioning restriction: E[ M_t * R_t * h_phi(z_{t-1}) ] = 0.
    """
    def __init__(self, p_in, n_assets, n_moments=6, hidden_dim=32):
        super().__init__()
        self.n_moments = n_moments
        self.n_assets = n_assets
        self.net = nn.Sequential(
            nn.Linear(p_in, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, n_assets * n_moments)
        )

    def forward(self, z):
        T = z.shape[0]
        out = self.net(z)
        out = out.view(T, self.n_assets, self.n_moments)
        return torch.tanh(out)


def train_gan_sdf(
    epochs=200,
    seed=42,
    lr_g=5e-4,
    lr_d=1e-3,
    save_model=True,
    verbose=True,
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    custom_splits=None
):
    """
    Trains the Paper 3 Minimax GAN SDF strictly on designated representation-learning pairs.
    Returns the trained generator, discriminator, normalization parameters, and training diagnostics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if custom_splits is not None:
        splits = custom_splits
    else:
        (f_tr, f_te, f_va), (p_tr, p_te, p_va) = load_raw_splits(data_dir=data_dir)
        splits = get_representation_and_inference_splits(f_tr, p_tr)

    factor_cols = splits["factor_cols"]
    port_cols = splits["port_cols"]

    # 1. FIT PREPROCESSING STRICTLY ON REPRESENTATION STATES (T_rep = 180)
    rep_states_raw = splits["rep"]["states_raw"]
    rep_returns_raw = splits["rep"]["returns_raw"]

    rep_medians = {}
    for col in factor_cols:
        col_med = rep_states_raw[col].median()
        rep_medians[col] = float(col_med) if not np.isnan(col_med) else 0.0

    # Impute representation states using representation medians
    Z_rep_clean = rep_states_raw.copy()
    for col in factor_cols:
        Z_rep_clean[col] = Z_rep_clean[col].fillna(rep_medians[col])

    Z_rep_arr = Z_rep_clean.values.astype(np.float32)
    R_rep_arr = rep_returns_raw.fillna(0.0).values.astype(np.float32)

    # State Normalization fitted STRICTLY on representation states
    z_mean = np.nanmean(Z_rep_arr, axis=0, keepdims=True)
    z_std = np.nanstd(Z_rep_arr, axis=0, keepdims=True) + 1e-5
    Z_rep_norm = np.clip((Z_rep_arr - z_mean) / z_std, -5.0, 5.0)

    T_rep = Z_rep_norm.shape[0]
    P_states = Z_rep_norm.shape[1]
    N_ports = R_rep_arr.shape[1]

    if verbose:
        print("=" * 72)
        print(" PAPER 3: DEEP LEARNING SDF TRAINING (MINIMAX GAN)")
        print(" Isolation: Fitted STRICTLY on Representation Sample (T_rep = 180 months)")
        print(f" State Period:  {splits['rep']['state_dates'][0]} to {splits['rep']['state_dates'][-1]}")
        print(f" Return Period: {splits['rep']['return_dates'][0]} to {splits['rep']['return_dates'][-1]}")
        print(f" Test Assets (N): {N_ports} portfolios | Factors (P): {P_states}")
        print(f" CPU Thread Lock: {torch.get_num_threads()} cores active")
        print("=" * 72 + "\n")

    R_tensor = torch.tensor(R_rep_arr, dtype=torch.float32)
    Z_tensor = torch.tensor(Z_rep_norm, dtype=torch.float32)

    bench_rmse = float(np.sqrt(np.mean(np.mean(R_rep_arr, axis=0) ** 2)))
    if verbose:
        print(f"  Benchmark Pricing Error (M = 1): RMSE = {bench_rmse:.6f}\n")

    # 2. ARCHITECTURES & OPTIMIZERS
    generator = SDFGenerator(p_in=P_states, n_out=N_ports, hidden_dim=64)
    discriminator = MomentDiscriminator(p_in=P_states, n_assets=N_ports, n_moments=6, hidden_dim=32)

    opt_G = optim.Adam(generator.parameters(), lr=lr_g, weight_decay=1e-5)
    opt_D = optim.Adam(discriminator.parameters(), lr=lr_d, weight_decay=1e-5)

    generator.train()
    discriminator.train()

    loss_history = []
    rmse_history = []
    exposure_history = []
    neg_m_history = []

    # 3. MINIMAX ADVERSARIAL OPTIMIZATION
    for epoch in range(1, epochs + 1):
        # Step A: Update Discriminator (Maximize pricing errors)
        opt_D.zero_grad()
        with torch.no_grad():
            w_t, _ = generator(Z_tensor)
            M_t = 1.0 - torch.sum(w_t * R_tensor, dim=1, keepdim=True)
            pricing_errors = M_t * R_tensor

        H_t = discriminator(Z_tensor)
        contracted = torch.einsum("tn,tnm->tm", pricing_errors, H_t)
        moments = torch.mean(contracted, dim=0)
        loss_D = -torch.sum(moments ** 2)

        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        opt_D.step()

        # Step B: Update Generator (Minimize pricing errors)
        opt_G.zero_grad()
        w_t, _ = generator(Z_tensor)
        M_t = 1.0 - torch.sum(w_t * R_tensor, dim=1, keepdim=True)
        pricing_errors = M_t * R_tensor

        with torch.no_grad():
            H_t = discriminator(Z_tensor)

        contracted = torch.einsum("tn,tnm->tm", pricing_errors, H_t)
        moments = torch.mean(contracted, dim=0)

        reg_weights = 1e-4 * torch.mean(w_t ** 2)
        loss_G = torch.sum(moments ** 2) + reg_weights

        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        opt_G.step()

        loss_val = float(loss_G.item())
        loss_history.append(loss_val)

        with torch.no_grad():
            pe_np = pricing_errors.detach().cpu().numpy()
            cur_rmse = float(np.sqrt(np.mean(np.mean(pe_np, axis=0) ** 2)))
            rmse_history.append(cur_rmse)

            gross_exp = float(torch.mean(torch.sum(torch.abs(w_t), dim=1)).item())
            exposure_history.append(gross_exp)

            m_np = M_t.detach().cpu().numpy().flatten()
            neg_pct = float(np.mean(m_np < 0) * 100)
            neg_m_history.append(neg_pct)

        if verbose and (epoch % 40 == 0 or epoch == 1):
            R_sdf = 1.0 - m_np
            sr = float(np.mean(R_sdf) / (np.std(R_sdf) + 1e-8) * np.sqrt(12))
            print(f"  Epoch [{epoch:3d}/{epochs}] | Minimax Loss: {loss_val:.6f} | "
                  f"Pricing RMSE: {cur_rmse:.6f} | Gross Exp: {gross_exp:.2f} | "
                  f"M<0: {neg_pct:.1f}% | SDF Sharpe: {sr:.2f}")

    checkpoint_data = {
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "z_mean": z_mean,
        "z_std": z_std,
        "rep_medians": rep_medians,
        "factor_cols": factor_cols,
        "port_cols": port_cols,
        "epochs": epochs,
        "seed": seed,
        "timing": "strictly_lagged_t_minus_1",
        "split_config": {
            "T_rep_months": T_REP_MONTHS,
            "T_inf_months": T_INF_MONTHS,
            "rep_return_start": int(splits["rep"]["return_dates"][0]),
            "rep_return_end": int(splits["rep"]["return_dates"][-1]),
            "inf_return_start": int(splits["inf"]["return_dates"][0]),
            "inf_return_end": int(splits["inf"]["return_dates"][-1]),
            "rep_state_start": int(splits["rep"]["state_dates"][0]),
            "rep_state_end": int(splits["rep"]["state_dates"][-1]),
            "inf_state_start": int(splits["inf"]["state_dates"][0]),
            "inf_state_end": int(splits["inf"]["state_dates"][-1]),
        },
        "loss_history": loss_history,
        "rmse_history": rmse_history,
        "exposure_history": exposure_history,
        "neg_m_history": neg_m_history,
        "final_train_rmse": rmse_history[-1],
        "final_gross_exposure": exposure_history[-1],
        "final_neg_m_pct": neg_m_history[-1]
    }

    if save_model:
        models_dir.mkdir(exist_ok=True)
        ckpt_path = models_dir / "gan_sdf_trained.pt"
        torch.save(checkpoint_data, ckpt_path)
        if verbose:
            print(f"\n[OK] Model checkpoint saved to: {ckpt_path}")

    return {
        "generator": generator,
        "discriminator": discriminator,
        "z_mean": z_mean,
        "z_std": z_std,
        "rep_medians": rep_medians,
        "checkpoint_data": checkpoint_data
    }


def main():
    train_gan_sdf(epochs=200, seed=42, save_model=True, verbose=True)


if __name__ == "__main__":
    main()
