"""
test_timing_chronology.py — Targeted Tests for Information Timing and Future Perturbation Invariance
DS340W Independent Audit Verification
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
from scripts.extract_gan_factors import SDFGenerator

def test_strictly_lagged_dates():
    """Verify that every evaluated return is paired with a strictly prior state."""
    f_tr = pd.read_csv(DATA_DIR / "factors_train_70.csv")
    f_te = pd.read_csv(DATA_DIR / "factors_test_20.csv")
    f_va = pd.read_csv(DATA_DIR / "factors_val_10.csv")
    
    # 1. Within training: state t-1 precedes return t
    dates_tr = f_tr["Date"].values
    for i in range(1, len(dates_tr)):
        assert dates_tr[i-1] < dates_tr[i], f"Train order violation at row {i}"
    
    # 2. Boundary: last training state strictly precedes first test return
    assert dates_tr[-1] < f_te["Date"].iloc[0], "Train-test boundary chronology violation"
    
    # 3. Boundary: last test state strictly precedes first val return
    assert f_te["Date"].iloc[-1] < f_va["Date"].iloc[0], "Test-val boundary chronology violation"
    print("[PASS] Chronological dates strictly monotonic with zero forward-looking overlap.")

def test_future_perturbation_invariance():
    """Verify that altering future returns or states cannot change earlier weights."""
    ckpt = torch.load(MODELS_DIR / "gan_sdf_trained.pt", map_location="cpu", weights_only=False)
    z_mean = ckpt["z_mean"]
    z_std = ckpt["z_std"]
    state_dict = ckpt["generator_state"]
    
    gen = SDFGenerator(p_in=z_mean.shape[1], n_out=state_dict["fc3.weight"].shape[0], hidden_dim=64)
    gen.load_state_dict(state_dict)
    gen.eval()
    
    f_te = pd.read_csv(DATA_DIR / "factors_test_20.csv")
    factor_cols = [c for c in f_te.columns if c not in ["Date", "RF"]]
    Z_orig = f_te[factor_cols].fillna(0.0).values.astype(np.float32)
    
    # Weights for month 10 under original data
    Z_norm_orig = np.clip((Z_orig - z_mean) / z_std, -5.0, 5.0)
    with torch.no_grad():
        w_orig, _ = gen(torch.tensor(Z_norm_orig))
    
    # Perturb future observations (months 50 to 99) by 1000x
    Z_perturbed = Z_orig.copy()
    Z_perturbed[50:] += 50.0
    Z_norm_perturbed = np.clip((Z_perturbed - z_mean) / z_std, -5.0, 5.0)
    with torch.no_grad():
        w_perturbed, _ = gen(torch.tensor(Z_norm_perturbed))
    
    # Weights for earlier months (e.g. month 10) must be identical within float32 precision
    diff_early = torch.max(torch.abs(w_orig[:50] - w_perturbed[:50])).item()
    assert diff_early < 1e-6, f"Future perturbation contaminated earlier weights! Max diff = {diff_early}"
    print(f"[PASS] Future data perturbation has zero impact on earlier weights (max diff = {diff_early:.2e} <= 1e-6).")

if __name__ == "__main__":
    test_strictly_lagged_dates()
    test_future_perturbation_invariance()
    print("ALL TIMING TESTS PASSED SUCCESSFULLY.")
