"""
test_pca_isolation.py — Targeted Tests for PCA Isolation on Representation Activations
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
from scripts.split_config import T_REP_MONTHS


def test_frozen_pca_invariance():
    """Verify that PCA loadings depend strictly on representation activations (T_rep = 180)."""
    frozen_pca = np.load(MODELS_DIR / "frozen_pca_basis.npz")
    pca_center = frozen_pca["pca_center"]
    pca_basis = frozen_pca["pca_basis"]
    
    ckpt = torch.load(MODELS_DIR / "gan_sdf_trained.pt", map_location="cpu", weights_only=False)
    z_mean = ckpt["z_mean"]
    z_std = ckpt["z_std"]
    rep_medians = ckpt.get("rep_medians", {})
    state_dict = ckpt["generator_state"]
    factor_cols = ckpt["factor_cols"]
    
    gen = SDFGenerator(p_in=z_mean.shape[1], n_out=state_dict["fc3.weight"].shape[0], hidden_dim=64)
    gen.load_state_dict(state_dict)
    gen.eval()
    
    # Load representation activations (T_rep = 180 months)
    f_tr = pd.read_csv(DATA_DIR / "factors_train_70.csv")
    clean_df = f_tr[factor_cols].copy()
    for col in factor_cols:
        clean_df[col] = clean_df[col].fillna(rep_medians.get(col, 0.0))
        
    Z_rep_states = clean_df.values.astype(np.float32)[:T_REP_MONTHS]
    Z_rep_norm = np.clip((Z_rep_states - z_mean) / z_std, -5.0, 5.0)
    
    with torch.no_grad():
        _, h_rep = gen(torch.tensor(Z_rep_norm))
    
    h_np = h_rep.numpy()
    computed_center = h_np.mean(axis=0, keepdims=True)
    assert np.allclose(pca_center, computed_center, atol=1e-5), "PCA center does not match representation activations"
    
    # Check that loadings matrix is orthonormal
    ortho_check = pca_basis.T @ pca_basis
    assert np.allclose(ortho_check, np.eye(3), atol=1e-4), "PCA basis columns are not orthonormal"
    
    # Check deterministic sign orientation (largest element in each column is positive)
    for j in range(3):
        max_idx = np.argmax(np.abs(pca_basis[:, j]))
        assert pca_basis[max_idx, j] > 0, f"PCA column {j} does not satisfy positive dominant sign convention"
        
    print("[PASS] Frozen PCA basis strictly matches representation activations and maintains deterministic sign convention.")


if __name__ == "__main__":
    test_frozen_pca_invariance()
    print("ALL PCA ISOLATION TESTS PASSED SUCCESSFULLY.")
