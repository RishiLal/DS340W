"""
Evaluation of the four frozen SDF models on Test and Validation partitions,
along with paired moving-block bootstrap uncertainty.
"""
import numpy as np
from extension_20261108.models import evaluate_sdf_moments, compute_unconditional_basis_points

def evaluate_period_models(frozen_models, F_period, Q_period, factor_scales, market_scale):
    """
    Evaluate all 4 models on a given period (Test or Validation) using the shared 308-moment dictionary.
    Returns:
        metrics: dict containing Loss, RMSE, MAE bps, RMSE bps for all 4 models,
                 as well as paired main contrast Delta and interaction I.
        MQ_dict: dict of (T x 308) discounted payoff matrices for each model (used for bootstrap)
    """
    T, d = Q_period.shape
    assert d == 308, f"Expected 308 evaluation moments, got {d}"
    assert F_period.shape[0] == T
    
    results = {}
    MQ_dict = {}
    
    for key, model_info in frozen_models.items():
        b = np.array(model_info["b"])
        ev = evaluate_sdf_moments(b, F_period, Q_period)
        bps = compute_unconditional_basis_points(ev["e_bar"][:154], factor_scales, market_scale)
        
        results[key] = {
            "name": model_info["name"],
            "lambda": model_info["lambda"],
            "loss": ev["loss"],
            "rmse": ev["rmse"],
            "mae_bps": bps["mae_bps"],
            "rmse_bps": bps["rmse_bps"],
            "e_bar": ev["e_bar"].tolist()
        }
        MQ_dict[key] = ev["MQ"]
        
    # Contrasts:
    # Main contrast Delta = L(C, G) - L(U, G)
    delta_goldilocks = results["C_G"]["loss"] - results["U_G"]["loss"]
    # Minimum contrast Delta_M = L(C, M) - L(U, M)
    delta_minimum = results["C_M"]["loss"] - results["U_M"]["loss"]
    # Interaction I = Delta_goldilocks - Delta_minimum
    interaction = delta_goldilocks - delta_minimum
    
    summary = {
        "models": results,
        "delta_goldilocks": float(delta_goldilocks),
        "delta_minimum": float(delta_minimum),
        "interaction": float(interaction),
        "num_months": T,
        "num_moments": d
    }
    
    return summary, MQ_dict

def run_paired_moving_block_bootstrap(MQ_dict, block_length=12, num_resamples=999, seed=340):
    """
    Paired moving-block bootstrap:
    Resamples complete aligned monthly rows across all 4 models simultaneously.
    For each resample draw, recomputes the period-mean moment vector, loss, Delta, and I.
    """
    rng = np.random.default_rng(seed)
    
    # All models have same T rows
    T = next(iter(MQ_dict.values())).shape[0]
    num_blocks_needed = int(np.ceil(T / block_length))
    max_start_idx = T - block_length + 1
    assert max_start_idx > 0, f"Block length {block_length} exceeds sample length {T}"
    
    # Pre-allocate arrays for bootstrap statistics
    boot_stats = {
        "L_U_M": np.zeros(num_resamples),
        "L_U_G": np.zeros(num_resamples),
        "L_C_M": np.zeros(num_resamples),
        "L_C_G": np.zeros(num_resamples),
        "delta_goldilocks": np.zeros(num_resamples),
        "delta_minimum": np.zeros(num_resamples),
        "interaction": np.zeros(num_resamples)
    }
    
    for b_idx in range(num_resamples):
        # Pick block starting positions
        start_indices = rng.integers(0, max_start_idx, size=num_blocks_needed)
        # Construct sample row indices
        sample_indices = []
        for s in start_indices:
            sample_indices.extend(range(s, s + block_length))
        sample_indices = np.array(sample_indices[:T])
        
        # Compute mean moment and loss for each model
        draw_losses = {}
        for key in ["U_M", "U_G", "C_M", "C_G"]:
            resampled_MQ = MQ_dict[key][sample_indices, :]
            resampled_e_bar = resampled_MQ.mean(axis=0)
            resampled_loss = float(np.mean(resampled_e_bar ** 2))
            draw_losses[key] = resampled_loss
            
        boot_stats["L_U_M"][b_idx] = draw_losses["U_M"]
        boot_stats["L_U_G"][b_idx] = draw_losses["U_G"]
        boot_stats["L_C_M"][b_idx] = draw_losses["C_M"]
        boot_stats["L_C_G"][b_idx] = draw_losses["C_G"]
        
        d_g = draw_losses["C_G"] - draw_losses["U_G"]
        d_m = draw_losses["C_M"] - draw_losses["U_M"]
        boot_stats["delta_goldilocks"][b_idx] = d_g
        boot_stats["delta_minimum"][b_idx] = d_m
        boot_stats["interaction"][b_idx] = d_g - d_m
        
    # Compute 95% bootstrap intervals
    ci_results = {}
    for stat_name, arr in boot_stats.items():
        ci_results[stat_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "p025": float(np.quantile(arr, 0.025)),
            "p50": float(np.quantile(arr, 0.50)),
            "p975": float(np.quantile(arr, 0.975))
        }
        
    return {
        "block_length": block_length,
        "num_resamples": num_resamples,
        "seed": seed,
        "ci_intervals": ci_results
    }
