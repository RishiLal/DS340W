"""
Forward inner validation tuning over 3 expanding folds:
Minimum (M) and Goldilocks (G) stability selection rules.
"""
import numpy as np
import pandas as pd
from extension_20261108.data import compute_volatility_state
from extension_20261108.clustering import fit_theme_representation
from extension_20261108.models import solve_ridge_sdf, evaluate_sdf_moments

def get_penalty_grid():
    """31 log-spaced penalties from 10^-8 to 10^1."""
    return np.logspace(-8, 1, 31)

def run_goldilocks_selection(rmse_curve, window_size=3):
    """
    Goldilocks stability selection:
    For adjacent windows of size S=3 (j = 0, ..., num_penalties - S):
        m_j = mean RMSE within window
        V_j = population variance of RMSE within window (ddof=0)
        min-max normalize m_j and V_j across all windows
        Score S_j = norm_m_j + norm_V_j
        Lowest-score window j* (ties resolve to first index)
        Within window j*, select penalty with lowest RMSE (ties resolve to first index)
    """
    K = len(rmse_curve)
    num_windows = K - window_size + 1
    m = np.zeros(num_windows)
    V = np.zeros(num_windows)
    
    for j in range(num_windows):
        window = rmse_curve[j : j + window_size]
        m[j] = np.mean(window)
        V[j] = np.var(window, ddof=0)
        
    # Min-max normalization
    m_range = np.max(m) - np.min(m)
    norm_m = (m - np.min(m)) / m_range if m_range > 0 else np.zeros(num_windows)
    
    V_range = np.max(V) - np.min(V)
    norm_V = (V - np.min(V)) / V_range if V_range > 0 else np.zeros(num_windows)
    
    scores = norm_m + norm_V
    
    # Lowest score window (ties choose first index)
    best_window_idx = int(np.argmin(scores))
    
    # Within chosen window, lowest RMSE (ties choose first index)
    window_rmses = rmse_curve[best_window_idx : best_window_idx + window_size]
    within_window_best = int(np.argmin(window_rmses))
    best_penalty_idx = best_window_idx + within_window_best
    
    return {
        "best_penalty_idx": best_penalty_idx,
        "best_window_idx": best_window_idx,
        "window_scores": scores.tolist(),
        "window_means": m.tolist(),
        "window_vars": V.tolist(),
        "norm_means": norm_m.tolist(),
        "norm_vars": norm_V.tolist()
    }

def run_forward_inner_tuning(factors_panel, market_series, full_market, train_dates):
    """
    Perform 3-fold forward expanding inner validation on Training months (Jan 1975 to Dec 2009).
    Folds:
        Fold 1: Train 210 months, Validate next 70 months
        Fold 2: Train 280 months, Validate next 70 months
        Fold 3: Train 350 months, Validate next 70 months
    Returns:
        tuning_results: dict containing full curves, selected penalties, and fold diagnostics
    """
    assert len(train_dates) == 420
    lambdas = get_penalty_grid()
    
    fold_configs = [
        {"fold": 1, "n_train": 210, "n_val": 70},
        {"fold": 2, "n_train": 280, "n_val": 70},
        {"fold": 3, "n_train": 350, "n_val": 70},
    ]
    
    # Storage for squared error curves: (3 folds x 31 lambdas)
    mse_curves = {
        "unconditional": np.zeros((3, len(lambdas))),
        "conditional": np.zeros((3, len(lambdas)))
    }
    
    fold_diagnostics = []
    
    for fold_idx, cfg in enumerate(fold_configs):
        n_tr = cfg["n_train"]
        n_val = cfg["n_val"]
        fold_train_dates = train_dates[:n_tr]
        fold_val_dates = train_dates[n_tr : n_tr + n_val]
        fold_all_dates = fold_train_dates + fold_val_dates
        
        # 1. State h_{t-1} fitted on fold_train_dates
        h_series, _, h_meta = compute_volatility_state(full_market, fold_all_dates, fold_train_dates)
        h_tr = h_series.loc[fold_train_dates].values
        h_val = h_series.loc[fold_val_dates].values
        
        # 2. Theme representation fitted on fold_train_dates
        tr_f = factors_panel.loc[fold_train_dates]
        tr_m = market_series.loc[fold_train_dates]
        val_f = factors_panel.loc[fold_val_dates]
        val_m = market_series.loc[fold_val_dates]
        all_f = factors_panel.loc[fold_all_dates]
        all_m = market_series.loc[fold_all_dates]
        
        F_all, X_all, scales, cluster_labels = fit_theme_representation(tr_f, tr_m, all_f, all_m, k=13)
        
        F_tr = F_all[:n_tr]
        X_tr = X_all[:n_tr]
        F_val = F_all[n_tr:]
        X_val = X_all[n_tr:]
        
        # Fitting dictionaries
        # Unconditional: Q = X (d=154)
        Q_tr_unc = X_tr
        # Conditional: Q = [X, h * X] (d=308)
        Q_tr_cond = np.hstack([X_tr, h_tr[:, np.newaxis] * X_tr])
        
        # Validation dictionary: SHARED full conditional dictionary [X_val, h_val * X_val] (d=308)
        Q_val_shared = np.hstack([X_val, h_val[:, np.newaxis] * X_val])
        assert Q_val_shared.shape == (n_val, 308)
        
        fold_diag = {
            "fold": cfg["fold"],
            "train_months": n_tr,
            "val_months": n_val,
            "train_dates_range": [fold_train_dates[0], fold_train_dates[-1]],
            "val_dates_range": [fold_val_dates[0], fold_val_dates[-1]],
            "volatility_c": h_meta["c_threshold"],
            "volatility_p_H": h_meta["p_H_train_frequency"],
            "val_high_vol_count": int((h_val > 0).sum())
        }
        fold_diagnostics.append(fold_diag)
        
        # Evaluate grid
        for l_idx, lam in enumerate(lambdas):
            # Unconditional model
            b_unc = solve_ridge_sdf(Q_tr_unc, F_tr, lam, N=154)
            ev_unc = evaluate_sdf_moments(b_unc, F_val, Q_val_shared)
            mse_curves["unconditional"][fold_idx, l_idx] = ev_unc["loss"]
            
            # Conditional model
            b_cond = solve_ridge_sdf(Q_tr_cond, F_tr, lam, N=154)
            ev_cond = evaluate_sdf_moments(b_cond, F_val, Q_val_shared)
            mse_curves["conditional"][fold_idx, l_idx] = ev_cond["loss"]
            
    # Pooled MSE: average equally over the 3 folds
    pooled_mse_unc = mse_curves["unconditional"].mean(axis=0)
    pooled_rmse_unc = np.sqrt(pooled_mse_unc)
    
    pooled_mse_cond = mse_curves["conditional"].mean(axis=0)
    pooled_rmse_cond = np.sqrt(pooled_mse_cond)
    
    # Minimum selection (smallest index on tie)
    min_idx_unc = int(np.argmin(pooled_rmse_unc))
    min_idx_cond = int(np.argmin(pooled_rmse_cond))
    
    # Goldilocks selection
    gold_res_unc = run_goldilocks_selection(pooled_rmse_unc, window_size=3)
    gold_res_cond = run_goldilocks_selection(pooled_rmse_cond, window_size=3)
    
    selection_summary = {
        "unconditional_minimum": {
            "lambda_idx": min_idx_unc,
            "lambda_val": float(lambdas[min_idx_unc]),
            "rmse": float(pooled_rmse_unc[min_idx_unc]),
            "loss": float(pooled_mse_unc[min_idx_unc])
        },
        "unconditional_goldilocks": {
            "lambda_idx": gold_res_unc["best_penalty_idx"],
            "lambda_val": float(lambdas[gold_res_unc["best_penalty_idx"]]),
            "rmse": float(pooled_rmse_unc[gold_res_unc["best_penalty_idx"]]),
            "loss": float(pooled_mse_unc[gold_res_unc["best_penalty_idx"]]),
            "goldilocks_details": gold_res_unc
        },
        "conditional_minimum": {
            "lambda_idx": min_idx_cond,
            "lambda_val": float(lambdas[min_idx_cond]),
            "rmse": float(pooled_rmse_cond[min_idx_cond]),
            "loss": float(pooled_mse_cond[min_idx_cond])
        },
        "conditional_goldilocks": {
            "lambda_idx": gold_res_cond["best_penalty_idx"],
            "lambda_val": float(lambdas[gold_res_cond["best_penalty_idx"]]),
            "rmse": float(pooled_rmse_cond[gold_res_cond["best_penalty_idx"]]),
            "loss": float(pooled_mse_cond[gold_res_cond["best_penalty_idx"]]),
            "goldilocks_details": gold_res_cond
        }
    }
    
    curves_data = {
        "lambdas": lambdas.tolist(),
        "unconditional_rmse": pooled_rmse_unc.tolist(),
        "conditional_rmse": pooled_rmse_cond.tolist(),
        "unconditional_fold_mse": mse_curves["unconditional"].tolist(),
        "conditional_fold_mse": mse_curves["conditional"].tolist()
    }
    
    return {
        "selection_summary": selection_summary,
        "curves_data": curves_data,
        "fold_diagnostics": fold_diagnostics
    }

def fit_and_freeze_full_models(factors_panel, market_series, full_market, train_dates, selection_summary):
    """
    Refit the four prespecified models on the full 420 Training months:
        1. (U, M): Unconditional with Minimum penalty
        2. (U, G): Unconditional with Goldilocks penalty
        3. (C, M): Conditional with Minimum penalty
        4. (C, G): Conditional with Goldilocks penalty
    Returns:
        frozen_models: dict of fitted coefficients and model metadata
        training_transforms: full training scales, clusters, and volatility state metadata
    """
    assert len(train_dates) == 420
    all_dates = factors_panel.index.tolist()
    
    # 1. Full training volatility state
    h_series, v_series, h_meta = compute_volatility_state(full_market, all_dates, train_dates)
    h_all = h_series.values
    h_train = h_series.loc[train_dates].values
    
    # 2. Full training theme representation
    tr_f = factors_panel.loc[train_dates]
    tr_m = market_series.loc[train_dates]
    
    F_all, X_all, scales, cluster_labels = fit_theme_representation(tr_f, tr_m, factors_panel, market_series, k=13)
    
    F_train = F_all[:420]
    X_train = X_all[:420]
    
    # Training dictionaries
    Q_train_unc = X_train
    Q_train_cond = np.hstack([X_train, h_train[:, np.newaxis] * X_train])
    
    # Selected lambdas
    lam_u_m = selection_summary["unconditional_minimum"]["lambda_val"]
    lam_u_g = selection_summary["unconditional_goldilocks"]["lambda_val"]
    lam_c_m = selection_summary["conditional_minimum"]["lambda_val"]
    lam_c_g = selection_summary["conditional_goldilocks"]["lambda_val"]
    
    # Fit coefficients
    b_u_m = solve_ridge_sdf(Q_train_unc, F_train, lam_u_m, N=154)
    b_u_g = solve_ridge_sdf(Q_train_unc, F_train, lam_u_g, N=154)
    b_c_m = solve_ridge_sdf(Q_train_cond, F_train, lam_c_m, N=154)
    b_c_g = solve_ridge_sdf(Q_train_cond, F_train, lam_c_g, N=154)
    
    frozen_models = {
        "U_M": {
            "name": "Unconditional_Minimum",
            "dictionary": "unconditional",
            "tuning": "minimum",
            "lambda": lam_u_m,
            "b": b_u_m.tolist()
        },
        "U_G": {
            "name": "Unconditional_Goldilocks",
            "dictionary": "unconditional",
            "tuning": "goldilocks",
            "lambda": lam_u_g,
            "b": b_u_g.tolist()
        },
        "C_M": {
            "name": "Conditional_Minimum",
            "dictionary": "conditional",
            "tuning": "minimum",
            "lambda": lam_c_m,
            "b": b_c_m.tolist()
        },
        "C_G": {
            "name": "Conditional_Goldilocks",
            "dictionary": "conditional",
            "tuning": "goldilocks",
            "lambda": lam_c_g,
            "b": b_c_g.tolist()
        }
    }
    
    training_transforms = {
        "factor_scales": scales["factor_scales"].tolist(),
        "market_scale": float(scales["market_scale"]),
        "factor_names": scales["factor_names"],
        "cluster_labels": cluster_labels.tolist(),
        "volatility_metadata": h_meta,
        "F_all": F_all,
        "X_all": X_all,
        "h_all": h_all,
        "all_dates": all_dates
    }
    
    return frozen_models, training_transforms
