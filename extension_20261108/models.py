"""
SDF model fitting, objective solving, and evaluation moment functions.
"""
import numpy as np

def compute_fitting_moments(Q, F):
    """
    Compute mean vector mu in R^d and cross-moment matrix A in R^(d x p).
    Q: (n x d) dictionary of test payoffs
    F: (n x p) theme payoffs
    """
    n = Q.shape[0]
    assert F.shape[0] == n, f"Length mismatch: Q has {n} rows, F has {F.shape[0]}"
    mu = Q.mean(axis=0)  # (d,)
    A = (Q.T @ F) / n    # (d x p)
    return mu, A

def solve_ridge_sdf(Q, F, lambda_val, N=154):
    """
    Solve normal equations:
        (A'A / N + lambda * I_p) b = A' mu / N
    N is fixed to 154 for both unconditional (d=154) and conditional (d=308) models
    to hold the per-asset restriction weight constant.
    """
    p = F.shape[1]
    mu, A = compute_fitting_moments(Q, F)
    
    # LHS: (p x p)
    LHS = (A.T @ A) / N + lambda_val * np.eye(p)
    # RHS: (p,)
    RHS = (A.T @ mu) / N
    
    b = np.linalg.solve(LHS, RHS)
    return b

def solve_augmented_least_squares(Q, F, lambda_val, N=154):
    """
    Alternative independent solver via augmented least squares:
        design = [A / sqrt(N); sqrt(lambda) * I_p]
        response = [mu / sqrt(N); 0]
    """
    p = F.shape[1]
    mu, A = compute_fitting_moments(Q, F)
    
    design = np.vstack([A / np.sqrt(N), np.sqrt(lambda_val) * np.eye(p)])
    response = np.concatenate([mu / np.sqrt(N), np.zeros(p)])
    
    b, residuals, rank, s = np.linalg.lstsq(design, response, rcond=None)
    return b

def evaluate_sdf_moments(b, F_eval, Q_eval):
    """
    Compute evaluation moments and loss on evaluation data:
        M_t = 1 - b' F_t
        e_bar = (1 / T) sum_{t=1}^T M_t Q_t in R^d
        Loss L = (1 / d) sum_{m=1}^d e_bar_m^2
        RMSE = sqrt(L)
    Q_eval: (T x d) dictionary (d=308 for shared conditional evaluation)
    F_eval: (T x p) theme payoffs (p=14)
    """
    T, d = Q_eval.shape
    assert F_eval.shape[0] == T, f"Row mismatch: Q has {T}, F has {F_eval.shape[0]}"
    
    # Discount factor M_t: shape (T,)
    M = 1.0 - F_eval @ b
    
    # Discounted payoffs M_t * Q_mt: shape (T x d)
    MQ = M[:, np.newaxis] * Q_eval
    
    # Period-mean moment error e_bar: shape (d,)
    e_bar = MQ.mean(axis=0)
    
    # Loss: mean of squared mean moments
    loss = float(np.mean(e_bar ** 2))
    rmse = float(np.sqrt(loss))
    
    return {
        "loss": loss,
        "rmse": rmse,
        "e_bar": e_bar,
        "M": M,
        "MQ": MQ
    }

def compute_unconditional_basis_points(e_bar_unconditional, factor_scales, market_scale):
    """
    Convert the first 154 ordinary test payoff errors back to unscaled percentage return errors,
    and compute MAE and RMSE in monthly basis points (1 bp = 0.01% = 0.0001 return).
    """
    assert len(e_bar_unconditional) == 154
    scales = np.concatenate([factor_scales, [market_scale]])
    
    # Unscaled return errors: e_bar_i * s_i
    unscaled_errors = e_bar_unconditional * scales
    
    # Convert to basis points (x 10,000)
    errors_bps = unscaled_errors * 10000.0
    mae_bps = float(np.mean(np.abs(errors_bps)))
    rmse_bps = float(np.sqrt(np.mean(errors_bps ** 2)))
    
    return {
        "mae_bps": mae_bps,
        "rmse_bps": rmse_bps,
        "errors_bps": errors_bps
    }
