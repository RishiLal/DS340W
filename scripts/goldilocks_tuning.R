#################################################################################
###  goldilocks_tuning.R
###  The 'Goldilocks Zone' Stability-Based Hyperparameter Calibration
###  Based on Ciganovic, D'Amario, and Tancioni (2026), Section 3
###  Audited & Corrected for:
###    - Strict pre-filtering length equality validation
###    - Explicit descriptive error on entirely non-finite tuning grids
###    - Safe handling of small, constant, or degenerate candidate windows
###    - Deterministic tie breaking and min-max normalization
#################################################################################

#' Goldilocks Zone Hyperparameter Selection
#'
#' Evaluates out-of-sample RMSE over candidate penalty parameters using a moving window
#' of size S. Computes local mean RMSE (R_bar) and local variance of RMSE (V_j),
#' normalizes them, and chooses the window minimizing total instability score S_j = V_tilde + R_tilde.
#' Within that optimal window, selects the penalty lambda* that minimizes RMSE.
#'
#' @param lambda_grid Ordered numeric vector of candidate penalties
#' @param rmse_vec Numeric vector of validation RMSE values corresponding to lambda_grid
#' @param S Window size for local stability evaluation (default S = 3)
#' @return List containing best_lambda, best_lambda_idx, best_window, and scores data frame
get_goldilocks_lambda <- function(lambda_grid, rmse_vec, S = 3) {
  # 1. Pre-filtering length checks
  if (length(lambda_grid) != length(rmse_vec)) {
    stop(sprintf("Length of lambda_grid (%d) and rmse_vec (%d) must match.",
                 length(lambda_grid), length(rmse_vec)))
  }
  if (length(lambda_grid) == 0) {
    stop("lambda_grid and rmse_vec must not be empty.")
  }
  
  # 2. Filter non-finite values
  valid_mask <- is.finite(lambda_grid) & is.finite(rmse_vec)
  lambda_grid <- lambda_grid[valid_mask]
  rmse_vec <- rmse_vec[valid_mask]
  
  M <- length(lambda_grid)
  if (M == 0) {
    stop("No finite candidate penalties or RMSE values found in tuning grid.")
  }
  
  # 3. Fallback for short grids (M < S)
  if (M < S) {
    min_idx <- which.min(rmse_vec)
    return(list(
      best_lambda = lambda_grid[min_idx],
      best_lambda_idx = min_idx,
      best_window = c(1, M),
      scores = NULL
    ))
  }
  
  num_windows <- M - S + 1
  R_bar <- numeric(num_windows)
  V_j <- numeric(num_windows)
  
  for (j in 1:num_windows) {
    window_rmse <- rmse_vec[j:(j + S - 1)]
    R_bar[j] <- mean(window_rmse)
    v_val <- var(window_rmse)
    V_j[j] <- if (is.finite(v_val)) v_val else 0.0
  }
  
  # Min-Max Normalization
  min_V <- min(V_j)
  max_V <- max(V_j)
  V_tilde <- if (max_V > min_V + 1e-12) (V_j - min_V) / (max_V - min_V) else rep(0, num_windows)
  
  min_R <- min(R_bar)
  max_R <- max(R_bar)
  R_tilde <- if (max_R > min_R + 1e-12) (R_bar - min_R) / (max_R - min_R) else rep(0, num_windows)
  
  S_score <- V_tilde + R_tilde
  
  # Select optimal window minimizing combined instability score
  j_star <- which.min(S_score)
  
  # Select optimal lambda* minimizing validation RMSE within window j*
  best_window_idx <- j_star:(j_star + S - 1)
  best_idx_in_window <- which.min(rmse_vec[best_window_idx])
  best_lambda_idx <- best_window_idx[best_idx_in_window]
  best_lambda <- lambda_grid[best_lambda_idx]
  
  scores_df <- data.frame(
    window = 1:num_windows,
    lambda_start = lambda_grid[1:num_windows],
    lambda_end = lambda_grid[(1:num_windows) + S - 1],
    mean_rmse = round(R_bar, 6),
    var_rmse = round(V_j, 8),
    norm_var = round(V_tilde, 4),
    norm_rmse = round(R_tilde, 4),
    goldilocks_score = round(S_score, 4)
  )
  
  return(list(
    best_lambda = best_lambda,
    best_lambda_idx = best_lambda_idx,
    best_window = c(j_star, j_star + S - 1),
    scores = scores_df
  ))
}
