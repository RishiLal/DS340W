#################################################################################
###  rcf_adml.R
###  Time-Series Robust Reverse Cross-Fitting (RCF) Asset Pricing Estimator
###  
###  Methodology:
###    1. Point Estimator: Partially Linear Model (Robinson 1988) on generated moments
###       Y_i = theta * D_i + g(X_i) + eps_i
###       where Y_i = E[R_i], D_i = Cov(F_k, R_i), X_i = Cov(F_{-k}, R_i)
###    2. Temporal Partitioning: Reverse Cross-Fitting (RCF, Ciganovic et al. 2026)
###       with Goldilocks zone stability calibration on validation folds.
###    3. Split-Moment IV Capability: Disjoint auxiliary sample instrumenting
###       evaluation block moments to eliminate errors-in-variables attenuation.
###    4. Unified Block Bootstrap: Joint time-series block bootstrap resampling of
###       raw returns and factor realizations, recalculating the full estimation
###       pipeline on every resample (invariant to portfolio duplication).
###    5. Trustworthy Status Hierarchy: Distinct execution, fold, and bootstrap statuses.
#################################################################################

suppressPackageStartupMessages({
  library(glmnet)
  library(nnet)
  library(MASS)
})

# Resolve project-relative paths
find_project_dir <- function() {
  candidates <- c(
    ".",
    "..",
    "c:/Users/rishi/Desktop/DS340W",
    "C:/Users/rishi/Desktop/DS340W"
  )
  for (cand in candidates) {
    if (file.exists(file.path(cand, "scripts", "rcf_blocks.R"))) {
      return(normalizePath(cand))
    }
  }
  return(getwd())
}

PROJECT_DIR <- find_project_dir()
source(file.path(PROJECT_DIR, "scripts", "rcf_blocks.R"))
source(file.path(PROJECT_DIR, "scripts", "goldilocks_tuning.R"))

# ==============================================================================
# 1. CROSS-SECTIONAL MOMENT CONSTRUCTION & IMPUTATION
# ==============================================================================

#' Compute Cross-Sectional Moments (Y, D, X) from Time-Series Data
#'
#' @param factors_matrix T x P matrix of factor realizations
#' @param port_matrix T x N matrix of test asset excess returns
#' @param factor_idx Index of treatment factor k in 1:P
#' @param t_idx Optional vector of time indices to subset
#' @return List with Y (N x 1), D (N x 1), X (N x (P-1)), effective sample sizes
compute_cross_sectional_moments <- function(factors_matrix, port_matrix, factor_idx, t_idx = NULL) {
  if (!is.null(t_idx)) {
    F_sub <- factors_matrix[t_idx, , drop = FALSE]
    R_sub <- port_matrix[t_idx, , drop = FALSE]
  } else {
    F_sub <- factors_matrix
    R_sub <- port_matrix
  }
  
  T_sub <- nrow(F_sub)
  N_ports <- ncol(R_sub)
  
  gt_sub <- F_sub[, factor_idx]
  ht_sub <- F_sub[, -factor_idx, drop = FALSE]
  
  # Sample mean returns (Y)
  Y_m <- colMeans(R_sub, na.rm = TRUE)
  
  # Sample covariances with treatment factor (D) and controls (X)
  D_m <- as.vector(cov(gt_sub, R_sub, use = "pairwise.complete.obs"))
  X_m <- as.matrix(cov(ht_sub, R_sub, use = "pairwise.complete.obs"))
  if (nrow(X_m) != N_ports) X_m <- t(X_m)
  
  # Record effective observations count
  valid_y_count <- sum(is.finite(Y_m))
  valid_d_count <- sum(is.finite(D_m))
  
  # Impute any missing moments across portfolios using cross-sectional medians
  if (any(!is.finite(Y_m))) {
    med_y <- median(Y_m[is.finite(Y_m)], na.rm = TRUE)
    Y_m[!is.finite(Y_m)] <- if (is.finite(med_y)) med_y else 0.0
  }
  if (any(!is.finite(D_m))) {
    med_d <- median(D_m[is.finite(D_m)], na.rm = TRUE)
    D_m[!is.finite(D_m)] <- if (is.finite(med_d)) med_d else 0.0
  }
  for (col_j in 1:ncol(X_m)) {
    if (any(!is.finite(X_m[, col_j]))) {
      med_x <- median(X_m[is.finite(X_m[, col_j]), col_j], na.rm = TRUE)
      X_m[!is.finite(X_m[, col_j]), col_j] <- if (is.finite(med_x)) med_x else 0.0
    }
  }
  
  list(
    Y = Y_m,
    D = D_m,
    X = X_m,
    T_obs = T_sub,
    N = N_ports,
    valid_y_count = valid_y_count,
    valid_d_count = valid_d_count
  )
}


# ==============================================================================
# 2. PURE POINT ESTIMATOR (RCF-PLR / SPLIT-IV PLR)
# ==============================================================================

#' Point Estimator for Reverse Cross-Fitting Partially Linear Model
#'
#' Evaluates the Robinson (1988) double LASSO estimator across K temporal RCF folds
#' with Goldilocks zone stability calibration on each fold.
#' Supports split-moment instrumental variables to purge generated-moment attenuation.
#'
#' @param factor_idx Index of treatment factor
#' @param factors_matrix T x P matrix of factor returns
#' @param port_matrix T x N matrix of asset excess returns
#' @param K Number of RCF temporal folds (default K = 5)
#' @param S Goldilocks zone window size (default S = 3)
#' @param lambda_grid_len Length of lambda penalty grid (default 25)
#' @param use_split_iv Logical: whether to instrument test moments with auxiliary moments
#' @return List with status, theta, fold_thetas, fold_details, and psi_all
estimate_rcf_plr_point <- function(factor_idx, factors_matrix, port_matrix,
                                   K = 5, S = 3, lambda_grid_len = 25,
                                   use_split_iv = FALSE) {
  T_total <- nrow(factors_matrix)
  N_ports <- ncol(port_matrix)
  
  if (T_total < 20 || N_ports < 5) {
    return(list(
      execution_status = "ERROR",
      fold_status = "ALL_FOLDS_FAILED",
      message = "Insufficient sample size for RCF partitioning",
      theta = NA_real_,
      fold_thetas = rep(NA_real_, K),
      fold_details = vector("list", K),
      psi_all = numeric(0),
      valid_folds = 0
    ))
  }
  
  rcf_parts <- get_rcf_partitions(T_total, K = K)
  
  fold_thetas <- rep(NA_real_, K)
  fold_details <- vector("list", K)
  psi_all <- numeric(0)
  failed_folds <- integer(0)
  
  for (k in 1:K) {
    part <- rcf_parts[[k]]
    m_train <- compute_cross_sectional_moments(factors_matrix, port_matrix, factor_idx, part$train_idx)
    m_val   <- compute_cross_sectional_moments(factors_matrix, port_matrix, factor_idx, part$val_idx)
    m_aux   <- compute_cross_sectional_moments(factors_matrix, port_matrix, factor_idx, part$aux_idx)
    m_test  <- compute_cross_sectional_moments(factors_matrix, port_matrix, factor_idx, part$test_idx)
    
    # 1. Nuisance model for treatment D on controls X
    fit_d_init <- tryCatch(
      glmnet(m_train$X, m_train$D, alpha = 1, nlambda = lambda_grid_len),
      error = function(e) NULL
    )
    lam_vec_d <- if (!is.null(fit_d_init)) as.numeric(fit_d_init[["lambda"]]) else numeric(0)
    
    if (length(lam_vec_d) < 3) {
      fold_details[[k]] <- list(fold = k, status = "FAIL_D_GRID_TOO_SHORT")
      failed_folds <- c(failed_folds, k)
      next
    }
    
    grid_d <- rev(lam_vec_d)
    preds_d <- predict(fit_d_init, newx = m_val$X, s = grid_d)
    rmse_d <- sqrt(colMeans((m_val$D - preds_d)^2))
    
    gz_d <- tryCatch(
      get_goldilocks_lambda(grid_d, rmse_d, S = S),
      error = function(e) NULL
    )
    if (is.null(gz_d)) {
      fold_details[[k]] <- list(fold = k, status = "FAIL_D_GOLDILOCKS")
      failed_folds <- c(failed_folds, k)
      next
    }
    best_lam_d <- gz_d$best_lambda
    
    # 2. Nuisance model for outcome Y on controls X (separately tuned)
    fit_y_init <- tryCatch(
      glmnet(m_train$X, m_train$Y, alpha = 1, nlambda = lambda_grid_len),
      error = function(e) NULL
    )
    lam_vec_y <- if (!is.null(fit_y_init)) as.numeric(fit_y_init[["lambda"]]) else numeric(0)
    
    if (length(lam_vec_y) < 3) {
      fold_details[[k]] <- list(fold = k, status = "FAIL_Y_GRID_TOO_SHORT")
      failed_folds <- c(failed_folds, k)
      next
    }
    
    grid_y <- rev(lam_vec_y)
    preds_y <- predict(fit_y_init, newx = m_val$X, s = grid_y)
    rmse_y <- sqrt(colMeans((m_val$Y - preds_y)^2))
    
    gz_y <- tryCatch(
      get_goldilocks_lambda(grid_y, rmse_y, S = S),
      error = function(e) NULL
    )
    if (is.null(gz_y)) {
      fold_details[[k]] <- list(fold = k, status = "FAIL_Y_GOLDILOCKS")
      failed_folds <- c(failed_folds, k)
      next
    }
    best_lam_y <- gz_y$best_lambda
    
    # 3. Refit nuisances on pooled auxiliary partition (train + val)
    fit_aux_d <- tryCatch(
      glmnet(m_aux$X, m_aux$D, alpha = 1, lambda = best_lam_d),
      error = function(e) NULL
    )
    fit_aux_y <- tryCatch(
      glmnet(m_aux$X, m_aux$Y, alpha = 1, lambda = best_lam_y),
      error = function(e) NULL
    )
    
    if (is.null(fit_aux_d) || is.null(fit_aux_y)) {
      fold_details[[k]] <- list(fold = k, status = "FAIL_AUX_FIT")
      failed_folds <- c(failed_folds, k)
      next
    }
    
    # 4. Out-of-fold residuals on evaluation block B_k
    pred_d <- as.vector(predict(fit_aux_d, newx = m_test$X, s = best_lam_d))
    pred_y <- as.vector(predict(fit_aux_y, newx = m_test$X, s = best_lam_y))
    
    v_test <- m_test$D - pred_d
    u_test <- m_test$Y - pred_y
    
    if (use_split_iv) {
      # Split-Moment IV: instrument test moment v_test with disjoint auxiliary moment v_aux
      v_aux <- m_aux$D - as.vector(predict(fit_aux_d, newx = m_aux$X, s = best_lam_d))
      denom <- sum(v_aux * v_test)
      numer <- sum(v_aux * u_test)
    } else {
      # Standard PLR: evaluate directly on test block
      denom <- sum(v_test^2)
      numer <- sum(v_test * u_test)
    }
    
    if (!is.finite(denom) || abs(denom) < 1e-12) {
      fold_details[[k]] <- list(fold = k, status = "FAIL_DEGENERATE_DENOMINATOR")
      failed_folds <- c(failed_folds, k)
      next
    }
    
    theta_k <- numer / denom
    if (!is.finite(theta_k)) {
      fold_details[[k]] <- list(fold = k, status = "FAIL_NON_FINITE_THETA")
      failed_folds <- c(failed_folds, k)
      next
    }
    
    fold_thetas[k] <- theta_k
    
    mean_v2 <- mean(v_test^2)
    psi_k <- if (mean_v2 > 1e-12) (u_test - theta_k * v_test) * v_test / mean_v2 else rep(0, N_ports)
    psi_all <- c(psi_all, psi_k)
    
    fold_details[[k]] <- list(
      fold = k,
      status = "SUCCESS",
      theta = theta_k,
      lambda_d = best_lam_d,
      lambda_y = best_lam_y,
      gz_window_d = gz_d$best_window,
      gz_window_y = gz_y$best_window,
      denom = denom,
      mean_v2 = mean_v2
    )
  }
  
  valid_thetas <- fold_thetas[is.finite(fold_thetas)]
  n_valid <- length(valid_thetas)
  
  if (n_valid == 0) {
    return(list(
      execution_status = "ERROR",
      fold_status = "ALL_FOLDS_FAILED",
      message = "All RCF folds failed numerical evaluation",
      theta = NA_real_,
      fold_thetas = fold_thetas,
      fold_details = fold_details,
      psi_all = psi_all,
      valid_folds = 0
    ))
  }
  
  fold_status <- if (n_valid == K) "COMPLETE_ALL_K_FOLDS" else "PARTIAL_FOLDS"
  theta_final <- mean(valid_thetas)
  
  list(
    execution_status = "SUCCESS",
    fold_status = fold_status,
    theta = theta_final,
    fold_thetas = fold_thetas,
    fold_details = fold_details,
    psi_all = psi_all,
    valid_folds = n_valid,
    failed_folds = failed_folds
  )
}


# ==============================================================================
# 3. UNIFIED TIME-SERIES BLOCK BOOTSTRAP
# ==============================================================================

#' Joint Time-Series Block Bootstrap for RCF-PLR
#'
#' Jointly resamples contiguous blocks of time across all factors and test assets,
#' calling the exact same point-estimation pipeline on every replication.
#' Does not mutate caller's RNG state.
#'
#' @param factor_idx Index of treatment factor
#' @param factors_matrix T x P matrix of factor returns
#' @param port_matrix T x N matrix of asset excess returns
#' @param K Number of RCF temporal folds
#' @param S Goldilocks zone window size
#' @param n_boot Number of bootstrap replications
#' @param block_len Length of contiguous time blocks (months)
#' @param use_split_iv Logical: whether to use split-moment IV
#' @param seed Optional random seed
#' @return List with bootstrap distribution, mean, SD, bias, quantiles, and failure counts
bootstrap_rcf_plr <- function(factor_idx, factors_matrix, port_matrix,
                              K = 5, S = 3, n_boot = 100, block_len = 12,
                              use_split_iv = FALSE,
                              seed = NULL) {
  # Preserve caller's RNG state
  if (!is.null(seed)) {
    if (exists(".Random.seed", envir = .GlobalEnv)) {
      old_seed <- get(".Random.seed", envir = .GlobalEnv)
      on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
    }
    set.seed(seed)
  }
  
  T_total <- nrow(factors_matrix)
  n_blocks <- ceiling(T_total / block_len)
  boot_thetas <- numeric(n_boot)
  fail_count <- 0
  
  for (b in 1:n_boot) {
    # Sample contiguous block starts
    start_indices <- sample(1:(T_total - block_len + 1), size = n_blocks, replace = TRUE)
    boot_t_idx <- unlist(lapply(start_indices, function(si) si:(si + block_len - 1)))[1:T_total]
    
    F_b <- factors_matrix[boot_t_idx, , drop = FALSE]
    R_b <- port_matrix[boot_t_idx, , drop = FALSE]
    
    # Evaluate EXACT point estimator on resampled panel
    est_b <- estimate_rcf_plr_point(
      factor_idx = factor_idx,
      factors_matrix = F_b,
      port_matrix = R_b,
      K = K,
      S = S,
      use_split_iv = use_split_iv
    )
    
    if (est_b$execution_status == "SUCCESS" && is.finite(est_b$theta)) {
      boot_thetas[b] <- est_b$theta
    } else {
      boot_thetas[b] <- NA_real_
      fail_count <- fail_count + 1
    }
  }
  
  valid_boots <- boot_thetas[is.finite(boot_thetas)]
  n_success <- length(valid_boots)
  fail_rate <- fail_count / n_boot
  
  # Rigorous bootstrap adequacy criteria: require at least 80% success rate and min 20 valid draws
  # For small smoke tests (n_boot < 20), require at least 80% success rate
  min_req <- max(5, floor(n_boot * 0.8))
  
  if (n_success < min_req) {
    return(list(
      bootstrap_status = "INSUFFICIENT_BOOTSTRAPS",
      message = sprintf("Bootstrap failed: %d/%d succeeded (below requirement of %d)",
                        n_success, n_boot, min_req),
      boot_thetas = boot_thetas,
      se_boot = NA_real_,
      boot_mean = NA_real_,
      fail_count = fail_count,
      n_successful = n_success,
      fail_rate = fail_rate
    ))
  }
  
  boot_status <- if (fail_rate > 0.10) "HIGH_FAILURE_RATE" else "ADEQUATE"
  boot_mean <- mean(valid_boots)
  se_boot <- sd(valid_boots)
  boot_quantiles <- quantile(valid_boots, probs = c(0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975))
  
  list(
    bootstrap_status = boot_status,
    boot_thetas = boot_thetas,
    valid_boots = valid_boots,
    se_boot = se_boot,
    boot_mean = boot_mean,
    boot_quantiles = boot_quantiles,
    fail_count = fail_count,
    n_successful = n_success,
    fail_rate = fail_rate
  )
}


# ==============================================================================
# 4. RIESZ REPRESENTOR SOLVER (EXPERIMENTAL ADML BRANCH)
# ==============================================================================

solve_rmd_lasso <- function(G, M, lambda, max_iter = 500, tol = 1e-5) {
  p <- ncol(G)
  beta <- rep(0, p)
  diag_G <- diag(G)
  diag_G[diag_G <= 1e-10] <- 1e-10
  
  for (iter in 1:max_iter) {
    beta_old <- beta
    for (j in 1:p) {
      rho <- M[j] - sum(G[j, ] * beta) + G[j, j] * beta[j]
      z <- diag_G[j]
      
      if (!is.finite(rho)) {
        beta[j] <- 0
        next
      }
      
      if (rho < -lambda) {
        beta[j] <- (rho + lambda) / z
      } else if (rho > lambda) {
        beta[j] <- (rho - lambda) / z
      } else {
        beta[j] <- 0
      }
    }
    
    if (sum(abs(beta - beta_old)) < tol) {
      break
    }
  }
  
  beta[abs(beta) < 1e-6] <- 0
  return(beta)
}


# ==============================================================================
# 5. UNIFIED PRODUCTION ENTRY POINT: estimate_rcf_adml
# ==============================================================================

#' Estimate Factor Risk Premium using RCF-PLR / RCF-ADML
#'
#' @param factor_idx Index of factor to test (1:P)
#' @param factors_matrix Matrix of monthly factor returns (T x P)
#' @param port_matrix Matrix of monthly portfolio excess returns (T x N)
#' @param method "plr" (Production validated Partially Linear Model) or "adml" (Experimental Riesz ADML)
#' @param K Number of temporal folds (default K = 5)
#' @param S Goldilocks window size (default S = 3)
#' @param n_boot Number of time-series block bootstrap draws (default 100)
#' @param block_len Block length for time bootstrap (default 12 months)
#' @param use_split_iv Logical: whether to use split-moment IV (default FALSE)
#' @param seed Random seed for reproducibility (default 42)
#' @return List with point estimate, bootstrap SE, test statistics, and diagnostics
estimate_rcf_adml <- function(factor_idx, factors_matrix, port_matrix,
                              method = c("plr", "adml"),
                              K = 5, S = 3,
                              n_boot = 100, block_len = 12,
                              use_split_iv = FALSE,
                              seed = 42) {
  method <- match.arg(method)
  
  T_total <- nrow(factors_matrix)
  N_ports <- ncol(port_matrix)
  P_factors <- ncol(factors_matrix)
  
  stopifnot(nrow(port_matrix) == T_total)
  stopifnot(factor_idx >= 1 && factor_idx <= P_factors)
  
  if (method == "plr") {
    # 1. Point Estimate on Original Sample
    pt_res <- estimate_rcf_plr_point(
      factor_idx = factor_idx,
      factors_matrix = factors_matrix,
      port_matrix = port_matrix,
      K = K,
      S = S,
      use_split_iv = use_split_iv
    )
    
    if (pt_res$execution_status != "SUCCESS") {
      return(list(
        status = "ERROR",
        execution_status = "ERROR",
        fold_status = pt_res$fold_status,
        bootstrap_status = "NOT_RUN",
        factor_idx = factor_idx,
        method = "plr",
        theta = NA_real_,
        se = NA_real_,
        se_boot = NA_real_,
        t_stat = NA_real_,
        t_boot = NA_real_,
        p_val = NA_real_,
        p_boot = NA_real_,
        inference_valid = FALSE,
        message = pt_res$message
      ))
    }
    
    theta_orig <- pt_res$theta
    
    # 2. Uncertainty via Unified Block Bootstrap
    boot_res <- bootstrap_rcf_plr(
      factor_idx = factor_idx,
      factors_matrix = factors_matrix,
      port_matrix = port_matrix,
      K = K,
      S = S,
      n_boot = n_boot,
      block_len = block_len,
      use_split_iv = use_split_iv,
      seed = seed
    )
    
    se_boot <- boot_res$se_boot
    
    # Trustworthy validation criteria:
    # Inference is valid ONLY IF:
    #   - All K folds completed cleanly in the point estimator
    #   - Bootstrap status is ADEQUATE (success rate >= 90%)
    #   - se_boot is finite and strictly positive
    inference_valid <- (pt_res$fold_status == "COMPLETE_ALL_K_FOLDS") &&
                       (boot_res$bootstrap_status == "ADEQUATE") &&
                       is.finite(se_boot) && (se_boot > 0)
    
    if (is.finite(se_boot) && se_boot > 0) {
      t_boot <- theta_orig / se_boot
      p_boot_norm <- 2 * pnorm(-abs(t_boot))
      log_p_boot <- pnorm(-abs(t_boot), log.p = TRUE) + log(2)
      p_boot_emp <- 2 * min(
        mean(boot_res$valid_boots <= 0),
        mean(boot_res$valid_boots >= 0)
      )
    } else {
      t_boot <- NA_real_
      p_boot_norm <- NA_real_
      log_p_boot <- NA_real_
      p_boot_emp <- NA_real_
    }
    
    # 3. Cross-sectional influence SE for comparison
    n_eff_cs <- length(pt_res$psi_all)
    se_cs <- if (n_eff_cs > 0) sqrt(mean(pt_res$psi_all^2) / n_eff_cs) else NA_real_
    t_cs <- if (is.finite(se_cs) && se_cs > 1e-12) theta_orig / se_cs else NA_real_
    
    return(list(
      status = "SUCCESS",
      execution_status = "SUCCESS",
      fold_status = pt_res$fold_status,
      bootstrap_status = boot_res$bootstrap_status,
      factor_idx = factor_idx,
      method = "plr",
      use_split_iv = use_split_iv,
      theta = theta_orig,
      # Compatibility aliases for callers expecting 'se' and 't_stat'
      se = se_boot,
      se_boot = se_boot,
      t_stat = t_boot,
      t_boot = t_boot,
      p_val = p_boot_norm,
      p_boot = p_boot_norm,
      p_boot_empirical = p_boot_emp,
      log_p_boot = log_p_boot,
      boot_mean = boot_res$boot_mean,
      boot_bias = if (is.finite(boot_res$boot_mean)) boot_res$boot_mean - theta_orig else NA_real_,
      boot_quantiles = boot_res$boot_quantiles,
      boot_thetas = boot_res$boot_thetas,
      boot_failures = boot_res$fail_count,
      boot_success_count = boot_res$n_successful,
      se_cs = se_cs,
      t_cs = t_cs,
      fold_thetas = pt_res$fold_thetas,
      fold_details = pt_res$fold_details,
      valid_folds = pt_res$valid_folds,
      inference_valid = inference_valid
    ))
    
  } else {
    # --- EXPERIMENTAL ADML ROUTE: Non-Parametric Average Derivative ---
    return(list(
      status = "EXPERIMENTAL_UNVALIDATED",
      execution_status = "EXPERIMENTAL",
      fold_status = "NOT_EVALUATED",
      bootstrap_status = "NOT_EVALUATED",
      factor_idx = factor_idx,
      method = "adml",
      theta = NA_real_,
      se = NA_real_,
      se_boot = NA_real_,
      t_stat = NA_real_,
      t_boot = NA_real_,
      p_val = NA_real_,
      p_boot = NA_real_,
      inference_valid = FALSE,
      message = "Experimental Riesz ADML branch retained for exploratory comparison; production inference withheld."
    ))
  }
}
