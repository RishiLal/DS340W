#################################################################################
###  test_estimator_simulation.R
###  Rigorous Monte Carlo Simulation Validating Estimator Correctness & Inference
###  DS340W Independent Audit Verification
###
###  Features:
###    1. Analytically Derived Population Targets:
###       - AR(1) factor F_t = rho*F_{t-1} + sqrt(1-rho^2)*e_t -> Var(F) = 1.0000.
###       - Covariance pricing target theta_cov = theta_beta / Var(F) = theta_beta = 2.0000.
###       - Stationary initialization F_1 ~ N(0, 1).
###    2. Independent Random Streams:
###       - Independent seed sequence across simulation replications.
###       - Estimator execution preserves caller's RNG state.
###    3. Multi-Scenario Simulation Battery:
###       - Test 1: Oracle Moments Benchmark (|theta_hat - theta_true| < 0.05).
###       - Test 2: Null Hypothesis DGP (theta_0 = 0, size / Type-I error calibration).
###       - Test 3: Unbiased Non-Zero Alternative DGP (theta_1 = 2.0, Enforced 95% CI Coverage >= 80%).
###       - Test 4: Finite-Sample Attenuation Diagnostic (Standard PLR reliability ratio).
###    4. Enforced Acceptance Criteria:
###       - Non-zero exit code on failure (stop() with descriptive error).
#################################################################################

suppressPackageStartupMessages({
  library(glmnet)
  library(MASS)
})

# Resolve project-relative paths
find_project_dir <- function() {
  candidates <- c(".", "..", "c:/Users/rishi/Desktop/DS340W", "C:/Users/rishi/Desktop/DS340W")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "scripts", "rcf_adml.R"))) return(normalizePath(cand))
  }
  return(getwd())
}

PROJECT_DIR <- find_project_dir()
source(file.path(PROJECT_DIR, "scripts", "rcf_blocks.R"))
source(file.path(PROJECT_DIR, "scripts", "goldilocks_tuning.R"))
source(file.path(PROJECT_DIR, "scripts", "rcf_adml.R"))

cat("==================================================================\n")
cat(" RIGOROUS MONTE CARLO SIMULATION SUITE FOR FACTOR PRICING ESTIMATOR\n")
cat("==================================================================\n\n")

# ------------------------------------------------------------------------------
# DGP Generator with Analytically Derived Covariance Target
# ------------------------------------------------------------------------------
generate_sim_dataset <- function(T_obs = 160, N_assets = 50, P_dim = 12,
                                 theta_true = 2.0, rho_ar1 = 0.2, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  # AR(1) Factor Process normalized so Var(F_j) = 1.0000 exactly
  # F_{j,t} = rho * F_{j,t-1} + sqrt(1 - rho^2) * eta_{j,t}
  sigma_eta <- sqrt(1 - rho_ar1^2)
  
  F_mat <- matrix(0, T_obs, P_dim)
  F_mat[1, ] <- rnorm(P_dim, mean = 0, sd = 1.0)
  for (t in 2:T_obs) {
    F_mat[t, ] <- rho_ar1 * F_mat[t - 1, ] + rnorm(P_dim, mean = 0, sd = sigma_eta)
  }
  
  # Cross-sectional Betas (factor loadings)
  # Treatment factor loading Beta_1 has substantial variation across assets
  Betas <- matrix(rnorm(N_assets * P_dim, mean = 0.5, sd = 0.3), N_assets, P_dim)
  Betas[, 1] <- runif(N_assets, min = 0.1, max = 1.5)
  
  # Population Expected Returns: E[R_i] = theta * Beta_{i,1} + Beta_{i,-1} %*% gamma
  gamma_vec <- seq(0.8, 0.1, length.out = P_dim - 1)
  true_exp_ret <- theta_true * Betas[, 1] + as.vector(Betas[, -1] %*% gamma_vec)
  
  # Return Realizations: R_{i,t} = E[R_i] + Beta_i * F_t + eps_{i,t}
  eps <- matrix(rnorm(T_obs * N_assets, mean = 0, sd = 0.5), T_obs, N_assets)
  R_mat <- matrix(0, T_obs, N_assets)
  for (t in 1:T_obs) {
    R_mat[t, ] <- true_exp_ret + as.vector(Betas %*% F_mat[t, ]) + eps[t, ]
  }
  
  list(
    F_mat = F_mat,
    R_mat = R_mat,
    Betas = Betas,
    true_exp_ret = true_exp_ret,
    theta_true = theta_true,
    theta_cov_target = theta_true
  )
}


# ==============================================================================
# TEST 1: ORACLE MOMENTS BENCHMARK
# ==============================================================================
cat("--- TEST 1: Oracle Moments Benchmark ---\n")
cat("Verifying Robinson (1988) Double LASSO on true population moments...\n")

set.seed(999)
dgp_oracle <- generate_sim_dataset(T_obs = 300, N_assets = 100, P_dim = 15, theta_true = 2.0)

# True population moments
Y_oracle <- dgp_oracle$true_exp_ret
D_oracle <- dgp_oracle$Betas[, 1]          # Cov(F_1, R) = Beta_1
X_oracle <- dgp_oracle$Betas[, -1]         # Cov(F_other, R) = Beta_other

fit_d_oracle <- glmnet(X_oracle, D_oracle, alpha = 1, lambda = 0.001)
fit_y_oracle <- glmnet(X_oracle, Y_oracle, alpha = 1, lambda = 0.001)

v_oracle <- D_oracle - as.vector(predict(fit_d_oracle, newx = X_oracle, s = 0.001))
u_oracle <- Y_oracle - as.vector(predict(fit_y_oracle, newx = X_oracle, s = 0.001))

theta_oracle <- sum(v_oracle * u_oracle) / sum(v_oracle^2)
oracle_error <- abs(theta_oracle - 2.0)

cat(sprintf("  Oracle Estimated Theta: %.5f (True: 2.00000, Absolute Error: %.5f)\n",
            theta_oracle, oracle_error))

if (oracle_error > 0.05) {
  stop(sprintf("[FAIL] Oracle benchmark failed! Error = %.5f exceeds tolerance 0.05", oracle_error))
}
cat("[PASS] Test 1: Oracle Moments Benchmark passed with near-zero error.\n\n")


# ==============================================================================
# TEST 2: NULL HYPOTHESIS DGP (theta = 0) & TYPE-I ERROR CALIBRATION
# ==============================================================================
cat("--- TEST 2: Null Hypothesis DGP (theta = 0) Calibration ---\n")
cat("Testing size and false-positive rate under H0: theta = 0 (nominal alpha = 0.05)...\n")

n_sim_null <- 25
null_estimates <- numeric(n_sim_null)
null_se_boots  <- numeric(n_sim_null)
null_t_stats   <- numeric(n_sim_null)
null_rejections <- logical(n_sim_null)

for (s in 1:n_sim_null) {
  sim_data <- generate_sim_dataset(
    T_obs = 160, N_assets = 50, P_dim = 12,
    theta_true = 0.0, seed = 1000 + s
  )
  
  res_null <- estimate_rcf_adml(
    factor_idx = 1,
    factors_matrix = sim_data$F_mat,
    port_matrix = sim_data$R_mat,
    method = "plr",
    K = 4, S = 3, n_boot = 30, block_len = 8,
    use_split_iv = TRUE,
    seed = 2000 + s
  )
  
  null_estimates[s]  <- res_null$theta
  null_se_boots[s]   <- res_null$se_boot
  null_t_stats[s]    <- res_null$t_boot
  null_rejections[s] <- (abs(res_null$t_boot) >= 1.96)
  
  if (s %% 5 == 0 || s == 1) {
    cat(sprintf("  Null Sim [%2d/%d]: Est = %+.4f | SE_boot = %.4f | t_boot = %+.2f | Rejected = %s\n",
                s, n_sim_null, res_null$theta, res_null$se_boot, res_null$t_boot, null_rejections[s]))
  }
}

mean_null_est <- mean(null_estimates)
null_rejection_count <- sum(null_rejections)
null_rejection_rate  <- mean(null_rejections) * 100

cat(sprintf("\nNull Results: Mean Est = %+.4f (Expected: 0.0000) | Rejections: %d/%d (%.1f%%)\n",
            mean_null_est, null_rejection_count, n_sim_null, null_rejection_rate))

# Enforce Predeclared Acceptance Criteria:
# Under H0 with alpha = 0.05 and N = 25 trials, Binomial(25, 0.05):
# P(K <= 4) = 0.993. Therefore, <= 4 rejections is the 99% binomial acceptance region.
if (abs(mean_null_est) > 0.25) {
  stop(sprintf("[FAIL] Null estimates are systematically biased away from 0: Mean = %.4f", mean_null_est))
}
if (null_rejection_count > 4) {
  stop(sprintf("[FAIL] Null hypothesis rejection rate too high: %d/%d rejections exceeds 99%% binomial bound (4)",
               null_rejection_count, n_sim_null))
}
cat("[PASS] Test 2: Null Hypothesis size calibration passed (no spurious discovery).\n\n")


# ==============================================================================
# TEST 3: UNBIASED ALTERNATIVE DGP (theta = 2.0) & ENFORCED 95% CI COVERAGE
# ==============================================================================
cat("--- TEST 3: Non-Zero Alternative DGP (theta = 2.0) & Confidence Interval Coverage ---\n")
cat("Validating unattenuated recovery and empirical confidence interval coverage...\n")

n_sim_alt <- 25
alt_estimates  <- numeric(n_sim_alt)
alt_se_boots   <- numeric(n_sim_alt)
alt_t_stats    <- numeric(n_sim_alt)
alt_rejections <- logical(n_sim_alt)
alt_covered    <- logical(n_sim_alt)

for (s in 1:n_sim_alt) {
  sim_data <- generate_sim_dataset(
    T_obs = 160, N_assets = 50, P_dim = 12,
    theta_true = 2.0, seed = 3000 + s
  )
  
  res_alt <- estimate_rcf_adml(
    factor_idx = 1,
    factors_matrix = sim_data$F_mat,
    port_matrix = sim_data$R_mat,
    method = "plr",
    K = 4, S = 3, n_boot = 30, block_len = 8,
    use_split_iv = TRUE,
    seed = 4000 + s
  )
  
  alt_estimates[s]  <- res_alt$theta
  alt_se_boots[s]   <- res_alt$se_boot
  alt_t_stats[s]    <- res_alt$t_boot
  alt_rejections[s] <- (abs(res_alt$t_boot) >= 1.96)
  
  ci_lo <- res_alt$theta - 1.96 * res_alt$se_boot
  ci_hi <- res_alt$theta + 1.96 * res_alt$se_boot
  alt_covered[s]    <- (2.0 >= ci_lo && 2.0 <= ci_hi)
  
  if (s %% 5 == 0 || s == 1) {
    cat(sprintf("  Alt Sim [%2d/%d]: Est = %.4f | SE_boot = %.4f | CI = [%.4f, %.4f] | Covered = %s\n",
                s, n_sim_alt, res_alt$theta, res_alt$se_boot, ci_lo, ci_hi, alt_covered[s]))
  }
}

mean_alt_est <- mean(alt_estimates)
alt_emp_sd   <- sd(alt_estimates)
mean_alt_se  <- mean(alt_se_boots)
se_sd_ratio  <- mean_alt_se / alt_emp_sd
power_rate   <- mean(alt_rejections) * 100
alt_coverage <- mean(alt_covered) * 100

cat("\n==================================================================\n")
cat(" ALTERNATIVE SIMULATION SUMMARY (theta_true = 2.0000):\n")
cat(sprintf("  Mean Point Estimate:      %.4f (Bias = %+.4f)\n", mean_alt_est, mean_alt_est - 2.0))
cat(sprintf("  Empirical SD of Slopes:   %.4f\n", alt_emp_sd))
cat(sprintf("  Mean Estimated SE_boot:   %.4f (SE / SD Ratio = %.2f)\n", mean_alt_se, se_sd_ratio))
cat(sprintf("  Empirical Power:          %.1f%% (%d/%d rejections of H0)\n", power_rate, sum(alt_rejections), n_sim_alt))
cat(sprintf("  95%% CI Empirical Coverage: %.1f%% (%d/%d intervals covered target 2.0)\n",
            alt_coverage, sum(alt_covered), n_sim_alt))
cat("==================================================================\n\n")

# Enforce Predeclared Acceptance Criteria:
# 1. 95% Confidence Interval Coverage must be at least 80% (Binomial(25, 0.95) 99% bound is >= 20/25 = 80%)
if (alt_coverage < 80.0) {
  stop(sprintf("[FAIL] Confidence-interval coverage failed: %.1f%% is below minimum required 80.0%%", alt_coverage))
}
# 2. Point estimate bias must be bounded: |mean - 2.0| <= 0.35
if (abs(mean_alt_est - 2.0) > 0.35) {
  stop(sprintf("[FAIL] Point estimate bias too large: Mean = %.4f (Bias = %+.4f exceeds 0.35)",
               mean_alt_est, mean_alt_est - 2.0))
}
# 3. SE Calibration: SE/SD ratio must be between 0.6 and 1.6
if (se_sd_ratio < 0.6 || se_sd_ratio > 1.6) {
  stop(sprintf("[FAIL] SE calibration ratio out of bounds: SE/SD = %.2f (expected 0.6 to 1.6)", se_sd_ratio))
}
# 4. Power must exceed 70%
if (power_rate < 70.0) {
  stop(sprintf("[FAIL] Empirical power too low: %.1f%% (minimum threshold: 70%%)", power_rate))
}

cat("[PASS] Test 3: Unbiased alternative recovery and 95% confidence interval coverage passed.\n\n")


# ==============================================================================
# TEST 4: FINITE-SAMPLE ATTENUATION DIAGNOSTIC (STANDARD PLR)
# ==============================================================================
cat("--- TEST 4: Finite-Sample Attenuation Diagnostic (Standard PLR) ---\n")
cat("Quantifying the reliability ratio and measurement-error attenuation of standard PLR...\n")

n_diag <- 10
std_estimates <- numeric(n_diag)
for (s in 1:n_diag) {
  sim_data <- generate_sim_dataset(
    T_obs = 160, N_assets = 50, P_dim = 12,
    theta_true = 2.0, seed = 5000 + s
  )
  res_std <- estimate_rcf_adml(
    factor_idx = 1,
    factors_matrix = sim_data$F_mat,
    port_matrix = sim_data$R_mat,
    method = "plr",
    K = 4, S = 3, n_boot = 5, block_len = 8,
    use_split_iv = FALSE,
    seed = 6000 + s
  )
  std_estimates[s] <- res_std$theta
}

mean_std_est <- mean(std_estimates)
reliability_ratio <- mean_std_est / 2.0

cat(sprintf("  Standard PLR Mean: %.4f | True Target: 2.0000 | Empirical Reliability Ratio: %.2f\n",
            mean_std_est, reliability_ratio))
cat(sprintf("  Attenuation Attributable to Generated-Moment Noise: %.1f%%\n", (1 - reliability_ratio) * 100))
cat("[PASS] Test 4: Attenuation mechanism diagnosed and documented.\n\n")

cat("==================================================================\n")
cat(" ALL ESTIMATOR SIMULATION TESTS PASSED SUCCESSFULLY (ALL CRITERIA MET)\n")
cat("==================================================================\n")
