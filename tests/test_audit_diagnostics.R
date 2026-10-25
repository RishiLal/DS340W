suppressPackageStartupMessages({
  library(glmnet)
  library(MASS)
})

source("scripts/rcf_blocks.R")
source("scripts/goldilocks_tuning.R")
source("scripts/rcf_adml.R")

factors_train <- read.csv("data_splits/factors_train_70.csv")
port_train    <- read.csv("data_splits/port_3x2_excess_train_70.csv")
gan_inf       <- read.csv("data_splits/gan_factors_inf_167.csv")

dates_eval <- gan_inf$Date
f_idx_eval <- which(factors_train$Date %in% dates_eval)
p_idx_eval <- which(port_train$Date %in% dates_eval)

factor_names <- colnames(factors_train)[!(colnames(factors_train) %in% c("Date", "RF"))]
factors_mat  <- as.matrix(factors_train[f_idx_eval, factor_names])
port_mat     <- as.matrix(port_train[p_idx_eval, colnames(port_train) != "Date"])

test_f <- c("MktRf", "SMB", "HML", "RMW", "CMA", "UMD", "ol", "ndf")

cat("=================================================================\n")
cat(" DIAGNOSTIC 1: GOLDILOCKS (S=3) VS ORDINARY MIN-RMSE (S=1)\n")
cat("=================================================================\n")
for (f_code in test_f) {
  f_idx <- which(factor_names == f_code)
  pt_gz  <- estimate_rcf_plr_point(f_idx, factors_mat, port_mat, K = 5, S = 3)
  pt_min <- estimate_rcf_plr_point(f_idx, factors_mat, port_mat, K = 5, S = 1)
  diff_val <- pt_gz$theta - pt_min$theta
  pct_diff <- if (abs(pt_min$theta) > 1e-4) (diff_val / pt_min$theta) * 100 else 0.0
  cat(sprintf("Factor %-6s | Theta_Goldilocks = %9.4f | Theta_MinRMSE = %9.4f | Diff = %+8.4f (%+.1f%%)\n",
              f_code, pt_gz$theta, pt_min$theta, diff_val, pct_diff))
}

cat("\n=================================================================\n")
cat(" DIAGNOSTIC 2: ABLATION OF CROSS-FITTING: RCF VS RANDOM SPLIT\n")
cat("=================================================================\n")
set.seed(42)
N_assets <- ncol(port_mat)
K <- 5
folds_random <- split(sample(N_assets, N_assets, replace = FALSE), as.factor(1:K))

# Classical Maasoumi-style cross-sectional DML on full-sample moments
m_full <- compute_cross_sectional_moments(factors_mat, port_mat, 1)
v_all <- numeric(N_assets)
u_all <- numeric(N_assets)

for (k in 1:K) {
  test_idx_cs <- folds_random[[k]]
  train_idx_cs <- setdiff(1:N_assets, test_idx_cs)
  
  fit_d <- glmnet(m_full$X[train_idx_cs, ], m_full$D[train_idx_cs], alpha = 1)
  cv_d <- cv.glmnet(m_full$X[train_idx_cs, ], m_full$D[train_idx_cs], alpha = 1)
  pred_d <- predict(fit_d, newx = m_full$X[test_idx_cs, ], s = cv_d$lambda.min)
  
  fit_y <- glmnet(m_full$X[train_idx_cs, ], m_full$Y[train_idx_cs], alpha = 1)
  cv_y <- cv.glmnet(m_full$X[train_idx_cs, ], m_full$Y[train_idx_cs], alpha = 1)
  pred_y <- predict(fit_y, newx = m_full$X[test_idx_cs, ], s = cv_y$lambda.min)
  
  v_all[test_idx_cs] <- m_full$D[test_idx_cs] - pred_d
  u_all[test_idx_cs] <- m_full$Y[test_idx_cs] - pred_y
}
theta_cs_dml <- sum(v_all * u_all) / sum(v_all^2)
psi_cs <- (u_all - theta_cs_dml * v_all) * v_all / mean(v_all^2)
se_cs_defective <- sqrt(mean(psi_cs^2) / N_assets)
t_cs_defective <- theta_cs_dml / se_cs_defective

cat(sprintf("MktRf Classical CS-DML:  Theta = %+.4f | SE_cs = %.4f | t_cs = %+.2f\n",
            theta_cs_dml, se_cs_defective, t_cs_defective))

pt_rcf <- estimate_rcf_plr_point(1, factors_mat, port_mat, K = 5, S = 3)
cat(sprintf("MktRf RCF-PLR (Point):   Theta = %+.4f\n", pt_rcf$theta))

cat("\n=================================================================\n")
cat(" DIAGNOSTIC 3: BOOTSTRAP SENSITIVITY (BLOCK LENGTH & REPLICATIONS)\n")
cat("=================================================================\n")
# Test MktRf with block_len = 6, 12, 24
for (bl in c(6, 12, 24)) {
  res_boot <- bootstrap_rcf_plr(1, factors_mat, port_mat, K = 5, S = 3, n_boot = 30, block_len = bl, seed = 42)
  cat(sprintf("Block Length %2d mo: Mean = %+.4f | SE_boot = %.4f | Range = [%.2f, %.2f]\n",
              bl, res_boot$boot_mean, res_boot$se_boot, min(res_boot$valid_boots), max(res_boot$valid_boots)))
}

cat("=================================================================\n")
cat(" DIAGNOSTIC COMPLETE\n")
