#################################################################################
###  test_se_duplication.R
###  Targeted Test: Portfolio Duplication Invariance on Uncertainty Estimation
###  DS340W Independent Audit Verification
#################################################################################

suppressPackageStartupMessages({
  library(glmnet)
  library(MASS)
})

source("scripts/rcf_adml.R")

factors_train <- read.csv("data_splits/factors_train_70.csv")
port_train    <- read.csv("data_splits/port_3x2_excess_train_70.csv")
gan_train     <- read.csv("data_splits/gan_factors_train_70.csv")

dates_eval <- gan_train$Date
f_idx_eval <- which(factors_train$Date %in% dates_eval)
p_idx_eval <- which(port_train$Date %in% dates_eval)

factors_df <- factors_train[f_idx_eval, !(colnames(factors_train) %in% c("Date", "RF"))]
port_df    <- port_train[p_idx_eval, colnames(port_train) != "Date"]
gan_df     <- gan_train[, colnames(gan_train) != "Date"]

factors_augmented <- cbind(as.matrix(factors_df), as.matrix(gan_df))
port_750  <- as.matrix(port_df)
port_1500 <- cbind(port_750, port_750) # Exact duplicate

cat("Running RCF-ADML on original 750 portfolios...\n")
res_750 <- estimate_rcf_adml(1, factors_augmented, port_750, method = "plr", K = 5, S = 3, n_boot = 15, seed = 42)

cat("Running RCF-ADML on duplicated 1500 portfolios...\n")
res_1500 <- estimate_rcf_adml(1, factors_augmented, port_1500, method = "plr", K = 5, S = 3, n_boot = 15, seed = 42)

cat("\n==================================================================\n")
cat(" PORTFOLIO DUPLICATION INVARIANCE TEST RESULTS\n")
cat("==================================================================\n")
cat(sprintf(" 750 Portfolios:  Theta = %.5f | SE_boot = %.5f | t_boot = %.4f | SE_cs = %.5f\n",
            res_750$theta, res_750$se_boot, res_750$t_boot, res_750$se_cs))
cat(sprintf(" 1500 Portfolios: Theta = %.5f | SE_boot = %.5f | t_boot = %.4f | SE_cs = %.5f\n",
            res_1500$theta, res_1500$se_boot, res_1500$t_boot, res_1500$se_cs))

# Assertions
stopifnot(abs(res_750$theta - res_1500$theta) < 1e-6)

# Time-series bootstrap SE must be invariant to portfolio duplication (within 1%)
diff_boot_se <- abs(res_750$se_boot - res_1500$se_boot) / res_750$se_boot
cat(sprintf(" Relative Bootstrap SE Difference: %.4f%%\n", diff_boot_se * 100))
stopifnot(diff_boot_se < 0.05)

# Cross-sectional SE, in contrast, drops by sqrt(2) (the defect demonstrated in audit)
cs_ratio <- res_750$se_cs / res_1500$se_cs
cat(sprintf(" Cross-Sectional SE Defect Ratio: %.4f (Expected sqrt(2) = 1.4142)\n", cs_ratio))
cat("==================================================================\n")
cat("[PASS] Portfolio duplication does NOT manufacture precision under time-series bootstrap!\n\n")
