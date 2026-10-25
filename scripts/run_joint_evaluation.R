#################################################################################
###  run_joint_evaluation.R
###  The Three-Paper Synthesis Factor Evaluation Engine
###  Audited & Corrected for:
###    - Strict Representation Isolation: Evaluates strictly on Inference Split
###      (T_inf = 167 return months: August 1991 to June 2005)
###    - Strict Date Matching: Exact date equality across factors, test assets, and GAN
###    - Reverse Cross-Fitting (RCF) with Goldilocks zone stability tuning
###    - Stationary time-series block bootstrap standard errors
###    - Full numerical precision storage: float thetas, SEs, t-stats, numeric p-values
#################################################################################

rm(list = ls())

Sys.setenv(OMP_NUM_THREADS = "4")
set.seed(42)

cat("=================================================================\n")
cat(" DS340W CAPSTONE: RIGOROUS THREE-PAPER SYNTHESIS BENCHMARK\n")
cat(" Evaluating Factor Pricing Contribution Under Corrected Engine\n")
cat(" Sample: Designated Econometric Inference Split (T_inf = 167 months)\n")
cat("=================================================================\n\n")

suppressPackageStartupMessages({
  library(glmnet)
  library(nnet)
  library(MASS)
})

# Project-relative path resolution
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

data_dir <- file.path(PROJECT_DIR, "data_splits")

# 1. Load Data Splits strictly matching the Inference Period (167 months)
cat("Loading training data and extracted deep representations for inference sample...\n")
factors_train <- read.csv(file.path(data_dir, "factors_train_70.csv"))
port_train    <- read.csv(file.path(data_dir, "port_3x2_excess_train_70.csv"))
gan_inf       <- read.csv(file.path(data_dir, "gan_factors_inf_167.csv"))

# Verify exact chronological alignment: 167 months from 19910830 to 20050630
dates_eval <- gan_inf$Date
stopifnot(length(dates_eval) == 167)
stopifnot(dates_eval[1] == 19910830)
stopifnot(dates_eval[167] == 20050630)

f_idx_eval <- which(factors_train$Date %in% dates_eval)
p_idx_eval <- which(port_train$Date %in% dates_eval)

stopifnot(length(f_idx_eval) == 167)
stopifnot(length(p_idx_eval) == 167)
stopifnot(all(factors_train$Date[f_idx_eval] == dates_eval))
stopifnot(all(port_train$Date[p_idx_eval] == dates_eval))

factor_names <- colnames(factors_train)[!(colnames(factors_train) %in% c("Date", "RF"))]
factors_mat  <- as.matrix(factors_train[f_idx_eval, factor_names])
port_mat     <- as.matrix(port_train[p_idx_eval, colnames(port_train) != "Date"])

# Deep controls extracted using frozen representation PCA
sdf_ret <- as.matrix(gan_inf$SDF_Return)
colnames(sdf_ret) <- "SDF_Return"
pcs_mat <- as.matrix(gan_inf[, c("GAN_F1", "GAN_F2", "GAN_F3")])

# Augmented control matrices
factors_pcs  <- cbind(factors_mat, pcs_mat)
factors_sdf  <- cbind(factors_mat, sdf_ret)
factors_full <- cbind(factors_mat, sdf_ret, pcs_mat)

T_obs <- nrow(factors_mat)
N_ports <- ncol(port_mat)
P_factors <- ncol(factors_mat)

cat(sprintf("  Evaluation Sample: %d months (%s to %s)\n", T_obs, min(dates_eval), max(dates_eval)))
cat(sprintf("  Test Assets (N):   %d portfolios\n", N_ports))
cat(sprintf("  Published Factors: %d\n", P_factors))
cat(sprintf("  Deep GAN Controls: %d (SDF_Return + 3 Latent PCs)\n\n", ncol(factors_full) - P_factors))

# 2. Benchmark Factors to Evaluate
test_factors <- c("MktRf", "SMB", "HML", "RMW", "CMA", "UMD", "ol", "ndf")
factor_labels <- c("Market", "Size (SMB)", "Value (HML)", "Profitability (RMW)", 
                   "Investment (CMA)", "Momentum (UMD)", "Operating Leverage (ol)", 
                   "Net Debt Finance (ndf)")

results_df <- data.frame(
  Factor_Code = character(),
  Factor_Name = character(),
  # Baseline (Linear 150 factors)
  Base_Theta = numeric(),
  Base_SE_Boot = numeric(),
  Base_t_Boot = numeric(),
  Base_p_val_num = numeric(),
  Base_log_p_val = numeric(),
  Base_p_val_str = character(),
  # Latent PCs Only
  PCs_Theta = numeric(),
  PCs_SE_Boot = numeric(),
  PCs_t_Boot = numeric(),
  PCs_p_val_num = numeric(),
  # SDF Return Only
  SDF_Theta = numeric(),
  SDF_SE_Boot = numeric(),
  SDF_t_Boot = numeric(),
  SDF_p_val_num = numeric(),
  # Full Deep Augmented (Papers 1+2+3)
  Deep_Theta = numeric(),
  Deep_SE_Boot = numeric(),
  Deep_t_Boot = numeric(),
  Deep_p_val_num = numeric(),
  Deep_log_p_val = numeric(),
  Deep_p_val_str = character(),
  # Diagnostics
  Deep_t_CS_Defective = numeric(),
  Significant_5pct = character(),
  Execution_Status = character(),
  Fold_Status = character(),
  Inference_Valid = character(),
  stringsAsFactors = FALSE
)

format_p <- function(p_val, log_p) {
  if (is.finite(p_val) && p_val >= 0.0001) {
    return(sprintf("%.4f", p_val))
  } else {
    exp_val <- log_p / log(10)
    return(sprintf("< 1e%d", ceiling(exp_val)))
  }
}

start_time <- Sys.time()

for (i in seq_along(test_factors)) {
  f_code <- test_factors[i]
  f_name <- factor_labels[i]
  
  f_idx <- which(factor_names == f_code)
  if (length(f_idx) == 0) {
    cat(sprintf("Skipping %s (not found)\n", f_code))
    next
  }
  
  cat(sprintf("[%d/%d] Evaluating %s (%s)...\n", i, length(test_factors), f_code, f_name))
  
  # Spec 1: Baseline (Published 150 factors only)
  est_base <- estimate_rcf_adml(
    factor_idx = f_idx,
    factors_matrix = factors_mat,
    port_matrix = port_mat,
    method = "plr",
    K = 5, S = 3, n_boot = 50, block_len = 12, seed = 42 + i
  )
  
  # Spec 2: Ablation with Latent PCs only
  est_pcs <- estimate_rcf_adml(
    factor_idx = f_idx,
    factors_matrix = factors_pcs,
    port_matrix = port_mat,
    method = "plr",
    K = 5, S = 3, n_boot = 30, block_len = 12, seed = 42 + i
  )
  
  # Spec 3: Ablation with SDF Return only
  est_sdf <- estimate_rcf_adml(
    factor_idx = f_idx,
    factors_matrix = factors_sdf,
    port_matrix = port_mat,
    method = "plr",
    K = 5, S = 3, n_boot = 30, block_len = 12, seed = 42 + i
  )
  
  # Spec 4: Full Deep Augmented (150 factors + SDF_Return + 3 Latent PCs)
  est_deep <- estimate_rcf_adml(
    factor_idx = f_idx,
    factors_matrix = factors_full,
    port_matrix = port_mat,
    method = "plr",
    K = 5, S = 3, n_boot = 50, block_len = 12, seed = 42 + i
  )
  
  base_p_str <- format_p(est_base$p_boot, est_base$log_p_boot)
  deep_p_str <- format_p(est_deep$p_boot, est_deep$log_p_boot)
  
  sig_status <- if (is.finite(est_deep$p_boot) && est_deep$p_boot < 0.05) "YES (*)" else "NO"
  
  cat(sprintf("    Base: Theta=%+.4f, SE_boot=%.4f, t_boot=%+.2f (p=%s)\n", 
              est_base$theta, est_base$se_boot, est_base$t_boot, base_p_str))
  cat(sprintf("    Deep: Theta=%+.4f, SE_boot=%.4f, t_boot=%+.2f (p=%s) [Defective t_cs=%+.2f] -> Sig: %s\n\n", 
              est_deep$theta, est_deep$se_boot, est_deep$t_boot, deep_p_str, est_deep$t_cs, sig_status))
  
  results_df <- rbind(results_df, data.frame(
    Factor_Code = f_code,
    Factor_Name = f_name,
    # Base
    Base_Theta = est_base$theta,
    Base_SE_Boot = est_base$se_boot,
    Base_t_Boot = est_base$t_boot,
    Base_p_val_num = est_base$p_boot,
    Base_log_p_val = est_base$log_p_boot,
    Base_p_val_str = base_p_str,
    # PCs
    PCs_Theta = est_pcs$theta,
    PCs_SE_Boot = est_pcs$se_boot,
    PCs_t_Boot = est_pcs$t_boot,
    PCs_p_val_num = est_pcs$p_boot,
    # SDF
    SDF_Theta = est_sdf$theta,
    SDF_SE_Boot = est_sdf$se_boot,
    SDF_t_Boot = est_sdf$t_boot,
    SDF_p_val_num = est_sdf$p_boot,
    # Deep Full
    Deep_Theta = est_deep$theta,
    Deep_SE_Boot = est_deep$se_boot,
    Deep_t_Boot = est_deep$t_boot,
    Deep_p_val_num = est_deep$p_boot,
    Deep_log_p_val = est_deep$log_p_boot,
    Deep_p_val_str = deep_p_str,
    # Diagnostics
    Deep_t_CS_Defective = est_deep$t_cs,
    Significant_5pct = sig_status,
    Execution_Status = est_deep$execution_status,
    Fold_Status = est_deep$fold_status,
    Inference_Valid = if (est_deep$inference_valid) "TRUE" else "FALSE",
    stringsAsFactors = FALSE
  ))
}

end_time <- Sys.time()
dur <- round(as.numeric(difftime(end_time, start_time, units = "secs")), 1)

cat("=================================================================\n")
cat(sprintf(" RIGOROUS JOINT EVALUATION FINISHED IN %s SECONDS\n", dur))
cat("=================================================================\n\n")

print(results_df[, c("Factor_Code", "Base_Theta", "Base_SE_Boot", "Base_t_Boot", 
                     "Deep_Theta", "Deep_SE_Boot", "Deep_t_Boot", "Deep_p_val_str", "Significant_5pct")],
      row.names = FALSE)

out_csv <- file.path(PROJECT_DIR, "scripts", "joint_evaluation_results.csv")
write.csv(results_df, file = out_csv, row.names = FALSE)
cat(sprintf("\nFull numerical precision results saved to: %s\n", out_csv))
