#################################################################################
###  run_rcf_adml_benchmark.R
###  Driver Script: Benchmark Evaluation of Novel RCF-ADML vs Baseline
#################################################################################

rm(list = ls())

cat("=================================================================\n")
cat(" DS340W CAPSTONE: RCF-ADML FACTOR PRICING BENCHMARK\n")
cat(" Time-Series Robust Automatic Debiased Machine Learning\n")
cat(" Combining Maasoumi et al. (Parent) & Ciganovic et al. (Paper 2)\n")
cat("=================================================================\n\n")

# Set working directory to replication files for data import
setwd("C:/Users/rishi/Desktop/DS340W/resources/replication_files_all/Identifying factors via automatic debiased machine learning (replication data)")

cat("Loading parent paper factor library and test assets...\n")
source("dataimport.R")

# Source our novel methodology modules
source("C:/Users/rishi/Desktop/DS340W/scripts/rcf_adml.R")

# Selected Benchmark Factors (Fama-French 5 + Carhart Momentum)
# Indices in factor library:
# 1: Mkt-RF (Market Excess Return)
# 21: SMB (Size)
# 22: HML (Value)
# 145: RMW (Operating Profitability)
# 146: CMA (Investment)
# 34: MOM (Momentum)
benchmark_indices <- c(1, 21, 22, 145, 146, 34)
factor_labels     <- c("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM")

cat(sprintf("Running RCF-ADML across %d benchmark risk factors (K = 5 folds, Goldilocks tuning S = 3)...\n\n", length(benchmark_indices)))

results_table <- data.frame(
  Factor_ID = integer(),
  Factor_Name = character(),
  RCF_Theta = numeric(),
  RCF_SE = numeric(),
  RCF_t_stat = numeric(),
  P_Value = numeric(),
  Significant_5pct = character(),
  stringsAsFactors = FALSE
)

start_time <- Sys.time()

for (idx in seq_along(benchmark_indices)) {
  f_idx <- benchmark_indices[idx]
  f_name <- factor_labels[idx]
  
  cat(sprintf("[%d/%d] Evaluating Factor #%d: %s ... ", idx, length(benchmark_indices), f_idx, f_name))
  t0 <- Sys.time()
  
  # Run RCF-ADML
  est <- estimate_rcf_adml(
    factor_idx = f_idx,
    factors_matrix = as.matrix(factors),
    port_matrix = as.matrix(port_3x2b),
    K = 5,
    S = 3
  )
  
  t1 <- Sys.time()
  dur <- round(as.numeric(difftime(t1, t0, units = "secs")), 1)
  
  p_val <- 2 * (1 - pnorm(abs(est$t_stat)))
  sig   <- if (p_val < 0.05) "YES (*)" else "NO"
  
  cat(sprintf("Done in %ss | Theta: %.4f | t-stat: %.2f | Sig: %s\n", 
              dur, est$theta, est$t_stat, sig))
  
  results_table <- rbind(results_table, data.frame(
    Factor_ID = f_idx,
    Factor_Name = f_name,
    RCF_Theta = round(est$theta, 4),
    RCF_SE = round(est$se, 4),
    RCF_t_stat = round(est$t_stat, 2),
    P_Value = round(p_val, 4),
    Significant_5pct = sig
  ))
}

end_time <- Sys.time()
total_dur <- round(as.numeric(difftime(end_time, start_time, units = "secs")), 1)

cat("\n=================================================================\n")
cat(sprintf(" BENCHMARK COMPLETED IN %s SECONDS\n", total_dur))
cat("=================================================================\n")
print(results_table, row.names = FALSE)
cat("=================================================================\n\n")

# Save results
out_path <- "C:/Users/rishi/Desktop/DS340W/scripts/rcf_adml_benchmark_results.csv"
write.csv(results_table, file = out_path, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", out_path))
