# ==============================================================================
# Script: split_adml_data.R
# Purpose: Clean, filter, and partition ADML replication data into:
#          - Training Set:   70% (348 months, 1976:07 - 2005:06)
#          - Testing Set:    20% (100 months, 2005:07 - 2013:10)
#          - Validation Set: 10% (50 months,  2013:11 - 2017:12)
# ==============================================================================

library(matlab)

rep_dir <- file.path("resources", "replication_files_all", 
                     "Identifying factors via automatic debiased machine learning (replication data)")
out_dir <- "data_splits"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

cat("==============================================================================\n")
cat("ADML Replication Data Split: 70% Train / 20% Test / 10% Validation\n")
cat("==============================================================================\n")

# 1. Load factors.csv
allfactors <- read.table(file.path(rep_dir, "factors.csv"), header = TRUE, sep = ",")
date <- allfactors[, 1]
rf <- allfactors[, 2]
factors <- allfactors[, 3:ncol(allfactors)]
T_total <- length(date)
P <- ncol(factors)

cat(sprintf("Loaded factors: %d months x %d factors (Time: %d to %d)\n", 
            T_total, P, date[1], date[T_total]))

# 2. Define split indices
n_train <- floor(0.70 * T_total)       # 348
n_test  <- 100                          # 100 (20.08%)
n_val   <- T_total - n_train - n_test   # 50 (10.04%)

idx_train <- 1:n_train
idx_test  <- (n_train + 1):(n_train + n_test)
idx_val   <- (n_train + n_test + 1):T_total

cat(sprintf("Splits: Train = 1:%d (%d), Test = %d:%d (%d), Val = %d:%d (%d)\n",
            max(idx_train), length(idx_train),
            min(idx_test), max(idx_test), length(idx_test),
            min(idx_val), max(idx_val), length(idx_val)))

# 3. Load & Clean Test Assets
# 3.1 Primary Test Assets: port_3x2 (bivariate sorted)
port_3x2_raw <- read.table(file.path(rep_dir, "port_3x2.csv"), sep = ",")
port_3x2 <- as.matrix(port_3x2_raw[, -1])
one_3x2  <- matrix(rep(1, ncol(port_3x2)), nrow = 1, ncol = ncol(port_3x2))
port_3x2_excess <- port_3x2 - rf %*% one_3x2

port_3x2_id <- read.table(file.path(rep_dir, "port_3x2_id.csv"), header = TRUE, sep = ",")
kk <- 10
include_3x2 <- t(find(port_3x2_id[, 4] >= kk))
port_3x2b <- matrix(nrow = T_total, ncol = 0)
for (i in 1:P) {
  if (i %in% include_3x2) {
    port_3x2b <- cbind(port_3x2b, port_3x2_excess[, (i * 6 - 5):(i * 6)])
  }
}

# 3.2 Robustness: port_5x5
port_5x5_raw <- read.table(file.path(rep_dir, "port_5x5.csv"), sep = ",")
port_5x5 <- as.matrix(port_5x5_raw[, -1])
one_5x5  <- matrix(rep(1, ncol(port_5x5)), nrow = 1, ncol = ncol(port_5x5))
port_5x5_excess <- port_5x5 - rf %*% one_5x5

port_5x5_id <- read.table(file.path(rep_dir, "port_5x5_id.csv"), header = TRUE, sep = ",")
include_5x5 <- t(find(port_5x5_id[, 4] >= kk))
port_5x5b <- matrix(nrow = T_total, ncol = 0)
for (i in 1:P) {
  if (i %in% include_5x5) {
    port_5x5b <- cbind(port_5x5b, port_5x5_excess[, (i * 25 - 24):(i * 25)])
  }
}

# 3.3 Robustness: port_202
port_202_raw <- read.table(file.path(rep_dir, "port202.csv"), sep = ",")
port_202 <- as.matrix(port_202_raw[, -1]) / 100
one_202  <- matrix(rep(1, ncol(port_202)), nrow = 1, ncol = ncol(port_202))
port_202_excess <- port_202 - rf %*% one_202 # 498 x 202

# 4. Partition into Train, Test, and Validation
data_splits <- list(
  train = list(
    date = date[idx_train],
    rf = rf[idx_train],
    factors = factors[idx_train, ],
    port_3x2 = port_3x2b[idx_train, ],
    port_5x5 = port_5x5b[idx_train, ],
    port_202 = port_202_excess[idx_train, ]
  ),
  test = list(
    date = date[idx_test],
    rf = rf[idx_test],
    factors = factors[idx_test, ],
    port_3x2 = port_3x2b[idx_test, ],
    port_5x5 = port_5x5b[idx_test, ],
    port_202 = port_202_excess[idx_test, ]
  ),
  val = list(
    date = date[idx_val],
    rf = rf[idx_val],
    factors = factors[idx_val, ],
    port_3x2 = port_3x2b[idx_val, ],
    port_5x5 = port_5x5b[idx_val, ],
    port_202 = port_202_excess[idx_val, ]
  )
)

save(data_splits, file = file.path(out_dir, "adml_data_splits_70_20_10.RData"))
cat(sprintf("Saved RData splits to %s/adml_data_splits_70_20_10.RData\n", out_dir))
cat("Done!\n")
