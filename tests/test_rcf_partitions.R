#################################################################################
###  test_rcf_partitions.R
###  Targeted Tests for Reverse Cross-Fitting Block Geometry and Symmetry
###  DS340W Independent Verification Suite
#################################################################################

source("scripts/rcf_blocks.R")

cat("Running comprehensive RCF partition validation tests...\n")

test_rcf_properties <- function(T_total, K) {
  parts <- get_rcf_partitions(T_total, K = K)
  stopifnot(length(parts) == K)
  
  # 1. Check all evaluation blocks partition 1:T_total without gaps or overlap
  all_test_idx <- unlist(lapply(parts, function(p) p$test_idx))
  stopifnot(length(all_test_idx) == T_total)
  stopifnot(setequal(all_test_idx, 1:T_total))
  
  # 2. Check disjointness and subsets for each fold
  for (k in 1:K) {
    p <- parts[[k]]
    # Evaluation block disjoint from auxiliary sample
    stopifnot(length(intersect(p$test_idx, p$aux_idx)) == 0)
    # Train and val disjoint
    stopifnot(length(intersect(p$train_idx, p$val_idx)) == 0)
    # Train + val = aux
    stopifnot(setequal(c(p$train_idx, p$val_idx), p$aux_idx))
    # Subsets are non-empty
    stopifnot(length(p$train_idx) > 0)
    stopifnot(length(p$val_idx) > 0)
  }
  
  # 3. Check symmetry and larger-side behavior
  if (K %% 2 == 1) {
    # Odd K: Middle fold (K+1)/2 must combine past and future
    mid_k <- (K + 1) / 2
    p_mid <- parts[[mid_k]]
    has_past <- any(p_mid$aux_idx < min(p_mid$test_idx))
    has_future <- any(p_mid$aux_idx > max(p_mid$test_idx))
    stopifnot(has_past && has_future)
    stopifnot(!p_mid$is_reversed)
  } else {
    # Even K: Every fold must follow larger side (no symmetric combined fold)
    for (k in 1:K) {
      p <- parts[[k]]
      num_left <- k - 1
      num_right <- K - k
      if (num_left > num_right) {
        # Past side larger -> all aux must be strictly in the past
        stopifnot(all(p$aux_idx < min(p$test_idx)))
        stopifnot(!p$is_reversed)
      } else if (num_right > num_left) {
        # Future side larger -> all aux must be strictly in the future
        stopifnot(all(p$aux_idx > max(p$test_idx)))
        stopifnot(p$is_reversed)
      }
    }
  }
}

# Test across various lengths and fold counts
test_rcf_properties(160, K = 4)  # Even K, divisible
test_rcf_properties(165, K = 4)  # Even K, non-divisible
test_rcf_properties(347, K = 5)  # Odd K, non-divisible
test_rcf_properties(348, K = 5)  # Odd K, non-divisible
test_rcf_properties(350, K = 5)  # Odd K, divisible
test_rcf_properties(100, K = 2)  # Minimal K = 2
test_rcf_properties(100, K = 3)  # Minimal odd K = 3
test_rcf_properties(498, K = 6)  # Even K = 6

# Test error handling on invalid inputs
expect_error <- function(expr) {
  res <- tryCatch(expr, error = function(e) TRUE)
  if (!isTRUE(res)) stop("Expected error was not raised!")
}

expect_error(get_rcf_partitions(10, K = 1))   # K < 2
expect_error(get_rcf_partitions(5, K = 5))    # T < K * 2
expect_error(get_rcf_partitions(-50, K = 5))  # Negative T
expect_error(get_rcf_partitions(100.5, K = 5))# Non-integer T
expect_error(get_rcf_partitions(100, K = 2.5))# Non-integer K

cat("[PASS] RCF partition geometry verified for odd K, even K, divisible/non-divisible T, and edge cases.\n")
