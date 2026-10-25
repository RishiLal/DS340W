#################################################################################
###  rcf_blocks.R
###  Time-Series Reverse Cross-Fitting (RCF) Block Generator
###  Based on Ciganovic, D'Amario, and Tancioni (2026), Section 2.1
###  Corrected for:
###    - Paper-faithful larger-side selection rule for both even and odd K
###    - Exact block-count symmetry check for odd K central fold
###    - Comprehensive edge-case handling (small T, invalid inputs, disjointness)
#################################################################################

#' Generate K-fold Reverse Cross-Fitting (RCF) Block Partitions
#'
#' Partitions a time series of length T_total into K contiguous evaluation blocks B_1, ..., B_K.
#' For each evaluation block B_k:
#'   - If (k - 1) < (K - k): Right side has more blocks -> Auxiliary sample A_k is the future (time-reversed).
#'   - If (k - 1) > (K - k): Left side has more blocks -> Auxiliary sample A_k is the past (forward).
#'   - If (k - 1) == (K - k): Equal number of blocks on both sides (odd K central fold) ->
#'     Auxiliary sample A_k combines left past and right future symmetrically.
#'
#' Note: Sample moments (means, covariances) computed over A_k are permutation-invariant;
#' reversing indices (rev()) preserves chronological boundary indexing but does not alter
#' static moment estimates.
#'
#' @param T_total Total number of time-series observations (must be >= K * 2)
#' @param K Number of temporal folds (must be >= 2, default K = 5)
#' @return List of K elements, each containing:
#'   - fold: Fold index k
#'   - test_idx: Evaluation block indices B_k
#'   - aux_idx: Auxiliary sample indices A_k
#'   - train_idx: Auxiliary training sub-block indices
#'   - val_idx: Auxiliary validation sub-block indices (adjacent to B_k)
#'   - is_reversed: Logical indicating whether A_k is time-reversed
get_rcf_partitions <- function(T_total, K = 5) {
  # Input validation
  if (!is.numeric(T_total) || length(T_total) != 1 || !is.finite(T_total) || T_total != as.integer(T_total)) {
    stop("T_total must be a single positive integer.")
  }
  if (!is.numeric(K) || length(K) != 1 || !is.finite(K) || K != as.integer(K)) {
    stop("K must be a single integer >= 2.")
  }
  T_total <- as.integer(T_total)
  K <- as.integer(K)
  
  if (K < 2) {
    stop("K must be at least 2.")
  }
  if (T_total < K * 2) {
    stop(sprintf("T_total (%d) is too small for K (%d) folds. Need at least %d observations.", T_total, K, K * 2))
  }
  
  block_size <- floor(T_total / K)
  blocks <- list()
  
  for (k in 1:K) {
    start_t <- (k - 1) * block_size + 1
    end_t <- if (k == K) T_total else k * block_size
    blocks[[k]] <- start_t:end_t
  }
  
  partitions <- list()
  
  for (k in 1:K) {
    B_k <- blocks[[k]]
    num_left_blocks  <- k - 1
    num_right_blocks <- K - k
    
    left_indices  <- if (num_left_blocks > 0) unlist(blocks[1:(k - 1)]) else integer(0)
    right_indices <- if (num_right_blocks > 0) unlist(blocks[(k + 1):K]) else integer(0)
    
    if (num_left_blocks > num_right_blocks) {
      # Left side is larger -> Auxiliary sample is Left (past, Forward mode)
      A_k <- left_indices
      is_rev <- FALSE
      val_len <- max(1, min(floor(length(left_indices) * 0.25), length(B_k)))
      val_idx <- tail(A_k, val_len)
      train_idx <- setdiff(A_k, val_idx)
    } else if (num_right_blocks > num_left_blocks) {
      # Right side is larger -> Auxiliary sample is Right (future, Time-Reversed mode)
      A_k <- rev(right_indices)
      is_rev <- TRUE
      val_len <- max(1, min(floor(length(right_indices) * 0.25), length(B_k)))
      val_idx <- head(right_indices, val_len)
      train_idx <- setdiff(A_k, val_idx)
    } else {
      # Equal number of blocks on both sides (Symmetric central fold in odd-K)
      A_k <- c(left_indices, right_indices)
      is_rev <- FALSE
      total_aux_len <- length(A_k)
      val_len <- max(2, min(floor(total_aux_len * 0.20), length(B_k)))
      val_left <- floor(val_len / 2)
      val_right <- ceiling(val_len / 2)
      val_idx <- c(tail(left_indices, val_left), head(right_indices, val_right))
      train_idx <- setdiff(A_k, val_idx)
    }
    
    # Assert structural integrity and disjointness
    stopifnot(length(intersect(B_k, A_k)) == 0)
    stopifnot(length(intersect(train_idx, val_idx)) == 0)
    stopifnot(setequal(c(train_idx, val_idx), A_k))
    stopifnot(length(train_idx) > 0)
    stopifnot(length(val_idx) > 0)
    
    partitions[[k]] <- list(
      fold = k,
      test_idx = B_k,
      aux_idx = A_k,
      train_idx = train_idx,
      val_idx = val_idx,
      is_reversed = is_rev
    )
  }
  
  return(partitions)
}
