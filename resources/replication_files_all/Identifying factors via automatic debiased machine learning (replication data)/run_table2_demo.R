setwd('c:/Users/rishi/Desktop/DS340W/resources/replication_files_all/Identifying factors via automatic debiased machine learning (replication data)')
suppressPackageStartupMessages(library(AER))

allfactors <- read.table('factors.csv', header=TRUE, sep=',')
factors <- allfactors[, 3:ncol(allfactors)]

models <- list(
  FF3   = c(1, 21, 22),
  FF5   = c(1, 21, 22, 145, 146),
  FF6   = c(1, 21, 22, 145, 146, 34),
  ADML3 = c(98, 132, 25),
  ADML5 = c(98, 132, 25, 96, 13),
  ADML6 = c(98, 132, 25, 96, 13, 119),
  OLS6  = c(88, 51, 64, 72, 148, 34),
  DSL6  = c(49, 32, 147, 51, 21, 72)
)

results <- data.frame(
  Model = character(),
  Mean_Abs_Alpha = numeric(),
  p_less_0.01 = integer(),
  Num_Factors = integer(),
  stringsAsFactors = FALSE
)

for (mname in names(models)) {
  mcols <- models[[mname]]
  x <- as.matrix(factors[, mcols])
  
  num_unexplained <- 0
  sum_alpha <- 0
  tested_count <- 0
  
  for (j in 1:150) {
    if (j %in% mcols) next
    fit <- lm(factors[, j] ~ x + 1)
    fit_adj <- coeftest(fit, vcov. = NeweyWest(fit, lag = floor(4 * (498 / 100)^(2 / 9))))
    
    if (fit_adj[1, 4] < 0.01) {
      num_unexplained <- num_unexplained + 1
    }
    sum_alpha <- sum_alpha + abs(summary(fit)$coef[1, 1])
    tested_count <- tested_count + 1
  }
  
  results <- rbind(results, data.frame(
    Model = mname,
    Mean_Abs_Alpha = round((sum_alpha / tested_count) * 100, 2),
    p_less_0.01 = num_unexplained,
    Num_Factors = tested_count
  ))
}

cat('\n=======================================================\n')
cat(' REPLICATION OF TABLE 2 (Panel A: All Factors)\n')
cat(' Paper: Identifying Factors via ADML (Maasoumi et al.)\n')
cat('=======================================================\n')
print(results, row.names = FALSE)
cat('=======================================================\n\n')
