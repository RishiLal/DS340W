# Register broom's glmnet tidier before the historical driver calls tidy().
# Uses the same package installation and the same native checkpoint runner.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1],winslash='/')
.libPaths(c(file.path(root,'env','R-library'), .Library))
Sys.setlocale("LC_COLLATE", "C")
stopifnot(requireNamespace('broom',quietly=TRUE))

# Compatibility shim: dplyr 1.1+ requires scalar summary values; patch plot_over_time when 0 - Functions.R is sourced
base_ns <- asNamespace("base")
orig_source <- base_ns$source
unlockBinding("source", base_ns)
base_ns$source <- function(file, ...) {
  ans <- orig_source(file, ...)
  if (is.character(file) && grepl("0 - Functions\\.R", file)) {
    args_list <- list(...)
    loc <- if (!is.null(args_list$local) && is.environment(args_list$local)) args_list$local else parent.frame()
    if (exists("plot_over_time", envir = loc, inherits = FALSE)) {
      f <- get("plot_over_time", envir = loc)
      body_text <- deparse(body(f))
      body_text <- gsub("mkt_vw_exc = unique(mkt_vw_exc)", "mkt_vw_exc = mean(mkt_vw_exc)", body_text, fixed = TRUE)
      body(f) <- parse(text = body_text)
      assign("plot_over_time", f, envir = loc)
      cat("Compatible repair: registered dplyr 1.1+ compatibility for plot_over_time\n")
    }
  }
  return(invisible(ans))
}
lockBinding("source", base_ns)

base::source(file.path(root,'tools','run_checkpointed.R'),local=.GlobalEnv)
