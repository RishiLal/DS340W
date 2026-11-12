# Rerender figures with explicit print calls for ggplot objects inside the saving block
# Reporting repair: Author code placed bare ggplot expressions inside if (save_figures) { ... }
# which does not auto-print to the active PDF device, resulting in 0-page PDFs.
# This script re-renders the freshly checkpointed objects without refitting any model.

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
root <- normalizePath(args[1], winslash='/')
run_dir <- normalizePath(args[2], winslash='/')
setwd(run_dir)

.libPaths(c(file.path(root, "env/R-library"), .Library))
Sys.setlocale("LC_COLLATE", "C")

suppressPackageStartupMessages({
  library(directlabels)
  library(xtable)
  library(zeallot)
  library(dendextend)
  library(RColorBrewer)
  library(rsample)
  library(lubridate)
  library(tidyverse)
  library(data.table)
  library(broom)
})
options(dplyr.summarise.inform = FALSE)

session <- new.env(parent = .GlobalEnv)

# Setup initial settings and variables by evaluating main.R setup lines (1 to 102)
main_lines <- readLines("main.R")
eval(parse(text = main_lines[1:102]), envir = session)

# 1. Source 0 - Functions.R with dplyr 1.1+ patch for plot_over_time
cat("Sourcing 0 - Functions.R...\n")
base::source("0 - Functions.R", local = session)
if (exists("plot_over_time", envir = session, inherits = FALSE)) {
  f <- get("plot_over_time", envir = session)
  body_text <- deparse(body(f))
  body_text <- gsub("mkt_vw_exc = unique(mkt_vw_exc)", "mkt_vw_exc = mean(mkt_vw_exc)", body_text, fixed = TRUE)
  body(f) <- parse(text = body_text)
  assign("plot_over_time", f, envir = session)
}

# 2. Source Stage 1 and Stage 2
cat("Sourcing 1 - Prepare Data.R...\n")
base::source("1 - Prepare Data.R", local = session)
cat("Sourcing 2 - Determine Clusters.R...\n")
base::source("2 - Determine Clusters.R", local = session)

# 3. Restore all 31 Analysis checkpoints
cat("Restoring 31 checkpoints of Stage 3...\n")
for (i in 1:31) {
  cp <- sprintf("checkpoints/analysis_%03d.rds", i)
  if (file.exists(cp)) {
    restored <- readRDS(cp)
    list2env(restored$bindings, session)
  } else {
    stop(paste("Missing checkpoint:", cp))
  }
}
cat("All checkpoints restored successfully.\n")

# 4. Execute lines 1-135 of 4 - Output.R to construct output$figures
cat("Constructing output$figures...\n")
output_code_lines <- readLines("4 - Output.R")
# Lines 1 to 135 build the figures list
eval(parse(text = output_code_lines[1:135]), envir = session)
cat("output$figures constructed successfully.\n")

# 5. Define output_fig and save all 37 figures with explicit print() calls
output_path <- "Figures"
h <- 5
w <- 9
format <- "pdf"

output_fig <- function(path, name, format, width, height) {
  path <- paste0(path, "/", name)
  pdf(str_c(path, ".pdf"), width = width, height = height)
}

with(session, {
  # 1. lit_comp
  output_fig(path=output_path, name = "lit_comp", format = format, width = w + 1, height = h)
  print(output$figures$lit_comp)
  dev.off()
  
  # 2. mt_factors
  output_fig(path=output_path, name = "mt_factors", format = format, width = w, height = h)
  print(output$figures$mt_factors + theme(text = element_text(size = 13), axis.text.x = element_blank()))
  dev.off()
  
  # 3. ci_many_fcts
  output_fig(path=output_path, name = "ci_many_fcts", format = format, width = w, height = h)
  print(output$figures$ci_many_fcts + theme(axis.text.y = element_text(size = 13), text = element_text(size = 13)))
  dev.off()
  
  # 4. mt_summary
  output_fig(path=output_path, name = "mt_summary", format = format, width = w, height = h)
  print(output$figures$mt_summary + 
    theme(
      text = element_text(size = 13),
      axis.title.x = element_blank(), 
      axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, size = 9)
    ))
  dev.off()
  
  # 5. gl_by_factor
  output_fig(path=output_path, name = "gl_by_factor", format = format, width = w, height = h)
  print(output$figures$gl_by_factor + theme(text = element_text(size = 13), legend.position = "right", axis.text.x = element_text(size = 5)))
  dev.off()
  
  # 6. gl_by_cluster
  output_fig(path=output_path, name = "gl_by_cluster", format = format, width = w, height = h)
  print(output$figures$gl_by_cluster + theme(text = element_text(size = 13)))
  dev.off()
  
  # 7. tpf
  output_fig(path=output_path, name = "tpf", format = format, width = w, height = h)
  print(output$figures$tpf + theme(text = element_text(size = 13), axis.text.x = element_text(size = 13), axis.text.y = element_text(size = 13)))
  dev.off()
  
  # 8. tpf_regions
  output_fig(path=output_path, name = "tpf_regions", format = format, width = w, height = h)
  print(output$figures$tpf_regions + theme(text = element_text(size = 13), axis.text.x = element_text(size = 13), axis.text.y = element_text(size = 13)))
  dev.off()
  
  # 9. tpf_size
  output_fig(path=output_path, name = "tpf_size", format = format, width = w, height = h)
  print(output$figures$tpf_size + theme(
    text = element_text(size = 13), 
    axis.text.x = element_text(size = 13), 
    axis.text.y = element_text(size = 11)))
  dev.off()
  
  # 10. over_time
  output_fig(path=output_path, name = "over_time", format = format, width = w, height = h)
  print(output$figures$overtime)
  dev.off()
  
  # 11. over_time_flex
  output_fig(path=output_path, name = "over_time_flex", format = format, width = w, height = h)
  print(output$figures$overtime_flex + theme(text = element_text(size = 13)))
  dev.off()
  
  # 12. over_time_flex_taus
  output_fig(path=output_path, name = "over_time_flex_taus", format = format, width = w, height = h)
  print(output$figures$overtime_flex_taus + theme(text = element_text(size = 13), legend.text = element_text(size = 12)))
  dev.off()
  
  # 13. size_overall
  output_fig(path=output_path, name = "size_overall", format = format, width = w, height = h)
  print(output$figures$size_overall + theme(
    axis.title.x = element_text(size = 17),
    axis.text.x = element_text(size = 17),
    axis.text.y = element_text(size = 17)))
  dev.off()
  
  # 14. size_clusters
  output_fig(path=output_path, name = "size_clusters", format = format, width = w, height = h+1)
  print(output$figures$size_clusters)
  dev.off()
  
  # 15. us_hcl
  output_fig(path=output_path, name = "us_hcl", format = format, width = w, height = h*2+1.5)
  output$figures$hcl_us()
  dev.off()
  
  # 16. us_hcl_pairwise_cor
  output_fig(path=output_path, name = "us_hcl_pairwise_cor", format = format, width = w-4, height = h)
  output$figures$hcl_us_val()
  dev.off()
  
  # 17. model_fdr
  output_fig(path=output_path, name = "model_fdr", format = format, width = w, height = h)
  print(output$figures$model_fdr + theme(text = element_text(size = 13)))
  dev.off()
  
  # 18. sim_fdr
  output_fig(path=output_path, name = "sim_fdr", format = format, width = w, height = h)
  print(output$figures$sim_fdr)
  dev.off()
  
  # 19. world_vs_us
  output_fig(path=output_path, name = "world_vs_us", format = format, width = h, height = h)
  print(output$figures$world_vs_us + theme(text = element_text(size = 13)))
  dev.off()
  
  # 20. is_pre_post
  output_fig(path=output_path, name = "is_pre_post", format = format, width = w/3, height = w/3)
  print(output$figures$is_oos + theme(
    text = element_text(size = 12), 
    plot.title = element_text(size = 10, vjust = -2), 
    plot.subtitle = element_text(size = 8, vjust = 0),
    plot.margin = unit(c(0,0,0,0), "cm")
  ))
  dev.off()
  
  # 21. is_post
  output_fig(path=output_path, name = "is_post", format = format, width = w/3, height = w/3)
  print(output$figures$is_post + theme(
    text = element_text(size = 12), 
    plot.title = element_text(size = 10, vjust = -2), 
    plot.subtitle = element_text(size = 8, vjust = 0),
    plot.margin = unit(c(0,0,0,0), "cm")
  ))
  dev.off()
  
  # 22. is_pre
  output_fig(path=output_path, name = "is_pre", format = format, width = w/3, height = w/3)
  print(output$figures$is_pre + theme(
    text = element_text(size = 12), 
    plot.title = element_text(size = 10, vjust = -2), 
    plot.subtitle = element_text(size = 8, vjust = 0),
    plot.margin = unit(c(0,0,0,0), "cm")
  ))
  dev.off()
  
  # 23. is_pre_post_quad
  output_fig(path=output_path, name = "is_pre_post_quad", format = format, width = w/3, height = w/3)
  print(output$figures$is_oos_quad + theme(
    text = element_text(size = 12), 
    plot.title = element_text(size = 10, vjust = -2), 
    plot.subtitle = element_text(size = 8, vjust = 0),
    plot.margin = unit(c(0,0,0,0), "cm")
  ))
  dev.off()
  
  # 24. is_post_quad
  output_fig(path=output_path, name = "is_post_quad", format = format, width = w/3, height = w/3)
  print(output$figures$is_post_quad + theme(
    text = element_text(size = 12), 
    plot.title = element_text(size = 10, vjust = -2), 
    plot.subtitle = element_text(size = 8, vjust = 0),
    plot.margin = unit(c(0,0,0,0), "cm")
  ))
  dev.off()
  
  # 25. is_pre_quad
  output_fig(path=output_path, name = "is_pre_quad", format = format, width = w/3, height = w/3)
  print(output$figures$is_pre_quad + theme(
    text = element_text(size = 12), 
    plot.title = element_text(size = 10, vjust = -2), 
    plot.subtitle = element_text(size = 8, vjust = 0),
    plot.margin = unit(c(0,0,0,0), "cm")
  ))
  dev.off()
  
  # 26. effect_regions
  output_fig(path=output_path, name = "effect_regions", format = format, width = w, height = h)
  print(output$figures$effect_regions + theme(
    axis.text.y = element_text(size = 13), 
    strip.text.x = element_text(size = 13),
    axis.text.x = element_text(size = 12),
    axis.title.x = element_text(size = 13)
  ))
  dev.off()
  
  # 27. effect_size
  output_fig(path=output_path, name = "effect_size", format = format, width = w, height = h)
  print(output$figures$effect_size + theme(
    axis.text.y = element_text(size = 13), 
    strip.text.x = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.title.x = element_text(size = 13)
  ))
  dev.off()
  
  # 28. repl_cluster_us
  output_fig(path=output_path, name = "repl_cluster_us", format = format, width = w, height = h)
  print(output$figures$repl_cluster_us + theme(
    axis.text.y = element_text(size = 13), 
    axis.text.x = element_text(size = 14), 
    text = element_text(size = 14)))
  dev.off()
  
  # 29. sim_harvey
  output_fig(path=output_path, name = "sim_harvey", format = format, width = w, height = h)
  print(output$figures$sim_harvey + theme(
    text = element_text(size = 13),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.text = element_text(size = 12)
  ))
  dev.off()
  
  # 30. sim_harvey_robustness
  output_fig(path=output_path, name = "sim_harvey_robustness", format = format, width = w, height = h)
  print(output$figures$sim_harvey_robustness + theme(
    text = element_text(size = 13),
    axis.text.x = element_text(size = 11),
    axis.text.y = element_text(size = 12),
    legend.text = element_text(size = 12)
  ))
  dev.off()
  
  # 31. cluster_plus_market
  output_fig(path=output_path, name = "cluster_plus_market", format = format, width = w, height = h)
  print(output$figures$tpf_factors_one_cluster)
  dev.off()
  
  # 32. excl_cluster
  output_fig(path=output_path, name = "excl_cluster", format = format, width = w, height = h)
  print(output$figures$tpf_factors_excl_one)
  dev.off()
  
  # 33. factor_tpf_imp
  output_fig(path=output_path, name = "factor_tpf_imp", format = format, width = w, height = h*1.5)
  print(output$figures$tpf_factors_imp + theme(axis.text.y = element_text(size = 6)))
  dev.off()
  
  # 34. factor_tpf_imp_cluster
  output_fig(path=output_path, name = "factor_tpf_imp_cluster", format = format, width = w, height = h*1.5)
  print(output$figures$tpf_factors_imp_cluster + theme(axis.text.y = element_text(size = 6)))
  dev.off()
  
  # 35. tpf_evolution
  output_fig(path=output_path, name = "tpf_evolution", format = format, width = w, height = h*0.8)
  print(output$figures$tpf_evolution)
  dev.off()
  
  # 36. marg_sig_oos
  output_fig(path=output_path, name = "marg_sig_oos", format = format, width = w, height = h*2/3)
  print(output$figures$marg_sig_oos)
  dev.off()
  
  # 37. eb_sig_oos
  output_fig(path=output_path, name = "eb_sig_oos", format = format, width = w, height = h*2/3)
  print(output$figures$eb_sig_oos)
  dev.off()
})

cat("All 37 figures re-rendered and saved to Figures/ with explicit print calls.\n")
