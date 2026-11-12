# Invoke the original US analysis on all months, using the published settings.
args <- commandArgs(trailingOnly=TRUE)
setwd(args[1])
.libPaths(c(normalizePath(args[2]),.Library))
stopifnot(requireNamespace('broom',quietly=TRUE))
options(warn=1)
model <- new.env(parent=.GlobalEnv)
model$source <- function(file, ...) {
  if(file %in% c('0 - Functions.R','1 - Prepare Data.R','2 - Determine Clusters.R')) {
    cat('Author workflow:', file, '\n')
    base::sys.source(file,envir=model)
  } else if(file=='3 - Analysis.R') {
    native <- parse(file,keep.source=FALSE)
    stopifnot(identical(native[[1]][[2]],as.name('search_list')),
              identical(native[[2]][[2]],as.name('eb_est')))
    eval(native[[1]],model)
    stopifnot(identical(names(model$search_list)[1], 'us'), length(model$search_list)==11,
              model$settings$eb$bs_samples==10000, model$settings$seed==1)
    model$search_list <- model$search_list['us']
    cat('Computing the original US model with all 10,000 bootstrap samples...\n')
    eval(native[[2]],model)
  } else if(file!='4 - Output.R') {
    stop('Unexpected source file: ',file)
  }
  invisible(NULL)
}
base::sys.source('main.R',envir=model)
fit <- model$eb_est$us
point <- setNames(fit$mle$ml_est, fit$mle$estimate)
ids <- model$char_info$characteristic[model$char_info$significance==1 &
                                        !is.na(model$char_info$significance)]
rows <- fit$factors[fit$factors$characteristic %in% ids,]
positive <- sum(rows$p025>0)
rate <- 100*positive/nrow(rows)
# The author's table_taus() displays the native two-layer tau_s as printed tau_w.
fresh <- c(unname(point['tau_c']), unname(point['tau_s']),rate)
printed <- c(.35,.21,82.4)
check <- data.frame(result=c('Table I: US tau_c','Table I: US tau_w','Figure 1: US EB replication rate'),
                    fresh_percent=round(fresh,5),published_percent=printed,
                    within_printed_precision=abs(fresh-printed)<=c(.005,.005,.05))
cat('\nUS benchmark | monthly percentage points for tau, percent of eligible factors for rate\n')
print(check,row.names=FALSE)
cat(sprintf('Positive lower posterior bound: %d of %d originally significant factors.\n',positive,nrow(rows)))
stopifnot(nrow(fit$factors)==153, nrow(rows)==119, positive==98, all(check$within_printed_precision))
jsonlite::write_json(list(scope='Fresh US empirical Bayes benchmark only',checks=check,
                         unique_factors=nrow(fit$factors),eligible=nrow(rows),positive=positive,
                         draws=10000,seed=1,R=R.version.string), 'result.json',auto_unbox=TRUE,pretty=TRUE,digits=12)
