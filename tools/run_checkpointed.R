# Run original expressions in their original order. Store completed expressions
# and RNG state so an interruption does not erase a finished native calculation.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1],winslash='/')
run <- normalizePath(args[2],winslash='/')
.libPaths(c(file.path(root,'env','R-library'), .Library))
stopifnot(normalizePath(.libPaths()[1]) == normalizePath(file.path(root, 'env', 'R-library')))
setwd(run)
options(warn=1)
started <- Sys.time()
session <- new.env(parent=.GlobalEnv)
state <- function(status, step, error=NULL) {
  result <- list(status=status, step=step, pid=Sys.getpid(), at=as.character(Sys.time()),
    elapsed_seconds=as.numeric(difftime(Sys.time(),started,units='secs')), error=error)
  jsonlite::write_json(result,'runtime_status.json',pretty=TRUE,auto_unbox=TRUE)
}
inputs <- sort(c(list.files(pattern='\\.(R|xlsx)$'),list.files('Data',full.names=TRUE),
  file.path(root,'tools','run_checkpointed.R')))
signature <- list(files=tools::md5sum(inputs), R=R.version.string,
  packages=installed.packages()[,c('Package','Version')], RNG=RNGkind())
if(file.exists('fingerprint.rds')) {
  if(!identical(signature,readRDS('fingerprint.rds'))) stop('Source, inputs or environment changed; use a new run directory.')
} else saveRDS(signature,'fingerprint.rds')
dir.create('checkpoints',showWarnings=FALSE)
dir.create('numeric',showWarnings=FALSE)
cat('Invocation',format(started),'PID',Sys.getpid(),'\n')
print(sessionInfo())
lhs_name <- function(x) {
  if(is.symbol(x)) return(as.character(x))
  if(is.call(x) && length(x)>=2) return(lhs_name(x[[2]]))
  character()
}
assigned <- function(x) {
  if(!is.call(x) || identical(x[[1]],as.name('function'))) return(character())
  head <- as.character(x[[1]])[1]
  here <- if(head %in% c('<-','<<-','=','for')) lhs_name(x[[2]]) else character()
  unique(c(here,unlist(lapply(as.list(x)[-1],assigned),use.names=FALSE)))
}
export_eb <- function() {
  for(nm in names(session$eb_est)) {
    fit <- session$eb_est[[nm]]
    write.csv(fit$factors,file.path('numeric',paste0('factors_',nm,'.csv')),row.names=FALSE)
    write.csv(fit$mle,file.path('numeric',paste0('hyperparameters_',nm,'.csv')),row.names=FALSE)
  }
  write.csv(session$char_info,'numeric/factor_metadata.csv',row.names=FALSE)
}
session$source <- function(file,...) {
  cat('Stage',file,format(Sys.time()),'\n')
  state('running',file)
  if(file!='3 - Analysis.R') {
    ans <- base::source(file,local=session,...)
    if(file=='2 - Determine Clusters.R') write.csv(session$cluster_labels,'numeric/cluster_labels.csv',row.names=FALSE)
    return(invisible(ans))
  }
  expressions <- parse(file,keep.source=FALSE)
  for(i in seq_along(expressions)) {
    expr <- expressions[[i]]
    desc <- paste0('3 - Analysis.R expression ',i,'/',length(expressions),': ',substr(deparse(expr)[1],1,65))
    checkpoint <- sprintf('checkpoints/analysis_%03d.rds',i)
    state('running',desc)
    if(file.exists(checkpoint)) {
      restored <- readRDS(checkpoint)
      if(!identical(restored$expression,expr)) stop('Checkpoint expression mismatch')
      list2env(restored$bindings,session)
      if(!is.null(restored$seed)) assign('.Random.seed',restored$seed,.GlobalEnv)
      cat('Restored',desc,'\n')
    } else {
      cat('Computing',desc,format(Sys.time()),'\n')
      before <- ls(session,all.names=TRUE)
      eval(expr,session)
      after <- ls(session,all.names=TRUE)
      bindings <- intersect(unique(c(assigned(expr),setdiff(after,before))),after)
      saved <- list(expression=expr,bindings=mget(bindings,envir=session,inherits=FALSE),
        seed=get0('.Random.seed',envir=.GlobalEnv,inherits=FALSE))
      saveRDS(saved,paste0(checkpoint,'.tmp'),compress=FALSE)
      stopifnot(file.rename(paste0(checkpoint,'.tmp'),checkpoint))
      rm(saved)
      cat('Completed',desc,format(Sys.time()),'\n')
    }
    if(i==2) export_eb()
  }
  invisible(NULL)
}
tryCatch({
  base::source('main.R',local=session,echo=TRUE)
  state('executed','main.R finished; numerical comparisons pending')
},error=function(err) {
  cat('Native failure:',conditionMessage(err),'\n')
  state('failed',as.character(conditionCall(err))[1],conditionMessage(err))
  quit(status=1)
})
