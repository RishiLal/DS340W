# A separate library; never install into the user's existing library.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1], winslash='/')
lib <- file.path(root, 'env', 'R-library')
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .Library))
stopifnot(normalizePath(.libPaths()[1]) == normalizePath(lib))
options(repos=c(CRAN='https://cloud.r-project.org'), timeout=300)
pkgs <- c('directlabels','xtable','zeallot','dendextend','RColorBrewer','rsample',
 'lubridate','tidyverse','data.table','mvtnorm','glmnet','stargazer','lmtest',
 'sandwich','corrplot','cowplot','ggrepel','jsonlite','remotes','broom','pdfCluster')
missing <- setdiff(pkgs, rownames(installed.packages()))
if(length(missing)) install.packages(missing, lib=lib, type=getOption('pkgType'))
if(!requireNamespace('drlib', quietly=TRUE))
  remotes::install_github('dgrtwo/drlib@1780f937f0d488b56c0e23db92e2e976cbd9c371', lib=lib, upgrade='never', dependencies=NA)
stopifnot(all(vapply(c(pkgs, 'drlib'),requireNamespace,logical(1),quietly=TRUE)))
write.csv(installed.packages()[,c('Package','Version','LibPath')],
          file.path(root,'evidence','R_packages.csv'), row.names=FALSE)
capture.output(sessionInfo(), file=file.path(root,'evidence','R_sessionInfo.txt'))


