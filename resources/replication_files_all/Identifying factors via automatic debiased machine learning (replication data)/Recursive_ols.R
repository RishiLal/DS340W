##########################################################################
###
###  This file provides how to obtain OLS estimator in expanding windows. 
###
##########################################################################

rm(list=ls())
library("AER")
library("pracma")
library("foreach")
library("doParallel")
library("foreign")
library("dplyr")
library("ggplot2")
library("quantreg")
library("MASS")
library("glmnet")	
library("grplasso")	
library("nnet")	
library("randomForest")
library("gglasso")
library("plotrix")
library("gridExtra")

source('dataimport.R')

# Number of cores
cores = 1

#calculate expanding window estimates
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)
chunk.size <- 150/cores
out_recursive<-matrix(nrow=20,ncol=450)
out_recursive.s<-matrix(0, nrow=20, ncol=(3*chunk.size))

out_recursive<- foreach(q=1:cores, .combine='cbind',.packages=c("mnormt","nnet","mvtnorm","lars","randomForest","glmnet","grplasso","MASS","quantreg")) %dopar%
{  
   for(jj in ((q-1)*chunk.size+1):(q*chunk.size)) 
   { 
      out.s<-matrix(nrow=20,ncol=3)
      for (kk in 1:20)
      {
         Ri<-port_3x2b[1:(270+((kk-1)*12)),]
         
         gt<-factors[1:(270+((kk-1)*12)),jj]
         ht<-factors[1:(270+((kk-1)*12)),-jj]
         
         n<-ncol(Ri)
         p<-ncol(ht)
         d<-ncol(gt)
         
         C_g<-t(cov(gt,Ri,use="complete.obs"))
         C_h<-t(cov(ht,Ri,use="complete.obs"))
         
         ER<-matrix(nrow = n,ncol = 1)
         for (i in 1:n)
         {
          ER[i,1]<- mean(Ri[,i])
         }
         
         Y=ER
         T=C_g
         X=C_h 
         
         #OLS result:
         fit<-lm(Y~C_g+C_h+1)
         out.s[kk,]<-c(summary(fit)[["coefficients"]][2,1],summary(fit)[["coefficients"]][2,2],summary(fit)[["coefficients"]][2,3])
      }
      out_recursive.s[,(3*((jj-(q-1)*chunk.size)-1)+1):(3*((jj-(q-1)*chunk.size)))]<-out.s
      
   }
   out_recursive.s
}

# output
write.table(out_recursive,file = "out_ols.xlsx")

