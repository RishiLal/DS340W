#################################################################################
###
###  This file provides how to obtain simulation results
###
#################################################################################

rm (list = ls(all=TRUE))
graphics.off()

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
library("lars")
library("mnormt")

source('primitives.R')
source('stage1.R')
source('stage2.R')

# load simulated factors
load("simulation_p100_n300.RData")

D_LB=0 #each diagonal entry of \hat{D} lower bounded by D_LB
D_add=0.2 #each diagonal entry of \hat{D} increased by D_add
max_iter=10 #max number iterations in Dantzig selector iteration over estimation and weights

set.seed(1) # for sample splitting

#alpha_estimator: 1 lasso
#gamma_estimator: 1 rf, 2 nn
alpha_estimator=1
gamma_estimator=2

# 2000 Monte Carlo runs
l<-2000

cores <- 1
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)
chunk.size <- l/cores
out<-matrix(0,nrow = l,ncol = 2)
out.s<-matrix(0, nrow=chunk.size, ncol=2)

out<- foreach(q=1:cores, .combine='rbind',.packages=c("mnormt","nnet","mvtnorm","lars","randomForest","glmnet","MASS","grplasso","quantreg")) %dopar%
  {  
    dml.s <- matrix(0, nrow=chunk.size, ncol=1)
    plug.s <- matrix(0, nrow=chunk.size, ncol=1)
    
    for(k in ((q-1)*chunk.size+1):(q*chunk.size)) {
      
      t<- 480
      rt<- matrix(nrow=t,ncol = n)
      p<- 100
      for (i in 1:t)
      {
        ut <- rmvt(1,sigma_ut,5,rep(0,n),type=c("shifted","Kshirsagar")[1])
        rt[i,] <- E_rt+t(ut)
      }
      
      mr<-matrix(nrow = n, ncol = 1)
      
      for (j in 1:n)
      {
        mr[j,1]<- mean(rt[,j])
      }
      
      Y=mr
      T=C_g[,1]
      X=cbind(C_g[,2],C_g[,3],C_h)
      
      # dictionary
      dict=b # b for partially linear model
      p=length(b(T[1],X[1,]))
      
      p0=ceiling(p/4) 
      if (p>60){
        p0=ceiling(p/40)
      }
      
      #bias: 0 DML, 1 plug-in
      bias<-0
      fit_0<-rrr(Y,T,X,p0,D_LB,D_add,max_iter,dict,alpha_estimator,gamma_estimator,bias)
      bias<-1
      fit_1<-rrr(Y,T,X,p0,D_LB,D_add,max_iter,dict,alpha_estimator,gamma_estimator,bias)
      
      dml.s[k-(q-1)*chunk.size,] <-(fit_0[1]-37.344486252115887)/fit_0[2]
      plug.s[k-(q-1)*chunk.size,] <-(fit_1[1]-37.344486252115887)/fit_1[2]
    }
    # return local results
    out.s<-cbind(dml.s,plug.s)
    out.s
  }

write.table(out,file = "simulation_p100_n300_t480.xlsx")

#draw the distribution
h<-rnorm(100000)
xfit<-seq(min(h),max(h),length=100)
yfit<-dnorm(xfit,mean(h),sd(h))
boundaries <- seq(-8, 8, by=0.2)
ADML<-out[,1]
hist(ADML, freq=FALSE, breaks=boundaries,col ="dimgray",ylim = c(0, 0.5))
lines(xfit,yfit,col="black",lwd=2)
boundaries <- seq(-500, 200, by=2)
Plug_in<-out[,2]
hist(Plug_in, freq=FALSE, breaks=boundaries,col ="dimgray",ylim = c(0, 0.5),xlim = c(-60, 10))



