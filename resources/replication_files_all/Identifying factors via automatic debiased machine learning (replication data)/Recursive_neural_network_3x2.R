########################################################################################################
###
###  This file provides how to obtain ADML estimator in expanding windows. 
###
###  Method: Neural Network
###  Testing Asset: 750 3¡Á2 portfolios
########################################################################################################

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
      
      source('primitives.R')
      source('stage1.R')
      
      # dictionary
      dict=b # b for partially linear model, shows up in stage1.R for NN
      p=length(b(T[1],X[1,]))
      
      #p0=dim(X0) used in low-dim dictionary in the stage 1 tuning procedure
      p0=ceiling(p/4) 
      if (p>60){
        p0=ceiling(p/40)
        }
      
      D_LB=0 #each diagonal entry of \hat{D} lower bounded by D_LB
      D_add=0.2 #each diagonal entry of \hat{D} increased by D_add
      max_iter=10 
      
      set.seed(1) # for sample splitting
      
      alpha_estimator=1  #alpha_estimator: 1 lasso
      gamma_estimator=2  #gamma_estimator: 1 rf, 2 nn
      bias=0  #bias: 0 DML, 1 plug-in
      
      source('stage2.R')
      
      #DML result:
      results<-rrr(Y,T,X,p0,D_LB,D_add,max_iter,dict,alpha_estimator,gamma_estimator,bias)
        
      out.s[kk,]<-results
      
    }
    out_recursive.s[,(3*((jj-(q-1)*chunk.size)-1)+1):(3*((jj-(q-1)*chunk.size)))]<-out.s
       
   }
   out_recursive.s
}

# output
write.table(out_recursive,file = "out_recursive_nn_3x2.xlsx")

# The result of the last row is an estimate of the whole sample. 
list1<-seq(1,448,3)
list2<-seq(2,449,3)
list3<-seq(3,450,3)
out_all_ape <- t(out_recursive[20,list1])
out_all_std <- t(out_recursive[20,list2])
out_all_tstat <- t(out_recursive[20,list3])
out_onebyone<-cbind(out_all_ape,out_all_tstat)

write.table(out_onebyone,file = "onebyone_nn_3x2.xlsx")
