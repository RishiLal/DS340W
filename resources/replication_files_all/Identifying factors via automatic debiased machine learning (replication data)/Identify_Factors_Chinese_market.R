########################################################################################################
###
###  This file provides how to obtain ADML estimator in Chinese stock market. 
###
###  Method: Neural Network
###  Testing Asset: 762 3¡Á2 portfolios
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
library("matlab")

# import data
# factors
allfactors <- read.table('factors_cn.csv',header=TRUE,sep =",")
date <- allfactors[ ,1]
factors <- allfactors[ ,2:ncol(allfactors)]
L <- length(date)
P<-  ncol(factors)

# test portfolios
testing <- read.table('testing_assets_cn.csv',header=TRUE,sep=",")
port_3x2b <- testing[ ,2:ncol(testing)]

cores <- 1
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)
chunk.size <- 132/cores
out<-matrix(nrow=1,ncol=396)
out.s<-matrix(0, nrow=1, ncol=(3*chunk.size))

out<- foreach(q=1:cores, .combine='cbind',.packages=c("mnormt","nnet","mvtnorm","lars","randomForest","glmnet","grplasso","MASS","quantreg")) %dopar%
  {  
    
    for(jj in ((q-1)*chunk.size+1):(q*chunk.size)) 
    { 
        Ri<-port_3x2b

        gt<-factors[,jj]
        ht<-factors[,-jj]
        
        n<-ncol(Ri)
        p<-ncol(ht)
        d<-ncol(gt)
        
        
        C_g<-t(cov(gt,Ri,use="complete.obs"))
        C_h<-t(cov(ht,Ri,use="complete.obs"))
        
        ER<-matrix(nrow = n,ncol = 1)
        for (i in 1:n)
        {
          ER[i,1]<- mean(Ri[,i],na.rm = TRUE)
        }
        
        
        Y=ER
        T=C_g
        X=C_h 
        
        source('primitives.R')
        source('stage1.R')
        
        # dictionary
        dict=b # b for partially linear model
        p=length(b(T[1],X[1,]))

        p0=ceiling(p/4) 
        if (p>60){
          p0=ceiling(p/40)
          
        }
        
        D_LB=0 #each diagonal entry of \hat{D} lower bounded by D_LB
        D_add=0.2 #each diagonal entry of \hat{D} increased by D_add
        max_iter=10 
        
        
        set.seed(1) #for sample splitting
        
        alpha_estimator=1  #alpha_estimator: 1 lasso
        gamma_estimator=2  #gamma_estimator: 1 rf, 2 nn
        bias=0  #bias: 0 DML, 1 plug-in

        source('stage2.R')
        
        results<-rrr(Y,T,X,p0,D_LB,D_add,max_iter,dict,alpha_estimator,gamma_estimator,bias)
      
        out.s[,(3*((jj-(q-1)*chunk.size)-1)+1):(3*((jj-(q-1)*chunk.size)))]<-results
        
    }
    out.s
  }

list1<-seq(1,394,3)
list2<-seq(2,395,3)
list3<-seq(3,396,3)
out_all_ape <- t(out[1,list1])
out_all_std <- t(out[1,list2])
out_all_tstat <- t(out[1,list3])
out_onebyone<-cbind(out_all_ape,out_all_tstat)

# output
write.table(out_onebyone,file = "Chinese_market.xlsx")

