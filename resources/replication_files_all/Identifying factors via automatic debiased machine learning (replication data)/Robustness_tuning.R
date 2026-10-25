#################################################################################
###
###  This file provides how to obtain ADML estimates of different tuning parameters.
###
#################################################################################
rm(list=ls())

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

source("dataimport.R")

cores <- 1
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)
chunk.size <- 150/cores
out_recursive<-matrix(nrow=100,ncol=450)
out_recursive.s<-matrix(0, nrow=100, ncol=(3*chunk.size))

out_recursive<- foreach(q=1:cores, .combine='cbind',.packages=c("mnormt","nnet","mvtnorm","lars","randomForest","glmnet","grplasso","MASS","quantreg")) %dopar%
{  
  for(jj in ((q-1)*chunk.size+1):(q*chunk.size)) 
  { 
    out.s<-matrix(nrow=100,ncol=3)
    for (sizes in 1:10) 
    {
      for (splits in 2:11) 
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
          ER[i,1]<- mean(Ri[,i])
        }
        
        
        Y=ER
        T=C_g
        X=C_h 
        
        source('primitives.R')
        
        #stage 1 function
        l=0.1
        
        RMD_dantzig <- function(M, G, D, lambda=0, sparse = TRUE) {
          
          p <- ncol(G)
          zp <- rep(0, p)
          L <-c(l,rep(1,p-1)) #dictionary is ordered (constant,T,X,TX)
          
          A <- solve(diag(D),G)
          R <- rbind(A, -A)
          
          a <- solve(diag(D),M)
          r <- c(a - lambda*L, -a - lambda*L)
          
          if(sparse) {
            Ip <- as(p, "matrix.diag.csr")
            R <- as.matrix.csr(R)
            f <- rq.fit.sfnc(Ip, zp, R = R, r = r)
          } else {
            Ip <- diag(p)
            f <- rq.fit.fnc(Ip, zp, R = R, r = r)
          }
          
          return(f)
        }
        
        RMD_lasso <- function(M, G, D, lambda=0, control = list(maxIter = 1000, optTol = 10^(-5), 
                                                                zeroThreshold = 10^(-6)), beta.start = NULL) {
          
          p <- ncol(G)
          
          Gt<-G
          Mt<-M
          
          L <-c(l,rep(1,p-1)) #dictionary is ordered (constant,...)
          lambda_vec=lambda*L*D 
          
          if (is.null(beta.start)) {
            beta <- rep(0,p) #vs low-dimensional initialization
          }
          else {
            beta <- beta.start
          }
          wp <- beta
          mm <- 1
          while (mm < control$maxIter) {
            beta_old <- beta
            for (j in 1:p) {
              rho=Mt[j]-Gt[j,]%*%beta+Gt[j,j]*beta[j]
              z=Gt[j,j]
              
              if (sum(is.na(rho)) >= 1) {
                beta[j] <- 0
                next
              }
              if (rho < -1 * lambda_vec[j]) 
                beta[j] <- (rho+lambda_vec[j])/z
              if (abs(rho) <= lambda_vec[j]) 
                beta[j] <- 0
              if (rho > lambda_vec[j]) 
                beta[j] <- (rho-lambda_vec[j])/z
            }
            wp <- cbind(wp, beta)
            if (sum(abs(beta - beta_old), na.rm = TRUE) < control$optTol) {
              break
            }
            mm <- mm + 1
          }
          w <- beta
          w[abs(w) < control$zeroThreshold] <- 0
          return(list(coefficients = w, coef.list = wp, num.it = mm))
        }
        
        get_D <- function(Y,T,X,m,rho_hat,b){
          n=nrow(X)
          p=length(b(T[1],X[1,]))
          
          df=matrix(0,p,n)
          for (i in 1:n){
            df[,i]=b(T[i],X[i,])*as.vector(rho_hat %*% b(T[i],X[i,]))-m(Y[i],T[i],X[i,],b)
          }
          df=df^2
          D2=rowMeans(df)
          
          D=sqrt(D2)
          return(D) #pass around D as vector
        }
        
        c=0.5
        alpha=0.1
        tol=1e-6
        
        RMD_stable<-function(Y,T,X,p0,D_LB,D_add,max_iter,b,is_alpha){
          
          k=1
          
          p=length(b(T[1],X[1,]))
          n=length(T)
          
          # low-dimensional moments
          X0=X[,1:p0]
          MNG0<-get_MNG(Y,T,X0,b)
          M_hat0=MNG0[[1]]
          N_hat0=MNG0[[2]]
          G_hat0=MNG0[[3]]
          
          # initial estimate
          rho_hat0=solve(G_hat0,M_hat0)
          rho_hat=c(rho_hat0,rep(0,p-ncol(G_hat0)))
          beta_hat0=solve(G_hat0,N_hat0)
          beta_hat=c(beta_hat0,rep(0,p-ncol(G_hat0)))
          
          # moments
          MNG<-get_MNG(Y,T,X,b)
          M_hat=MNG[[1]]
          N_hat=MNG[[2]]
          G_hat=MNG[[3]]
          
          # penalty
          lambda=c*qnorm(1-alpha/(2*p))/sqrt(n) # snippet
          
          if(is_alpha){ 
            ###########
            # alpha_hat
            ###########
            diff_rho=1
            while(diff_rho>tol & k<=max_iter){
              
              # previous values
              rho_hat_old=rho_hat+0
              
              # normalization
              D_hat_rho=get_D(Y,T,X,m,rho_hat_old,b)
              D_hat_rho=pmax(D_LB,D_hat_rho)
              D_hat_rho=D_hat_rho+D_add
              
              # RMD estimate
              rho_hat=RMD_lasso(M_hat, G_hat, D_hat_rho, lambda)$coefficients
        
              
              # difference
              diff_rho=two.norm(rho_hat-rho_hat_old)
              k=k+1
              
            }
            
            print(paste0('k: '))
            print(paste0(k))
            return(rho_hat)
            
          } else { 
            ###########
            # gamma_hat
            ###########
            diff_beta=1
            while(diff_beta>tol & k<=max_iter){
              
              # previous values
              beta_hat_old=beta_hat+0
              
              # normalization
              D_hat_beta=get_D(Y,T,X,m,beta_hat_old,b)
              D_hat_beta=pmax(D_LB,D_hat_beta)
              D_hat_beta=D_hat_beta+D_add
              
              # RMD estimate
              beta_hat=RMD_lasso(N_hat, G_hat, D_hat_beta, lambda)$coefficients
        
              
              # difference
              diff_beta=two.norm(beta_hat-beta_hat_old)
              k=k+1
              
            }
            
            print(paste0('k: '))
            print(paste0(k))
            return(beta_hat)
            
          }
        }
        
        arg_Forest<- list(clas_nodesize=1, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
        arg_Nnet<- list(size=sizes, maxit=1000, decay=0.01, MaxNWts=10000,  trace=FALSE)
        
        get_stage1<-function(Y,T,X,p0,D_LB,D_add,max_iter,b,alpha_estimator,gamma_estimator){
          
          p=length(b(T[1],X[1,]))
          n=length(T)
          MNG<-get_MNG(Y,T,X,b)
          B=MNG[[4]]
          
          ###########
          # alpha hat
          ###########
          if(alpha_estimator==1){ # lasso
            
            rho_hat=RMD_stable(Y,T,X,p0,D_LB,D_add,max_iter,b,1)
            alpha_hat<-function(d,z){
              return(b(d,z)%*%rho_hat)
            }
            
          }
          
          ###########
          # gamma hat
          ###########
        
          if(gamma_estimator==1){ # random forest
            
            forest<- do.call(randomForest, append(list(x=B,y=Y), arg_Forest))
            gamma_hat<-function(d,z){
              return(predict(forest,newdata=b(d,z), type="response"))
            }
            
          } else if(gamma_estimator==2){ # neural net
            
            # scale down, de-mean, run NN, scale up, remean so that NN works well
            maxs_B <- apply(B, 2, max)
            mins_B <- apply(B, 2, min)
            
            maxs_Y<-max(Y)
            mins_Y<-min(Y)
            
            # hack to ensure that constant covariates do not become NA in the scaling
            const=maxs_B==mins_B
            keep=(1-const)*1:length(const)
            
            NN_B<-B
            NN_B[,keep]<-scale(NN_B[,keep], center = mins_B[keep], scale = maxs_B[keep] - mins_B[keep])
            
            NN_Y<-scale(Y, center = mins_Y, scale = maxs_Y - mins_Y)
            
            nn<- do.call(nnet, append(list(x=NN_B,y=NN_Y), arg_Nnet)) 
            gamma_hat<-function(d,z){
              
              test<-t(as.vector(b2(d,z)))
              NN_b<-test
              NN_b[,keep]<-scale(t(NN_b[,keep]), 
                                 center = mins_B[keep], 
                                 scale = maxs_B[keep] - mins_B[keep])
              
              NN_Y_hat<-predict(nn,newdata=NN_b)
              Y_hat=NN_Y_hat*(maxs_Y-mins_Y)+mins_Y
              
              return(Y_hat)
            }
            
          }
          
          return(list(alpha_hat,gamma_hat))
          
        }
        
        L <- splits
        rrr<-function(Y,T,X,p0,D_LB,D_add,max_iter,dict,alpha_estimator,gamma_estimator,bias){
          
          n=nrow(X)
          folds <- split(sample(n, n,replace=FALSE), as.factor(1:L))
          
          Psi_tilde=numeric(0)
          Y_new=numeric(0)
          T_new=numeric(0)
          
          for (l in 1:L){
            
            Y.l=Y[folds[[l]]]
            Y.nl=Y[-folds[[l]]]
            
            T.l=T[folds[[l]]]
            T.nl=T[-folds[[l]]]
            
            X.l=X[folds[[l]],]
            X.nl=X[-folds[[l]],]
            
            n.l=length(T.l)
            n.nl=length(T.nl)
            
            # get stage 1 (on nl)
            stage1_estimators<-get_stage1(Y.nl,T.nl,X.nl,p0,D_LB,D_add,max_iter,dict,alpha_estimator,gamma_estimator)
            alpha_hat=stage1_estimators[[1]]
            gamma_hat=stage1_estimators[[2]]
            
            # debugging
            print(paste0('fold: ',l))
            #print(paste0('beta_hat: '))
            #print(paste0(round(beta_hat,2)))
            #print(paste0('rho_hat: '))
            #print(paste0(round(rho_hat,2)))
            
            #get stage 2 (on l) 
            
            #psi_star
            Psi_tilde.l=rep(0,n.l)
            for (i in 1:n.l){
              if(bias){ #plug-in
                Psi_tilde.l[i]=psi_tilde_bias(Y.l[i],T.l[i],X.l[i,],m,alpha_hat,gamma_hat) # without subtracting theta_hat
              }else{ #DML
                Psi_tilde.l[i]=psi_tilde(Y.l[i],T.l[i],X.l[i,],m,alpha_hat,gamma_hat) # without subtracting theta_hat
              }
            }
            
            Psi_tilde=c(Psi_tilde,Psi_tilde.l)
            Y_new=c(Y_new,Y.l)
            T_new=c(T_new,T.l)
            #print(paste0(round(mean(Psi_tilde.l),2)))
          }
          
          #point estimation
          est=mean(Psi_tilde)
          
          #influences
          Psi=Psi_tilde-est
          
          #V
          v=mean(Psi^2)
        
          se=sqrt(v/n)
          
          out<-c(est,se,est/se)
          
          return(out)
        }
        
        printer<-function(spec1){
          print(paste(" estimation ",spec1[1], "standard error", spec1[2],"t-statistic", spec1[3]))
        }
        
        
        dict=b # b for partially linear model, b2 for interacted model. 
        p=length(b(T[1],X[1,]))
        
        p0=ceiling(p/4) 
        if (p>60){
          p0=ceiling(p/40)
          
        }
        
        D_LB=0 #each diagonal entry of \hat{D} lower bounded by D_LB
        D_add=0.2 #each diagonal entry of \hat{D} increased by D_add
        max_iter=10 #max number iterations in Dantzig selector iteration over estimation and weights
        
        ###########
        # algorithm
        ###########
        
        set.seed(1) # for sample splitting
        
        alpha_estimator=1 #alpha_estimator: 1 lasso
        gamma_estimator=2 #gamma_estimator: 1 rf, 2 nn
        bias=0 #bias: 0 DML, 1 plug-in
        
        results<-rrr(Y,T,X,p0,D_LB,D_add,max_iter,dict,alpha_estimator,gamma_estimator,bias)
        
        out.s[10*(sizes-1)+(splits-1),]<-results
        
     }
    }
    out_recursive.s[,(3*((jj-(q-1)*chunk.size)-1)+1):(3*((jj-(q-1)*chunk.size)))]<-out.s

  }
  out_recursive.s
}


write.table(out_recursive,file = "robustness_tuning.xlsx")

