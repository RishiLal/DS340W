library(lars)
double_selection_lasso<- function(mr,C_g,C_h,k)
{
  #first step lasso
  object<-lars(C_h,mr,type="lasso") 
  cv_sol<-cv.lars(C_h,mr,K=k,mode="step",plot.it = FALSE,max.steps=80)
  fra<-cv_sol$index[which.min(cv_sol$cv)]
  index <-object$beta[fra,]
  
  i<-seq(1,length(index))
  i1<-i[index>0]
  
  #second step lasso
  object<-lars(C_h,C_g[,1],type="lasso") 
  cv_sol<-cv.lars(C_h,C_g[,1],K=k,mode="step",plot.it = FALSE,max.steps=80)
  fra<-cv_sol$index[which.min(cv_sol$cv)]
  index <-object$beta[fra,]
  i<-seq(1,length(index))
  i2_1<-i[index>0]
  
  #OLS regression
  I<- sort(unique(c(i1,i2_1)))
  fit<- lm(mr~C_g+C_h[,I])
  return(fit)
}