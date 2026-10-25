## primitive functions

two.norm <- function(x){
  return(sqrt(x %*% x))
} 

one.norm<-function(x){
  return(sum(x%*%sign(x)))
}

one.norm.grad<-function(x){
  return(sign(x))
}

# intercept
b<-function(d,z){
  return(c(1,d,z))
}

# intercept and interaction
b2<-function(d,z){
  return(c(1,d,z,d*z))
}

obj<-function(rho,G,M,r){
  return(t(rho)%*%G%*%rho-2*t(M)%*%rho+2*r*one.norm(rho))
}

m<-function(y,d,z,gamma){ 
  dx<-0.000005
  return((gamma(d+dx,z)-gamma(d-dx,z))/(2*dx))
}

psi_tilde<-function(y,d,z,m,alpha,gamma){
  return(m(y,d,z,gamma)+alpha(d,z)*(y-gamma(d,z)))
}

psi_tilde_bias<-function(y,d,z,m,alpha,gamma){
  return(m(y,d,z,gamma))
}

get_MNG<-function(Y,T,X,b){
  
  p=length(b(T[1],X[1,]))
  n.nl=length(T)
  
  B=matrix(0,n.nl,p)
  M=matrix(0,p,n.nl)
  N=matrix(0,p,n.nl)
  
  for (i in 1:n.nl){
    B[i,]=b(T[i],X[i,])
    M[,i]=m(Y[i],T[i],X[i,],b) 
    N[,i]=m(Y[i],T[i],X[i,],b)
  }
  
  M_hat=rowMeans(M)
  N_hat=rowMeans(N)
  G_hat=t(B)%*%B/n.nl
  
  return(list(M_hat,N_hat,G_hat,B))
}