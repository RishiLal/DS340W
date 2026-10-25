## Import and clean data 

library(matlab)

# factors
allfactors <- read.table('factors.csv',header=TRUE,sep =",")
date <- allfactors[ ,1]
rf <- allfactors[ ,2]
factors <- allfactors[ ,3:ncol(allfactors)]
L <- length(date)
P<-  ncol(factors)

# test portfolios
port_3x2 <- read.table('port_3x2.csv',sep=",")
port_3x2 <- port_3x2[ ,-1]
one<-matrix(rep(1,ncol(port_3x2)),nrow = 1,ncol = ncol(port_3x2))
port_3x2 <- port_3x2-rf%*%one

port_5x5 <- read.table('port_5x5.csv',sep=",")
port_5x5 <- port_5x5[ ,-1]
one<-matrix(rep(1,ncol(port_5x5)),nrow = 1,ncol = ncol(port_5x5))
port_5x5 <- port_5x5-rf%*%one

port_202 <- read.table('port202.csv',sep=",")
port_202 <- port_202[ ,-1]/100
one<-matrix(rep(1,ncol(port_202)),nrow = 1,ncol = ncol(port_202))
port_202 <- port_202-rf%*%one


#other information
summary<- read.table('summary.csv',header=TRUE,sep=",")
factorname<-summary[,1]
factorname_full<-summary[,10]
year_pub<-summary[,7]
year_end<-summary[,9]

port_3x2_id<-read.table('port_3x2_id.csv',header=TRUE,sep=",")
port_5x5_id<-read.table('port_5x5_id.csv',header=TRUE,sep=",")

#form a smaller set of portfolios for bivariate sorted porfolios
#set minimun number of stocks in a portfolio
kk<-10
include_3x2 <- t(find(port_3x2_id[,4]>=kk))
port_3x2b<-matrix()

for (i in 1:P)
{
 if (i%in%include_3x2) 
  {port_3x2b<-cbind(port_3x2b,port_3x2[,(i*6-5):(i*6)])} 
}
port_3x2b<-port_3x2b[,-1]


include_5x5 <- t(find(port_5x5_id[,4]>=kk))
port_5x5b<-matrix()

for (i in 1:P)
{
  if (i%in%include_5x5) 
  {port_5x5b<-cbind(port_5x5b,port_5x5[,(i*25-24):(i*25)])} 
}
port_5x5b<-port_5x5b[,-1]








