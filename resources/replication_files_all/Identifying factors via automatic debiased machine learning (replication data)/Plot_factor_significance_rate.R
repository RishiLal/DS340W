####################################################################################################
###
###  This file provides how to calculate factor significance rate.
###
####################################################################################################
rm(list=ls())
library("ggplot2")

# Calculate factor significance rate
out_recursive <- read.table('out_recursive_nn_3x2.xlsx',header=TRUE)
tstat_recursive<-out_recursive[,seq(3,450,3)]
significant<-matrix(0,nrow =150, ncol = 1)

for (i in 1:150) 
{
  for (j in 1:20)
  {
    if((tstat_recursive[j,i]>=3.59)||(tstat_recursive[j,i]<=-3.59))
    {
      significant[i,1]<-significant[i,1]+1
    }
  }
  significant[i,1]<-significant[i,1]/20
}

factor_significance<-cbind(seq(1,150,1),significant)

#plot factor significance rate
factor_significance<-as.data.frame(factor_significance)
colnames(factor_significance)[1]<-"factor_id"
colnames(factor_significance)[2]<-"significance_rate"
ggplot(data=factor_significance,aes(factor_id,significance_rate))+geom_bar(stat = "identity",position =position_dodge(0.3))+scale_colour_gradient(low = "white", high = "black")



