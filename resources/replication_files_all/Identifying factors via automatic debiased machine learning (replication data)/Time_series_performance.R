####################################################################################################
###
###  This file provides how to calculate the time series performance
###
####################################################################################################

rm(list=ls())

# Calculate factor significance rate
out_recursive <- read.table('out_recursive_nn_3x2.xlsx',header=TRUE)
tstat_recursive<-out_recursive[,seq(3,450,3)]

# Calculate time series performance
momentum<-tstat_recursive[,c(5,23,24,25,34,45,46,91,100,110,111,123)]
value_versus_growth<-tstat_recursive[,c(1,3,4,7,8,19,20,22,27,29,30,32,48,49,67,74,105,106,107,108,121,135,137,141)]
investment<-tstat_recursive[,c(31,59,60,64,66,68,69,71,72,73,77,78,79,80,81,82,83,84,85,86,87,88,89,92,95,96,97,98,99,103,109,113,114,115,133,134,142,146,147)]
profitability<-tstat_recursive[,c(26,33,42,43,47,50,90,116,117,118,119,120,127,138,145,148)]
intangibles<-tstat_recursive[,c(9,10,11,12,13,14,15,16,17,36,37,38,39,40,41,44,55,56,57,58,65,70,75,76,101,102,104,112,125,126,129,130,132,136,139,143,144,150)]
trading_frictions<-tstat_recursive[,c(2,6,18,21,28,35,51,52,53,54,61,62,63,93,94,122,124,128,131,140,149)]

sum_mom<-matrix(0,nrow = 20,ncol = 1)
sum_vvg<-matrix(0,nrow = 20,ncol = 1)
sum_inv<-matrix(0,nrow = 20,ncol = 1)
sum_pro<-matrix(0,nrow = 20,ncol = 1)
sum_int<-matrix(0,nrow = 20,ncol = 1)
sum_taf<-matrix(0,nrow = 20,ncol = 1)

for (i in 1:20)
{
  for (j in 1:12)
  {
    if((momentum[i,j]>=3.59)||(momentum[i,j]<=-3.59))
    {
      sum_mom[i,1]<-sum_mom[i,1]+1
    }
  }
  for (j in 1:24)
  {
    if((value_versus_growth[i,j]>=3.59)||(value_versus_growth[i,j]<=-3.59))
    {
      sum_vvg[i,1]<-sum_vvg[i,1]+1
    }
  }
  
  for (j in 1:39)
  {
    if((investment[i,j]>=3.59)||(investment[i,j]<=-3.59))
    {
      sum_inv[i,1]<-sum_inv[i,1]+1
    }
  }
  
  for (j in 1:16)
  {
    if((profitability[i,j]>=3.59)||(profitability[i,j]<=-3.59))
    {
      sum_pro[i,1]<-sum_pro[i,1]+1
    }
  }
  
  for (j in 1:38)
  {
    if((intangibles[i,j]>=3.59)||(intangibles[i,j]<=-3.59))
    {
      sum_int[i,1]<-sum_int[i,1]+1
    }
  }
  
  for (j in 1:21)
  {
    if((trading_frictions[i,j]>=3.59)||(trading_frictions[i,j]<=-3.59))
    {
      sum_taf[i,1]<-sum_taf[i,1]+1
    }
  }
}

sum<-cbind(sum_mom,sum_vvg,sum_inv,sum_pro,sum_int,sum_taf)

# output
write.table(sum,file = "sum_nn_3x2.xlsx")




