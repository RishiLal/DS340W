#################################################################################
###
###  Spanning test
###
#################################################################################

rm(list=ls())

library(AER)

allfactors <- read.table('factors.csv',header=TRUE,sep =",")
factors <- allfactors[ ,3:ncol(allfactors)]

# Fama-French factor models
x_ff3<- as.matrix(factors[,c(1,21,22)])
x_ff5<- as.matrix(factors[,c(1,21,22,145,146)])
x_ff6<- as.matrix(factors[,c(1,21,22,145,146,34)])
list_ff3 <- c(1,21,22)
list_ff5 <- c(1,21,22,145,146)
list_ff6 <- c(1,21,22,145,146,34)

# Factor models selected by ADML
x_adml_3<-as.matrix(factors[,c(98,132,25)])
list_adml_3<-c(98,132,25)
x_adml_5<-as.matrix(factors[,c(98,132,25,96,13)])
list_adml_5<-c(98,132,25,96,13)
x_adml_6<-as.matrix(factors[,c(98,132,25,96,13,119)])
list_adml_6<-c(98,132,25,96,13,119)

# Factor model selected by Double-selection Lasso
x_dl_6<-as.matrix(factors[,c(49,32,147,51,21,72)])
list_dl_6<-c(49,32,147,51,21,72)

# Factor model selected by OLS
x_ols_6<-as.matrix(factors[,c(88,51,64,72,148,34)])
list_ols_6<-c(88,51,64,72,148,34)

# initialization
index<-0
num<-0
momentum<-0
value_versus_growth<-0
investment<-0
profitability<-0
intangibles<-0
trading_frictions<-0

num0 <-0
num1 <-0
num2 <-0
num3 <-0
num4 <-0
num5 <-0
num6 <-0

alpha_all<-0
alpha_momentum<-0
alpha_value_versus_growth<-0
alpha_investment<-0
alpha_profitability<-0
alpha_intangibles<-0
alpha_trading_frictions<-0

# Select the factor model to be tested
list <- list_adml_6
x <- x_adml_6

for (j in 1:150)
{
  if (j %in% list) next 
  fit_all<-lm(factors[,j]~ x+1) 
  fit_all_adj<-coeftest(fit_all, vcov. = NeweyWest(fit_all, lag = floor(4*(498/100)^(2/9))))
  
  if (fit_all_adj[1,4]<0.01)  # Significance level: 0.01, 0.05
  {
    # Counting the number of unexplainable factors in each category
    if(j %in% c(5,23,24,25,34,45,46,91,100,110,111,123))
    {
      momentum<-momentum+1
    }
    if(j %in% c(1,3,4,7,8,19,20,22,27,29,30,32,48,49,67,74,105,106,107,108,121,135,137,141))
    {
      value_versus_growth<-value_versus_growth+1
    }
    if(j %in% c(31,59,60,64,66,68,69,71,72,73,77,78,79,80,81,82,83,84,85,86,87,88,89,92,95,96,97,98,99,103,109,113,114,115,133,134,142,146,147))
    {
      investment<-investment+1
    }
    if(j %in% c(26,33,42,43,47,50,90,116,117,118,119,120,127,138,145,148))
    {
      profitability<-profitability+1
    }
    if(j %in% c(9,10,11,12,13,14,15,16,17,36,37,38,39,40,41,44,55,56,57,58,65,70,75,76,101,102,104,112,125,126,129,130,132,136,139,143,144,150))
    {
      intangibles<-intangibles+1
    }
    if(j %in% c(2,6,18,21,28,35,51,52,53,54,61,62,63,93,94,122,124,128,131,140,149))
    {
      trading_frictions<-trading_frictions+1
    }
    index<-append(index,j)  
    num<-num+1
    }
  
  # Accumulate the value of alpha  
  if(j %in% c(5,23,24,25,34,45,46,91,100,110,111,123))
  {
    alpha_momentum<-alpha_momentum+abs(summary(fit_all)$coef[1,1])
    num1<-num1+1
  }
  if(j %in% c(1,3,4,7,8,19,20,22,27,29,30,32,48,49,67,74,105,106,107,108,121,135,137,141))
  {
    alpha_value_versus_growth<-alpha_value_versus_growth+abs(summary(fit_all)$coef[1,1])
    num2<-num2+1
  }
  if(j %in% c(31,59,60,64,66,68,69,71,72,73,77,78,79,80,81,82,83,84,85,86,87,88,89,92,95,96,97,98,99,103,109,113,114,115,133,134,142,146,147))
  {
    alpha_investment<-alpha_investment+abs(summary(fit_all)$coef[1,1])
    num3<-num3+1
  }
  if(j %in% c(26,33,42,43,47,50,90,116,117,118,119,120,127,138,145,148))
  {
    alpha_profitability<-alpha_profitability+abs(summary(fit_all)$coef[1,1])
    num4<-num4+1
  }
  if(j %in% c(9,10,11,12,13,14,15,16,17,36,37,38,39,40,41,44,55,56,57,58,65,70,75,76,101,102,104,112,125,126,129,130,132,136,139,143,144,150))
  {
    alpha_intangibles<-alpha_intangibles+abs(summary(fit_all)$coef[1,1])
    num5<-num5+1
  }
  if(j %in% c(2,6,18,21,28,35,51,52,53,54,61,62,63,93,94,122,124,128,131,140,149))
  {
    alpha_trading_frictions<-alpha_trading_frictions+abs(summary(fit_all)$coef[1,1])
    num6<-num6+1
  }
  alpha_all<-alpha_all+abs(summary(fit_all)$coef[1,1])
  num0 <-num0+1
}

# Output the number of unexplainable factors in each category
summary<-c(num,momentum,value_versus_growth,investment,profitability,intangibles,trading_frictions)
summary

# Output the mean absolute alpha
alpha<-c(alpha_all/num0,alpha_momentum/num1,alpha_value_versus_growth/num2,alpha_investment/num3,alpha_profitability/num4,alpha_intangibles/num5,alpha_trading_frictions/num6)*100
alpha

# Output the number of factors being tested
c(num0,num1,num2,num3,num4,num5,num6)

