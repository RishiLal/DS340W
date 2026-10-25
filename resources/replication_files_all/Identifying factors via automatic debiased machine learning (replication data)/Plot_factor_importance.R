#################################################################################
###
###  This file provides how to draw the factor importance figure.
###
#################################################################################
rm(list=ls())

library(ggplot2)
library(reshape) 
library(openxlsx)
library(gplots)
library(ggpubr)

out <- read.table('out_recursive_nn_3x2.xlsx',header=TRUE)
list1<-seq(1,448,3)
list2<-seq(2,449,3)
list3<-seq(3,450,3)
out_all_ape1 <- t(out[20,list1])
out_all_std1 <- t(out[20,list2])
out_all_tstat1 <- t(out[20,list3])
out <- read.table('out_recursive_nn_5x5.xlsx',header=TRUE)
list1<-seq(1,448,3)
list2<-seq(2,449,3)
list3<-seq(3,450,3)
out_all_ape2 <- t(out[20,list1])
out_all_std2 <- t(out[20,list2])
out_all_tstat2 <- t(out[20,list3])
out <- read.table('out_recursive_nn_202.xlsx',header=TRUE)
list1<-seq(1,448,3)
list2<-seq(2,449,3)
list3<-seq(3,450,3)
out_all_ape3 <- t(out[20,list1])
out_all_std3 <- t(out[20,list2])
out_all_tstat3 <- t(out[20,list3])
out <- read.table('out_recursive_rf_3x2.xlsx',header=TRUE)
list1<-seq(1,448,3)
list2<-seq(2,449,3)
list3<-seq(3,450,3)
out_all_ape4 <- t(out[20,list1])
out_all_std4 <- t(out[20,list2])
out_all_tstat4 <- t(out[20,list3])
out <- read.table('out_recursive_rf_5x5.xlsx',header=TRUE)
list1<-seq(1,448,3)
list2<-seq(2,449,3)
list3<-seq(3,450,3)
out_all_ape5 <- t(out[20,list1])
out_all_std5 <- t(out[20,list2])
out_all_tstat5 <- t(out[20,list3])
out <- read.table('out_recursive_rf_202.xlsx',header=TRUE)
list1<-seq(1,448,3)
list2<-seq(2,449,3)
list3<-seq(3,450,3)
out_all_ape6 <- t(out[20,list1])
out_all_std6 <- t(out[20,list2])
out_all_tstat6 <- t(out[20,list3])

#rank data
out<-cbind(seq(1,150),cbind(abs(out_all_tstat1),abs(out_all_tstat2),abs(out_all_tstat3),abs(out_all_tstat4),abs(out_all_tstat5),abs(out_all_tstat6)))
out<-out[order(out[,2],decreasing = T),]
out<-cbind(out,seq(1,150))
out<-out[order(out[,3],decreasing = T),]
out<-cbind(out,seq(1,150))
out<-out[order(out[,4],decreasing = T),]
out<-cbind(out,seq(1,150))
out<-out[order(out[,5],decreasing = T),]
out<-cbind(out,seq(1,150))
out<-out[order(out[,6],decreasing = T),]
out<-cbind(out,seq(1,150))
out<-out[order(out[,7],decreasing = T),]
out<-cbind(out,seq(1,150))

out<-out[,c(1,8,9,10,11,12,13)]
out<-out[order(out[,1]),]
factor_importance<-t(out[,2:7])

summary<- read.table('summary.csv',header=TRUE,sep=",")
factorname<-summary[,1]
colnames(factor_importance) <- factorname
rownames(factor_importance) <- c("NN3+LASSO(3X2)","NN3+LASSO(5X5)","NN3+LASSO(202)","RF+LASSO(3X2)","RF+LASSO(5X5)","RF+LASSO(202)")

importance<-cbind(t(factor_importance),t(factor_importance)[,1]+t(factor_importance)[,2]+t(factor_importance)[,3]+t(factor_importance)[,4]+t(factor_importance)[,5]+t(factor_importance)[,6])
importance<-importance[order(importance[,7],decreasing = T),]

factor_importance1<-t(importance[1:75,1:6])
factor_importance1 <- melt(as.matrix(factor_importance1))
ggp <- ggplot(factor_importance1, aes(X1, X2)) +                   
  geom_tile(aes(fill = value))
g1<-ggp+ggtitle("factor_importance")+theme(plot.title = element_text(hjust = 0.5))+scale_fill_gradient2(low="darkblue", high="slategray1", mid="skyblue", midpoint=75)

g1

factor_importance2<-t(importance[76:150,1:6])
factor_importance2 <- melt(as.matrix(factor_importance2))
ggp <- ggplot(factor_importance2, aes(X1, X2)) +                   
  geom_tile(aes(fill = value))
g1<-ggp+ggtitle("factor_importance")+theme(plot.title = element_text(hjust = 0.5))+scale_fill_gradient2(low="darkblue", high="slategray1", mid="skyblue", midpoint=75)

g1
