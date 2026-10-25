#################################################################################
###
###  This file provides how to draw the heatmap figure.
###
#################################################################################
rm(list=ls())

library(ggplot2)
library(reshape) 
library(openxlsx)
library(gplots)
library(ggpubr)

out <- read.table('robustness_tuning.xlsx',header=TRUE)
list1<-seq(1,448,3)
list2<-seq(2,449,3)
list3<-seq(3,450,3)
out_all_ape <- out[,list1]
out_all_std <- out[,list2]
out_all_tstat <- out[,list3]

#Convert the original results
matrixs<-list()
for (i in 1:150) {
  matrixs[[i]]<-matrix(nrow = 10,ncol = 10)
}

for (i in 1:150) 
{
  for (j in 1:10)
  {
  matrixs[[i]][j,]<-t(out_all_tstat[(10*(j-1)+1):(10*j),i])
  }
}

factor<- abs(matrixs[[98]])
colnames(factor) <- as.character(2:11)   
rownames(factor) <- as.character(1:10) 
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g1<-ggp+ggtitle("Net debt finance")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[132]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g2<-ggp+ggtitle("Operating Leverage")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[25]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g3<-ggp+ggtitle("36-month momentum")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[96]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g4<-ggp+ggtitle("Composite Equity Issuance")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[13]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g5<-ggp+ggtitle("% change sales-to-inventory")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[119]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g6<-ggp+ggtitle("Industry-adjusted change in asset turnover")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[110]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g7<-ggp+ggtitle("Abnormal earnings announcement volume")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[3]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g8<-ggp+ggtitle("Earnings to price")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[37]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g9<-ggp+ggtitle("% change in sales - % change in inventory")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

factor<- abs(matrixs[[73]])
colnames(factor) <- as.character(2:11)
rownames(factor) <- as.character(1:10)
factor <- melt(as.matrix(factor))
ggp <- ggplot(factor, aes(factor(X1), factor(X2)))+geom_tile(aes(fill = value))
g10<-ggp+ggtitle("Changes in Net Operating Assets")+theme(plot.title = element_text(hjust = 0.5))+xlab("unit")+ylab("split")+scale_fill_gradient2(low="yellow", high="royalblue4", mid="royalblue", midpoint=3)

# draw the heatmap figure
ggarrange(g1, g2,g3, g4,g5, g6,g7, g8,g9, g10,ncol = 2, nrow = 5)




