rm(list=ls())

# Load the library and set working dir.
setwd("C:/Users/eugene/Desktop/Machine_Learning_with_R/NBC")
source('fn/nbc.R')
source('fn/predict.nbc.R')

# Load the data
train <- read.table("data/trn.txt")
test <- read.table("data/tst.txt")

# Check the data
str(train) # 1 ~ 13th are variables and 14th is response variable 
str(test)

train.x <- train[,1:13]
train.y <- train[,14]

test.x <- test[,1:13]
test.y <- test[,14]

################
##  Modeling  ##
################
obj <- nbc(train.x, train.y) # Train
obj2 <- predict.nbc(obj, test.x) # Test

######################
### Error analysis ###
######################
# Empirical error on the test data
predict.y <- obj2$predict.y
error <- (1 - mean(predict.y == test.y))*100
error
y
#Confusion matrix
true <- test.y[predict.y == test.y]
false <- test.y[predict.y != test.y]

TP <- sum(true == 1)
TN <- sum(true == 0)
FP <- sum(false == 0)
FN <- sum(false == 1)

FPR <- FP/(TN+FP)
TPR <- TP/(TP+FN)

confusion_m <- matrix(c(TP,FP,FN,TN),nrow=2)
colnames(confusion_m)<-c("P","N")
rownames(confusion_m)<-c("P","N")
confusion_m

#The false positive and false negative error rates can vary by changing the prior probabilities.
likelihood <- obj2$likelihood
FNR_function <- function(x){
  
  #changing prior1 probability
  prediction <- function(x1){
    prob0 <- likelihood(x1, mean = obj$m0, cov = obj$cov0)*(1-x)
    prob1 <- likelihood(x1, mean = obj$m1, cov = obj$cov1)*(x)
    if (prob1 >= prob0){
      return (1)
    }
    else {
      return (0)
    }
  }
  
  #Empirical error on the test data
  predict.y <- apply(test.x,1,prediction)
  
  true <- test.y[predict.y == test.y]
  false <- test.y[predict.y != test.y]
  
  TP <- sum(true == 1)
  TN <- sum(true == 0)
  FP <- sum(false == 0)
  FN <- sum(false == 1)
  
  FPR <- FP/(TN+FP)
  TPR <- TP/(TP+FN)
  
  return(c(FPR,TPR))
}

# Report the ROC curve and the equal error rate (EER) of your model.
grid <- matrix(seq(0,1,0.01))
result <- apply(grid, 1, FNR_function)

plot(result[1,],result[2,],type="o",col="blue")

line <- lines(x = c(0,1), y = c(1,0), col = 2, lty = 2, lwd = 2)

## In the graph, the EER is shown.
#(when prior probability, for output=1, is 0.96)



