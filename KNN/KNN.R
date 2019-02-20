##########################
# KNN for CIFAR dataset ##
##########################
rm(list = ls())

## Load train and prediction function
setwd('C:/Users/eugene/Desktop/Machine_Learning_with_R/KNN')
source('fn/train.knn.R')   
source('fn/predict.knn.R')
source('fn/predict.knn.par.R')

## How did knn_sample.RData is saved ##
## (The way I saved the sample CIFAR data) ##
library(keras)
cifar10 <- dataset_cifar10()
str(cifar10) # Check the structure of data

#### Load train_x,y ####
train_x <- cifar10$train$x # 50000
train_y <- cifar10$train$y # 10000

# Set the number of train and valid set
n.train <- 5000
n.valid <- 3000
# assign random index for train and validation set
set.seed(1)
idx <- sample(1:50000, n.train + n.valid, replace = FALSE)
trn.idx <- idx[1:n.train]
val.idx <- idx[c(n.train+1):length(idx)]
# Train set
train.x <- train_x[trn.idx,,,]
train.y <- train_y[trn.idx]
# Validation set
val.x <- train_x[val.idx,,,]
val.y <- train_y[val.idx]

#### Load test_x,y ####
test_x <- cifar10$test$x
test_y <- cifar10$test$y

# Sampling test set
set.seed(1)
n.test <- 1000
tst.idx <- sample(1:10000, n.test)
test.x <- test_x[tst.idx,,,]
test.y <- test_y[tst.idx]

#### Save Data ####
save(train.x, train.y, val.x, val.y, test.x, test.y, file="C:/Users/eugene/Desktop/Machine_Learning_with_R/KNN/knn_sample.RData")

################################
## Practice KNN on CIFAR data ##
################################
####========= load Data =========####
load(file="knn_sample.RData")

# Check the data set
dim(train.x) # 5000 training set
dim(test.x) # 1000 test set
dim(val.x) # 3000 validation set

###========= Performance Check btw non-parallel and parallel =========###
obj <- train.knn(train.x, train.y)
k <- 2

# (1) using original KNN
system.time(
obj.trn <- predict.knn(obj, test.x, k = k, distance = "L1")
)# 721.02 seconds elapsed

# (2) using parallelized KNN
system.time(
obj.trn.par <- predict.knn.par(obj, test.x, k = k, distance = "L1")
) # 524 seconds elapsed

###========= Validation set =========###
# Evaluate on validation set
correct <- matrix(0,10,1)
for(kk in 1:10){
  obj.val <- predict.knn.par(obj, test.x, k = kk, distance = "L1")
  correct[kk] <- sum(test.y == obj.val$predict)
}
opt.k <- which.max(correct)
print(paste0("The Accuracy of k = ",which.max(correct)," is ", max(correct)/length(test.y)))

###========= Test set =========###
# Pick one hyper parameter and apply to test set
obj.tst <- predict.knn(obj, test.x, k = opt.k, distance = "L1")

# Check the performance by accuarcy
obj.tst$predict
sum(test.y == obj.tst$predict)
length(test.y)








