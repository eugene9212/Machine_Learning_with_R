##########################
# KNN for CIFAR dataset ##
##########################
rm(list = ls())
# install.packages("keras")
library(keras)
# install_keras()

## Load train and prediction function
setwd('C:/Users/eugene/Desktop/Machine_Learning_with_R/KNN')
source('fn/train.knn.R')   
source('fn/predict.knn.R') 

###################
## Load the Data ##
###################
cifar10 <- dataset_cifar10()
str(cifar10) # Check the structure of data

# Setting hyperparameter K with train, validation, and test sets.

#### Load train_x,y ####
train_x <- cifar10$train$x
train_y <- cifar10$train$y

# Sampling train set
set.seed(1)
n.train <- 2000
n.valid <- 1000
idx <- sample(50000, n.train + n.valid, replace = FALSE)
trn.idx <- idx[1:n.train]
train.x <- train_x[trn.idx,,,]
train.y <- train_y[trn.idx]

# Sampling validation set
set.seed(1)
val.idx <- idx[(n.train+1):(n.train + n.valid)]
valid.x <- train_x[val.idx,,,]
valid.y <- train_y[val.idx]

# Check whether samples are balanced
table(train.y)

#### Load test_x,y ####
test_x <- cifar10$test$x
test_y <- cifar10$test$y

# Sampling test set
set.seed(1)
n.test <- 500
tst.idx <- sample(10000, n.test)
test.x <- test_x[tst.idx,,,]
test.y <- test_y[tst.idx]

#### Save Data ####
save(train.x, train.y, test.x, test.y, file="C:/Users/eugene/Desktop/cs231n/knn/knn_1000_500.RData")

# Check the data structure
test.x[1,,,1] # test.x[sample number, row, column, channel number]

#### load Data ####
load(file="knn_1000_500.RData")

obj <- train.knn(train.x, train.y)

### Train set ###
# Try several Hyperparameter with train set
obj <- train.knn(train.x, train.y)
obj.trn <- predict.knn(obj, test.x, k = 2, distance = "L1")

# Check the performance by accuarcy
obj.tst$predict
sum(test.y == obj2$predict)
length(test.y)

### Validation set ###
# Evaluate on validation set
obj.val <- predict.knn(obj, valid.x, k = 2, distance = "L1")

# Check the performance by accuarcy
obj.val$predict
sum(valid.y == obj.val$predict)
length(valid.y)

### Test set ###
# Pick one hyper parameter and apply to test set
obj.tst <- predict.knn(obj, test.x, k = 2, distance = "L1")

# Check the performance by accuarcy
obj.tst$predict
sum(test.y == obj.tst$predict)
length(test.y)








