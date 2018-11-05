##########################
# KNN for CIFAR dataset ##
##########################
rm(list = ls())
# install.packages("keras")
library(keras)
# install_keras()

## Load train and prediction function
setwd('C:/Users/eugene/Desktop/cs231n/knn')
source('train.knn.R')     # GP with identity covariance (beta)
source('predict.knn.R')  # GP with trivial covariance  (beta)

###################
## Load the Data ##
###################
cifar10 <- dataset_cifar10()
str(cifar10) # Check the structure of data

## Load train_x,y
train_x <- mnist_lst$train$x
train_y <- mnist_lst$train$y

## Load test_x,y
test_x <- mnist_lst$test$x
test_y <- mnist_lst$test$y

## Data Flattening (2D array --> 1D array)
train_x <- array(as.numeric(train_x), dim = c(dim(train_x)[[1]], 784))
test_x <- array(as.numeric(test_x), dim = c(dim(test_x)[[1]], 784))
