##################################
## NN for binary classification ##
##################################
# This is for practice (Guideline of how to use uploaded Neural Net code)
# Load the functions related with Neaural Net
rm(list=ls())
setwd("C:/Users/eugene/Desktop/NN")
source('fn/train.NN.R')    # Train Neural Network
source('fn/w.init.R')      # weight initialization
source('fn/cd.weight.R')   # CD Algorithm for weight Adjustment (used in train function)
source('fn/w.update.R')    # Weight update (used in train function)
source('fn/sigmoid.R')     # Calculate Sigmoid (used in train function)
source('fn/predict.NN.R')  # predict Neural Network
source('fn/ROC.R')         # Roc function and confusion matrix

# Load the data
train <- read.table("data/trn.txt")
test <- read.table("data/tst.txt")

# split input and output
in.trn <- as.matrix(train[,1:13])
out.trn <- as.matrix(train[,14])

# shape of train data
trn.shape <- dim(in.trn) # 60290?? ??????, 13????(60290 X 13)

# TRAIN
obj.train.nn <- train.NN(train.x = in.trn, train.y = out.trn, 
                         w.type = "RBM", s.d = 0, layer.units = c(3,4,3), 
                         update = "Adam", learning = 0.001, hyper.p = c(0.9,0.999), batch = 1, epoch = 1)

# TEST
in.tst <- as.matrix(test[,1:13])
out.tst <- as.matrix(test[,14])
obj.test.nn <- predict.NN(test.x = in.tst, obj = obj.train.nn)

# Make a histogram for predicted value
# table(obj.test.nn$output)
hist(obj.test.nn$output*10^6)

predict_y  <-  matrix(1, nrow = dim(obj.test.nn$output)[1],ncol = dim(obj.test.nn$output)[2])

######################
## create ROC curve ##
######################
# find out the optimal cutting point
rslt <- ROC(x = 0.507, predicted.y = obj.test.nn$output, true.y = out.tst)
rslt$error
rslt$ROC
rslt$confusion
rslt$error
dots <- rbind(ROC(0, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC, 
             ROC(0.51605214, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             ROC(0.516052165, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             ROC(0.51605216511, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             ROC(0.7, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             ROC(1, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC)
colnames(dots) <- c("FPR","TPR")
dots <- data.frame(dots)

# Draw ROC curve
plot(dots,type="o",col="blue")
line <- lines(x = c(0,1), y = c(1,0), col = 2, lty = 2, lwd = 2)


for(i in 1:dim(output)[1])
{
  if (output[i]>=0.50543){
    predict_y[i] <- "1"
  }
  else predict_y[i] <- "0"
}

# Calculate the error rate
error <- (1-mean(predict_y==out.tst))*100
error


# Confusion matrix
rslt$confusion

## Conclusion ##
# With this test set Adam update showed the best performance

