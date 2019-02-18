########
## NN ##
########
rm(list=ls())
setwd("C:/Users/eugene/Desktop/Machine_Learning_with_R/NN")
source('fn/train.NN.R')    # Train Neural Network
source('fn/w.init.R')      # weight initialization
source('fn/cd.weight.R')   # CD Algorithm for weight Adjustment (used in train function)
source('fn/w.update.R')    # Weight update (used in train function)
source('fn/sigmoid.R')     # Calculate Sigmoid (used in train function)
source('fn/predict.NN.R')  # predict Neural Network
source('fn/ROC.R')         # Roc function and confusion matrix

##Load the data
train=read.table("data/trn.txt")
test=read.table("data/tst.txt")

# split input and output
in.trn = as.matrix(train[,1:13])
out.trn = as.matrix(train[,14])

# shape of train data
trn.shape <- dim(in.trn) # 60290개 데이터, 13차원(60290 X 13)

# TRAIN
obj.train.nn <- train.NN(train.x = in.trn, train.y = out.trn, 
                         w.type = "RBM", s.d = 0, layer.units = c(3,4,3), 
                         update = "momentum", learning = 0.01, momen = 0.007, epoch = 5)

# TEST
in.tst <- as.matrix(test[,1:13])
out.tst <- as.matrix(test[,14])
obj.test.nn <- predict.NN(test.x = in.tst, test.y = out.tst, obj = obj.train.nn)

# table(obj.test.nn$output)
hist(obj.test.nn$output*10^6)

predict_y = matrix(1, nrow = dim(obj.test.nn$output)[1],ncol = dim(obj.test.nn$output)[2])

######################
## create ROC curve ##
######################

rslt <- roc_function(x = 0.1156, predicted.y = obj.test.nn$output, true.y = out.tst)
rslt$error
rslt$ROC
rslt$confusion
rslt$error
dots = rbind(roc_function(0, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC, 
             roc_function(0.72470386782, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             roc_function(0.7247038678239, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             roc_function(0.72470386782365, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             roc_function(0.7247038679, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             roc_function(0.72470386794, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC, 
             roc_function(0.72470386794, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC,
             roc_function(1, predicted.y = obj.test.nn$output, true.y = out.tst)$ROC)
colnames(dots)<-c("FPR","TPR")
dots<-data.frame(dots)
plot(dots,type="o",col="blue")
line=lines(x = c(0,1), y = c(1,0), col = 2, lty = 2, lwd = 2)


for(i in 1:dim(output)[1])
{
  if (output[i]>=0.50543){
    predict_y[i]="1"
  }
  else predict_y[i]="0"
}


error = (1-mean(predict_y==out.tst))*100
error


#Confusion matrix
rslt$confusion

  