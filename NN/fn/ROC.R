# ROC function #
roc_function <-function(x, predicted.y, true.y){
  
  predict_y = matrix(1, nrow = dim(predicted.y)[1],ncol = dim(predicted.y)[2])
  
  for(i in 1:dim(predicted.y)[1])
  {
    if (predicted.y[i]>=x){
      predict_y[i]=1
    }
    else predict_y[i]=0
  }

  true = out.tst[predict_y == true.y]
  false = out.tst[predict_y != true.y]
  
  TP = sum(true==1)
  TN = sum(true==0)
  FP = sum(false==0)
  FN = sum(false==1)
  
  FPR = FP/(TN+FP)
  TPR = TP/(TP+FN)
  
  # confusion matrix
  confusion_m <- matrix(c(TP,FP,FN,TN),nrow=2)
  colnames(confusion_m)<-c("P","N")
  rownames(confusion_m)<-c("P","N")
  
  # error
  error = (1-mean(predict_y==out.tst))*100
  
  # return
  obj <- list(ROC = c(FPR,TPR), error = error, confusion = confusion_m)
}