#Confusion matrix
true = out.tst[predict_y == out.tst]
false = out.tst[predict_y != out.tst]
length(true)
length(false)
sum(length(true),length(false))
TP = sum(true==1)
TN = sum(true==0)
FP = sum(false==0)
FN = sum(false==1)

FPR = FP/(TN+FP)
FPR
TPR = TP/(TP+FN)
TPR

confusion_m=matrix(c(TP,FP,FN,TN),nrow=2)
confusion_m
colnames(confusion_m)<-c("P","N")
rownames(confusion_m)<-c("P","N")
confusion_m