########
## NN ##
########
rm(list=ls())
setwd("C:/Users/eugene/Desktop/Machine_Learning_with_R/NN")

##Load the data
train=read.table("data/trn.txt")
test=read.table("data/tst.txt")

#split input and output
in.trn=train[,1:13]
out.trn=train[,14]
in.tst=test[,1:13]
out.tst=test[,14]

## SETTING an hidden units
desig.unit = c(3, 4, 3)

# create a weight1 vectors for each hidden nodes
for (i in 1:desig.unit[1]){
  assign(paste("weight1",i,sep = "_"),as.matrix(sample(-100:100, 13 , replace=T)/10^6))
}

# form a matrix of weights
weight1 = matrix(c(weight1_1, weight1_2, weight1_3),nrow=desig.unit[1],ncol=13,byrow=T)

# create bias vector for v and h
bias_v = matrix(1,nrow = 1,ncol = 13)
bias_h = matrix(1,nrow = 1,ncol = desig.unit[1])
learning = 0.01
############################
## Start CD-1 for weight1 ##
############################

for (m in 1:dim(in.trn)[1])
{
  v1_value = as.matrix(in.trn[m, ])
  
  # Probability for h1
  if (dim(weight1)[1] == 13) {
    weight1 = weight1
  }
  else
    weight1 = t(weight1)
  
  h1_prob = 1 / (1 + exp(-(v1_value %*% weight1 + bias_h)))
  
  # value of h1
  h1_value = matrix(0, nrow = 1, ncol = desig.unit[1]) ## need correction
  for (i in 1:desig.unit[1])
  {
    if (h1_prob[i] >= runif(1, min = 0, max = 1)) {
      h1_value[i] = 1
    }
    else
      h1_value[i] = 0
  }
  
  # probability for v2
  weight1 = t(weight1)
  v2_prob = 1 / (1 + exp(-(h1_value %*% weight1 + bias_v)))
  
  # value of v2
  v2_value = matrix(0, nrow = 1, ncol = 13)
  for (i in 1:length(v2_prob))
  {
    if (v2_prob[i] >= runif(1, min = 0, max = 1)) {
      v2_value[i] = 1
    }
    else
      v2_value[i] = 0
  }
  
  # Probability for h2
  weight1 = t(weight1)
  
  h2_prob = 1 / (1 + exp(-(v2_value %*% weight1 + bias_h)))
  
  # value of h1
  h2_value = matrix(0, nrow = 1, ncol = desig.unit[1])
  for (i in 1:length(h2_value))
  {
    if (h2_prob[i] >= runif(1, min = 0, max = 1)) {
      h2_value[i] = 1
    }
    else
      h2_value[i] = 0
  }

  # weight1 update
  for (i in 1:13) {
    for (j in 1:desig.unit[1]) {
      weight1[i, j] = weight1[i, j] - learning * (v1_value[i] * h1_prob[j]
                                                  - v2_value[i] * h2_prob[j])
      bias_v = bias_v + learning * (v1_value - v2_value)
      bias_h = bias_h + learning * (h1_prob - h2_prob)
    }
  }
}

######################
## CD-1 for weight2 ##
######################

# create a weight1 vectors for each hidden nodes
for (i in 1:desig.unit[2]){
  assign(paste("weight2",i,sep = "_"),as.matrix(sample(-100:100, desig.unit[1] , replace=T)/10^6))
}

# form a matrix of weights
weight2 = matrix(c(weight2_1, weight2_2, weight2_3, weight2_4),nrow=desig.unit[1],ncol=desig.unit[2],byrow=T)

# create bias vector for v and h
bias_v = matrix(1,nrow = 1,ncol = desig.unit[1])
bias_h = matrix(1,nrow = 1,ncol = desig.unit[2])

# calculate an input
in.trn = as.matrix(in.trn)
input = 1/(1+exp(-in.trn%*%weight1))

for (m in 1:dim(input)[1])
{
  v1_value = t(as.matrix(input[m, ]))
  
  # Probability for h1
  if (dim(weight2)[1] == desig.unit[1]) {
    weight2 = weight2
  }
  else
    weight2 = t(weight2)
  
  h1_prob = 1 / (1 + exp(-(v1_value %*% weight2 + bias_h)))
  
  # value of h1
  h1_value = matrix(0, nrow = 1, ncol = desig.unit[2]) ## need correction
  for (i in 1:desig.unit[2])
  {
    if (h1_prob[i] >= runif(1, min = 0, max = 1)) {
      h1_value[i] = 1
    }
    else
      h1_value[i] = 0
  }
  
  # probability for v2
  weight2 = t(weight2)
  v2_prob = 1 / (1 + exp(-(h1_value %*% weight2 + bias_v)))
  
  # value of v2
  v2_value = matrix(0, nrow = 1, ncol = 2)
  for (i in 1:length(v2_prob))
  {
    if (v2_prob[i] >= runif(1, min = 0, max = 1)) {
      v2_value[i] = 1
    }
    else
      v2_value[i] = 0
  }
  
  # Probability for h2
  weight2 = t(weight2)
  
  h2_prob = 1 / (1 + exp(-(v2_value %*% weight2 + bias_h)))
  
  # value of h1
  h2_value = matrix(0, nrow = 1, ncol = desig.unit[2])
  for (i in 1:length(h2_value))
  {
    if (h2_prob[i] >= runif(1, min = 0, max = 1)) {
      h2_value[i] = 1
    }
    else
      h2_value[i] = 0
  }

  # weight1 update
  for (i in 1:desig.unit[1]) {
    for (j in 1:desig.unit[2]) {
      weight2[i, j] = weight2[i, j] - learning * (v1_value[i] * h1_prob[j]
                                                  - v2_value[i] * h2_prob[j])
      bias_v = bias_v + learning * (v1_value - v2_value)
      bias_h = bias_h + learning * (h1_prob - h2_prob)
    }
  }
}


##################
## third layer ##
##################

# create a weight3 vectors for each hidden nodes
for (i in 1:desig.unit[3]){
  assign(paste("weight3",i,sep = "_"),as.matrix(sample(-100:100, desig.unit[2] , replace=T)/10^6))
}

# form a matrix of weights
weight3 = matrix(c(weight1_1, weight1_2, weight1_3),nrow=desig.unit[2],ncol=desig.unit[3],byrow=T)

# create bias vector for v and h
bias_v = matrix(1,nrow = 1,ncol = desig.unit[2])
bias_h = matrix(1,nrow = 1,ncol = desig.unit[3])

# calculate an input
input = 1/(1+exp(-in.trn%*%weight1))
input1 = 1/(1+exp(-input%*%weight2))


for (m in 1:dim(input1)[1])
{
  v1_value = t(as.matrix(input1[m, ]))
  
  # Probability for h1
  if (dim(weight3)[1] == desig.unit[2]) {
    weight3 = weight3
  }
  else
    weight3 = t(weight3)
  
  h1_prob = 1 / (1 + exp(-(v1_value %*% weight3 + bias_h)))
  
  # value of h1
  h1_value = matrix(0, nrow = 1, ncol = desig.unit[3]) ## need correction
  for (i in 1:desig.unit[3])
  {
    if (h1_prob[i] >= runif(1, min = 0, max = 1)) {
      h1_value[i] = 1
    }
    else
      h1_value[i] = 0
  }
  
  # probability for v2
  weight3 = t(weight3)
  v2_prob = 1 / (1 + exp(-(h1_value %*% weight3 + bias_v)))
  
  # value of v2
  v2_value = matrix(0, nrow = 1, ncol = 3)
  for (i in 1:length(v2_prob))
  {
    if (v2_prob[i] >= runif(1, min = 0, max = 1)) {
      v2_value[i] = 1
    }
    else
      v2_value[i] = 0
  }
  
  # Probability for h2
  weight3 = t(weight3)
  
  h2_prob = 1 / (1 + exp(-(v2_value %*% weight3 + bias_h)))
  
  # value of h1
  h2_value = matrix(0, nrow = 1, ncol = desig.unit[3])
  for (i in 1:length(h2_value))
  {
    if (h2_prob[i] >= runif(1, min = 0, max = 1)) {
      h2_value[i] = 1
    }
    else
      h2_value[i] = 0
  }

  # weight1 update
  for (i in 1:desig.unit[2]) {
    for (j in 1:desig.unit[3]) {
      weight3[i, j] = weight3[i, j] - learning * (v1_value[i] * h1_prob[j]
                                                  - v2_value[i] * h2_prob[j])
      bias_v = bias_v + learning * (v1_value - v2_value)
      bias_h = bias_h + learning * (h1_prob - h2_prob)
    }
  }
}

##################
## Fourth layer ##
##################

# form a matrix of weights
weight4 = matrix(sample(-100:100, desig.unit[3] , replace=T)/10^6,nrow=desig.unit[3],1,byrow=T)

# create bias vector for v and h
bias_v = matrix(1,nrow = 1,ncol = desig.unit[3])
bias_h = matrix(1,nrow = 1,ncol = 1)

# calculate an input
input = 1/(1+exp(-in.trn%*%weight1))
input1 = 1/(1+exp(-input%*%weight2))
input2 = 1/(1+exp(-input1%*%weight3))

for (m in 1:dim(input2)[1])
{
  v1_value = t(as.matrix(input2[m, ]))
  
  # Probability for h1
  if (dim(weight4)[1] == desig.unit[3]) {
    weight4 = weight4
  }
  else
    weight4 = t(weight4)
  
  h1_prob = 1 / (1 + exp(-(v1_value %*% weight4 + bias_h)))
  
  # value of h1
  h1_value = matrix(0, nrow = 1, ncol = 1) ## need correction
  for (i in 1:1)
  {
    if (h1_prob[i] >= runif(1, min = 0, max = 1)) {
      h1_value[i] = 1
    }
    else
      h1_value[i] = 0
  }
  
  # probability for v2
  weight4 = t(weight4)
  v2_prob = 1 / (1 + exp(-(h1_value %*% weight4 + bias_v)))
  
  # value of v2
  v2_value = matrix(0, nrow = 1, ncol = desig.unit[3])
  for (i in 1:length(v2_prob))
  {
    if (v2_prob[i] >= runif(1, min = 0, max = 1)) {
      v2_value[i] = 1
    }
    else
      v2_value[i] = 0
  }
  
  # Probability for h2
  weight4 = t(weight4)
  
  h2_prob = 1 / (1 + exp(-(v2_value %*% weight4 + bias_h)))
  
  # value of h1
  h2_value = matrix(0, nrow = 1, ncol = 1)
  for (i in 1:length(h2_value))
  {
    if (h2_prob[i] >= runif(1, min = 0, max = 1)) {
      h2_value[i] = 1
    }
    else
      h2_value[i] = 0
  }
  #    prev_w = weight1[1, 1]
  # weight1 update
  for (i in 1:desig.unit[3]) {
    for (j in 1:1) {
      weight4[i, j] = weight4[i, j] - learning * (v1_value[i] * h1_prob[j]
                                                  - v2_value[i] * h2_prob[j])
      bias_v = bias_v + learning * (v1_value - v2_value)
      bias_h = bias_h + learning * (h1_prob - h2_prob)
    }
  }
}

#################
## Feedforward ##
#################

## Each node's input and output
for (j in 1:length(desig.unit)){
  assign(paste("layer",j,"nodeIN",sep = ""),matrix(0, nrow=1, ncol=desig.unit[j]))
  assign(paste("layer",j,"nodeOUT",sep = ""),matrix(0, nrow=1, ncol=desig.unit[j]))
}

## bias for each node
for (j in 1:length(desig.unit)){
  assign(paste("layer",j,"nodeINbias",sep = ""),matrix(1, nrow=1, ncol=desig.unit[j]))
}
yINbias = matrix(1,nrow =1, ncol =1)

# preprocessing
layer0nodeOUT = as.matrix(in.trn)
weight1 = as.matrix(weight1)
weight2 = as.matrix(weight2)
weight3 = as.matrix(weight3)
weight4 = as.matrix(weight4)

yout = matrix(0,nrow=1,ncol=dim(in.trn)[1])
out.trn = as.matrix(out.trn)

# setting momentum and learning rate
learn = 0.001 # learning rate
momen = 0.0007

pre_delta1 = matrix(0, nrow= dim(weight1)[1] , ncol= dim(weight1)[2]) 
pre_delta2 = matrix(0, nrow= dim(weight2)[1] , ncol= dim(weight2)[2])
pre_delta3 = matrix(0, nrow= dim(weight3)[1] , ncol= dim(weight3)[2])
pre_delta4 = matrix(0, nrow= dim(weight4)[1] , ncol= dim(weight4)[2])
## Feedforward
for (m in 1:dim(in.trn)[1]){
  layer0nodeOUT = in.trn[m,]
  layer1nodeIN = layer0nodeOUT%*%weight1
  layer1nodeOUT = 1/(1+exp(-layer1nodeIN-layer1nodeINbias))
  
  layer2nodeIN = layer1nodeOUT%*%weight2
  layer2nodeOUT = 1/(1+exp(-layer2nodeIN-layer2nodeINbias))
  
  layer3nodeIN = layer2nodeOUT%*%weight3
  layer3nodeOUT = 1/(1+exp(-layer3nodeIN-layer3nodeINbias))
  
  yIN = layer3nodeOUT%*%weight4
  #  yIN = as.matrix(yIN)
  yout[m] = 1/(1+exp(-yIN-yINbias)) # This is prob. of predict_y
  
  
  #####################
  ## Backpropagation ##
  #####################
  
  result = out.trn[m]
  error4 = result-yout[m]
  error3 = error4*t(weight4)*layer3nodeOUT*(1-layer3nodeOUT)
  error2 = error3%*%t(weight3)*layer2nodeOUT*(1-layer2nodeOUT)
  error1 = error2%*%t(weight2)*layer1nodeOUT*(1-layer1nodeOUT)
  
  
  # weight update
  for (i in 1:dim(weight1)[1]){
    for (j in 1:dim(weight1)[2]){
      weight1[i,j] = weight1[i,j] + learn*layer0nodeOUT[i]*error1[j]+ momen*pre_delta1[i,j]
      layer1nodeINbias[j] = layer1nodeINbias[j] + learn*layer0nodeOUT[i]*error1[j]
      pre_delta1[i,j] = learn*layer0nodeOUT[i]*error1[j]
    }
  }
  for (i in 1:dim(weight2)[1]){
    for (j in 1:dim(weight2)[2]){
      weight2[i,j] = weight2[i,j] + learn*layer1nodeOUT[i]*error2[j]+ momen*pre_delta2[i,j]
      layer2nodeINbias[j] = layer2nodeINbias[j] + learn*layer1nodeOUT[i]*error2[j]
      pre_delta2[i,j] = learn*layer1nodeOUT[i]*error2[j]
    }
  }
  for (i in 1:dim(weight3)[1]){
    for (j in 1:dim(weight3)[2]){
      weight3[i,j] = weight3[i,j] + learn*layer2nodeOUT[i]*error3[j]+ momen*pre_delta3[i,j]
      layer3nodeINbias[j] = layer3nodeINbias[j] + learn*layer2nodeOUT[i]*error3[j]
      pre_delta3[i,j] = learn*layer2nodeOUT[i]*error3[j]
    }
  }
  for (i in 1:dim(weight4)[1]){
    for (j in 1:dim(weight4)[2]){
      weight4[i,j] = weight4[i,j] + learn*layer3nodeOUT[i]*error4[j] + momen*pre_delta4[i,j]
      yINbias = yINbias + learn*layer3nodeOUT[i]*error4[j]
      pre_delta4[i,j] = learn*layer3nodeOUT[i]*error4[j]
    }
  }
}

##############
## predict y##
##############
in.tst = as.matrix(in.tst)
out.tst = as.matrix(out.tst)
layer1nodeINbias1 = matrix(rep(layer1nodeINbias,each =1, time= 18490),nrow=18490,ncol=desig.unit[1],byrow=T)
layer2nodeINbias1 = matrix(rep(layer2nodeINbias,each =1, time= 18490),nrow=18490,ncol=desig.unit[2],byrow=T)
layer3nodeINbias1 = matrix(rep(layer3nodeINbias,each =1, time= 18490),nrow=18490,ncol=desig.unit[3],byrow=T)
yINbias1 = matrix(rep(yINbias,each =1, time= 18490),nrow=18490,ncol=1,byrow=T)

layer1 = 1/(1+exp(-in.tst %*%weight1-layer1nodeINbias1))
layer2 = 1/(1+exp(-layer1%*%weight2-layer2nodeINbias1))
layer3 = 1/(1+exp(-layer2%*%weight3-layer3nodeINbias1))
output = 1/(1+exp(-layer3%*%weight4-yINbias1))

predict_y = matrix(1, nrow = dim(output)[1],ncol = dim(output)[2])



######################
## create ROC curve ##
######################

roc_function <-function(x){
  predict_y = matrix(1, nrow = dim(output)[1],ncol = dim(output)[2])
  for(i in 1:dim(output)[1])
  {
    if (output[i]>=x){
      predict_y[i]="1"
    }
    else predict_y[i]="0"
  }
  true = out.tst[predict_y == out.tst]
  false = out.tst[predict_y != out.tst]
  
  TP = sum(true==1)
  TN = sum(true==0)
  FP = sum(false==0)
  FN = sum(false==1)
  
  FPR = FP/(TN+FP)
  TPR = TP/(TP+FN)
  
  return(c(FPR,TPR))
}
roc_function(0.50543)
dots = rbind(roc_function(0),roc_function(0.1),roc_function(0.504),
             roc_function(0.504599),roc_function(0.504999),roc_function(0.5051999),
             roc_function(0.518),roc_function(0.52)
            ,roc_function(1))
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


