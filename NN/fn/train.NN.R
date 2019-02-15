#########
# train #
#########
train.NN <- function(train.x, train.y, w.dist = "normal", s.d = 0.05, layer.units = c(3, 3), learning = 0.01, momen = 0.007 ){
  
  # shape of train data
  input.dim <- dim(train.x)[2]
  
  # hyper parameters
  layers = c(input.dim, layer.units, 1)
  learning = learning # learning rate
  momen = momen
  
  # number of weights btw layers
  num.w <- length(layer.units) + 1
  
  # weight initialization
  if(w.dist == "normal"){
    for (i in 1:num.w){
      if(i == num.w){
        eval(parse(text=paste0(
          "weight",i,"<- matrix(rnorm(", layers[i]*layers[i+1],", mean=0, sd=",s.d,"),nrow =", layers[i],", ncol =", layers[i+1],")")))
        eval(parse(text=paste0(
          "bias_v",i,"<- matrix(1,nrow = 1,ncol = ", layers[i],")")))
        eval(parse(text=paste0(
          "bias_h",i,"<- matrix(1,nrow = 1,ncol = ", layers[i+1],")")))
      } else if (i != num.w){
        eval(parse(text=paste0(
          "weight",i,"<- matrix(rnorm(", layers[i]*layers[i+1],", mean=0, sd=",s.d,"),nrow =", layers[i],", ncol =", layers[i+1],")")))
        eval(parse(text=paste0(
          "bias_v",i,"<- matrix(1,nrow = 1,ncol = ", layers[i],")")))
        eval(parse(text=paste0(
          "bias_h",i,"<- matrix(1,nrow = 1,ncol = ", layers[i+1],")")))
      }
    }
  } else if (w.dist == "uniform"){ # uniform distribution sample(-100:100, 13 , replace=T)/10^6)
    for (i in 1:num.w){
      if(i == num.w){
        eval(parse(text=paste0(
          "weight",i,"<- matrix(runif(n=", layers[i]*layers[i+1],", min = -100, max= 100)/10^6,nrow =", layers[i],", ncol =", layers[i+1],")")))
        eval(parse(text=paste0(
          "bias_v",i,"<- matrix(1,nrow = 1,ncol = ", layers[i],")")))
        eval(parse(text=paste0(
          "bias_h",i,"<- matrix(1,nrow = 1,ncol = ", layers[i+1],")")))
      } else if (i != num.w){
        eval(parse(text=paste0(
          "weight",i,"<- matrix(runif(n=", layers[i]*layers[i+1],", min = -100, max= 100)/10^6,nrow =", layers[i],", ncol =", layers[i+1],")")))
        eval(parse(text=paste0(
          "bias_v",i,"<- matrix(1,nrow = 1,ncol = ", layers[i],")")))
        eval(parse(text=paste0(
          "bias_h",i,"<- matrix(1,nrow = 1,ncol = ", layers[i+1],")")))
      }
    }
  }
  
  # RBM for weight initialization (CD algorithm)
  for (j in 1:num.w){
    if(j == 1){
      # CD for weight1
      input1 <- in.trn
      weight.1 <- cd.weight(input = input1, weight = weight1, 
                            bias_v = bias_v1, bias_h = bias_h1, learning = learning)
      cd.weight1 <- weight.1$weight
    } else{
      # CD for weight2 ~
      eval(parse(text=paste0(
        "input",j,"<- sigmoid(input",j-1,",cd.weight",j-1,")")))
      eval(parse(text=paste0(
        "weight.",j,"<- cd.weight(input",j,", weight",j,", bias_v",j,", bias_h",j,", learning)")))
      eval(parse(text=paste0(
        "cd.weight",j,"<- weight.",j,"$weight")))
    }
  }
  
  #################
  ## Feedforward ##
  #################
  ## Storage for Each node's input and output
  for (j in 1:num.w){
    assign(paste("layer",j,"IN",sep = ""),matrix(0, nrow=1, ncol=layers[j+1]))
    assign(paste("layer",j,"OUT",sep = ""),matrix(0, nrow=1, ncol=layers[j+1]))
    assign(paste("layer",j,"INbias",sep = ""),matrix(1, nrow=1, ncol=layers[j+1]))
  }
  
  # Storage for pre_delta
  for (j in 1:num.w){
    eval(parse(text=paste0(
      "pre_delta",j,"<- matrix(0, nrow= dim(weight",j,")[1] , ncol= dim(weight",j,")[2]",")")))
  }
  
  ###################################
  # Feedforward and Backpropagation #
  ###################################
  for (m in 1:dim(in.trn)[1]){
    layer0OUT = in.trn[m,]
    
    ## Feedforward
    if(m==1){
      
      for (j in 1:num.w){
        eval(parse(text=paste0(
          "layer",j,"IN", "<-layer",j-1,"OUT%*%cd.weight",j)))
        eval(parse(text=paste0(
          "layer",j,"OUT", "<-sigmoid(layer",j-1,"OUT, cd.weight",j,", b0 = layer",j,"INbias)")))
      }
      # yout = layer4OUT / yINbias = layer4INbias
      # layer4OUT : prob. of predict_y
      
    } else {
      
      for (j in 1:num.w){
        eval(parse(text=paste0(
          "layer",j,"IN", "<-layer",j-1,"OUT%*%obj",j,"$weight")))
        eval(parse(text=paste0(
          "layer",j,"OUT", "<-sigmoid(layer",j-1,"OUT,obj",j,"$weight, b0 = obj",j,"$layer2INbias)")))
      }
    }
    
    
    #####################
    ## Backpropagation ##
    #####################
    result = out.trn[m]
    
    for(ii in num.w:1){
      if(ii == num.w) eval(parse(text=paste0("error",ii,"<- result-layer",ii,"OUT")))
      else {
        eval(parse(text=paste0(
          "error",ii,"<- error",ii+1,"%*%t(weight",ii+1,")*layer",ii,"OUT*(1-layer",ii,"OUT)")))
      }
    }
    
    # weight update
    for(ii in 1:num.w){
      eval(parse(text=paste0(
        "obj",ii,"<- w.update(weight",ii,", layer",ii-1,"OUT, layer",ii,"INbias, error",ii,", pre_delta",ii,", momen, learning)")))
    }
  }
  
  # Weight and Bias Result
  weight <- as.list(1:num.w)
  bias <- as.list(1:num.w)
  for(jj in 1:num.w){
    eval(parse(text=paste0(
      "weight[[",jj,"]] <- obj",jj,"$weight")))
    eval(parse(text=paste0(
      "bias[[",jj,"]] <- obj",jj,"$layer2INbias")))
  }
  
  # return
  obj <- list(weight = weight, bias = bias,
              layer.units = layer.units, learning = learning, momen = momen)
  
}
