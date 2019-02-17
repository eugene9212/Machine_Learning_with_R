w.init <- function(w.type, layers, in.trn, learning, s.d = NA){
  
  # number of weights between the layers
  num.w <- length(layers) - 1
  
  # assign types of weight matrix
  if(w.type == "normal"){ 
    
    ##==== (1) Normal distribution with user defined standard deviation ====##
    for (i in 1:num.w){
      
        eval(parse(text=paste0(
          "weight",i,"<- matrix(rnorm(", layers[i]*layers[i+1],", mean=0, sd=",s.d,"),nrow =", layers[i],", ncol =", layers[i+1],")")))
        eval(parse(text=paste0(
          "bias_v",i,"<- matrix(1,nrow = 1,ncol = ", layers[i],")")))
        eval(parse(text=paste0(
          "bias_h",i,"<- matrix(1,nrow = 1,ncol = ", layers[i+1],")")))

    }
  } else if (w.type == "RBM"){ 
    
    ##==== (2) using one of Recurrent Boltzmann Machine ====##
    for (i in 1:num.w){
        eval(parse(text=paste0(
          "weight",i,"<- matrix(sample(-100:100,", layers[i]*layers[i+1],", replace=T)/10^6, nrow =", layers[i],", ncol =", layers[i+1],")")))
        eval(parse(text=paste0(
          "bias_v",i,"<- matrix(1,nrow = 1,ncol = ", layers[i],")")))
        eval(parse(text=paste0(
          "bias_h",i,"<- matrix(1,nrow = 1,ncol = ", layers[i+1],")")))
    }
    
    # using CD(Contrastive Divergence) algorithm for RBM weight initialization
    for (j in 1:num.w){
      if(j == 1){
        # CD for weight1
        input1 <- in.trn
        weight.1 <- cd.weight(input = input1, weight = weight1, 
                              bias_v = bias_v1, bias_h = bias_h1, learning = learning)
        weight1 <- weight.1$weight
      } else{
        # CD for other weights
        eval(parse(text=paste0(
          "input",j,"<- sigmoid(input",j-1,",weight",j-1,")")))
        eval(parse(text=paste0(
          "weight.",j,"<- cd.weight(input",j,", weight",j,", bias_v",j,", bias_h",j,", learning)")))
        eval(parse(text=paste0(
          "weight",j,"<- weight.",j,"$weight")))
      }
    }
    
  } else if (w.type == "Xavier"){ 
    
    ##==== (3) Xavier weight initialization typically for sigmoid activation function ====##
    # assign standard deviation of each weights
    X.sd <- sqrt(1/layers)
    
    # weight initialization
    for (i in 1:num.w){
        eval(parse(text=paste0(
          "weight",i,"<- matrix(rnorm(", layers[i]*layers[i+1],", mean=0, sd=",X.sd[i],"),nrow =", layers[i],", ncol =", layers[i+1],")")))
        eval(parse(text=paste0(
          "bias_v",i,"<- matrix(1,nrow = 1,ncol = ", layers[i],")")))
        eval(parse(text=paste0(
          "bias_h",i,"<- matrix(1,nrow = 1,ncol = ", layers[i+1],")")))
      } 
    } else if (w.type == "He"){ 
    
    ##==== (4) He weight initialization typically for sigmoid activation function ====##
    # assign standard deviation of each weights
    He.sd <- sqrt(2/layers)
    
    # weight initialization
    for (i in 1:num.w){
      if(i == num.w){
        eval(parse(text=paste0(
          "weight",i,"<- matrix(rnorm(", layers[i]*layers[i+1],", mean=0, sd=",He.sd[i],"),nrow =", layers[i],", ncol =", layers[i+1],")")))
        eval(parse(text=paste0(
          "bias_v",i,"<- matrix(1,nrow = 1,ncol = ", layers[i],")")))
        eval(parse(text=paste0(
          "bias_h",i,"<- matrix(1,nrow = 1,ncol = ", layers[i+1],")")))
      } 
    }
  }
  
  # Weight and Bias Result
  weight <- as.list(1:num.w)
  bias_v <- as.list(1:num.w)
  bias_h <- as.list(1:num.w)
  for(jj in 1:num.w){
    eval(parse(text=paste0(
      "weight[[",jj,"]] <- weight",jj)))
    eval(parse(text=paste0(
      "bias_v[[",jj,"]] <- bias_v",jj)))
    eval(parse(text=paste0(
      "bias_h[[",jj,"]] <- bias_h",jj)))
  }
  
  # return
  obj <- list(weight = weight, bias_v = bias_v, bias_h = bias_h)
}