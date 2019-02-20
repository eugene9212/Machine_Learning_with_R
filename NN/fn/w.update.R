# weight update
w.update <-function(update = "momentum", weight, layer1OUT, layer2INbias, error, learning, hyper.p = 0, pre_delta1, pre_delta2 = NA){
  
  if (update == "momentum"){
    
    # Momentum
    momen <- hyper.p
    for (i in 1:dim(weight)[1]){
      for (j in 1:dim(weight)[2]){
        pre_delta1[i,j] <- momen*pre_delta1[i,j] + learning*layer1OUT[i]*error[j]
        weight[i,j] <- weight[i,j] + pre_delta1[i,j]
        layer2INbias[j] <- layer2INbias[j] + learning*layer1OUT[i]*error[j]
      }
    }
  } else if (update == "AdaGrad"){
    
    # AdaGrad
    decay = hyper.p  # usually 0.9 for decay rate
    for (i in 1:dim(weight)[1]){
      for (j in 1:dim(weight)[2]){
        pre_delta1[i,j] <- decay*pre_delta1[i,j] + (1-decay)*(layer1OUT[i]*error[j])^2
        weight[i,j] <- weight[i,j] + learning*(1/(sqrt(pre_delta1[i,j]) + 1e-8))*layer1OUT[i]*error[j]
        layer2INbias[j] <- layer2INbias[j] + learning*layer1OUT[i]*error[j]
      }
    }
  } else if (update == "Adam"){
      
      # Adam
      beta1 <- hyper.p[1]
      beta2 <- hyper.p[2]
      # usually beta1 = 0.9; beta2 = 0.999 for Adam
      # pre_delta1 for m; pre_delta2 for v
      for (i in 1:dim(weight)[1]){
        for (j in 1:dim(weight)[2]){
          pre_delta1[i,j] <- beta1*pre_delta1[i,j] + (1-beta1)*(layer1OUT[i]*error[j])
          pre_delta2[i,j] <- beta2*pre_delta2[i,j] + (1-beta2)*(layer1OUT[i]*error[j])^2
          weight[i,j] <- weight[i,j] + learning*pre_delta1[i,j]/(sqrt(pre_delta2[i,j])+1e-8)
          layer2INbias[j] <- layer2INbias[j] + learning*layer1OUT[i]*error[j]
        }
      }
    }
  
  # return
  obj <- list(weight = weight, layer2INbias=layer2INbias, pre_delta1=pre_delta1, pre_delta2=pre_delta2)
}

