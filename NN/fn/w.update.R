# weight update
w.update <-function(update = "momentum", weight, layer1OUT, layer2INbias, error, pre_delta, learning, momen = 0){
  
  if (update == "momentum"){
    
    # Momentum Update
    for (i in 1:dim(weight)[1]){
      for (j in 1:dim(weight)[2]){
        pre_delta[i,j] = momen*pre_delta[i,j] - learning*layer1OUT[i]*error[j]
        weight[i,j] = weight[i,j] + pre_delta[i,j]
        layer2INbias[j] = layer2INbias[j] - learning*layer1OUT[i]*error[j]
      }
    }
  } else if (update == "AdaGrad"){
    
    # AdaGrad
    for (i in 1:dim(weight)[1]){
      for (j in 1:dim(weight)[2]){
        pre_delta[i,j] = pre_delta[i,j] + (layer1OUT[i]*error[j])^2
        if(pre_delta[i,j] == 0) weight[i,j] = weight[i,j] - learning*layer1OUT[i]*error[j]
        else weight[i,j] = weight[i,j] - learning*(1/sqrt(pre_delta[i,j]))*layer1OUT[i]*error[j]
        layer2INbias[j] = layer2INbias[j] - learning*layer1OUT[i]*error[j]
      }
    }
  }
  
  # return
  obj <- list(weight = weight, layer2INbias=layer2INbias, pre_delta=pre_delta)
}

