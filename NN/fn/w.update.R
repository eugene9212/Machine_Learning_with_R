# weight update
w.update <-function(update = "momentum", weight, layer1OUT, layer2INbias, error, pre_delta, learning, hyper.p = 0){
  
  if (update == "momentum"){
    
    # Momentum
    momen <- hyper.p
    for (i in 1:dim(weight)[1]){
      for (j in 1:dim(weight)[2]){
        pre_delta[i,j] = momen*pre_delta[i,j] + learning*layer1OUT[i]*error[j]
        weight[i,j] = weight[i,j] + pre_delta[i,j]
        layer2INbias[j] = layer2INbias[j] + learning*layer1OUT[i]*error[j]
      }
    }
  } else if (update == "AdaGrad"){
    
    # AdaGrad
    decay = hyper.p  # usually 0.9 for decay rate
    for (i in 1:dim(weight)[1]){
      for (j in 1:dim(weight)[2]){
        pre_delta[i,j] = decay*pre_delta[i,j] + (1-decay)*(layer1OUT[i]*error[j])^2
        weight[i,j] = weight[i,j] + learning*(1/(sqrt(pre_delta[i,j]) + 1e-8))*layer1OUT[i]*error[j]
        layer2INbias[j] = layer2INbias[j] + learning*layer1OUT[i]*error[j]
      }
    }
  } else if (update == "Adam"){
      
      # Adam
      beta1 <- hyper.p[1]
      beta2 <- hyper.p[2]
      # usually beta1 = 0.9; beta2 = 0.999 for Adam
      for (i in 1:dim(weight)[1]){
        for (j in 1:dim(weight)[2]){
          m[i,j] = beta1*m[i,j] + (1-beta1)*(layer1OUT[i]*error[j])
          v[i,j] = beta2*v[i,j] + (1-beta2)*(layer1OUT[i]*error[j])^2
          weight[i,j] = weight[i,j] + learning*m/(sqrt(v)+1e-8)
          layer2INbias[j] = layer2INbias[j] + learning*layer1OUT[i]*error[j]
        }
      }
    }
  
  # return
  obj <- list(weight = weight, layer2INbias=layer2INbias, pre_delta=pre_delta)
}

