# weight update
w.update <-function(weight, layer1OUT, layer2INbias, error, pre_delta, momen, learning){
  for (i in 1:dim(weight)[1]){
    for (j in 1:dim(weight)[2]){
      weight[i,j] = weight[i,j] + learning*layer1OUT[i]*error[j]+ momen*pre_delta[i,j]
      layer2INbias[j] = layer2INbias[j] + learning*layer1OUT[i]*error[j]
      pre_delta[i,j] = learning*layer1OUT[i]*error[j]
    }
  }
  # return
  obj <- list(weight = weight, layer2INbias=layer2INbias, pre_delta=pre_delta)
}

