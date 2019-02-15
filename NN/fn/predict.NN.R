##############
## predict y##
##############
predict.NN <- function(test.x, test.y, obj){
  
  layers <- c(obj$layer.units, 1)
  num.w <- length(layers)
  n.trn <- dim(test.x)[1]
  
  for(ii in 1:num.w){
    eval(parse(text=paste0(
      "layer",ii,"INbias1", "<- matrix(rep(obj$bias[[",ii,"]],each =1, time= ",n.trn,"),nrow=",n.trn,"
      , ncol=layers[",ii,"],byrow=T)")))
  }
  
  layer0 <- in.tst
  for(ii in 1:num.w){
    eval(parse(text=paste0(
      "layer",ii, "<- 1/(1+exp(-layer",ii-1,"%*%obj$weight[[",ii,"]]-layer",ii,"INbias1))")))
    if(ii==num.w) eval(parse(text=paste0("output <- layer",ii))) 
  }
  
  # return
  obj <- list(output = output)
}




