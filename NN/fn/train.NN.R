####################################
# train function for neural network#
####################################
train.NN <- function(train.x, train.y, w.type = "RBM", s.d = 0, 
                     layer.units = c(3, 3), update, learning = 0.01, hyper.p = 0, epoch = 1 ){
  
  # shape of train data
  input.dim <- dim(train.x)[2]
  
  # Create layers
  layers <- c(input.dim, layer.units, 1)
  
  # number of weights btw layers
  num.w <- length(layer.units) + 1
  
  # weight initialization
  w.obj <- w.init(w.type = w.type, layers = layers, in.trn = train.x, learning = learning)
  
  for(jj in 1:num.w){
    eval(parse(text=paste0(
      "weight",jj," <- w.obj$weight[[",jj,"]]")))
    eval(parse(text=paste0(
      "bias_v",jj," <- w.obj$bias_v[[",jj,"]]")))
    eval(parse(text=paste0(
      "bias_h",jj," <- w.obj$bias_h[[",jj,"]]")))
  }
    
  ##====== Storage pre-allocation ======##
  # Storage for Each node's input and output
  for (j in 1:num.w){
    assign(paste("layer",j,"IN",sep = ""),matrix(0, nrow=1, ncol=layers[j+1]))
    assign(paste("layer",j,"OUT",sep = ""),matrix(0, nrow=1, ncol=layers[j+1]))
    assign(paste("layer",j,"INbias",sep = ""),matrix(1, nrow=1, ncol=layers[j+1]))
  }
  
  # Storage for pre_delta1 and 2
  for (j in 1:num.w){
    eval(parse(text=paste0(
      "pre_delta1",j,"<- matrix(0, nrow= dim(weight",j,")[1] , ncol= dim(weight",j,")[2]",")")))
    eval(parse(text=paste0(
      "pre_delta2",j,"<- matrix(0, nrow= dim(weight",j,")[1] , ncol= dim(weight",j,")[2]",")")))
  }
  
  ##############################################################
  # Feedforward and Backpropagation for each samples (N epoch) #
  ##############################################################
  
  for (ee in 1:epoch) {
    
    #=================================== EPOCH 1 START =========================================#
    for (m in 1:dim(in.trn)[1]){
      layer0OUT = in.trn[m,]
      
      ##====== Feedforward ======##
      if(m==1 && ee==1){
        
        for (j in 1:num.w){
          eval(parse(text=paste0(
            "layer",j,"IN", "<-layer",j-1,"OUT%*%weight",j)))
          eval(parse(text=paste0(
            "layer",j,"OUT", "<-sigmoid(layer",j-1,"OUT, weight",j,", b0 = layer",j,"INbias)")))
        }
        # yout = layer4OUT / yINbias = layer4INbias
        # layer4OUT : prob. of predict_y
        
      } else {
        
        for (j in 1:num.w){
          # weight assign
          eval(parse(text=paste0(
            "weight",j," <- obj",j,"$weight")))
          # layer2INbias assign
          eval(parse(text=paste0(
            "layer",j,"INbias <- obj",j,"$layer2INbias")))
          # layer IN n OUT assign
          eval(parse(text=paste0(
            "layer",j,"IN", "<-layer",j-1,"OUT%*%weight",j)))
          eval(parse(text=paste0(
            "layer",j,"OUT", "<-sigmoid(layer",j-1,"OUT,obj",j,"$weight, b0 = layer",j,"INbias)")))
        }
      }
      
      ##====== Backpropagation ======##
      result = out.trn[m]
      
      # Calculate the errors
      for(ii in num.w:1){
        if(ii == num.w) eval(parse(text=paste0("error",ii,"<- result-layer",ii,"OUT")))
        else {
          eval(parse(text=paste0(
            "error",ii,"<- error",ii+1,"%*%t(weight",ii+1,")*layer",ii,"OUT*(1-layer",ii,"OUT)")))
        }
      }
      
      # weight update
      for(ii in 1:num.w){
        if (update != "Adam"){
          eval(parse(text=paste0(
            "obj",ii,"<- w.update(update, weight",ii,", layer",ii-1,"OUT, layer",ii,"INbias, error",ii,", 
                                  learning, hyper.p = hyper.p, pre_delta1",ii,")")))
          eval(parse(text=paste0(
            "pre_delta1",ii,"<- obj",ii,"$pre_delta1")))
        } else {
          eval(parse(text=paste0(
            "obj",ii,"<- w.update(update, weight",ii,", layer",ii-1,"OUT, layer",ii,"INbias, error",ii,", 
                                learning, hyper.p = hyper.p, pre_delta1",ii,", pre_delta2",ii," )")))
          eval(parse(text=paste0(
            "pre_delta1",ii,"<- obj",ii,"$pre_delta1")))
          eval(parse(text=paste0(
            "pre_delta2",ii,"<- obj",ii,"$pre_delta2")))
          
        }
      }
    }#=================================== EPOCH 1 END =========================================#
      
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
              layer.units = layer.units, learning = learning, hyper.p = hyper.p)
}
