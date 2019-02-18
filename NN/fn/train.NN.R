####################################
# train function for neural network#
####################################
train.NN <- function(train.x, train.y, w.type = "RBM", s.d = 0, 
                     layer.units = c(3, 3), update, learning = 0.01, momen = 0, epoch = 1 ){
  
  # shape of train data
  input.dim <- dim(train.x)[2]
  
  # hyper parameters
  layers = c(input.dim, layer.units, 1)
  learning = learning # learning rate
  momen = momen
  
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
  
  # Storage for pre_delta
  for (j in 1:num.w){
    eval(parse(text=paste0(
      "pre_delta",j,"<- matrix(0, nrow= dim(weight",j,")[1] , ncol= dim(weight",j,")[2]",")")))
  }
  
  ##############################################################
  # Feedforward and Backpropagation for each samples (1 epoch) #
  ##############################################################
  
  for (ee in 1:epoch) {
    
    if(ee == 1){
      #=================================== EPOCH 1 START =========================================#
      for (m in 1:dim(in.trn)[1]){
        layer0OUT = in.trn[m,]
        
        ##====== Feedforward ======##
        if(m==1){
          
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
            eval(parse(text=paste0(
              "layer",j,"IN", "<-layer",j-1,"OUT%*%obj",j,"$weight")))
            eval(parse(text=paste0(
              "layer",j,"OUT", "<-sigmoid(layer",j-1,"OUT,obj",j,"$weight, b0 = obj",j,"$layer2INbias)")))
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
          eval(parse(text=paste0(
            "obj",ii,"<- w.update(update, weight",ii,", layer",ii-1,"OUT, layer",ii,"INbias, error",ii,", pre_delta",ii,", learning, momen)")))
          if(update == "Adagrad"){
            eval(parse(text=paste0(
              "pre_delta",ii,"<- 0.9*obj",ii,"$pre_delta"))) # for Adagrad Update (0.9 : RMSProp)
          } else{
            eval(parse(text=paste0(
              "pre_delta",ii,"<- obj",ii,"$pre_delta"))) # for Momentum update
          }
        }
      }#=================================== EPOCH 1 END =========================================#
    } else {
    #=================================== EPOCH 2 ~ START =========================================#
    for (m in 1:dim(in.trn)[1]){
      layer0OUT = in.trn[m,]
      
      ##====== Feedforward ======##
      for (j in 1:num.w){
        eval(parse(text=paste0(
          "layer",j,"IN", "<-layer",j-1,"OUT%*%obj",j,"$weight")))
        eval(parse(text=paste0(
          "layer",j,"OUT", "<-sigmoid(layer",j-1,"OUT,obj",j,"$weight, b0 = obj",j,"$layer2INbias)")))
      }
      
      ##====== Backpropagation ======##
      result = out.trn[m]
      
      # Calculate the errors
      for(ii in num.w:1){
        if(ii == num.w) eval(parse(text=paste0("error",ii,"<- result-layer",ii,"OUT")))
        else {
          eval(parse(text=paste0(
            "error",ii,"<- error",ii+1,"%*%t(obj",ii,"$weight)*layer",ii,"OUT*(1-layer",ii,"OUT)")))
        }
      }
      
      # weight update
      for(ii in 1:num.w){
        eval(parse(text=paste0(
          "obj",ii,"<- w.update(update, obj",ii,"$weight, layer",ii-1,"OUT, obj",ii,"$layer",ii,"INbias, error",ii,", pre_delta",ii,", learning, momen)")))
        if(update == "Adagrad"){
          eval(parse(text=paste0(
            "pre_delta",ii,"<- 0.9*obj",ii,"$pre_delta"))) # for Adagrad Update (0.9 : RMSProp)
        } else{
          eval(parse(text=paste0(
            "pre_delta",ii,"<- obj",ii,"$pre_delta"))) # for Momentum update
        }
      }
    }
    #=================================== EPOCH 2 ~ END =========================================#    
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
