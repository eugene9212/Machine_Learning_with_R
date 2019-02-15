# CD Algorithm
cd.weight <- function(input, weight, bias_v, bias_h, learning){
  # input = data, output = weight()
  
  # Demensionality Check!!
  if (dim(input)[2] != dim(weight)[1]) output <- t(weight)
  if (dim(input)[2] != dim(weight)[1]) warning("Dimensions of input and output are not identical")
  
  for (m in 1:dim(input)[1]){
    # m <- 1
    v1_value = matrix(as.matrix(input[m,]),ncol = dim(weight)[1])
    
    # Probability for h1
    h1_prob = 1 / (1 + exp(-(v1_value %*% weight + bias_h)))
    
    # value of h1
    h1_value = matrix(0, nrow = 1, ncol = dim(weight)[2]) ## need correction
    for (i in 1:dim(weight)[2]) 
    {
      if (h1_prob[i] >= runif(1, min = 0, max = 1)) h1_value[i] = 1
      else h1_value[i] = 0
    }
    
    # probability for v2
    t.weight = t(weight)
    v2_prob = 1 / (1 + exp(-(h1_value %*% t.weight + bias_v)))
    
    # value of v2
    v2_value = matrix(0, nrow = 1, ncol = dim(weight)[1])
    for (i in 1:length(v2_prob))
    {
      if (v2_prob[i] >= runif(1, min = 0, max = 1)) v2_value[i] = 1
      else v2_value[i] = 0
    }
    
    # Probability for h2
    h2_prob = 1 / (1 + exp(-(v2_value %*% weight + bias_h)))
    
    # value of h1
    h2_value = matrix(0, nrow = 1, ncol = dim(weight)[2])
    for (i in 1:length(h2_value))
    {
      if (h2_prob[i] >= runif(1, min = 0, max = 1)) h2_value[i] = 1
      else h2_value[i] = 0
    }
    
    # weight1 update
    for (i in 1:dim(weight)[1]) {
      for (j in 1:dim(weight)[2]) {
        weight[i, j] = weight[i, j] - learning * (v1_value[i] * h1_prob[j]
                                                    - v2_value[i] * h2_prob[j])
        bias_v = bias_v + learning * (v1_value - v2_value)
        bias_h = bias_h + learning * (h1_prob - h2_prob)
      }
    }
  }
  
  # return
  obj <- list(weight = weight, bias_v, bias_h)
  
}

