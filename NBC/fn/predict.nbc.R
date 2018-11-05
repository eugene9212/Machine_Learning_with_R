# Naive Bayes Classification (test)#

predict.nbc <- function(obj, test.x){
  
  # Bring the parameters from the train set
  prior0 <- obj$prior[1]
  prior1 <- obj$prior[2]
  
  m0 <- obj$m0
  m1 <- obj$m1 
  
  cov0 <- obj$cov0
  cov1 <- obj$cov1
  
  len <- dim(cov0)[1]
  
  # likelihood function
  likelihood <- function(x, mean, cov){
    x <- matrix(x)
    n.c <- ncol(x); n.r <- nrow(x)
    mean <- matrix(mean, nrow = n.r)
    diff <- matrix(x - mean, nrow = n.r)
    if (dim(diff)[2] == dim(cov)[1]){
      tmp <- (diff %*% solve(cov)) %*% t(diff)
    } else {
      tmp <- (t(diff) %*% solve(cov)) %*% diff
    }
    
    
    numerator <- exp(-tmp/2)
    denominator <- sqrt((2*pi)^n.c*det(cov))
    
    y <- numerator/denominator
    return(y)
  }
  
  # prediction by Bayes Classifier
  prediction <- function(x){
    prob0 = likelihood(x, m0, cov0)*prior0
    prob1 = likelihood(x, m1, cov1)*prior1
    
    if (prob1 >= prob0){
      return (1)
    }
    else {
      return (0)
    }
  }
  
  #Empirical error on the test data
  predict.y <- apply(test.x, 1, prediction)
  
  # return
  obj <- list(predict.y = predict.y, likelihood = likelihood)
  
}