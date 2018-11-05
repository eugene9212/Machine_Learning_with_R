# Naive Bayes Classification (train)#

nbc <- function(train.x, train.y){
  
  # Divide the data set
  train.x.0 <- train.x[which(train.y==0),]
  train.x.1 <- train.x[which(train.y==1),]
  
  # Estimate mean and covariance for class 1 and 0
  m0 <- colMeans(train.x.0)
  m1 <- colMeans(train.x.1)
  
  cov0 <- cov(train.x.0)
  cov1 <- cov(train.x.1)
  
  #prior probability
  prior0 <- dim(train.x.0)[1]/length(train.y)
  prior1 <- dim(train.x.1)[1]/length(train.y)

  # return
  obj <- list(prior = c(prior0, prior1), cov0 = cov0, cov1 = cov1, m0 = m0, m1 = m1)
  
}