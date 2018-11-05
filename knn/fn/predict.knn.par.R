
# Prediction code for KNN with Parallel Computing#

predict.knn.par <- function(obj, test.x, k =1, distance = "L1"){
  
  
  # Bring out the train data
  x <- obj$train.x
  y <- obj$train.y
  
  # length
  n.test <- dim(test.x)[1]
  n.train <- dim(x)[1]
  
  # Distance
  dist <- distance
  predict <- matrix(0, 1, n.test)
  
  if (dist == "L1"){
    # L1 Distance
    working <- function(i){
      result <- matrix(0, 1, n.train)
      
      for(j in 1:n.train){
        result[j] <- sum(abs(x[j,,,] - test.x[i,,,]))
      }
      
      idx <- which(order(result) %in% c(1:k))
      predict[i] <- as.numeric(names(sort(table(y[idx]),decreasing=TRUE)[1]))
      
      return(predict)
    }
  } else if (dist == "L2"){
    # L1 Distance
    working <- function(i){
      result <- matrix(0, 1, n.train)
      
      for(j in 1:n.train){
        result[j] <- sqrt(sum((x[j,,,]^2 - test.x[i,,,]^2)^2))
      }
      
      idx <- which(order(result) %in% c(1:k))
      predict[i] <- as.numeric(names(sort(table(y[idx]),decreasing=TRUE)[1]))
      
      return(predict)
    }
  } else message("Distance is not assigned properly")
  
  require(doParallel)
  require(foreach)
  t <- detectCores(all.tests = FALSE, logical = TRUE)
  
  cl <- makeCluster(t/2)  # choose the number of cores
  registerDoParallel(cl) # register clusters
  
  mm <- foreach(i=1:n.test, .combine=rbind) %dopar%
    working(i)
  
  
  # return
  obj <- list(predict = predict, dist = distance)
}

