# Prediction code for KNN with Parallel Computing#
predict.knn.par <- function(obj, test.x, k =1, distance = "L1", core = NA){
  
  # Bring out the train data
  x <- obj$train.x
  y <- obj$train.y
  
  # length
  n.test <- dim(test.x)[1]
  n.train <- dim(x)[1]
  n.array <- length(dim(test.x))
  
  # Distance
  predict <- matrix(0, 1, n.test)
  
  if (distance == "L1"){
    # L1 Distance
    working <- function(i){
      result <- matrix(0, n.train,1)
      
      for(j in 1:n.train){
        
        assign(paste("result[",j,"]",sep = ""), 
               paste("sum(abs(x[",j, strrep(",",n.array-1),"]-test.x[",i, strrep(",",n.array-1),"]"))
        # result[j] <- sum(abs(x[j,,,] - test.x[i,,,]))
      }
      
      idx <- which(order(result) %in% c(1:k))
      predict <- as.numeric(names(sort(table(y[idx]),decreasing=TRUE)[1]))
      
      return(predict)
    }
  } else if (distance == "L2"){
    # L1 Distance
    working <- function(i){
      result <- matrix(0, n.train,1)
      
      for(j in 1:n.train){
        assign(paste("result[",j,"]",sep = ""), 
               paste("sqrt(sum((x[",j, strrep(",",n.array-1),"]^2-test.x[",i, strrep(",",n.array-1),"]^2)^2))"))
        # result[j] <- sqrt(sum((x[j,,,]^2 - test.x[i,,,]^2)^2))
      }
      
      idx <- which(order(result) %in% c(1:k))
      predict <- as.numeric(names(sort(table(y[idx]),decreasing=TRUE)[1]))
      
      return(predict)
    }
  } else message("Distance is not assigned properly")
  
  # Parallel Computing #
  require(doParallel)
  require(foreach)
  
  if(is.na(core)){
    t <- detectCores(all.tests = FALSE, logical = TRUE) # Detect the computer's available cores
    cl <- makeCluster(t-2)  # choose the number of cores
  } else {
    t <- core
    cl <- makeCluster(t)  # choose the number of cores
  }
  
  registerDoParallel(cl) # register clusters
  
  predicted <- foreach(i=1:n.test, .combine=rbind) %dopar%
    working(i)
  
  stopCluster(cl)
  
  # return
  obj <- list(predict = predicted, dist = distance, k = k, cl = cl)
}

