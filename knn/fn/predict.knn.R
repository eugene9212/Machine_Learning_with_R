# Prediction code for KNN without Parallel Computing#
predict.knn <- function(obj, test.x, k =1, distance = "L1"){
  
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
    for(i in 1:n.test){
      result <- matrix(0, n.train,1)
      
      for(j in 1:n.train){
        assign(paste("result[",j,"]",sep = ""), 
               paste("sum(abs(x[",j, strrep(",",n.array-1),"]-test.x[",i, strrep(",",n.array-1),"]"))
      }
      
      idx <- which(order(result) %in% c(1:k))
      predict[i] <- as.numeric(names(sort(table(y[idx]),decreasing=TRUE)[1]))
    }
  } else if (distance == "L2"){
    # L2 Distance
    for(i in 1:n.test){
      result <- matrix(0, n.train,1)
      
      for(j in 1:n.train){
        assign(paste("result[",j,"]",sep = ""), 
               paste("sqrt(sum((x[",j, strrep(",",n.array-1),"]^2-test.x[",i, strrep(",",n.array-1),"]^2)^2))"))
        # result[j] <- sqrt(sum((x[j,,,]^2 - test.x[i,,,]^2)^2))
      }
      
      idx <- which(order(result) %in% c(1:k))
      predict[i] <- as.numeric(names(sort(table(y[idx]),decreasing=TRUE)[1]))
    }
  } else message("Distance is not assigned properly")
  
  # return
  obj <- list(predict = predict, dist = distance, k = k)
}

