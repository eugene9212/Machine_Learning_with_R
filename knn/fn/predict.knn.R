
# Prediction code for KNN #

predict.knn <- function(obj, test.x, distance = "L1"){

  # Bring out the train data
  x <- obj$train.x
  y <- obj$train.y
  
  # length
  n <- dim(test.x)[1]
  n.train <- dim(x)[1]
  
  # Distance
  result <- matrix()
  # L1 Distance
  for (i in 1:n){
    for(j in 1:n.train){
      result1[j] <- sum(abs(x[j,,,] - test.x[i]))
    }
    result[i] <- x - test.x[i]
  }

}