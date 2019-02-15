# sigmoid function
sigmoid <- function(input, output, b0 = 0){
  return(1/(1+exp(-input%*%output - b0)))
}
