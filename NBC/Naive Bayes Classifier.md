# Naive Bayes Classifier

## Introduction
Naive Bayes Classifier is a classification method using Bayes theorem. The algorithm is quite simple. As it uses Bayes theorem, it decides the class of the sample by comparing the posterior probability. This classifier is probabilistic model, since it assumes the probability distribution of the data and uses iid(identical and independently distributed) assumption for constructing likelihood : If covariates are continuous, we assume Normal distribution and if discrete, Bernoulli or mulinomial distribution.

## Algorithm
The code that I posted is about the continuous covariate case and the logic is as follows.

1. Split the train and test data
	train : $D = {x_i,y_i}^{N}_{i=1}$
    $N_k$ = the number of class k
2. Construct the model
	- prior probability for class 0 and 1\\
		* P(class 0) = $N_0/N$, P(class 1) = $N_1/N$
	- MLE(Maximum Likelihood Extimator) for parameters($\mu$, $\sigma^2$)
		* MLE : Estimator of parameter which maximize the likelihood of data.(Find the parameters that is the most likely to happen in this data)
		* MLE of $\mu$ = $\sum x_i / N$
		* MLE of $\sigma^2$ = $\sum(x_i - \mu)^2 / N$
3. Prediction
	Use Bayes theorem and decide the predicted class label.
    - Bayes Theorem (Compare the posterior probabilities on each class)
		* P(x|class 0)P(class0) < P(x|class 1)P(class1) then x can be estimated as class 1.

