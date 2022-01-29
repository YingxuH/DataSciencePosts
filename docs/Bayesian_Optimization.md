1. Develop a surrogate function and find the x that minimizes the surrogate function. 
2. Compute the real response y with the chosen set of hyperparameters x.
3. Include the new observation into the historical observations and update the parameters of the surrogate function.  

# Expected Improvement

One of the surrogate function used to pick the best x. It compromises the difference with y and the current best y value and p(y|x).

## Gaussian Process

Calculating the EI using p(y|x). Possibly need multiple GPs for different set of parameters (different layers of a neural network). 

## Tree-structured Parzen Estimators

Calculating the EI using p(x|y) and p(y). Tree-structue allows all the variables to be estimated simultaneously. Two sets of population density for observations lower or higher than a certain theshold. 

https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
