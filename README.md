# GPConstr - Gaussian Process regression with linear operator constraints
Python module for constrained GP regression. 

Code based on the paper [_C. Agrell (2019) Gaussian processes with linear operator inequality constraints_](https://arxiv.org/abs/1901.03134). The current implementation covers boundedness of the function to estimate, combined with bounds on its first order partial derivatives, using the RBF kernenel. 

### Prerequisites
Besides the standard numpy/scipy libraries, [rpy2](https://pypi.org/project/rpy2/) is used to access some useful R packages for working with the truncated multivariate normal distribution. The code has been tested with the following requirements: 

__Python 3 (3.6.3 64bit)__
- __numpy (1.14.0)__
- __scipy (1.1.0)__
- __pandas (0.22.0)__
- __sklearn (0.19.1)__ _Only uses the function sklearn.metrics.pairwise.euclidean_distances from this package for fast computation of Gram matrices (and could easily be replaced by custom code if needed)_
- __rpy2 (2.8.6)__ _Used to acces R for computation involving the truncated multivariate normal. See the Python wrapper in /GPConstr/r_functions/ for details_

__R (3.4.3)__
- __tmvtnorm (1.4.10)__
- __mvtnorm (1.0.7)__
- __TruncatedNormal (1.0)__
- __truncnorm (1.0.8)__

### Examples
Some examples are given in jupyter notebooks
- __Example_1.ipynb__ _1D example of boundedness and monotonicity constraints_

### Further work
We will be including other types of constraints and kernels as needed, either buidling on the current implementation or on a suitable GP library with good functionality for kernel manupulation such as e.g. [GPflow](https://github.com/GPflow/GPflow)
