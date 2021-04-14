
# For checking that importing the model with R dependency works ok

from GPConstr.model import GPmodel, Constraint
from GPConstr.kern import kernel_RBF

import pyDOE
import numpy as np

# Function to emulate
def fun(x1, x2):
    return (x1*2)**2 + np.sin(x2*5)

# Define a model and print it
def main():
    print('testing..')

    # Design data
    n_samples = 5
    x_design = pyDOE.lhs(2, samples = n_samples, criterion = 'maximin', iterations = 1000)
    y_design = np.array([fun(x[0], x[1]) for x in x_design])

    # Initial parameters
    gp_mean = 0 # Constant mean function
    gp_likelihood = 0.000001 # Gaussian noise
    kernel_variance = 1
    kernel_lengthscale = [1, 1]

    # Set up model
    ker = kernel_RBF(variance = kernel_variance, lengthscale = kernel_lengthscale)
    model = GPmodel(kernel = ker, likelihood = gp_likelihood, mean = gp_mean)

    # Training data
    model.X_training = x_design
    model.Y_training = y_design

    # Optimize
    model.optimize(include_constraint = False, fix_likelihood = True)
        
    print(model)


if __name__ == "__main__":
    main()