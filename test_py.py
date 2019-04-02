
from gp_model import kernel_RBF, GPmodel
import pyDOE
import numpy as np
import itertools

# import scipy as sp
# from scipy import optimize

#import numpy as np
#from r_functions.python_wrappers import tmvsampler, pmvnorm_sgn

# Function to emulate
def fun(x1, x2):
    return (x1*2)**2 + np.sin(x2*5)

def Dfun(x1, x2):
    return 8*x1

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