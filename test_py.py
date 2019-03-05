
# from gp_model import kernel_RBF, GPmodel
# import pyDOE
# import numpy as np
# import itertools

# import scipy as sp
# from scipy import optimize

import numpy as np
from r_functions.python_wrappers import tmvsampler, pmvnorm_sgn



# Function to emulate
def fun(x1, x2):
    return (x1*2)**2 + np.sin(x2*5)

def Dfun(x1, x2):
    return 8*x1

def main():
    print('testing..')

    #_load_R()

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
    model.optimize_unconstrained(method = 'ML', fix_likelihood = True)
        
    print(model)

    # Derivative data
    num_grid_der = 3
    s_der = np.array(list(itertools.product(np.linspace(0, 1, num_grid_der), np.linspace(0, 1, num_grid_der))))

    deriv_lik = 0.0001

    # Update model
    model.DX_training = s_der
    model.DY_training = np.array([Dfun(x[0], x[1]) for x in s_der])

    model.sgn_DY_training = np.ones(len(model.DX_training))

    # Set noise on derivatives
    model.deriv_likelihood = deriv_lik

    # Run prediction
    c_x1 = 0.5
    c_x2 = 0.5
    N = 100
    x1_test = np.array([[x1, x2] for x1, x2 in zip(np.linspace(0, 1, N), c_x2*np.ones(N))])
    x2_test = np.array([[x1, x2] for x1, x2 in zip(c_x1*np.ones(N), np.linspace(0, 1, N))])
    X_test = np.concatenate([x1_test, x2_test])

    mean_all, var_all, perc_all, samples_all = model.calc_posterior_sgn_constrained_Z(X_test, num_samples = 1000 )

# def _scipyopt_test():

#     def rosen(x):
#         """The Rosenbrock function"""
#         return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

#     def rosen_der(x):
#         xm = x[1:-1]
#         xm_m1 = x[:-2]
#         xm_p1 = x[2:]
#         der = np.zeros_like(x)
#         der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
#         der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
#         der[-1] = 200*(x[-1]-x[-2]**2)
#         return der

#     x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
#     res = optimize.minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
#     print(res)

# def main_3():
#     _scipyopt_test()
#     from r_functions.python_wrappers import tmvsampler
#     _scipyopt_test()

# def main_2():
#     print('Testing scipy.optimize')
    
#     x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
#     #res = sp.optimize.minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
#     res = optimize.minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
#     print('res', res)
#     from r_functions.python_wrappers import tmvsampler

#     res = optimize.minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
#     print('res', res)

#     main()

def main_4():
    mu = np.array([0, 0])
    sigma = np.matrix([[1, 0.95], [0.95, 1]])
    sgn = np.array([1, 1])

    t = pmvnorm_sgn(mu, sigma, sgn)

    print(t)

    z_prime_samples = tmvsampler(10, mu, sigma, sgn, algorithm = 'rejection')

    print(z_prime_samples)

if __name__ == "__main__":
    main_4()