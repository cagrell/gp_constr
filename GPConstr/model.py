### Dependent packages ###
import time
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from scipy import optimize

# Can replace this with custom code that does the same..
from sklearn.metrics.pairwise import euclidean_distances as sklear_euclidean_distances

### Custom imports ###
print('Loading constrained GP module from ' + os.path.dirname(os.path.realpath('__file__')))
from .util.div import formattime, len_none
from .util.linalg import jitchol, try_jitchol, triang_solve, mulinv_solve, chol_inv, traceprod, nearestPD
from .util.stats import norm_cdf_int, norm_cdf_int_approx, normal_cdf_approx, mode_from_samples, trunc_norm_moments_approx_corrfree

##################################################################################
### Loading R functions -- this is a hack to make R and scipy not crash....... ###
### (need to run scipy.optimize once before loading R)                         ###
def _scipyopt_test():

    def rosen(x):
        """The Rosenbrock function"""
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    def rosen_der(x):
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return der

    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = optimize.minimize(rosen, x0, method='BFGS', jac=rosen_der)


print('Loading R wrapper...')
_scipyopt_test()
from .r_functions.python_wrappers import rtmvnorm, pmvnorm, mtmvnorm
##################################################################################

class kernel_RBF():
    """
    RBF kernel
    """
    
    def __init__(self, variance, lengthscale):
        
        self.lengthscale = self._convert_to_array(lengthscale)
        self.variance = variance
        self.dim = len(self.lengthscale)
        
        assert np.isscalar(variance), 'variance must be scalar'
       
    def __str__(self):
        """ What to show when the object is printed """
        return '  type = {} \n   input dim = {} \n   lenghtscale = {} \n   variance = {}'.format(
            'RBF', self.dim, self.lengthscale, self.variance)
    
    def _convert_to_array(self, v):
        if np.isscalar(v):
            return np.array([v])
        else:
            return np.array(v)
    
    def _euclidian_dist_squared(self, X1, X2):
        """ Return gram matrix with ||x - y||^2 for each pair of points """
               
        # Use function from sklearn - can replace this later to avoid dependence on sklearn
        return sklear_euclidean_distances(X1, X2, squared = True)         
    
    def set_params(self, theta):
        """
        Set all kernel parameters from a single array theta
        .. Used in optimization
        """
        assert self.dim == (len(theta) - 1), 'Parameter array does not match kernel dimension'
        self.variance = theta[0]
        self.lengthscale = theta[1:]
        
    def get_params(self):
        """
        Get all kernel parameters in a single array
        .. Used in optimization
        """
        return np.array([self.variance] + list(self.lengthscale))
    
    def K_gradients(self, X1, X2):
        """
        Return kernel gradients w.r.t hyperparameters
        
        Returns:
        List of Gram matrices of derivatives of K w.r.t. the hyperparameters in the ordering 
        given by get_params and set_params
        """
        
        R = self.R(X1, X2)
        K_R = self.K_R(R)
        
        # W.r.t variance
        dK_dv = K_R/self.variance
        
        # W.r.t lengthscales
        dK_dR = -0.5*K_R
        
        dK_dl = []
        for i in range(len(self.lengthscale)):
            t1, t2 = np.meshgrid(X1[:,i], X2[:,i])
            dR_dli = ((-2/self.lengthscale[i])*((t1 - t2)/self.lengthscale[i])**2).T

            dK_dl.append(dK_dR*dR_dli)

        return [dK_dv] + dK_dl
    
    def R(self, X1, X2):
        """ 
        Return scaled distances squared 
        For RBF kernel: K(X1, X2) = variance * exp(-0.5 * R)
        """
        return self._euclidian_dist_squared(X1 / self.lengthscale, X2 / self.lengthscale)
    
    def K_R(self, R):
        """ Kernel as a function of squared distances """
        return self.variance*np.exp(-0.5*R)
    
    def Ri(self, X1, X2, i):
        """ 
        Returns dR/dX1_i 
        Note: dR/dX2_j(X1, X2) = Ri(X2, X1, j).T
        """
        return (2/self.lengthscale[i]**2)*(X1[:,i].reshape(-1, 1) - X2[:,i].reshape(-1, 1).T)
    
    def Ki0(self, X1, X2, i, R = None, K = None):
        """ For K = K(X1, X2), X1 = [X1_1, X1_2, ..], X2 = [X2_1, X2_2, ..] etc., return dK/dX1_i """
        
        # Make use of K or R if they exist
        if K is None:
            if R is None:
                K = self.K_R(self.R(X1, X2))
            else:
                K = self.K_R(R)
        
        return -0.5*K*self.Ri(X1, X2, i)
    
    def Kij(self, X1, X2, i, j, R = None, K = None):
        """ For K = K(X1, X2), X1 = [X1_1, X1_2, ..], X2 = [X2_1, X2_2, ..] etc., return d^2K/dX1_i*dX2_j """
        
        # Make use of K or R if they exist
        if K is None:
            if R is None:
                K = self.K_R(self.R(X1, X2))
            else:
                K = self.K_R(R)
        
        F = 1/self.lengthscale[i]**2 if i == j else 0
        
        return K*((1/4)*self.Ri(X1, X2, i)*self.Ri(X1 = X2, X2 = X1, i = j).T + F)    
            
    
    def K(self, X1, X2):
        """ Returns Gram matrix of k(x1, x2) """
        return self.K_R(self.R(X1, X2))
        
    def K_diag(self, X):
        """ Returns diagonal of Gram matrix of k(x, x) """
        return np.ones(len(X))*self.variance


class Constraint():
    """ 
    Stores virtual observations and bounds functions for some constraint
    
    LB(Xv) <= Lf(Xv) <= UB(Xv) for some linear operator L
    
    NB! Bound functions needs to be vectorized!
    """
    
    def __init__(self, LB, UB):
        
        self.LB = LB # Function from R^n -> R
        self.UB = UB # Function from R^n -> R
        self.Xv = None # Virtual observation locations
        
    def _check(self, dim, txt):
        """ Check that constraint is ok """
        
        if self.Xv is not None:
            assert len(self.Xv.shape) == 2, 'Error in constraint {} : Data Xv must be 2d array'.format(txt)
            assert self.LBXV().shape == (self.Xv.shape[0],), 'Error in constraint {} : Error in LB function'.format(txt)
            assert self.UBXV().shape == (self.Xv.shape[0],), 'Error in constraint {} : Error in UB function'.format(txt)
            assert self.Xv.shape[1] == dim, 'Error in constraint {} : Xv dimension incorrect'.format(txt)
        
    def LBXV(self):
        """ Lower bound evaluated at Xv """
        return self.LB(self.Xv)
    
    def UBXV(self):
        """ Upper bound evaluated at Xv """
        return self.UB(self.Xv)
    
    def add_XV(self, X):
        """ Add a single point X to XV """
        if self.Xv is None:
            self.Xv = np.atleast_2d(X)
        else:
            self.Xv = np.append(self.Xv, np.atleast_2d(X), axis = 0)
    
    
class GPmodel():
    """ GP model """
    
    def __init__(self, kernel, likelihood = 1, mean = 0, constr_likelihood = 1E-6, verbatim = True):
        
        ### Prior model input ##################################
        
        # GP parameters
        self.kernel = kernel # Object containing kernel function and its derivatives
        self.mean = mean # Constant mean function
        self.likelihood = likelihood
        self.constr_likelihood = constr_likelihood # 0 = noise free observations of constraint. Can increase this for stability
        
        # Design data
        self.X_training = None
        self.Y_training = None
        
        # Constraints
        self.constr_bounded = None # Boundedness constraint
        self.constr_deriv = None # List with partial derivative constraints 
              
        ### Cached data from intermediate calculations ###########
        
        # Depending only on X
        self.K_w = None # K_w = K_x_x + sigma^2*I
        self.K_w_chol = None # Cholesky factor L s.t. L*L.T = K_w
        
        # Depending only on Y
        self.Y_centered = None
        
        # Depending on (X, Y)
        self.LLY = None # Only used in the unconstrained calculations
        
        # Depending on (X, XS)
        self.v2 = None
        self.A2 = None
        self.B2 = None
        
        # Depending on (X, XV)
        self.v1 = None
        self.A1 = None
        self.B1 = None
        self.L_1 = None
        self._p2 = False
        
        # Depending on (X, Y, XV)
        self.C_sim = None
        
        ### Other ##############################################
        self.verbatim = verbatim # Print info during execution
        
    # Parameters that need calculation reset
    @property
    def X_training(self): return self.__X_training

    @property
    def Y_training(self): return self.__Y_training

    @X_training.setter
    def X_training(self, value):
        self.K_w = None
        self.K_w_chol = None
        self.LLY = None
        self.v2 = None
        self.A2 = None
        self.B2 = None
        
        self.reset_XV()
        
        self.__X_training = value

    @Y_training.setter
    def Y_training(self, value):
        self.Y_centered = None
        self.LLY = None
        self.C_sim = None
        self.__Y_training = value

    def __str__(self):
        """ What to show when the object is printed """
        txt = '----- GP model ----- \n mean = {} \n likelihood = {} \n '.format(self.mean, self.likelihood)
        txt += 'kernel: \n {} \n'.format(self.kernel.__str__())
        txt += ' constraint: \n' 
        
        if self.constr_bounded is None and self.constr_deriv is None:
            txt += '   No constraints \n'
        else:
            ls = []
            if self.constr_bounded is not None:
                ls.append('f [{}]'.format(len_none(self.constr_bounded.Xv)))
            if self.constr_deriv is not None:
                ls = ls + ['df/dx_' + str(i+1) + ' [{}]'.format(len_none(self.constr_deriv[i].Xv)) for i in range(len(self.constr_deriv))]
        
            txt += '   ' + ', '.join(ls) + ' \n'
            txt += '   constr_likelihood = {} \n'.format(self.constr_likelihood)
            
        txt += '---------------------'
        return txt
    
    def reset(self):
        """ Reset model. I.e. forget all older calculations """
        self.K_w = None
        self.K_w_chol = None
        self.Y_centered = None
        self.LLY = None
        self.v2 = None
        self.A2 = None
        self.B2 = None
        
        self.reset_XV()
    
    def reset_XV(self):
        """ Reset everything that depends on the virtual observations XV """
        self.v1 = None
        self.A1 = None
        self.B1 = None
        self.L_1 = None
        self._p2 = False
        self.C_sim = None
        
    def calc_posterior_unconstrained(self, XS, full_cov = True):
        """
        Calculate pridictive posterior distribution f* | Y
        
        Returns: mean, cov (full or only diagonal)
        """
        
        # Check input
        self._check_XY_training()
        assert len(XS.shape) == 2, 'Test data XS must be 2d array'
        
        # Start timer
        t0 = time.time()
        
        # Run pre calcs
        self._prep_Y_centered()
        self._prep_K_w(verbatim = self.verbatim)
        self._prep_K_w_factor(verbatim = self.verbatim)
        self._prep_LLY()
        
        if self.verbatim: print("..Calculating f* | Y ...", end = '')
        
        # Kernel matrices needed
        K_x_xs = np.matrix(self.kernel.K(self.X_training, XS))
        
        v2 = triang_solve(self.K_w_chol, K_x_xs) 
        
        # Calculate mean
        mean = self.mean + K_x_xs.T*self.LLY
        
        # Calculate cov
        if full_cov:
            K_xs_xs = np.matrix(self.kernel.K(XS, XS))
            cov = K_xs_xs - v2.T*v2
        else:
            K_xs_xs_diag = self.kernel.K_diag(XS)
            cov = np.matrix(K_xs_xs_diag - np.square(v2).sum(0)).T
                    
        if self.verbatim: print(' DONE - Total time: {}'.format(formattime(time.time() - t0)))
        
        return mean, cov
    
    def calc_posterior_constrained(self, XS, compute_mode = False, num_samples = 1000, save_samples = 10, percentiles = [10, 50, 90], algorithm = 'minimax_tilting', resample = False):
        """
        Calculate constrained predictive posterior distribution f* | Y, C
        
        Returns: mean, variance, percentiles, mode, samples
        
        algorithm = 'rejection', 'gibbs' or 'minimax_tilting' 
        resample = False -> use old samples of constraint distribution if available
        """
        
        # Check that there are any constraints
        assert self.__has_xv(), 'No constraints or no virtual points specified for any constraint'
        
        # Check input
        self._check_XY_training()
        self._check_constraints()
        assert len(XS.shape) == 2, 'Test data XS must be 2d array'
        assert save_samples < num_samples, 'save_samples must be larger or equal to num_samples'
        
        # Start timer
        t0 = time.time()
        
        # Calculations only depending on (X, Y)
        self._prep_Y_centered()
        self._prep_K_w(verbatim = self.verbatim)
        self._prep_K_w_factor(verbatim = self.verbatim)
       
        # Calculations only depending on (X, XS) - v2, A2 and B2
        self._prep_1(XS, verbatim = self.verbatim)
        
        # Calculations only depending on (X, XV) - v1, A1 and B1
        self._prep_2(verbatim = self.verbatim)
        
        # Calculate mean of constraint distribution (covariance is B1)
        Lmu, constr_mean = self._calc_constr_mean()
        
        # Get bound vectors for constraint distribution
        LB, UB = self._calc_constr_bounds()
        
        # Calculate mean and covariance of constrained GP - A, B and Sigma
        self._prep_3(XS, verbatim = self.verbatim)
        
        ### Sample from truncated constraint distribution ###
        self._sample_constr_XV(m = num_samples, mu = constr_mean, sigma = self.B1, LB = LB, UB = UB, algorithm = algorithm, resample = resample, verbatim = self.verbatim)
            
        ### Sample from constrained GP ###
        t1 = time.time()
        if self.verbatim: print("..sampling {} times from constrained GP f*|C, Y".format(num_samples), end = '')
        
        # Draw from standard normal
        dim = XS.shape[0]
        U_sim = np.matrix(sp.random.multivariate_normal(np.zeros(dim), np.eye(dim), size = num_samples)).T

        # Using SVD to find Q.T*Q = Sigma
        # SVD decomposition of covariance matrix
        #(u, s, vh) = sp.linalg.svd(self.Sigma)
        #U, V = np.matrix(u), np.matrix(vh).T
        #Q = V*np.multiply(np.sqrt(s)[:,None], V.T)
        
        # Find matrix s.t. Q.T*Q = Sigma
        sigma_PD, Q = try_jitchol(self.Sigma)
        
        if not sigma_PD:
            Sigma_n = nearestPD(self.Sigma)
            err_pd = abs(Sigma_n - self.Sigma).max()
            Q = jitchol(Sigma_n)
        
        # Compute samples of f*
        fs_sim = self.mean + self.B*self.Y_centered + self.A*(self.C_sim - Lmu) + Q*U_sim
        
        # This corresponds to degenerate Sigma
        #print('Using degenerate Sigma!!')
        #fs_sim = self.mean + self.B*self.Y_centered + self.A*(self.C_sim - Lmu)
        
        if self.verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t1)))
        
        t1 = time.time()
        if self.verbatim: print('..computing statistics from samples', end = '')
        
        # Compute mode from samples
        mode = None
        if compute_mode: mode = mode_from_samples(fs_sim)
                
        # Save some of the samples
        randints = np.random.choice(num_samples, save_samples)
        samples = fs_sim[:, randints]

        # Calculate mean and percentiles
        mean = np.matrix(fs_sim.mean(axis = 1).reshape(-1, 1))
        var = np.matrix(fs_sim.var(axis = 1).reshape(-1, 1))
        perc = np.percentile(fs_sim, np.array(percentiles), axis = 1)
        
        if self.verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t1)))
        if self.verbatim: print(' DONE - Total time: {}'.format(formattime(time.time() - t0)))
            
        if not sigma_PD:
            print('WARNING: covariance matrix not PD! -- used closest PD matrix, error = {}'.format(err_pd))
            
        return mean, var, perc, mode, samples

    def calc_posterior_constrained_moments(self, XS, corr_free_approx = False):
        """
        Calculate first two moments of constranied predictive posterior distribution f* | Y, C
        
        corr_free_approx = True -> Uses correlation free approximation

        Returns: mean = E[f* | Y, C], cov  = cov[f* | Y, C]
        
        """
        
        # Check that there are any constraints
        assert self.__has_xv(), 'No constraints or no virtual points specified for any constraint'
        
        # Check input
        self._check_XY_training()
        self._check_constraints()
        assert len(XS.shape) == 2, 'Test data XS must be 2d array'
        
        # Start timer
        t0 = time.time()
        
        # Calculations only depending on (X, Y)
        self._prep_Y_centered()
        self._prep_K_w(verbatim = self.verbatim)
        self._prep_K_w_factor(verbatim = self.verbatim)
       
        # Calculations only depending on (X, XS) - v2, A2 and B2
        self._prep_1(XS, verbatim = self.verbatim)
        
        # Calculations only depending on (X, XV) - v1, A1 and B1
        self._prep_2(verbatim = self.verbatim)
        
        # Calculate mean of constraint distribution (covariance is B1)
        Lmu, constr_mean = self._calc_constr_mean()
        
        # Get bound vectors for constraint distribution
        LB, UB = self._calc_constr_bounds()
        
        # Calculate mean and covariance of constrained GP - A, B and Sigma
        self._prep_3(XS, verbatim = self.verbatim)
        
        # Compute moments of truncated variables (the virtual observations subjected to the constraint)
        t1 = time.time()
        if self.verbatim: print("..computing moments of C~|C, Y (from truncated Gaussian)", end = '')
        
        if corr_free_approx:
            # Using correlation free approximation
            tmu, tvar = trunc_norm_moments_approx_corrfree(mu = np.array(constr_mean).flatten(), sigma = self.B1, LB = LB, UB = UB)
            trunc_mu, trunc_cov = np.matrix(tmu).T, np.matrix(np.diag(tvar))
        else:
            # Using mtmvnorm algorithm 
            trunc_moments = mtmvnorm(mu = constr_mean, sigma = self.B1, a = LB, b = UB)
            trunc_mu, trunc_cov = np.matrix(trunc_moments[0]).T, np.matrix(trunc_moments[1])

        if self.verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t1)))

        # Compute moments of f* | Y, C
        t1 = time.time()
        if self.verbatim: print("..computing moments of f*|C, Y", end = '')

        mean = self.mean + self.B*self.Y_centered + self.A*(trunc_mu - Lmu) 
        cov = self.Sigma + self.A*trunc_cov*self.A.T
        
        if self.verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t1)))
        if self.verbatim: print(' DONE - Total time: {}'.format(formattime(time.time() - t0)))
                      
        return mean, cov
    
    def constrprob_Xv(self, nu = 0, posterior = True, algorithm = 'minimax_tilting', n = 10E4):
        """        
        Calculate the probability that the constraint holds at XV 
        
        posterior = False : Return P(C)
        posterior = True  : Return P(C | Y)

        algorithm = 'GenzBretz' or 'minimax_tilting'
        """

        # Check input
        self._check_constraints()

        assert algorithm in ['GenzBretz', 'minimax_tilting'], 'unknown algorithm = ' + algorithm
        
        # Get bound vectors for constraint distribution
        LB, UB = self._calc_constr_bounds()
        
        # Widen intervals with nu
        LB = LB - nu
        UB = UB + nu 
        
        if posterior:
        
            # Check input
            self._check_XY_training()

            # Calculations only depending on (X, Y)
            self._prep_Y_centered()
            self._prep_K_w(verbatim = False)
            self._prep_K_w_factor(verbatim = False)

            # Calculations only depending on (X, XV) - v1, A1 and B1
            self._prep_2(verbatim = False)

            # Calculate mean of constraint distribution (covariance is B1)
            Lmu, constr_mean = self._calc_constr_mean()

            # Calculate probability that the constraint holds at XV
            return pmvnorm(constr_mean, self.B1, LB, UB, algorithm, n)
        
        else:
            
            # Mean 
            Lmu = self._Lmu()
            
            # Covariance
            L1L2T_K_xv_xv = self._calc_L1L2()            
            n = L1L2T_K_xv_xv.shape[0]
            cov = L1L2T_K_xv_xv + self.constr_likelihood*np.identity(n)  
            
            # Calculate probability that the constraint holds at XV
            return pmvnorm(Lmu, cov, LB, UB, algorithm, n)

                
    def optimize(self, include_constraint = True, fix_likelihood = False, bound_min = 1e-6, conditional = False, pc_alg = 'minimax_tilting', n = 100):
        """
        Optimize hyperparameters using MLE
        
        include_constraint = True -> optimize P(Y)*P(C|Y)
        fix_likelihood = False -> Don't optimize GP likelihood parameter self.likelihood
        bound_min = minimum value in parameter bounds = (bound_min, ...)
        conditional = False -> used only for constrained optimization

        pc_alg = 'GenzBretz' or 'minimax_tilting' -> algorithm used to compute P(C)
        n -> number of samples used in pc_alg
        """
        
        has_constr = self.__has_xv() # If there are any virtual points specified
        
        if has_constr and include_constraint:
            # Optimize with constraint
            self._optimize_constrained(fix_likelihood = fix_likelihood, bound_min = bound_min, conditional = conditional, algorithm = pc_alg, n = n)
            
        else:
            # Optimize without constraint
            if include_constraint and not has_constr:
                if self.verbatim: print('No virtual points found for constraint')
        
            self._optimize_unconstrained(method = 'ML', fix_likelihood = fix_likelihood, bound_min = bound_min)
            
    
    def find_XV_subop(self, bounds, p_target, i_range = None, nu = None, max_iterations = 200, moment_approximation = False, num_samples = 1000, min_prob_unconstr_xv = 1E-10, sampling_alg = 'minimax_tilting', moment_alg = 'correlation-free', opt_method = 'differential_evolution', print_intermediate = True):
        """
        Find the set of virtual observations needed for a set of sub-operators
        
        Input:
        
        bounds = bounds on input space
        p_target = target constraint probability
        i_range = list of indices of sub-operators, e.g. i_range = [0, 2] -> find XV for L = [f, df/dx_2]
                  *** if i_range = None then all sub-operators are included ***
        
        max_iterations = maximum number of iterations

        print_intermediate = True -> Print intermediate steps
        
        min_prob_unconstr_xv = Minimum probability that the constraint holds at XV using the unconstrained distribution
        (using this as a stopping criterion when rejection sampling is used)

        Global optimizer:
        opt_method = 'differential_evolution' or 'basinhopping'

        --- The choice of algorithm used to compute the constraint probability ---

        moment_approximation = False -> Estimate constraint probability using samples of the constraint process
            num_samples = number of samples to use in estimation of constraint probability
            sampling_alg = algorithm used to sample from truncated Gaussian ('rejection', 'gibbs' or 'minimax_tilting')

        moment_approximation = True -> Use moment approximation (Assume Gaussian distribution using moments of the constraint process)
            moment_alg = 'correlation-free', 'mtmvnorm'

        """
        
        # Set list of sub-operators if not specified
        if i_range is None:
            i_range = []
            
            if self.constr_bounded is not None:
                i_range.append(0)
                
            if self.constr_deriv is not None:
                i_range = i_range + [i+1 for i in range(len(self.constr_deriv))] 
        
        if min_prob_unconstr_xv < 1E-6 and not moment_approximation and sampling_alg == 'rejection':
            if self.verbatim: print('WARNING: very low acceptance rate criterion for rejection sampling. min_prob_unconstr_xv = ' + str(min_prob_unconstr_xv))
        
        # Start timer
        t0 = time.time()

        # Set nu parameter for wider bounds
        # Will use LB - nu and UB + nu in constraint probability calculation
        if nu is None: nu = max(self.constr_likelihood*sp.stats.norm.ppf(p_target), 0)

        # Print start message
        label = []
        for i in i_range:
            F = 'f' if i == 0 else 'df/dx_{}'.format(i)
            label.append(F)

        label = '[' + ', '.join(label) + ']'

        if self.verbatim: print('Searching for points XV s.t. P(a - nu < Lf < b + nu) > p_target = {} for Lf = {} and nu = {} ...'.format(p_target, label, nu))

        # For storing results
        row = []
    
        # Just in case..
        self.reset_XV()
    
        pc_min = None
    
        for j in range(max_iterations):

            tj = time.time()

            # Check for criteria on minimum probability at XV using the unconstrained distribution
            # If this is too small the sampling becomes difficult
            if j == 0:
                pc_xv = self.constrprob_Xv(nu) if self.__has_xv() else 1

            if pc_xv < min_prob_unconstr_xv:
                if self.verbatim: print('ABORTED: Too low acceptance rate ({}) - Found {} points. Min. constraint prob = {}. Total time spent = {}'.format(pc_xv, j, pc_min, formattime(time.time() - t0)))
                break

            # Run global optimization for each sub-operator in the list
            pc_min_i = []
            x_min_i = []
            for i in i_range:

                success, x_min, pc_min = self._argmin_pc_subop(i, nu, bounds, opt_method, moment_approximation, sampling_alg, moment_alg, False, num_samples)

                if success:
                    pc_min_i.append(pc_min)
                    x_min_i.append(x_min)

                else:
                    print('ERROR: Optimizer failed after {} points found'.format(j))
                    
                    if j == 0: return None
                
                    df_out = pd.DataFrame(row)
                    df_out.columns = ['num_Xv', 'update_constr'] + ['Xv[{}]'.format(i+1) for i in range(len(x_min))] + ['pc_{}'.format(i+1) for i in i_range] + ['acc_rate']
                    return df_out

            # Choose sub-operator with smallest probability
            pc_min = min(pc_min_i)
            
            i_min = pc_min_i.index(pc_min)
            x_min = x_min_i[i_min]
            
            if self.constr_bounded is None: i_min = i_min + 1
            

            # Store results
            row.append([j, i_min] + list(x_min) + pc_min_i + [pc_xv])
            
            if pc_min >= p_target:
                if self.verbatim: print('DONE - Found {} points. Min. constraint prob = {}. Total time spent = {}'.format(j, pc_min, formattime(time.time() - t0)))
                break

            else:
                if self.verbatim and print_intermediate: print('i = {}, XV[{}] = {}, prob = {}, acc. rate = {}, optimization time = {}'.format(i_min, j+1, x_min, pc_min, pc_xv, formattime(time.time() - tj)))

                # Add point
                if i_min == 0:
                    self.constr_bounded.add_XV(x_min)
                else:
                    self.constr_deriv[i_min-1].add_XV(x_min)
                
                # Reset computations depending on XV
                self.reset_XV()
                
                # Compute new acceptance rate
                if j != 0:
                    pc_xv = self.constrprob_Xv(nu) 
                
                if j+1 == max_iterations:
                    if self.verbatim: print('DONE - Found {} points. Min. constraint prob = {}. Total time spent = {}'.format(j+1, pc_min, formattime(time.time() - t0)))
                    break

        # Put results in dataframe and return
        df_out = pd.DataFrame(row)
        df_out.columns = ['num_Xv', 'update_constr'] + ['Xv[{}]'.format(i+1) for i in range(len(x_min))] + ['pc_{}'.format(i+1) for i in i_range] + ['acc_rate']

        return df_out
        
    
    def _optimize_unconstrained(self, method = 'ML', fix_likelihood = False, bound_min = 1e-6):
        """
        Optimize hyperparameters of unconstrained GP
        
        method = 'ML' -> maximum marginal likelihood
        method = 'CV' -> cross validation
        
        fix_likelihood = False -> Don't optimize GP likelihood parameter self.likelihood
        bound_min = minimum value in parameter bounds = (bound_min, ...)
        """
        
        # Start timer
        t0 = time.time()
        if self.verbatim: print("..Running optimization for unconstrained GP ...", end = '')
        
        # Run optimization
        if method == 'ML':
            res = self._optimize_unconstrained_ML(fix_likelihood, bound_min)
        elif method == 'CV':
            print('TODO...')
            raise NotImplementedError 
        else:
            raise NotImplementedError
        
        # Save results
        self.__setparams(res.x, not fix_likelihood)

        if self.verbatim:
            if res.success:
                print(' DONE - Total time: {}'.format(formattime(time.time() - t0)))
            else:
                print('WARNING -- NO CONVERGENCE IN OPTIMIZATION -- Total time: {}'.format(formattime(time.time() - t0)))
    
    
    def _optimize_constrained(self, fix_likelihood = False, opt_method = 'differential_evolution', bound_min = 1e-6, conditional = False, algorithm = 'minimax_tilting', n = 10E4):
        """
        Optimize hyperparameters of unconstrained GP
        
        fix_likelihood = False -> Don't optimize GP likelihood parameter self.likelihood
        bound_min = minimum value in parameter bounds = (bound_min, ...)
        conditional = False -> maximize P(Y, C), otherwise maximize P(Y|C)

        algorithm = 'GenzBretz' or 'minimax_tilting' -> algorithm used to compute P(C)
        n -> number of samples used in 'GenzBretz' or 'minimax_tilting'
        """
        
        # Start timer
        t0 = time.time()
        if self.verbatim: 
            if not conditional: 
                print("..Running optimization for constrained GP - max P(Y, C) ...", end = '')
            else:
                print("..Running optimization for constrained GP - max P(Y | C) ...", end = '')
                
            
        # Define wrapper function for optimization
        def optfun(theta, *args):
            self.reset()
            self.__setparams(theta, not args[0])
            
            loglik_unconstr = self._loglik_unconstrained() # P(Y)
            loglik_constr = np.log(self.constrprob_Xv(posterior = True, algorithm = args[2], n = args[3])) # P(C|Y)
            
            if args[1] == False:
                return -(loglik_unconstr + loglik_constr) # P(Y, C)
            else:
                loglik_constr_cond = np.log(self.constrprob_Xv(posterior = False, algorithm = args[2], n = args[3])) # P(C)
                return -(loglik_constr + loglik_constr_cond - loglik_unconstr) # P(Y|C)
        
        # Initial guess (not used for differential_evolution)
        if fix_likelihood:
            theta = np.array(self.kernel.get_params())
        else:
            theta = np.array([self.likelihood] + list(self.kernel.get_params()))
        
        # Define bounds
        # theta = [(likelihood), kernel_var, kernel_len_1, ...]
        likelihood_scale = 10
        ker_var_scale = 10
        ker_len_scale = 2

        if fix_likelihood:
            bounds = [(bound_min, ker_var_scale*theta[0])]
            bounds = bounds + [(bound_min, ker_len_scale*theta[i+1]) for i in range(len(theta)-1)]

        else:
            bounds = [(bound_min, likelihood_scale*theta[0]), (bound_min, ker_var_scale*theta[1])]
            bounds = bounds + [(bound_min, ker_len_scale*theta[i+2]) for i in range(len(theta)-2)]
        
        # Run global optimization
        args = (fix_likelihood, conditional, algorithm, n)
        
        if opt_method == 'differential_evolution':
            res = optimize.differential_evolution(optfun, bounds = bounds, args = args)
        else:
            res = optimize.basinhopping(optfun, theta, minimizer_kwargs = {'args':args, 'bounds': bounds})
            res = res.lowest_optimization_result
        
        
        # Save results
        self.__setparams(res.x, not fix_likelihood)

        if self.verbatim:
            if res.success:
                print(' DONE - Total time: {}'.format(formattime(time.time() - t0)))
            else:
                print('WARNING -- NO CONVERGENCE IN OPTIMIZATION -- Total time: {}'.format(formattime(time.time() - t0)))
    
    
    def _argmin_pc_subop(self, i, nu, bounds, opt_method = 'differential_evolution', moment_approximation = False, sampling_alg = 'minimax_tilting', moment_alg = 'correlation-free', verbatim = False, num_samples = 1000):
        """
        Finds smallest probability that the constraint is satisfied for
        the i-th sub-operator
        
        i = 0: boundedness
        i > 0: df/dx_i
        
        Global optimizer:
        opt_method = 'differential_evolution' or 'basinhopping'
        
        moment_approximation = False -> Use sampling based method
            sampling_alg = 'rejection', 'gibbs' or 'minimax_tilting'
        
        moment_approximation = True -> Use moment approximation
            moment_alg = 'correlation-free', 'mtmvnorm'

        Returns:
        sucess = True/False
        x = argmin
        y = f(x)
        """
        
        min_prob_log = 1E-10 # Cap the constraint prob at this lower limit (for log transform)
        
        assert opt_method in ['differential_evolution', 'basinhopping'], 'unknown opt_method = ' + opt_method
        
        # Calculations only depending on (X, Y)
        self._prep_Y_centered()
        self._prep_K_w(verbatim = False)
        self._prep_K_w_factor(verbatim = False)
        
        label = 'a < f < b' if i == 0 else 'a < df/dx_{} < b'.format(i)
        if verbatim: print('Finding argmin(p_c) sub-operator ' + label)
            
        # Define function to optimize
        if self._no_const():
            if verbatim: print('No previous constraints found -- optimizing using unconstrained GP')
            
            args = (i, nu) 
            
            def optfun(x, *args):
                i = args[0]
                nu = args[1]

                p_c = self._constrprob_xs_1(np.array(x).reshape(1, -1), i, nu)[0]
                if p_c < min_prob_log: p_c = min_prob_log
                return np.log(p_c)
                
        else:
            if verbatim: print('Optimizing using estimated constraint probability with {} samples'.format(num_samples))
            
            if moment_approximation:
                # Use moment approximation of constraint probability
                
                args = (i, nu, moment_alg)
                
                def optfun(x, *args):
                    i = args[0]
                    nu = args[1]
                    alg = args[2]
                    
                    p_c = self._constrprob_xs_2_momentapprox(np.array(x).reshape(1, -1), i, nu, alg, verbatim = False)[0]
                    if p_c < min_prob_log: p_c = min_prob_log
                    return np.log(p_c)

            else:
                # Estimate constraint probability from samples of the constrained process

                args = (i, nu, num_samples, sampling_alg)
                
                def optfun(x, *args):
                    i = args[0]
                    nu = args[1]
                    num_samples = args[2]
                    alg = args[3]
                    
                    p_c = self._constrprob_xs_2(np.array(x).reshape(1, -1), i, nu, num_samples, alg, verbatim = False)[0]
                    if p_c < min_prob_log: p_c = min_prob_log
                    return np.log(p_c)
        
        # Run global optimization
        if opt_method == 'differential_evolution':
            res = optimize.differential_evolution(optfun, bounds = bounds, args = args)
        else:
            x0 = [0.5*(x[0] + x[1]) for x in bounds]
            res = optimize.basinhopping(optfun, x0, minimizer_kwargs = {'args':args, 'bounds': bounds})
            res = res.lowest_optimization_result
        
        if verbatim:
            if res.success: 
                print('Global optimization completed - found x = {}, p_c = {}'.format(res.x, np.exp(res.fun)))
            else:
                print('ERROR IN GLOBAL OPTIMIZATION - ' + opt_method)
        
        # Return
        return res.success, res.x, np.exp(res.fun)

    
    def _constr_posterior_dist_1(self, XS, i):
        """
        Return mean and covariance of the i-th constraint at XS
        
        C~(XS) | Y
        """
        
        # Calculations only depending on (X, Y)
        self._prep_Y_centered()
        self._prep_K_w(verbatim = False)
        self._prep_K_w_factor(verbatim = False)
    
        # c_v2, c_A2 and c_B2
        c_v2, c_A2, c_B2 = self._constr_prep_1(XS, i)
        
        # Prior mean
        if i == 0:
            # Boundedness
            Lmu = np.matrix(np.zeros(len(XS))).T
            
        else:
            # Derivative
            Lmu = np.matrix(self.mean*np.ones(len(XS))).T
            
        # Posterior mean
        mu = Lmu + c_A2*self.Y_centered
        
        # Return posterior mean and covariance
        return mu, c_B2
    
    
    def _constrprob_xs_1(self, XS, i, nu):
        """
        Return the probability that the i-th constraint is satisfied at XS
        
        C~(XS) | Y
        """
        
        # Get mean and cov
        mu, cov = self._constr_posterior_dist_1(XS, i)
        std = np.sqrt(np.diagonal(cov))

        # Get bound vectors for constraint distribution
        LB, UB = self.calc_constr_bounds_subop(XS, i)
       
        # Widen intervals with nu
        LB = LB - nu
        UB = UB + nu
    
        # Calculate probability that the constraint holds at each XV
        #return norm_cdf_int(np.array(mu)[:,0], std, LB, UB) # Exact
        return norm_cdf_int_approx(np.array(mu)[:,0], std, LB, UB) # Aprroximation within E-7 error
        
    def _constrprob_xs_2(self, XS, i, nu, num_samples, algorithm, verbatim = False):
        """
        Return the probability that the i-th constraint is satisfied at XS
        
        C~(XS) | Y, C
        """
        
        # Calculations only depending on (X, Y)
        self._prep_Y_centered()
        self._prep_K_w(verbatim = verbatim)
        self._prep_K_w_factor(verbatim = verbatim)
        
        # Calculations only depending on (X, XV) - v1, A1 and B1
        self._prep_2(verbatim = verbatim)
        
        # Calculate mean of constraint distribution at XV (covariance is B1)
        Lmu_XV, constr_mean = self._calc_constr_mean()
        
        # Get bound vectors for constraint distribution
        LB, UB = self._calc_constr_bounds()
        
        # Sample from truncated constraint distribution 
        self._sample_constr_XV(m = num_samples, mu = constr_mean, sigma = self.B1, LB = LB, UB = UB, algorithm = algorithm, resample = False, verbatim = verbatim)

        # c_v2, c_A2 and c_B2
        c_v2, c_A2, c_B2 = self._constr_prep_1(XS, i)
        
        # c_A, c_B and c_Sigma
        c_A, c_B, c_Sigma = self._constr_prep_2(XS, i, c_v2, c_A2, c_B2)
        
        # Get bound vectors for constraint distribution
        LB, UB = self.calc_constr_bounds_subop(XS, i)
        
        # Widen intervals with nu
        LB = LB - nu
        UB = UB + nu
        
        # Prior mean
        if i == 0:
            # Boundedness
            Lmu = np.matrix(np.zeros(len(XS))).T
            
        else:
            # Derivative
            Lmu = np.matrix(self.mean*np.ones(len(XS))).T
        
        # Posterior mean
        mu = Lmu + c_A*(self.C_sim - Lmu_XV) + c_B*self.Y_centered
        
        # Posterior standard deviation
        std = np.sqrt(np.diagonal(c_Sigma))
        
        # Calculate probability that the constraint holds at each XS individually 
        # for each sample C_j and take the average over C_j
        if XS.shape[0] == 1:
            
            # Faster for single input
            probs = norm_cdf_int_approx(np.array(mu)[0], std, LB, UB)
            probs = np.array([probs.mean()])
            
        else:
            probs = np.apply_along_axis(norm_cdf_int_approx, axis = 0, arr = np.array(mu), std = std, LB = LB, UB = UB)
            probs = probs.mean(axis = 1)
            
        # Return probability
        return probs
        

    def _constrprob_xs_2_momentapprox(self, XS, i, nu, algorithm, verbatim = False):
        """
        Return the probability that the i-th constraint is satisfied at XS using moment approximation

        algorithm =  'correlation-free' -> Using correlation free approximation
        algorithm =  'mtmvnorm' -> Using R-package mtvmnorm 
        algorithm =  'Genz' -> NOT YET IMPLEMENTED! (Using Genz approximation)

        C~(XS) | Y, C
        """

        assert algorithm in ['correlation-free', 'mtmvnorm'], 'unknown algorithm = ' + algorithm

        # Calculations only depending on (X, Y)
        self._prep_Y_centered()
        self._prep_K_w(verbatim = verbatim)
        self._prep_K_w_factor(verbatim = verbatim)
        
        # Calculations only depending on (X, XV) - v1, A1 and B1
        self._prep_2(verbatim = verbatim)
        
        # Calculate mean of constraint distribution at XV (covariance is B1)
        Lmu_XV, constr_mean = self._calc_constr_mean()

        # Get bound vectors for constraint distribution
        LB, UB = self._calc_constr_bounds()
        
        # c_v2, c_A2 and c_B2
        c_v2, c_A2, c_B2 = self._constr_prep_1(XS, i)
        
        # c_A, c_B and c_Sigma
        c_A, c_B, c_Sigma = self._constr_prep_2(XS, i, c_v2, c_A2, c_B2)
               
        # Compute moments of truncated variables (the virtual observations subjected to the constraint)
        t1 = time.time()
        if verbatim: print("..computing moments of C~|C, Y (from truncated Gaussian)", end = '')
        
        if algorithm =='correlation-free':
            # Using correlation free approximation
            tmu, tvar = trunc_norm_moments_approx_corrfree(mu = np.array(constr_mean).flatten(), sigma = self.B1, LB = LB, UB = UB)
            trunc_mu, trunc_cov = np.matrix(tmu).T, np.matrix(np.diag(tvar))
        else:
            # Using mtmvnorm algorithm 
            trunc_moments = mtmvnorm(mu = constr_mean, sigma = self.B1, a = LB, b = UB)
            trunc_mu, trunc_cov = np.matrix(trunc_moments[0]).T, np.matrix(trunc_moments[1])

        if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t1)))

        # Compute moments of Lf* | Y, C
        t1 = time.time()
        if verbatim: print("..computing moments of Lf*|C, Y", end = '')

        # Prior mean
        if i == 0:
            # Boundedness
            Lmu = np.matrix(np.zeros(len(XS))).T
            
        else:
            # Derivative
            Lmu = np.matrix(self.mean*np.ones(len(XS))).T
        
        # Posterior mean
        mean = Lmu + c_B*self.Y_centered + c_A*(trunc_mu - Lmu_XV)
        
        # Posterior standard deviation
        std = np.sqrt(np.diagonal(c_Sigma + c_A*trunc_cov*c_A.T))
        
        if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t1)))

        # Get bound vectors for constraint distribution
        LB, UB = self.calc_constr_bounds_subop(XS, i)

        # Widen intervals with nu
        LB = LB - nu
        UB = UB + nu

        # Calculate probability that the constraint holds at each XS individually 
        # for each sample C_j and take the average over C_j
        if XS.shape[0] == 1:
            # Faster for single input
            probs = norm_cdf_int_approx(np.array(mean)[0], std, LB, UB)
            
        else:
            probs = np.apply_along_axis(norm_cdf_int_approx, axis = 0, arr = np.array(mean), std = std, LB = LB, UB = UB)
            
        # Return probability
        #return probs, mean, std
        return probs

    def _sample_constr_XV(self, m, mu, sigma, LB, UB, algorithm, resample = False, verbatim = True):
        """ 
        Generate m samples from the constraint distribution
        
        Input: 
        m -- number of samples
        mu, sigma, LB, UB -- distribution parameters of truncated Gaussian
        algorithm -- name of sampling algorithm ('rejection', 'gibbs' or 'minimax_tilting')
        resample -- resample = False -> Use old samples if they exist
        
        """
        
        # Check if we should just use the old samples
        if self.C_sim is None: 
            generate_samples = True
        else:
            if m == self.C_sim.shape[1]:
                generate_samples = resample
            else:
                generate_samples = True
                    

        if generate_samples:
            # Generate samples
            
            # Start timer
            t0 = time.time()
            
            if verbatim: print("..sampling {} times from truncated constraint distribution C~|C, Y".format(m), end = '')
            self.C_sim = rtmvnorm(n = m, mu = mu, sigma = sigma, a = LB, b = UB, algorithm = algorithm).T
            if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t0)))
            
        else:
            # Use old
            if verbatim: print('..using old samples from truncated constraint distribution C~|C, Y')
        
        
    def _check_XY_training(self):
        """
        Check that X_training and Y_training are OK
        """
        assert self.X_training is not None, 'Training data not found. Use model.X_training = ...' 
        assert len(self.X_training.shape) == 2, 'Training data X_training must be 2d array'
        assert self.Y_training is not None, 'Training data not found. Use model.Y_training = ...' 
        assert len(self.Y_training.shape) == 1, 'Training data Y_training must be 1d array'
        assert self.X_training.shape[0] == len(self.Y_training), 'Number of points in X_training and Y_training does not match'
     
    def _check_constraints(self):
        
        assert self.__has_xv(), 'No constraints found'
        
        if self.constr_bounded is not None: self.constr_bounded._check(self.kernel.dim, 'Bounded')
        
        if self.constr_deriv is not None:
            i = 1
            for c in self.constr_deriv:
                c._check(self.kernel.dim, 'Derivative ' + str(i))
                i+=1
           
    def _optimize_unconstrained_ML(self, fix_likelihood = False, bound_min = 1e-6):
        """
        Optimize hyperparameters of unconstrained GP using ML
        
        fix_likelihood = False -> Don't optimize GP likelihood parameter self.likelihood
        bound_min = minimum value in parameter bounds = (bound_min, ...)
        """
        
        # Define wrapper functions for optimization
        def optfun(theta, fix_likelihood):
            self.reset()
            self.__setparams(theta, not fix_likelihood)
            return -self._loglik_unconstrained()
        
        def optfun_grad(theta, fix_likelihood):
            self.reset()
            self.__setparams(theta, not fix_likelihood)
            grad = -np.array(self._loglik_grad_unconstrained())
            if fix_likelihood: 
                return grad[1:]
            else:
                return grad
            
        # Define bounds
        num_params = self.kernel.dim + 2
        if fix_likelihood: num_params -= 1
        bounds = [(bound_min, None)]*num_params
        
        # Initial guess
        if fix_likelihood:
            theta = np.array(self.kernel.get_params())
        else:
            theta = np.array([self.likelihood] + list(self.kernel.get_params()))
        
        # Run optimizer
        res = optimize.minimize(optfun, theta, args=fix_likelihood, jac = optfun_grad, bounds=bounds, method = 'L-BFGS-B')

        return res
    
    def _loglik_unconstrained(self):
        """
        Calculates log marginal likelihood
        
        I.e. log(P(Y_training | X_training))
        """
        # Check input
        self._check_XY_training()
               
        # Run pre calcs
        self._prep_Y_centered()
        self._prep_K_w(verbatim = False)
        self._prep_K_w_factor(verbatim = False)
        self._prep_LLY()
        
        ### Calculate log marginal likelihood ###
        n = self.X_training.shape[0]
        loglik = -0.5*self.Y_centered.T*self.LLY - np.log(np.diag(self.K_w_chol)).sum() - (n/2)*np.log(2*np.pi)
        loglik = loglik[0,0]
        
        return loglik
    
    def _loglik_grad_unconstrained(self):
        """
        Calculates gradient of log marginal likelihood w.r.t hyperparameters
        """
        # Check input
        self._check_XY_training()
               
        # Run pre calcs
        self._prep_Y_centered()
        self._prep_K_w(verbatim = False)
        self._prep_K_w_factor(verbatim = False)
        self._prep_LLY()
        
        # Invert K_w using the Cholesky factor
        K_w_inv = chol_inv(self.K_w_chol)
         
        # Partial derivative of K_w w.r.t. likelihood
        n = self.X_training.shape[0]
        dKw_dlik = np.matrix(np.identity(n))
        
        # Partial derivative of K_w w.r.t. kernel parameters
        dK_dpar = self.kernel.K_gradients(self.X_training, self.X_training)
        
        # Calculate gradient
        alpha = K_w_inv*self.Y_centered
        tmp = alpha*alpha.T - K_w_inv
        
        Dloglik_lik = 0.5*traceprod(tmp, dKw_dlik)                # W.r.t. GP likelihood parameter
        Dloglik_ker = [0.5*traceprod(tmp, K) for K in dK_dpar]    # W.r.t. kernel parameters
        
        Dloglik = [Dloglik_lik] + Dloglik_ker
        
        return Dloglik
        
    def __setparams(self, theta, includes_likelihood):
        """
        Set model parameters from single array theta
        """
        if includes_likelihood:
            self.likelihood = theta[0]
            self.kernel.set_params(theta[1:])
        else:
            self.kernel.set_params(theta)
    
    def _prep_K_w(self, verbatim = False):
        """ 
        Calculate K_w = K_x_x + likelihood 
        
        *** Need to run this if one of the following arrays are changed : ***
            - X_training
        
        """
        
        if verbatim: print('..Running calculation of K_w ...', end = '')

        if self.K_w is None:
            
            # Start timer
            t0 = time.time()
            
            n = len(self.X_training)
            self.K_w = np.matrix(self.kernel.K(self.X_training, self.X_training) + self.likelihood*np.identity(n)) 

            if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t0)))

        else:
            pass
            if verbatim: print(' SKIP - (cached)')
                
    def _prep_K_w_factor(self, verbatim = False):
        """
        Calculate matrix L s.t. L*L.T = K_w 
        
        *** Need to run this if one of the following arrays are changed : ***
            - X_training
        """
        
        if verbatim: print('..Running calculation of Cholesky factor for K_w ...', end = '')

        if self.K_w_chol is None:

            # Start timer
            t0 = time.time()

            # Cholesky
            self.K_w_chol = np.matrix(jitchol(self.K_w)) 

            if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t0)))

        else:
            if verbatim: print(' SKIP - (cached)')
                    
   
    def _prep_LLY(self):
        """
        Calculate LLY = L.T \ L \ Y_centered
        
        *** Need to run this if one of the following arrays are changed : ***
            - X_training
            - Y_training
        
        """
              
        if self.LLY is None:
            # Run calculation
            self.LLY = mulinv_solve(self.K_w_chol, self.Y_centered, triang = True)
            
    def _prep_Y_centered(self):
        """
        Calculate Y_centered
        """
        if self.Y_centered is None: self.Y_centered = self.Y_training.reshape(-1, 1) - self.mean
            
    def _prep_1(self, XS, verbatim = False):
        """
        Preparation step 1 - calculate matrices depending only on (XS, X)
        
        Updates self.v2, self.A2, self.B2
        """
        
        if verbatim: print('..Running preparation step 1 - dependence on (XS, X) ...', end = '')
        
        # Start timer
        t0 = time.time()
        
        K_x_xs = np.matrix(self.kernel.K(self.X_training, XS))
        K_xs_xs = np.matrix(self.kernel.K(XS, XS))
        
        self.v2 = triang_solve(self.K_w_chol, K_x_xs) 
        self.B2 = K_xs_xs - self.v2.T*self.v2
        self.A2 = triang_solve(self.K_w_chol, self.v2, trans = True).T 
            
        if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t0)))
            
    def _prep_2(self, verbatim = False):
        """
        Preparation step 2 - calculate matrices depending only on (XV, X)
        
        Updates self.v1, self.A1, self.B1
        """
        
        if verbatim: print('..Running preparation step 2 - dependence on (XV, X) ...', end = '')
        
        if self._p2 == False:
            
            # Start timer
            t0 = time.time()

            # Calculate kernel matrices
            L2T_K_x_xv = self._calc_L2T(self.X_training)
            L1L2T_K_xv_xv = self._calc_L1L2()
            
            # Calculate v1, A1 and B1
            self.v1 = triang_solve(self.K_w_chol, L2T_K_x_xv) 
            self.A1 = triang_solve(self.K_w_chol, self.v1, trans = True).T 
            
            n = L1L2T_K_xv_xv.shape[0]
            self.B1 = L1L2T_K_xv_xv + self.constr_likelihood*np.identity(n) - self.v1.T*self.v1            
            
            self._p2 = True
            if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t0)))

        else:
            if verbatim: print(' SKIP - (cached)')
    
   
    def _prep_3(self, XS, verbatim = False):
        """
        Preparation step 3 - calculate A, B and Sigma
        
        Updatese self.L_1, self.v3, self.B3, self.A, self.B, and self.Sigma
        """

        if verbatim: print('..Running preparation step 3 - dependence on (XS, XV, X) ...', end = '')
        
        # Start timer
        t0 = time.time()
        
        self._prep_L1() # Compute L_1
        
        L2T_K_xs_xv = self._calc_L2T(XS)
        self.B3 = L2T_K_xs_xv - self.v2.T*self.v1
        
        self.v3 = triang_solve(self.L_1, self.B3.T) 
        
        self.A = triang_solve(self.L_1, self.v3, trans = True).T 
        self.B = self.A2 - self.A*self.A1
        
        self.Sigma = self.B2 - self.v3.T*self.v3
        
        if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t0)))
        
    
    def _prep_L1(self):
        """ Cholesky factorization of B1 """
        
        if self.L_1 is None:
            self.L_1 = np.matrix(jitchol(self.B1)) 
    
    def _constr_prep_1(self, XS, i):
        """
        Return c_v2, c_A2 and c_B2 for constraint distribution
        """
        
        if i == 0:
            # Boundedness
            L2T_K_X_XS = np.matrix(self.kernel.K(self.X_training, XS))
            L1L2T_K_XS_XS = np.matrix(self.kernel.K(XS, XS))
        
        else:
            L2T_K_X_XS = np.matrix(self.kernel.Ki0(XS, self.X_training, i-1)).T
            L1L2T_K_XS_XS = np.matrix(self.kernel.Kij(XS, XS, i-1, i-1))

        c_v2 = triang_solve(self.K_w_chol, L2T_K_X_XS) 
        c_A2 = triang_solve(self.K_w_chol, c_v2, trans = True).T 
        c_B2 = L1L2T_K_XS_XS - c_v2.T*c_v2
        
        return c_v2, c_A2, c_B2
    
    def _constr_prep_2(self, XS, i, c_v2, c_A2, c_B2):
        """
        Return c_A, c_B and c_Sigma for constraint distribution
        """
        
        L1L2T_XS_XV = self._calc_FiL2T(XS, i)
        
        #Check
        #self.TMP_FiL2T_XS_XV = L1L2T_XS_XV
        #self.TMP_L1L2T_XS_XV = self._calc_L1L2()
        
        c_B3 = L1L2T_XS_XV - c_v2.T*self.v1
        
        self._prep_L1() # Compute L_1
        c_v3 = triang_solve(self.L_1, c_B3.T) 
        
        c_A = triang_solve(self.L_1, c_v3, trans = True).T 
        c_B = c_A2 - c_A*self.A1
        
        c_Sigma = c_B2 - c_v3.T*c_v3
        
        return c_A, c_B, c_Sigma
        
    
    def _calc_constr_mean(self):
        """ 
        Calculate mean of constraint distribution 
        
        Returns
        Lmu : Linear operator applied to GP mean
        constr_mean : Mean of constraint distribution = Lmu + A_1(Y - mu)
        """
        Lmu = self._Lmu()
        
        constr_mean = Lmu + self.A1*self.Y_centered
        
        return Lmu, constr_mean
    
    def _Lmu(self):
        """
        Returns
        Lmu : Linear operator applied to GP mean
        """
        m_tot = self._num_virtuial_pts()
        
        if not self.__has_xv_bounded():
            # Only derivative constraint 
            Lmu = np.matrix(np.zeros(m_tot)).T
            
        else:
            if not self.__has_xv_deriv():
                # Only boundedness constraint
                Lmu = np.matrix(self.mean*np.ones(m_tot)).T
                
            else:
                # Both constraints 
                m_0 = self.constr_bounded.Xv.shape[0] # Number of virtual points - boundedness
                m_1 = m_tot - m_0 # Number of virtual points - derivatives
        
                # Operator applied to mean
                Lmu = np.matrix(np.concatenate((self.mean*np.ones(m_0), np.zeros(m_1)), axis=0)).T
        
        return Lmu
    
    def _calc_constr_bounds(self):
        """ Return lower/upper bounds for constraint """
        
        if self.__has_xv_bounded():
            LB = [self.constr_bounded.LBXV()]
            UB = [self.constr_bounded.UBXV()]
        else:
            LB = []
            UB = []
        
        if self.constr_deriv is not None:
            for constr in self.constr_deriv:
                if constr.Xv is not None:
                    LB.append(constr.LBXV())
                    UB.append(constr.UBXV())

        return np.concatenate(LB), np.concatenate(UB)
    
    def calc_constr_bounds_subop(self, XS, i):
        """ Return lower/upper bounds for the i-th suboperator only at XS """
        
        if i == 0:
            LB = self.constr_bounded.LB(XS)
            UB = self.constr_bounded.UB(XS)
        else:
            LB = self.constr_deriv[i-1].LB(XS)
            UB = self.constr_deriv[i-1].UB(XS)
        
        return LB, UB
            
    def _num_virtuial_pts(self):
        """ Return total number of virtual points """
        
        n = 0
        if self.__has_xv_bounded():
            n = self.constr_bounded.Xv.shape[0]
        
        if self.constr_deriv is not None:
            for constr in self.constr_deriv:
                if constr.Xv is not None:
                    n = n + constr.Xv.shape[0]
            
        return n
    
    def _no_const(self):
        """
        Returns TRUE if there are no constraints or only constraints with no virtual points
        """
        return not self.__has_xv()
        
    def _calc_L2T(self, XX):
        """ Calculate L2^T K_XX_XV for XX = X or XX = XS """
    
        ls = [] # List of block matrices to concatenate

        if self.__has_xv_bounded():
            ls.append(np.matrix(self.kernel.K(XX, self.constr_bounded.Xv)))

        if self.__has_xv_deriv():
            i = 0
            for constr in self.constr_deriv:
                if constr.Xv is not None:
                    ls.append(np.matrix(self.kernel.Ki0(constr.Xv, XX, i)).T)
                i+= 1

        return np.block(ls)
    
    def _calc_FiL2T(self, XS, i):
        """ Calculate FiL2^T K_S_XV -- i.e. only the i-th row-block of L1L2^T K_XS_XV """
        
        if i == 0:
            return self._calc_L2T(XS)
        
        # i > 0
        ls = [] # List of block matrices to concatenate

        if self.__has_xv_bounded():
            ls.append(np.matrix(self.kernel.Ki0(XS, self.constr_bounded.Xv, i-1)))

        if self.__has_xv_deriv():
            j = 0
            for constr in self.constr_deriv:
                if constr.Xv is not None:
                    ls.append(np.matrix(self.kernel.Kij(XS, constr.Xv, i-1, j)))

                j+= 1

        return np.block(ls)


    def _calc_L1L2(self):
        """ Calculate L1L2^T K_XV_XV """

        if self.__has_xv_bounded():
            
            # Calculate boundedness constraint matrix
            K_xv = np.matrix(self.kernel.K(self.constr_bounded.Xv, self.constr_bounded.Xv))

            if self.__has_xv_deriv():
                # Calculate cross terms
                ls = []
                i = 0
                for constr in self.constr_deriv:
                    if constr.Xv is not None:
                        ls.append(np.matrix(self.kernel.Ki0(constr.Xv, self.constr_bounded.Xv, i)).T)
                    i+= 1

                K01_xv = np.block(ls)

            else:
                # Only boundedness constraint
                return K_xv

        if self.__has_xv_deriv():
            # Calculate derivative constraint matrix
            ls = []

            for i in range(len(self.constr_deriv)):
                if self.constr_deriv[i].Xv is not None:
                    ls_row = []
                    for l in range(len(self.constr_deriv) - i):
                        j = l + i
                        if self.constr_deriv[j].Xv is not None:
                            ls_row.append(np.matrix(self.kernel.Kij(self.constr_deriv[i].Xv, self.constr_deriv[j].Xv, i, j)))
                    ls.append(ls_row)

            # Create blocks
            blocks = [[np.block(ls[0])]]
            n_cols = blocks[0][0].shape[1]
            for i in range(len(ls) - 1):
                tmp = np.block(ls[i+1])
                n_rows = tmp.shape[0]
                blanks = np.matrix(np.zeros((n_rows, n_cols - tmp.shape[1])))
                blocks.append([np.block([blanks, tmp])])

            K11_xv = np.block(blocks)

            if not self.__has_xv_bounded():
                # Only derivative constraints, return K11_xv
                i_lower = np.tril_indices(K11_xv.shape[0], -1)
                K11_xv[i_lower] = K11_xv.T[i_lower] 
                return K11_xv

        # Compute full matrix and return
        blanks = np.matrix(np.zeros((K01_xv.shape[1], K01_xv.shape[0])))
        K = np.block([[K_xv, K01_xv], [blanks, K11_xv]])
        i_lower = np.tril_indices(K.shape[0], -1)
        K[i_lower] = K.T[i_lower] 
        return K
    
    def _delete_xv(self):
        """ Delete all virtual points """
        if self.__has_xv_bounded():
            self.constr_bounded.Xv = None
            
        if self.__has_xv_deriv():
            for constr in self.constr_deriv:
                constr.Xv = None
    
    def __has_xv_bounded(self):
        """ Check if there are virtual points for boundedness constraint """
        if self.constr_bounded is None:
            return False
        else:
            return False if self.constr_bounded.Xv is None else True
        
    def __has_xv_deriv(self):
        """ Check if there are virtual points for derivative constraints """
        if self.constr_deriv is None:
            return False
        else:
            for constr in self.constr_deriv:
                if constr.Xv is not None:
                    return True
            return False
    
    def __has_xv(self):
        """ Check if there are any virtual points """
        return self.__has_xv_bounded() or self.__has_xv_deriv()