from sklearn.neighbors import KernelDensity
import numpy as np
import scipy as sp

def norm_cdf_int(mu, std, LB, UB):
    """ Return P(LB < X < UB) for X Normal(mu, std) """
    rv = sp.stats.norm(mu, std)
    return rv.cdf(UB) - rv.cdf(LB)

def norm_cdf_int_approx(mu, std, LB, UB):
    """ 
    Return P(LB < X < UB) for X Normal(mu, std) using approximation of Normal CDF 
    
    Input: All inputs as 1-D arrays
    """
    l = normal_cdf_approx((LB - mu)/std)
    u = normal_cdf_approx((UB - mu)/std)
    return u - l

def normal_cdf_approx(x):
    """ 
    Approximation of standard normal CDF
    
    Input: x = array
    
    Polynomial approximation from Abramowitz and Stegun p. 932
    http://people.math.sfu.ca/~cbm/aands/frameindex.htm
    
    Absolute error < 7.5*10^-8
    """
    p = 0.2316419
    b = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    
    xx = abs(x) # Approximation only works for x > 0, return 1 - p otherwise
    
    t = 1/(1 + p*xx)
    Z = (1/(np.sqrt(2*np.pi)))*np.exp(-(x*x)/2)
    pol = b[0]*t + b[1]*(t**2) + b[2]*(t**3) + b[3]*(t**4) + b[4]*(t**5)
    
    prob = 1 - Z*pol # For x > 0
    prob[x < 0] = 1 - prob[x < 0] # Change when x < 0
    
    return prob

def mode_from_samples(samples, bandwidth_fraction = 0.1):
    """
    Compute the mode for each set of samples in 'samples'
    
    Using kernel density estimation
    
    Input:
    samples -- m x n array with n samples for each of m univariate random variables
    bandwidth_fraction -- kde will use bandwidth = [range of dataseries (max - min)] * bandwidth_fraction
    """

    # Function to optimize for finding the mode
    # -- the kernel density estimator
    def optfun(x, *args):
        kde = args[0]
        return -kde.score_samples(x.reshape(-1, 1))

    mode = np.zeros(samples.shape[0])

    for i in range(samples.shape[0]):
        data = samples[i].T
        min_x, max_x = data.min(), data.max()
        bandwidth = bandwidth_fraction*(max_x - min_x)

        # Fit kde
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)

        # Find argmax of density
        args = (kde, )
        bounds = [(min_x, max_x)]

        res = sp.optimize.differential_evolution(optfun, bounds = bounds, args = args)

        mode[i] = res.x[0]
        
    return mode

def trunc_norm_moments_approx_corrfree(mu, sigma, LB, UB, inf_num = 1E100):
    """ 
    Correlation free approximation of truncated moments of multivariate Gaussian
    
    If X~N(mu, sigma), compute expectation and variance of X | LB <= X <= UB
    
    Input: 
    mu, LB, UB : 1D numpy arrays
    sigma : numpy matrix
    inf_num : inf values are replaced with this number in calculations
    
    Returns:
    tmu, tvar (expectation and variance of truncated variable)
    """
    
    s2 = np.diag(sigma)
    s = np.sqrt(s2)
    a = (LB - mu )/s
    b = (UB - mu )/s
    
    # Replace inf and -inf by numbers
    a[a == float('inf')] = inf_num
    a[a == float('-inf')] = -inf_num
    b[b == float('inf')] = inf_num
    b[b == float('-inf')] = -inf_num
    
    phi_a = sp.stats.norm.pdf(a)
    phi_b = sp.stats.norm.pdf(b)
    PHI_diff = normal_cdf_approx(b) - normal_cdf_approx(a)
    
    tmu = mu + s*(phi_a - phi_b)/PHI_diff
    tvar = s2*(1 + (a*phi_a - b*phi_b)/PHI_diff - ((phi_a - phi_b)/PHI_diff)**2)
    
    return tmu, tvar