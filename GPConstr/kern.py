####
# Kernels for use in the GPconstr model 
# RBF and Matern5/2
####

import numpy as np

# Can replace this with custom code that does the same..
from sklearn.metrics.pairwise import euclidean_distances as sklear_euclidean_distances

class kernel_Stationary():
    """
    Superclass for stationary kernels
    """
    
    kernel_name = ''
    
    def __init__(self, variance, lengthscale):

        self.lengthscale = self._convert_to_array(lengthscale)
        self.variance = variance
        self.dim = len(self.lengthscale)

        assert np.isscalar(variance), 'variance must be scalar'
    
    def __str__(self):
        """ What to show when the object is printed """
        return '  type = {} \n   input dim = {} \n   lenghtscale = {} \n   variance = {}'.format(
            self.kernel_name, self.dim, self.lengthscale, self.variance)

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
        
        r = self.r(X1, X2)
        K_r = self.K_r(r)
        
        # W.r.t variance
        dK_dv = K_r/self.variance
        
        # W.r.t lengthscales
        dK_dr = self.dK_dr(r)
        dr_dl = self.dr_dl(X1, X2)
        
        dK_dl = [dK_dr*dr_dl_i for dr_dl_i in dr_dl]

        return [dK_dv] + dK_dl

    def R(self, X1, X2):
        """ 
        Return scaled distances squared 
        """
        return self._euclidian_dist_squared(X1 / self.lengthscale, X2 / self.lengthscale)
    
    def r(self, X1, X2):
        """
        Return scaled distances
        """
        return np.sqrt(self.R(X1, X2))
    
    def dr_dl(self, X1, X2):
        """
        Derivative of r w.r.t. length scales
        """
        
        # r
        r = self.r(X1, X2)
        
        # dr_dR
        dr_dR = self.dr_dR(r)
        
        # dr_dl
        dr_dl = []
        for i in range(len(self.lengthscale)):
            t1, t2 = np.meshgrid(X1[:,i], X2[:,i])
            dR_dli = ((-2/self.lengthscale[i])*((t1 - t2)/self.lengthscale[i])**2).T
            dr_dl.append(dr_dR*dR_dli)
            
        return dr_dl
    
    def dr_dR(self, r):
        """dr / dR"""
        f_div_zero = np.vectorize(lambda x: 0.0 if x == 0.0 else 1/(2*x))
        return f_div_zero(r)
    
    def Ri(self, X1, X2, i):
        """ 
        Returns dR/dX1_i 
        Note: dR/dX2_j(X1, X2) = Ri(X2, X1, j).T
        """
        return (2/self.lengthscale[i]**2)*(X1[:,i].reshape(-1, 1) - X2[:,i].reshape(-1, 1).T)
    
    def K(self, X1, X2):
        """ Returns Gram matrix of k(x1, x2) """
        return self.K_r(self.r(X1, X2))
        
    def K_diag(self, X):
        """ Returns diagonal of Gram matrix of k(x, x) """
        return np.ones(len(X))*self.variance
    
    def Ki0(self, X1, X2, i):
        """ For K = K(X1, X2), X1 = [X1_1, X1_2, ..], X2 = [X2_1, X2_2, ..] etc., return dK/dX1_i """
        
        r = self.r(X1, X2)
        
        dK_dr = self.dK_dr(r)
        dr_dR = self.dr_dR(r)
        dR_dxi = self.Ri(X1, X2, i)
        
        return dK_dr*dr_dR*dR_dxi
    
    def Kij(self, X1, X2, i, j):
        """ For K = K(X1, X2), X1 = [X1_1, X1_2, ..], X2 = [X2_1, X2_2, ..] etc., return d^2K/dX1_i*dX2_j """
        
        r = self.r(X1, X2) + 1E-50 # make nonzero to avoid handling singularity
        dK_dr = self.dK_dr(r)
        d2K_drdr = self.d2K_drdr(r)
        
        dr_dR = self.dr_dR(r)
        dR_dxi = self.Ri(X1, X2, i)
        dR_dxj = self.Ri(X1 = X2, X2 = X1, i = j).T
        dr_dxi = dr_dR*dR_dxi
        dr_dxj = dr_dR*dR_dxj
        
        d2R_dxidxj = -2/self.lengthscale[i]**2 if i == j else 0
        d2r_dxidxj = d2R_dxidxj*dr_dR + dR_dxi*dR_dxj*(-2*dr_dR**3)
        
        return d2r_dxidxj*dK_dr + dr_dxi*dr_dxj*d2K_drdr
    
    def K_r(self, r):
        """ Kernel as a function of scaled distances """
        raise NotImplementedError('Need to implement K_r, dK_dr and d2K_drdr for stationary kernels')
     
    def dK_dr(self, r):
        """
        Derivative w.r.t scaled distances
        """
        raise NotImplementedError('Need to implement K_r, dK_dr and d2K_drdr for stationary kernels')
    
    def d2K_drdr(self, r):
        """
        Double derivative w.r.t scaled distances
        """
        raise NotImplementedError('Need to implement K_r, dK_dr and d2K_drdr for stationary kernels')
    
    def Kii_diag(self, X, i):
        """ Returns diagonal of Gram matrix of d^2K/dX1_i*dX2_i """
        raise NotImplementedError('Need to implement Kii_diag')
    
class kernel_RBF(kernel_Stationary):
    """
    RBF kernel
    """
    
    kernel_name = 'RBF'
    
    def K_r(self, r):
        """ Kernel as a function of scaled distances """
        return self.variance*np.exp(-0.5*r**2)
     
    def dK_dr(self, r):
        """
        Derivative w.r.t scaled distances
        """
        return -r*self.K_r(r)
    
    def d2K_drdr(self, r):
        """
        Double derivative w.r.t scaled distances
        """
        return (r**2 - 1)*self.K_r(r)
    
    
    def Ki0(self, X1, X2, i):
        """ Overload generic with faster alternative """
        
        # Include K and r as input to use this
        # Make use of K or r if they exist 
        ##if K is None:
        ##    if r is None:
        ##        K = self.K(X1, X2)
        ##    else:
        ##        K = self.K_r(r)
        
        return -0.5*self.K(X1, X2)*self.Ri(X1, X2, i)
        
    
    def Kij(self, X1, X2, i, j):
        """ Overload generic with faster alternative """
        
        # Include K and r as input to use this
        # Make use of K or r if they exist 
        ##if K is None:
        ##    if r is None:
        ##        K = self.K(X1, X2)
        ##    else:
        ##        K = self.K_r(r)
        
        F = 1/self.lengthscale[i]**2 if i == j else 0
        K = self.K(X1, X2)
        return K*((1/4)*self.Ri(X1, X2, i)*self.Ri(X1 = X2, X2 = X1, i = j).T + F)    

    def Kii_diag(self, X, i):
        """ Returns diagonal of Gram matrix of d^2K/dX1_i*dX2_i """
        const = self.variance/(self.lengthscale[i]**2)
        return np.ones(len(X))*const


class kernel_Matern52(kernel_Stationary):
    """
    Matern 5/2 kernel
    """
    
    kernel_name = 'Matern52'
    
    def K_r(self, r):
        """ Kernel as a function of scaled distances """
        return self.variance*(1 + np.sqrt(5)*r + 5/3*r**2)*np.exp(-np.sqrt(5)*r)
     
    def dK_dr(self, r):
        """
        Derivative w.r.t scaled distances
        """
        return -5/3*self.variance*(r + np.sqrt(5)*r**2)*np.exp(-np.sqrt(5)*r)
        
    def d2K_drdr(self, r):
        """
        Double derivative w.r.t scaled distances
        """
        return -5/3*self.variance*(1 + np.sqrt(5)*r - 5*r**2)*np.exp(-np.sqrt(5)*r)
    
    def Kii_diag(self, X, i):
        """ Returns diagonal of Gram matrix of d^2K/dX1_i*dX2_i """
        const = self.variance*(5/3)*(1/self.lengthscale[i]**2)
        return np.ones(len(X))*const
    
            
class kernel_RBF_generic(kernel_Stationary):
    """
    RBF kernel, use generic set-up for testing
    """
    
    kernel_name = 'RBF_generic'
    
    def K_r(self, r):
        """ Kernel as a function of scaled distances """
        return self.variance*np.exp(-0.5*r**2)
     
    def dK_dr(self, r):
        """
        Derivative w.r.t scaled distances
        """
        return -r*self.K_r(r)
    
    def d2K_drdr(self, r):
        """
        Double derivative w.r.t scaled distances
        """
        return (r**2 - 1)*self.K_r(r)

# Old kernel class
# class kernel_RBF():
#     """
#     RBF kernel
#     """
    
#     def __init__(self, variance, lengthscale):
        
#         self.lengthscale = self._convert_to_array(lengthscale)
#         self.variance = variance
#         self.dim = len(self.lengthscale)
        
#         assert np.isscalar(variance), 'variance must be scalar'
       
#     def __str__(self):
#         """ What to show when the object is printed """
#         return '  type = {} \n   input dim = {} \n   lenghtscale = {} \n   variance = {}'.format(
#             'RBF', self.dim, self.lengthscale, self.variance)
    
#     def _convert_to_array(self, v):
#         if np.isscalar(v):
#             return np.array([v])
#         else:
#             return np.array(v)
    
#     def _euclidian_dist_squared(self, X1, X2):
#         """ Return gram matrix with ||x - y||^2 for each pair of points """
               
#         # Use function from sklearn - can replace this later to avoid dependence on sklearn
#         return sklear_euclidean_distances(X1, X2, squared = True)         
    
#     def set_params(self, theta):
#         """
#         Set all kernel parameters from a single array theta
#         .. Used in optimization
#         """
#         assert self.dim == (len(theta) - 1), 'Parameter array does not match kernel dimension'
#         self.variance = theta[0]
#         self.lengthscale = theta[1:]
        
#     def get_params(self):
#         """
#         Get all kernel parameters in a single array
#         .. Used in optimization
#         """
#         return np.array([self.variance] + list(self.lengthscale))
    
#     def K_gradients(self, X1, X2):
#         """
#         Return kernel gradients w.r.t hyperparameters
        
#         Returns:
#         List of Gram matrices of derivatives of K w.r.t. the hyperparameters in the ordering 
#         given by get_params and set_params
#         """
        
#         R = self.R(X1, X2)
#         K_R = self.K_R(R)
        
#         # W.r.t variance
#         dK_dv = K_R/self.variance
        
#         # W.r.t lengthscales
#         dK_dR = -0.5*K_R
        
#         dK_dl = []
#         for i in range(len(self.lengthscale)):
#             t1, t2 = np.meshgrid(X1[:,i], X2[:,i])
#             dR_dli = ((-2/self.lengthscale[i])*((t1 - t2)/self.lengthscale[i])**2).T

#             dK_dl.append(dK_dR*dR_dli)

#         return [dK_dv] + dK_dl
    
#     def R(self, X1, X2):
#         """ 
#         Return scaled distances squared 
#         For RBF kernel: K(X1, X2) = variance * exp(-0.5 * R)
#         """
#         return self._euclidian_dist_squared(X1 / self.lengthscale, X2 / self.lengthscale)
    
#     def K_R(self, R):
#         """ Kernel as a function of squared distances """
#         return self.variance*np.exp(-0.5*R)
    
#     def Ri(self, X1, X2, i):
#         """ 
#         Returns dR/dX1_i 
#         Note: dR/dX2_j(X1, X2) = Ri(X2, X1, j).T
#         """
#         return (2/self.lengthscale[i]**2)*(X1[:,i].reshape(-1, 1) - X2[:,i].reshape(-1, 1).T)
    
#     def Ki0(self, X1, X2, i, R = None, K = None):
#         """ For K = K(X1, X2), X1 = [X1_1, X1_2, ..], X2 = [X2_1, X2_2, ..] etc., return dK/dX1_i """
        
#         # Make use of K or R if they exist
#         if K is None:
#             if R is None:
#                 K = self.K_R(self.R(X1, X2))
#             else:
#                 K = self.K_R(R)
        
#         return -0.5*K*self.Ri(X1, X2, i)
    
#     def Kij(self, X1, X2, i, j, R = None, K = None):
#         """ For K = K(X1, X2), X1 = [X1_1, X1_2, ..], X2 = [X2_1, X2_2, ..] etc., return d^2K/dX1_i*dX2_j """
        
#         # Make use of K or R if they exist
#         if K is None:
#             if R is None:
#                 K = self.K_R(self.R(X1, X2))
#             else:
#                 K = self.K_R(R)
        
#         F = 1/self.lengthscale[i]**2 if i == j else 0
        
#         return K*((1/4)*self.Ri(X1, X2, i)*self.Ri(X1 = X2, X2 = X1, i = j).T + F)    
            
    
#     def K(self, X1, X2):
#         """ Returns Gram matrix of k(x1, x2) """
#         return self.K_R(self.R(X1, X2))
        
#     def K_diag(self, X):
#         """ Returns diagonal of Gram matrix of k(x, x) """
#         return np.ones(len(X))*self.variance