import numpy as np
import scipy as sp 
from numpy.core.umath_tests import inner1d

def is_symPD_svd(M, verbatim = False, sym_tol = 1E-5, psd_tol = 1E-5):
    """ Check if matrix is symmetric and positive definite up to some error using SVD """
    
    # Symmetric
    symdiff = abs(M - M.T).max()
    is_sym = symdiff < sym_tol
    
    # Eigenvalues
    eigvals = np.linalg.eigvals(M)
    eig_real_max, eig_real_min = eigvals.real.max(), eigvals.real.min()
    eig_imag_max, eig_imag_min = eigvals.imag.max(), eigvals.imag.min()
    
    # SVD
    # For M = U*S*V.T check if M = V*S*V.T also
    (u, s, vh) = sp.linalg.svd(M)
    vsvt = np.dot(vh.T * s, vh)
    svddiff = abs(M - vsvt).max()
    is_psd = svddiff < psd_tol
    
    if verbatim: 
        if is_sym: 
            print('Symmetric: error = {} < {}, OK'.format(symdiff, sym_tol))
        else:
            print('WARNING! Symmetric: error = {} > {}, NOT OK'.format(symdiff, sym_tol))
            
        print('Eigenvalues real part: min = {}, max = {}'.format(eig_real_min, eig_real_max))
        print('Eigenvalues imag part: min = {}, max = {}'.format(eig_imag_min, eig_imag_max))
        
        if is_psd: 
            print('Postive definite, M - VSV.T where M = USV.T (SVD): error = {} < {}, OK'.format(svddiff, psd_tol))
        else:
            print('WARNING! Postive definite, M - VSV.T where M = USV.T (SVD): error = {} > {}, NOT OK'.format(svddiff, psd_tol))
    
    return is_sym & is_psd

def isPD_chol(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = sp.linalg.cholesky(B)
        return True
    except sp.linalg.LinAlgError:
        return False
    
def try_jitchol(B):
    """Returns true AND cholesky factor when input is positive-definite, via Cholesky"""
    try:
        L = jitchol(B)
        return True, L
    except sp.linalg.LinAlgError:
        return False, None

def isPD_det(B):
    """Returns true when input is positive-definite, via Determinant"""
    if sp.linalg.det(B) <= 0: return False
    return True

def nearestPD(A, check = 'chol'):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    
    *** Modified code to include different checks for PD depending on use ***
    Can check using 
    Cholesky    - check = 'chol'
    SVD         - check = 'svd'
    Determinant - check = 'det'
    *************************************************************************
    """

    assert check in ['chol', 'svd', 'det'], 'Unknown checking function'
    
    B = (A + A.T) / 2
    _, s, V = sp.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2
    
    # Checking function
    if check == 'chol':
        isPD = isPD_chol
    elif check == 'svd':
        isPD = is_symPD_svd
    else:
        isPD = isPD_det
    
    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(sp.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def jitchol(A, maxtries = 5):
    """ Cholesky with jitter """
    A = np.ascontiguousarray(A)
    L, info = sp.linalg.lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise sp.linalg.LinAlgError("not pd: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = sp.linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                print('(!chol jitter added : '+ jitter + ')', end = '')
                return L
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise sp.linalg.LinAlgError("not positive definite, even with jitter.")
    import traceback
    try: raise
    except:
        print('\n'.join(['Added jitter of {:.10e}'.format(jitter),
            '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
    return L

def triang_solve(A, B, lower = True, trans = False, unitdiag = False):
    """
    Wrapper for lapack dtrtrs function
    DTRTRS solves a triangular system of the form
        A * X = B  or  A**T * X = B,
    where A is a triangular matrix of order N, and B is an N-by-NRHS
    matrix.  A check is made to verify that A is nonsingular.
    :param A: Matrix A(triangular)
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :param trans: calculate A**T * X = B (true) or A * X = B (false)
    
    :returns: Solution to A * X = B or A**T * X = B
    """
    
    lower_num = 1 if lower else 0
    trans_num = 1 if trans else 0
    unitdiag_num = 1 if unitdiag else 0
   
    A = np.asfortranarray(A)
    #Note: B does not seem to need to be F ordered!
    return np.matrix(sp.linalg.lapack.dtrtrs(A, B, lower=lower_num, trans=trans_num, unitdiag=unitdiag_num)[0])

def mulinv_solve(F, B, triang = True):
    """
    Returns C = A^{-1} * B where A = F*F^{T}
    
    triang = True -> when F is LOWER triangular. This gives faster calculation
    
    """
    if triang:
        tmp = triang_solve(F, B) # F*tmp = B
        C = triang_solve(F, tmp, trans = True) # F.T*C = tmp 
        
    else:
        tmp = np.matrix(sp.linalg.solve(F, B)) # F*tmp = B
        C = np.matrix(sp.linalg.solve(F.T, tmp)) # F.T*C = tmp 
        
    return C

def mulinv_solve_rev(F, B, triang = True):
    """
    Reversed version of mulinv_solve
    
    Returns C = B * A^{-1} where A = F*F^{T}
    
    triang = True -> when F is LOWER triangular. This gives faster calculation
    
    """
    return mulinv_solve(F, B.T, triang).T
    

def symmetrify(A, upper=False):
    """ Create symmetric matrix from triangular matrix """
    triu = np.triu_indices_from(A,k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]

def chol_inv(L):
    """
    Return inverse of matrix A = L*L.T where L is lower triangular
    Uses LAPACK function dpotri
    """
    A_inv, info = sp.linalg.lapack.dpotri(L, lower=1)
    A_inv = np.matrix(A_inv)
    symmetrify(A_inv)
    return A_inv

def traceprod(A, B):
    """
    Calculate trace(A*B) for two matrices A and B
    """
    return np.sum(np.core.umath_tests.inner1d(np.array(A), np.array(B).T))