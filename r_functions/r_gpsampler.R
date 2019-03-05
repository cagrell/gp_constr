
library(tmvtnorm)
library(mvtnorm)
library(TruncatedNormal)
library(truncnorm)

#' Create n samples from truncated multivariate normal with 
#' mean = mu and covariance matrix = sigma
rtmvnorm_w <- function(n, mu, sigma, a, b, algorithm = "gibbs"){
  
  # Some different available algorithms:
  #'rejection'       - rejection sampling (from mvtnorm)
  #'gibbs'           - Gibbs sampling (from mvtnorm)
  #'minimax_tilting' - Minimax tilting (from TruncatedNormal)
  
  # For univariate distributions use rtruncnorm instead
  if (length(mu) == 1) {
    return(matrix(rtruncnorm(n = n, a = a, b = b, mean = mu, sd = sigma), ncol = 1))
  }
  
  if (algorithm == "gibbs") {
    return(rtmvnorm(n=n, mean=mu, sigma=sigma, lower=a, upper=b, burn.in.samples=100, algorithm=algorithm))
  }
  
  if (algorithm == "rejection") {
    return(rtmvnorm(n=n, mean=mu, sigma=sigma, lower=a, upper=b, algorithm=algorithm))
  }
  
  if (algorithm == "minimax_tilting") {
    Z <- mvrandn(a - mu, b - mu, sigma, n)
    return(t(Z + mu))
  }
  
}

# For calculating probability over rectangle given by [a, b]
# (Same as acceptance rate for rejection sampling)
pmvnorm_w <- function(mu, sigma, a, b) {
  
  rownames(sigma) <- colnames(sigma)
  
  return(as.numeric(pmvnorm(lower=a, upper=b, mean=mu, sigma=sigma)))
  
}

# For calculating moments of truncated multivariate normal
mtmvnorm_w <- function(mu, sigma, a, b) {
  
  rownames(sigma) <- colnames(sigma)
  
  moments <- mtmvnorm(mean=mu, sigma=sigma, lower=a, upper=b)
  return(moments)
  
}