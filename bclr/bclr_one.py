import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from polyagamma import random_polyagamma
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

inv = np.linalg.inv
det = np.linalg.det
    
class BayesCC:
    """
    This class implements the Bayesian Changepoint via Logistic Regression (bclr) method 
    described in Thomas, Jauch, and Matteson (2023).
    """
    def __init__(self, X, prior_mean, prior_cov, n_iter, prior_kappa=None,
                 scaled=True, burn_in=None, incl_last=True):
        """

        Parameters
        ----------
        X : array-like
            n x d array of n d-dimensional vectors to assess changepoints.
        prior_mean : ndarray of length d
            Array containing prior mean of Normal distribution.
        prior_cov : ndarray of shape (d,d)
            Symmetric positive (semi)definite covariance matrix. 
        prior_kappa : ndarray of length n (or n-1 if incl_last=False) nonnegative values
            This should consist of at 
        n_iter : int
            Number of iterations to run Gibbs sampler. 
        scaled : bool, optional
            If False, each column of data will be mean centered and normalized to have variance 1. The default is True.
        burn_in : int, optional
            Number of initial iterations of the Gibbs sampler to discard. The default is None.
        incl_last : bool, optional
            Whether or not to include the last observation in changepoint calculations. The default is True.

        Raises
        ------
        ValueError
            Array must be two-dimensional.

        Returns
        -------
        None.

        """
        if len(X.shape) > 2: 
            raise ValueError("Array must be 2-dimensional")
    
        self.n, self.p = X.shape
    
        if not scaled:
            self.X = StandardScaler().fit_transform(X)
        else: 
            self.X = X
            
        if burn_in == None:
            burn_in = n_iter/2
            
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        if prior_kappa is None:
            self.prior_kappa = np.repeat(1, self.n)
        else:
            if np.any(prior_kappa < 0):
                raise ValueError("Must have nonnegative weights for prior.")
            elif len(prior_kappa) != (self.n+int(incl_last)-1):
                raise IndexError("Length of prior distribution must be as long as number of potential changepoints")
            self.prior_kappa = prior_kappa
            
        self.n_iter = n_iter
        self.burn_in = int(burn_in)
        self.incl_last = incl_last
    
    def fit(self, init_k = None, init_beta = None):
        """
        Fit BayesCC class, meaning implement the Gibbs sampler discussed for drawing posterior changepoints and coefficients in 
        Thomas, Jauch, and Matteson (2023).

        Parameters
        ----------
        init_k : int, optional
            The initial changepoint estimate to start the Markov chain. The default is None, in which case it is set
            to be equal to the midpoint of the sequence.
        init_beta : array_like of shape (d,) (where d is dimensionality of the data), optional
            Initial value of the coefficient vector beta. The default is None, in which case it is set to be equal 
            to the all zero coefficient vector.
            
        Returns
        -------
        None.

        """
        
        if init_k == None:
            init_k = self.n/2
        
        if init_beta is None:
            init_beta = np.repeat(0, self.p)
        
        self.k_draws_ = np.empty(self.n_iter)
        self.k_draws_[0] = int(init_k)
        
        self.beta_draws_ = np.empty((self.n_iter, self.p))
        self.beta_draws_[0] = init_beta
        
        self.omega_draws_ = np.zeros((self.n_iter, self.n))
        
        for t in range(1, self.n_iter):
            zv = np.squeeze(self.X @ self.beta_draws_[t-1,None].T)
            self.omega_draws_[t] = random_polyagamma(h=1, z=zv, size=self.n)
            
            inv_cov = inv(self.prior_cov)
            V_omega = inv(self.X.T @ np.diag(self.omega_draws_[t]) @ self.X + inv_cov)
            kappa = np.expand_dims(np.concatenate([np.repeat(-1/2, self.k_draws_[t-1]), np.repeat(1/2, self.n-self.k_draws_[t-1])]), axis=1)
            m_omega = np.squeeze(V_omega @ (self.X.T @ kappa + inv_cov @ np.expand_dims(self.prior_mean, 1)))
            self.beta_draws_[t] = multivariate_normal.rvs(mean=m_omega, cov=V_omega, size=1)
                                                                                                                                                                          
            y_pvec = 1/(1+np.exp(np.squeeze(-self.X @ self.beta_draws_[t,None].T)))
            if self.incl_last:
                num_elem = self.n
            else:
                num_elem = self.n-1
                
            k_lpvec = np.empty(num_elem)
                
            for k in range(self.n-1):
                k_lpvec[k] = np.prod(1-y_pvec[:(k+1)])*np.prod(y_pvec[(k+1):])*self.prior_kappa[k]

            if self.incl_last:            
                k_lpvec[self.n-1] = np.prod(1-y_pvec)*self.prior_kappa[self.n-1]
                
            k_pvec = k_lpvec/np.sum(k_lpvec)
            self.k_draws_[t] = np.random.choice(np.arange(1, num_elem+1), size=1, p=k_pvec)
            
        self.post_k = self.k_draws_[self.burn_in:]
        self.post_beta = self.beta_draws_[self.burn_in:, :]
            
    def transform(self, verbose=True):
        """
        Calculate the draws of kappa and beta outside of the burn-in period and calculate probabilities. 
        Optional displays a table with estimates of kappa and corresponding probabilities.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print out a table of posterior changepoint estimates. The default is True.

        Returns
        -------
        None.

        """
        check_is_fitted(self)
        #Here we create the values outside of the burn_in and calculate probabilities
        post_k_vals, post_k_counts = np.unique(self.post_k, return_counts=True)
        self.post_k_mode = post_k_vals[np.argmax(post_k_counts)]
        self.post_mode_prob = np.max(post_k_counts/(self.n_iter-self.burn_in))
        self.post_beta_mean = np.mean(self.post_beta, axis=0)
        
        if verbose:
            table = [post_k_vals, post_k_counts/(self.n_iter-self.burn_in)]
            print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        
    def plot_k(self):
        """
        Plots the posterior probability mass function of the changepoints.

        Returns
        -------
        None.

        """
        check_is_fitted(self)
        bins = np.arange(1, self.n+2)
        plt.hist(self.post_k, rwidth=0.9, density=True, align='left', bins=bins)
        plt.show()
        
    

