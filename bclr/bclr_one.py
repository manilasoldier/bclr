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
    def __init__(self, X, prior_mean, prior_cov, n_iter, 
                 prior_kappa=None, scaled=False, burn_in=None):
        """

        Parameters
        ----------
        X : array-like
            n x d array of n d-dimensional vectors to assess changepoints.
        prior_mean : ndarray of length d
            Array containing prior mean of Normal distribution.
        prior_cov : ndarray of shape (d,d)
            Symmetric positive (semi)definite covariance matrix. 
        prior_kappa : ndarray of length n-1 nonnegative values.
            Prior distribution for the changepoint kappa. 
        n_iter : int
            Number of iterations to run Gibbs sampler. 
        scaled : bool, optional
            If False, each column of data will be mean centered and 
            standardized to have variance 1. The default is False.
        burn_in : int, optional
            Number of initial iterations of the Gibbs sampler to discard. 
            The default is None.

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
            self.std_sc = StandardScaler()
            self.std_sc.fit(X)
            self.X = self.std_sc.transform(X)
        else: 
            self.X = X
        self.scaled = scaled
            
        if burn_in == None:
            burn_in = n_iter/2
            
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        if prior_kappa is None:
            self.prior_kappa = np.repeat(1, self.n-1)
        else:
            if np.any(prior_kappa < 0):
                raise ValueError("Must have nonnegative weights for prior.")
            elif len(prior_kappa) != (self.n-1):
                raise IndexError("Length of prior distribution must be as long as number of potential changepoints")
            self.prior_kappa = prior_kappa
            
        self.n_iter = n_iter
        self.burn_in = int(burn_in)
    
    def fit(self, init_k = None, init_beta = None, small_probs = True, tol = 1e-4, c = 1e-2):
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
                                                                                                                                                                          
            k_lpvec = np.zeros(self.n-1)
            logk_lpvec = np.zeros(self.n-1)
            prior_kappa_pos = np.where(self.prior_kappa > tol)[0]
            
            if small_probs:
                logk_lpvec[prior_kappa_pos[0]] = np.log(c)
                for k in range(1,len(prior_kappa_pos)):
                    k1 = prior_kappa_pos[k]
                    k0 = prior_kappa_pos[k-1]
                    logk_lpvec[k1] = logk_lpvec[k0] - self.X[k1,:] @ self.beta_draws_[t,None].T + np.log(self.prior_kappa[k1]) - np.log(self.prior_kappa[k0])
                
                v = np.max(logk_lpvec)
                # to avoid overflow, we truncate log posterior kappa values to 100
                if v > 100:
                    logk_lpvec = logk_lpvec - v + 100
                    
                k_lpvec = np.exp(logk_lpvec)
                k_pvec = k_lpvec/np.sum(k_lpvec)
                
            else:
                y_pvec = 1/(1+np.exp(np.squeeze(-self.X @ self.beta_draws_[t,None].T)))
                for k in range(self.n-1):
                    k_lpvec[k] = np.prod(1-y_pvec[:(k+1)])*np.prod(y_pvec[(k+1):])*self.prior_kappa[k]
                k_pvec = k_lpvec/np.sum(k_lpvec)
            
            self.k_draws_[t] = np.random.choice(np.arange(1, self.n), size=1, p=k_pvec)
            
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
        beta_original: bool, optional
            Whether or not to produces beta terms on the original scale. The default is False.
            
        Returns
        -------
        None.

        """
        check_is_fitted(self)
        #Here we create the values outside of the burn_in and calculate probabilities
        post_k_vals, post_k_counts = np.unique(self.post_k, return_counts=True)
        
        arr0 = post_k_counts/(self.n_iter-self.burn_in)
        arr = arr0[arr0 > 0]
        if np.array_equal(arr, np.array([1.])):
            self.norm_entr = 0
        else:
            self.norm_entr = np.sum(-arr * np.log(arr))/np.log(self.n-1)
            
        self.post_k_mode = post_k_vals[np.argmax(post_k_counts)]
        self.post_mode_prob = np.max(arr)
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
        bins = np.arange(1, self.n+1)
        plt.hist(self.post_k, rwidth=0.9, density=True, align='left', bins=bins)
        plt.show()
        
    

