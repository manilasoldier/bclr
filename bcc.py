import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tabulate import tabulate
from polyagamma import random_polyagamma
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

inv = np.linalg.inv
det = np.linalg.det

def polyagamma_int(X, omega, prior_cov):
    """
    Calculate integral of pi(kappa | x) directly using formula seen in the Appendix of Thomas, Jauch, and Matteson (2023).

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    omega : TYPE
        DESCRIPTION.
    prior_cov : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    omega_diag = np.diag(omega)
    n, d = X.shape
    V_omega = inv(X.T @ omega_diag @ X + inv(prior_cov))
    dVw = det(V_omega)**(1/2)
    
    #create the kappa/omega matrix
    nzs = np.full((n,n), -1)
    zkappa = (np.tril(nzs)+1/2)/np.sqrt(omega)
    
    X_omega = np.sqrt(omega_diag) @ X
    som = np.sum((1/8)*omega**(-1))
    
    zk = np.diagonal(-(zkappa @ inv(np.identity(n) + X_omega @ prior_cov @ X_omega.T) @ zkappa.T))
    return dVw*np.exp(zk+som)

def BayesCC_kappa(X, n_iter, prior_cov, print_res=True, n_jobs=None):
    """
    Parameters
    ----------
    X : array-like
        Ought to be an nxd array consisting of n d-dimensional observations
    n_iter : int
        Number of monte carlo iterations for Monte Carlo integration
    prior_cov : array-like
        Symmetric positive (semi)definite covariance matrix
    print_res: bool
        If True, prints indices and probabilities above 1e-3
        

    Returns
    -------
    ret : dict
        Dictionary consisting of raw MC integrands and as well as 
        normalized estimated probabilities in a Pandas DataFrame
        
    """
    n,_ = X.shape
    omegas = random_polyagamma(h=1, z=0, size=(n_iter,n))
    kappa_terms = np.zeros((n_iter, n))
    
    kappa_terms = Parallel(n_jobs=n_jobs)(delayed(lambda i: 
        polyagamma_int(X, omegas[i], prior_cov=prior_cov))(om) for om in range(n_iter))  

    #for om in range(n_iter):
    #    kappa_terms[om,:] = polyagamma_int(X, omegas[om], prior_cov=prior_cov)
    
    mk = np.mean(np.array(kappa_terms), axis=0)
    mk_norm = mk/np.sum(mk)
    ret = {'raw': mk, 
           'probs': pd.DataFrame({'Probability': mk_norm}, index=np.arange(1, n+1))}
    
    if print_res:
        for i,j in enumerate(np.round(mk_norm, 4)):
            if j > 1e-3:
                print(i+1, j)
    return ret
    
class BayesCC:
    """
    This class implements the Logistic Regression described in Thomas, Jauch, and Matteson (2023) for Bayesian changepoint analysis.
    """
    def __init__(self, X, prior_mean, prior_cov, n_iter, scaled=True, burn_in=None):
        """

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        prior_mean : TYPE
            DESCRIPTION.
        prior_cov : TYPE
            DESCRIPTION.
        n_iter : TYPE
            DESCRIPTION.
        scaled : TYPE, optional
            DESCRIPTION. The default is True.
        burn_in : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if len(X.shape) > 2: 
            raise ValueError("Array must be 2-dimensional")
    
        if not scaled:
            self.X = StandardScaler().fit_transform(X)
        else: 
            self.X = X
            
        if burn_in == None:
            burn_in = n_iter/2
            
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.n_iter = n_iter
        self.n, self.p = X.shape
        self.burn_in = int(burn_in)
    
    def fit(self, init_k = None, init_beta = None):
        
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
            k_lpvec = np.empty(self.n)
            
            for k in range(self.n-1):
                k_lpvec[k] = np.prod(1-y_pvec[:(k+1)])*np.prod(y_pvec[(k+1):])

            k_lpvec[self.n-1] = np.prod(1-y_pvec)
                
            k_pvec = k_lpvec/np.sum(k_lpvec)
            self.k_draws_[t] = np.random.choice(np.arange(1, self.n+1), size=1, p=k_pvec)
            
        self.post_k = self.k_draws_[self.burn_in:]
        self.post_beta = self.beta_draws_[self.burn_in:, :]
            
    def transform(self, verbose=True):
        """
        Calculate the draws of kappa and beta outside of the burn-in period and calculate probabilities. 
        Optional displays a table with estimates of kappa and corresponding probabilities.

        Parameters
        ----------
        verbose : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        #Here we create the values outside of the burn_in and calculate probabilities
        post_k_vals, post_k_counts = np.unique(self.post_k, return_counts=True)
        self.post_k_mode = post_k_vals[np.argmax(post_k_counts)]
        self.post_beta_mean = np.mean(self.post_beta, axis=0)
        
        if verbose:
            table = [post_k_vals, post_k_counts/(self.n_iter-self.burn_in)]
            print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        
    def plot_k(self):
        """

        Returns
        -------
        None.

        """
        check_is_fitted(self)
        bins = np.arange(1, self.n+2)
        plt.hist(self.post_k, rwidth=0.9, density=True, align='left', bins=bins)
        plt.show()
    
    def plot_post_mean(self):
        pass
        
    def trace_plot(self):
        pass
        
    

