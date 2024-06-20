import numpy as np
from .bclr_one import BayesCC
from .bclr_helper import uni_binom
import ruptures as rpt

inv = np.linalg.inv
det = np.linalg.det

class MultiBayesCC(rpt.Dynp):
    """
    Here we utilize an off-the-shelf changepoint method and post-process using bclr for better understanding of changepoint
    uncertainty and the best representation of said changepoint. This uses the dynamic programming implementation in the ruptures package.

    """
    def __init__(self, X, K, prior_cov, n_iter, lam=1,
                 model='rbf', min_size = 5, jump=1):
        super().__init__(model=model, min_size=min_size, jump=1)
        super().fit(X)
        self.bkps = super().predict(n_bkps=K)
        self.bkps.insert(0, 0)
        self.prior_cov = prior_cov
        self.n_iter = n_iter
        self.lam = lam
        self.n, self.p = X.shape
        self.X = X
        self.K = K
        
    def fit(self):
        prior_mean = np.repeat(0, self.p)
        self.cps = np.empty(self.K)
        self.prior_kappas = [uni_binom(n=self.bkps[i+2]-self.bkps[i]-1, 
                                       p=(self.bkps[i+1]-self.bkps[i])/(self.bkps[i+2]-self.bkps[i]), 
                                       lam=self.lam) for i in range(self.K)]
        self.bccs_ = [BayesCC(self.X[self.bkps[i]:self.bkps[i+2], :], prior_mean, 
                             self.prior_cov, self.n_iter, prior_kappa=self.prior_kappas[i])
                     for i in range(self.K)]
        for i in range(self.K):
            self.bccs_[i].fit()
    
    def transform(self):
        for i in range(self.K):
            #verify that this is how its supposed to work...
            self.bccs_[i].post_k = self.bccs_[i].post_k+self.bkps[i]
            self.bccs_[i].transform(verbose=False)

        