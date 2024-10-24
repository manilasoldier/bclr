import numpy as np
import pandas as pd
from .bclr_one import BayesCC
from .bclr_helper import uni_binom, prob_mode

inv = np.linalg.inv
det = np.linalg.det

class MultiBayesCC:
    """
    Here we utilize an off-the-shelf changepoint method and post-process using bclr for better understanding of changepoint
    uncertainty and the best representation of said changepoint. This uses the dynamic programming implementation in the ruptures package.

    """
    def __init__(self, X, cps, prior_cov, n_iter, lam=0, min_size=10):
        
        tcps  = type(cps)
        assert tcps == int or tcps == list
        
        if tcps == int:
            self.bkps = list(np.linspace(0, len(X), cps+2, dtype=np.int64))
        else:
            self.bkps = [0] + cps + [len(X)]               
        
        self.K = len(self.bkps)-2
        self.prior_cov = prior_cov
        self.n_iter = n_iter
        self.lam = lam
        self.n, self.p = X.shape
        self.X = X
        self.transformed = False
        self.min_size = min_size
        
    def fit(self):
        prior_mean = np.repeat(0, self.p)
        self.transformed = False
        self.prior_kappas = [uni_binom(n=self.bkps[i+2]-self.bkps[i]-1, 
                                       p=(self.bkps[i+1]-self.bkps[i])/(self.bkps[i+2]-self.bkps[i]), 
                                       lam=self.lam) for i in range(self.K)]
        self.bccs_ = [BayesCC(self.X[self.bkps[i]:self.bkps[i+2], :], prior_mean, 
                             self.prior_cov, self.n_iter, prior_kappa=self.prior_kappas[i])
                     for i in range(self.K)]
        for i in range(self.K):
            self.bccs_[i].fit()
    
    def transform(self):
        if self.transformed:
            pass
        else:
            for i in range(self.K):
                #verify that this is how its supposed to work...
                self.bccs_[i].post_k = self.bccs_[i].post_k+self.bkps[i]
                self.bccs_[i].transform(verbose=False)
        
        self.transformed = True

    def cps_df(self, offset=0):
        ### Add in functionality to restrict kappa_j - kappa_{j-1} > Delta with self.min_size
        bc_info = []
        for i, bc in enumerate(self.bccs_):
            if i >= 1:
                mode_val = prob_mode(bc.post_k[np.logical_and(bc.post_k > bc_info[i-1][0]-offset + self.min_size, bc.post_k < self.bkps[i+2]-self.min_size)])
                bc_info.append((mode_val+offset, bc.post_mode_prob, bc.norm_entr))
            else:
                bc_info.append((bc.post_k_mode+offset, bc.post_mode_prob, bc.norm_entr))
        
        df = pd.DataFrame(bc_info, columns = ['Location', 'Posterior Probability', 'Normalized Entropy'])
        return df
    
    def warm_up(self, n_iter_w=10, random_init=False, reps=1):
        self.n_iter_init = self.n_iter
        self.n_iter = n_iter_w
        best_n_entr = self.K
        if random_init:
            for i in range(reps):
                cps = list(np.sort(np.random.choice(np.arange(1, self.n), size=self.K, replace=False)).astype(np.int32))
                self.bkps = [0] + cps + [self.n]
                self.fit()
                self.transform()
                self.transformed = False
                try:
                    cps_df = self.cps_df()
                    entr_now = cps_df['Normalized Entropy'].sum()
                    cps_vals = [0] + list(cps_df['Location'].astype(np.int32)) + [self.n]
                    if entr_now < best_n_entr and np.min(np.diff(cps_vals)) > self.min_size:
                        best_brk = cps_vals
                        best_n_entr = entr_now
                except ValueError:
                    pass
                
            self.bkps = best_brk
            
        else:
            self.fit()
            self.transform()
            self.transformed = False
            self.bkps = [0] + list(self.cps_df()['Location'].astype(np.int32)) + [self.n]
        
        self.n_iter = self.n_iter_init
        
        