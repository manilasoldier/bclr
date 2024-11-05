import numpy as np
import pandas as pd
from .bclr_one import BayesCC
from .bclr_helper import uni_binom, prob_mode, SegmentationWarning, sample_sep
import warnings

inv = np.linalg.inv
det = np.linalg.det

class MultiBayesCC:
    """
    Here we utilize an off-the-shelf changepoint method and post-process using bclr for better understanding of changepoint
    uncertainty and the best representation of said changepoint. This uses the dynamic programming implementation in the ruptures package.

    """
    def __init__(self, X, cps, prior_cov, n_iter, lam=0, min_size=2):
        """

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        cps : TYPE
            DESCRIPTION.
        prior_cov : TYPE
            DESCRIPTION.
        n_iter : TYPE
            DESCRIPTION.
        lam : TYPE, optional
            DESCRIPTION. The default is 0.
        min_size : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        """
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
        """

        Returns
        -------
        None.

        """
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
        """

        Returns
        -------
        None.

        """
        if self.transformed:
            pass
        else:
            for i in range(self.K):
                #verify that this is how its supposed to work...
                self.bccs_[i].post_k = self.bccs_[i].post_k+self.bkps[i]
                self.bccs_[i].transform(verbose=False)
        
        self.transformed = True
    
    def proc(self):
        self.fit()
        self.transform()
        self.transformed = False

    def cps_df(self, offset=0, thr=None):
        """
        Returns estimated changepoints (posterior modes), along with posterior 'probability', and normalized entropy. Values coded nan (removed due to  min_size constraints are not returned).

        Parameters
        ----------
        offset : TYPE, optional
            DESCRIPTION. The default is 0.
        thr : TYPE. optional
            DESCRIPTION. The default is None.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        ### Add in functionality to restrict kappa_j - kappa_{j-1} > Delta with self.min_size
        bc_info = []
        prev = 0
        for i, bc in enumerate(self.bccs_):
            if i >= 1:
                pot = bc_info[i-1][0]
                if np.isnan(pot):
                    pass
                else:
                    prev = pot
                mode_val = prob_mode(bc.post_k[np.logical_and(bc.post_k > prev-offset + self.min_size, bc.post_k < self.bkps[i+2]-self.min_size)])
                bc_info.append((mode_val+offset, bc.post_mode_prob, bc.norm_entr))
            else:
                bc_info.append((bc.post_k_mode+offset, bc.post_mode_prob, bc.norm_entr))
        
        df = pd.DataFrame(bc_info, columns = ['Location', 'Posterior Probability', 'Normalized Entropy'])
        df_red = df[df['Location'].notnull()]
        if thr is None:
            return df_red
        else:
            return df_red[df_red['Normalized Entropy'] < thr]
    
    def warm_up(self, n_iter_w=10, reps=1, max_iter=100):
        """
        EDIT THIS 11/5/24
        
        Parameters
        ----------
        n_iter_w : TYPE, optional
            DESCRIPTION. The default is 10.
        reps : TYPE, optional
            DESCRIPTION. The default is 1.
        max_iter : TYPE, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        None.

        """
        self.n_iter_init = self.n_iter
        self.n_iter = n_iter_w
        best_n_entr = self.K
        iter_tr = 0 
        i = 0
        while i < reps and iter_tr < max_iter:
            iter_tr += 1
            cps = list(sample_sep(self.n, self.K, delta=self.min_size).astype(np.int32))
            self.bkps = [0] + cps
            self.proc()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cps_df = self.cps_df()
            entr_now = cps_df['Normalized Entropy'].sum()
            cps_vals = [0] + list(cps_df['Location'].astype(np.int32)) + [self.n]
            i += 1
            if entr_now < best_n_entr:
                best_brk = cps_vals
                best_n_entr = entr_now
        
        self.bkps = best_brk
        if len(best_brk)-2 < self.K:
            warnings.warn_explicit(message="""Number of changepoints reduced due to nan values owing to min_size constraints specified in MultiBayesCC... \n""",
                   category=SegmentationWarning, filename="bclr_multi.py", lineno=173)
            
        self.K = len(best_brk)-2
        self.n_iter = self.n_iter_init
            # try:
            #     self.bkps = best_brk
            # except UnboundLocalError:
            #     warnings.warn_explicit(message="""Desired segmentation not found in desired number of reps and max_iter. Argument min_size may be too large. Setting random_init=False and re-running... \n""",
            #                   category=SegmentationWarning, filename="bclr_multi.py", lineno=172)
            #     self.proc()
            #     self.bkps = list(np.linspace(0, self.n, self.K+2, dtype=np.int64))
            
        # else:
        #     self.proc()
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")
        #         dfM = self.cps_df()['Location']
        #     dfM_nan = dfM[np.logical_not(np.isnan(dfM))]
        #     self.bkps = [0] + list(dfM_nan.astype(np.int32)) + [self.n]
        #     if len(dfM_nan) < self.K:
        #         warnings.warn_explicit(message="""Appropriate number of changepoints could not be found based on min_size constraints. Returning breakpoints found... \n""",
        #                       category=SegmentationWarning, filename="bclr_multi.py", lineno=188)
        #         self.K = len(dfM_nan)
        
        
        