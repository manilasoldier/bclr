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
    def __init__(self, X, cps, prior_cov, n_iter=1000, lam=0, min_size=10, rng = None, warnings=True):
        """

        Parameters
        ----------
        X : array-like of shape n x d
            Array of n d-dimensional vectors to assess changepoints.
        cps : int or list
            Number of changepoints to seed or list of initial changepoints. 
            If list of changepoints given, note that it should be in terms of the indices
            series {, 2, ..., len(X)}.
        prior_cov : ndarray of shape (d,d)
            Symmetric positive (semi)definite covariance matrix, 
            for each segment of series.
        n_iter : int
            Number of iterations to run Gibbs sampler for each segment.
        lam : float
            Interpolation parameter for "uni-binomial" prior. Between 0 and 1.
        min_size : int, optional
            Minimum distance between changepoints. The default is 10.
        rng : np.random._generator.Generator, optional
            Random number generator to ensure reproducibility. The default is None.
        warnings : bool, optional
            If False, suppress warnings. 

        Returns
        -------
        None.

        """
        if rng == None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        
        assert type(self.rng) == np.random._generator.Generator
        
        tcps  = type(cps)
        assert tcps == int or tcps == list
        
        if tcps == int:
            self.bkps = list(np.linspace(0, len(X), cps+2, dtype=np.int64))
        else:
            self.bkps = [0] + cps + [len(X)]              
        
        self.K = len(self.bkps)-2
        self.n, self.p = X.shape
        if np.floor(self.n/(2*self.K + 2)) <= min_size:
            raise ValueError("min_size is too large for any changepoints to be estimated")
            
        self.prior_cov = prior_cov
        self.n_iter = n_iter
        self.lam = lam
        self.X = X
        self.transformed = False
        self.min_size = min_size
        self.warnings = warnings
        
    def fit(self):
        """
        Fit MultiBayesCC class, meaning implement the Gibbs sampler on each consecutive segment, 
        according to the multiple changepoint formulation discussed for drawing posterior changepoints 
        and coefficients in Thomas, Jauch, and Matteson (2025).
    
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
            self.bccs_[i].fit(rng = self.rng)
    
    def transform(self):
        """
        Calculate posterior distributions and summaries for each segment defined by the 
        endpoints of ``bkps`` defined above. 

        Returns
        -------
        None.

        """
        if self.transformed:
            pass
        else:
            for i in range(self.K):
                self.bccs_[i].post_k = self.bccs_[i].post_k+self.bkps[i]
                self.bccs_[i].transform(verbose=False)
        
        self.transformed = True
    
    def proc(self):
        self.fit()
        self.transform()
        self.transformed = False

    def cps_df(self, thr=None, offset=0):
        """
        Returns estimated changepoints (posterior modes), along with posterior 'probability', and normalized entropy. 
        Values coded nan (removed due to  ``min_size`` constraints are not returned). 

        Parameters
        ----------
        offset : float, optional
            For use in time series with indexing representing time: e.g. years, etc. Adds this
            value to the estimated changepoints. The default is 0.
        thr : float, optional
            Remove changepoints from DataFrame with normalized entropy greater than threshold.
            Between 0 and 1. The default is None.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with the above described information. 

        """
        bc_info = []
        prev = 0
        for i, bc in enumerate(self.bccs_):
            if i >= 1:
                pot = bc_info[i-1][0]
                if np.isnan(pot):
                    pass
                else:
                    prev = pot
                mode_val = prob_mode(bc.post_k[np.logical_and(bc.post_k >= prev-offset + self.min_size, bc.post_k <= self.bkps[i+2]-self.min_size)])
                bc_info.append((mode_val+offset, bc.post_mode_prob, bc.norm_entr))
            else:
                bc_info.append((bc.post_k_mode+offset, bc.post_mode_prob, bc.norm_entr))
        
        df = pd.DataFrame(bc_info, columns = ['Location', 'Posterior Probability', 'Normalized Entropy'])
        df_red = df[df['Location'].notnull()]
        if thr is None:
            return df_red
        else:
            return df_red[df_red['Normalized Entropy'] < thr]
    
    def fit_predict(self, iter_sch = [100, 250], thr_sch = [0.75, 0.5], offset=0):
        """
        Predict changepoints after two successive warm-up periods of increasing "complexity".

        Parameters
        ----------
        iter_sch : list of increasing positive int, optional
            List of increasing number of iterations to run Gibbs sampler in first two warm-up periods. 
            The default is [100, 250].
        thr_sch : List of decreasing float between 0 and 1, optional
            List of decreasing entropy thresholds. The default is [0.75, 0.5].

        Returns
        -------
        pd.DataFrame
            Estimated changepoints, posterior probability and normalized entropy.

        """
        self.warm_up(n_iter_w=iter_sch[0], thr=thr_sch[0])
        self.warm_up(n_iter_w=iter_sch[1], thr=thr_sch[1])
        self.fit()
        self.transform()
        return self.cps_df(offset=offset)
    
    def fit_transform(self):
        """
        Fits, then transforms.

        Returns
        -------
        None.

        """
        self.fit()
        self.transform()
    
    def warm_up(self, n_iter_w=100, random_init=False, thr=None, reps=10):
        """
        Runs the chain with various initializations according to Section 6.1 of Thomas, Jauch, and Matteson (2025).
        Argument thr can be useful for enhanced estimation ability by removing spurious changepoints. Resets ``bkps`` and
        may reduce number of changepoints ``K``. 

        Parameters
        ----------
        n_iter_w : int, optional
            Number of iterations to run each "warm-up" chain. The default is 100.
        random_init : bool, optional
            Whether or not to randomly choose ``bkps`` according to 
            random sampling respecting ``min_size`` constraints. The default is False.
        thr : float, optional
            Remove changepoints with normalized entropy greater than threshold.
            Between 0 and 1. The default is None.
        reps : int, optional
            Number of random initializations to try. Ignored if ``random_init``=False. The default is 10.

        Returns
        -------
        None.

        """
        self.n_iter_init = self.n_iter
        self.n_iter = n_iter_w
        if random_init:
            best_n_entr = self.K
            i = 0
            while i < reps:
                cps = list(sample_sep(self.n, self.K, delta=self.min_size).astype(np.int32))
                self.bkps = [0] + cps
                self.proc()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cps_df = self.cps_df(thr)
                entr_now = cps_df['Normalized Entropy'].sum()
                cps_vals = [0] + list(cps_df['Location'].astype(np.int32)) + [self.n]
                i += 1
                if entr_now < best_n_entr:
                    best_brk = cps_vals
                    best_n_entr = entr_now
            
            self.bkps = best_brk
            if len(best_brk)-2 < self.K:
                if self.warnings:
                    warnings.warn_explicit(message="""Number of changepoints reduced due to nan values owing to min_size constraints specified in MultiBayesCC... \n""",
                        category=SegmentationWarning, filename="bclr_multi.py", lineno=178)
                self.K = len(best_brk)-2
            
        else:
            self.proc()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dfM = self.cps_df(thr)['Location']
            dfM_nan = dfM[np.logical_not(np.isnan(dfM))]
            self.bkps = [0] + list(dfM_nan.astype(np.int32)) + [self.n]
            if len(dfM_nan) < self.K:
                if self.warnings:
                    warnings.warn_explicit(message="""Number of changepoints reduced due to nan values owing to min_size constraints specified in MultiBayesCC... \n""",
                              category=SegmentationWarning, filename="bclr_multi.py", lineno=192)
                self.K = len(dfM_nan)
        
        self.n_iter = self.n_iter_init
        
        
        