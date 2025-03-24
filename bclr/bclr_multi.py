import numpy as np
import pandas as pd
from .bclr_one import BayesCC
from .bclr_helper import uni_binom, prob_mode, SegmentationWarning, _proc_cov, _fit_Class
from joblib import Parallel, delayed
import warnings

inv = np.linalg.inv
det = np.linalg.det

class MultiBayesCC:
    """
    Here we utilize an off-the-shelf changepoint method and post-process using bclr for better understanding of changepoint
    uncertainty and the best representation of said changepoint. This uses the dynamic programming implementation in the ruptures package.

    """
    def __init__(self, X, cps, prior_cov, n_iter=1000, lam=0,
                 min_size=10, rng = None, warnings=True):
        """
        
        Parameters
        ----------
        X : array-like of shape n x d
            Array of n d-dimensional vectors to assess changepoints.
        cps : int or list
            Number of changepoints to seed or list of initial changepoints. 
            If list of changepoints given, note that it should be in terms of the indices
            series {1, 2, ..., len(X)}.
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
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        
        if not isinstance(self.rng, np.random._generator.Generator):
            raise TypeError("rng should be of type np.random._generator.Generator")
        
        tcps  = type(cps)
        if not (tcps is int or tcps is list):
            raise TypeError("cps should be an integer or a list of changepoint locations")
        
        if tcps is int:
            self.bkps = list(np.linspace(0, len(X), cps+2, dtype=np.int64))
        else:
            self.bkps = [0] + cps + [len(X)]
        
        self.K = len(self.bkps)-2
        self.n, self.p = X.shape
        if np.floor(self.n/(2*self.K + 2)) <= min_size:
            raise ValueError("min_size is too large for any changepoints to be estimated")
            
        if n_iter <= 0:
            raise ValueError("Number of MC iterations should be positive")
        self.prior_cov = _proc_cov(prior_cov, self.p)
        self.n_iter = n_iter
        
        if lam < 0 or lam > 1:
            raise ValueError("Lambda must be nonnegative and no greater than 1")
        else:
            self.lam = lam
        
        self.X = X
        self.transformed = False
        
        if min_size <= 1:
            raise ValueError("min_size should be greater than 1")

        self.min_size = int(min_size)
        self.warnings = warnings
        
    def fit(self, n_jobs=None):
        """
        Fit MultiBayesCC class, meaning implement the Gibbs sampler on each consecutive segment, 
        according to the multiple changepoint formulation discussed for drawing posterior changepoints 
        and coefficients in Thomas, Jauch, and Matteson (2025).

        Parameters
        ----------
        n_jobs : int or None, optional, default is ``None``
                The number of jobs to use for the computation. ``None`` means 1 unless
                in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
                processors.

        Returns
        -------
        None.

        """
        prior_mean = np.repeat(0, self.p)
        self.transformed = False
        self.prior_kappas = [uni_binom(n=self.bkps[i+2]-self.bkps[i]-1,
                                       p=(self.bkps[i+1]-self.bkps[i])/(self.bkps[i+2]-self.bkps[i]),
                                       lam=self.lam) for i in range(self.K)]
        
        new_gens = self.rng.spawn(self.K)
        self.bccs_ = Parallel(n_jobs = n_jobs)(delayed(_fit_Class)(BayesCC, new_gens[i], self.X[self.bkps[i]:self.bkps[i+2], :], 
                                                                   prior_mean, self.prior_cov, self.n_iter, 
                                                                   self.prior_kappas[i]) for i in range(self.K))
    
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
    
    def proc(self, n_jobs=None):
        """
        Similar to fit_transform, but resets transformed attribute.

        Parameters
        ----------
        n_jobs : int or None, optional, default is ``None``
            The number of jobs to use for the computation. ``None`` means 1 unless
            in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.

        Returns
        -------
        None.

        """
        self.fit(n_jobs=n_jobs)
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
                mode_val = prob_mode(bc.post_k[np.logical_and(bc.post_k >= prev-offset + self.min_size,
                                                              bc.post_k <= self.bkps[i+2]-self.min_size)])
                bc_info.append((mode_val+offset, bc.post_mode_prob, bc.norm_entr))
            else:
                bc_info.append((bc.post_k_mode+offset, bc.post_mode_prob, bc.norm_entr))
        
        df = pd.DataFrame(bc_info, columns = ['Location', 'Posterior Probability', 'Normalized Entropy'])
        df_red = df[df['Location'].notnull()]
        if thr is None:
            return df_red
        else:
            if thr < 0 or thr > 1:
                raise ValueError("Threshold thr should be between 0 and 1 (inclusive)")
            else:
                return df_red[df_red['Normalized Entropy'] < thr]
    
    def fit_predict(self, iter_sch=None, thr_sch=None, n_jobs=None, offset=0):
        """
        Predict changepoints after two successive warm-up periods of increasing "complexity".

        Parameters
        ----------
        iter_sch : list of increasing positive int, optional
            List of increasing number of iterations to run Gibbs sampler in first two warm-up periods. 
            The default is None, which becomes [100, 250].
        thr_sch : List of decreasing float between 0 and 1, optional
            List of decreasing entropy thresholds. The default is None, which becomes [0.75, 0.5].
        n_jobs : int or None, optional, default is ``None``
            The number of jobs to use for the computation. ``None`` means 1 unless
            in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.
        offset : float, optional
            For use in time series with indexing representing time: e.g. years, etc. Adds this
            value to the estimated changepoints. The default is 0.

        Returns
        -------
        pd.DataFrame
            Estimated changepoints, posterior probability and normalized entropy.

        """
        if iter_sch is None:
            iter_sch = [100, 250]
            
        if thr_sch is None:
            thr_sch = [0.75, 0.5]
        
        if len(iter_sch) != 2 or len(thr_sch) != 2:
            raise ValueError("Please ensure iter_sch and thr_sch are both of length 2")
        
        self.warm_up(n_iter_w=iter_sch[0], thr=thr_sch[0], n_jobs=n_jobs)
        self.warm_up(n_iter_w=iter_sch[1], thr=thr_sch[1], n_jobs=n_jobs)
        self.fit(n_jobs=n_jobs)
        self.transform()
        return self.cps_df(offset=offset)
    
    def fit_transform(self, n_jobs=None):
        """
        Fits, then transforms.
        
        Parameters
        ----------
        n_jobs : int or None, optional, default is ``None``
            The number of jobs to use for the computation. ``None`` means 1 unless
            in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.

        Returns
        -------
        None.

        """
        self.fit(n_jobs=n_jobs)
        self.transform()
    
    def warm_up(self, n_iter_w=100, thr=None, n_jobs=None):
        """
        Runs the chain with various initializations according to Section 6.1 of Thomas, Jauch, and Matteson (2025).
        Argument thr can be useful for enhanced estimation ability by removing spurious changepoints. Resets ``bkps`` and
        may reduce number of changepoints ``J``. 

        Parameters
        ----------
        n_iter_w : int, optional
            Number of iterations to run each "warm-up" chain. The default is 100 and the minimum value is 50.
        thr : float, optional
            Remove changepoints with normalized entropy greater than threshold.
            Between 0 and 1. The default is None.
        n_jobs : int or None, optional, default is ``None``
            The number of jobs to use for the computation. ``None`` means 1 unless
            in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
            processors.

        Returns
        -------
        None.

        """
        n_iter_w = int(max(50, n_iter_w))
        self.n_iter_init = self.n_iter
        self.n_iter = n_iter_w
            
        self.proc(n_jobs=n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfM = self.cps_df(thr)['Location']
        dfM_nan = dfM[np.logical_not(np.isnan(dfM))]
        self.bkps = [0] + list(dfM_nan.astype(np.int32)) + [self.n]
        if len(dfM_nan) < self.K:
            if self.warnings:
                warnings.warn_explicit(message="""Number of changepoints reduced due to nan values owing to 
                                       min_size constraints specified in MultiBayesCC... \n""",
                          category=SegmentationWarning, filename="bclr_multi.py", lineno=246)
            self.K = len(dfM_nan)
        
        self.n_iter = self.n_iter_init

