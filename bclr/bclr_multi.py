import numpy as np
from . import BayesCC

inv = np.linalg.inv
det = np.linalg.det

def bin_seg_bclr(bclr_obj, thr, start=0, 
                 min_seg=20, buff=0):
    """

    Parameters
    ----------
    bclr_obj : BayesCC object
        A fitted and transformed BayesCC object. 
    thr : float
        DValue between 0 and 1 to threshold posterior mode to stop binary segmentation.
    start : int, optional
        Prepended index to maintain record of location of changepoints. The default is 0.
    min_seg : int, optional
        Minimum changepoint segment size to consider. The default is 20.
    buff : int, optional
        Do not consider any changepoint within buff of endpoints of segment. The default is 0.

    Yields
    ------
    loc
        Estimated changepoint.
    prob
        Posterior mode probability. .
    loc_mean
        Posterior mean changepoint location.
    loc_std
        Posterior standard deviation of changepoint location.
    beta
        Coefficients of posterior mean of estimated betas for segment

    """
    
    incl_last = bclr_obj.incl_last
    mode = int(bclr_obj.post_k_mode)
    pm, psd = np.mean(bclr_obj.post_k), np.std(bclr_obj.post_k)
    yield (start + mode, bclr_obj.post_mode_prob, start+pm, psd, *bclr_obj.post_beta_mean)
    
    dat1 = bclr_obj.X[:mode, :]
    dat2 = bclr_obj.X[mode:, :]
    bclr1 = BayesCC(dat1, prior_mean=bclr_obj.prior_mean, 
                         prior_cov=bclr_obj.prior_cov, n_iter=bclr_obj.n_iter, 
                         scaled=False, incl_last=incl_last)
    
    bclr2 = BayesCC(dat2, prior_mean=bclr_obj.prior_mean, 
                         prior_cov=bclr_obj.prior_cov, n_iter=bclr_obj.n_iter, 
                         scaled=False, incl_last=incl_last)
    bclr1.fit()
    bclr2.fit()
    bclr1.transform(verbose=False)
    bclr2.transform(verbose=False)
    
    for i, bclrA in enumerate([bclr1, bclr2]):
        if incl_last:
            last = bclrA.n
        else:
            last = bclrA.n - 1
            
        if bclrA.n < min_seg:
            pass
        else:
            d = min(abs(bclrA.post_k_mode-1), 
                    abs(bclrA.post_k_mode-last))
            if d <= buff:
                pass
            elif bclrA.post_mode_prob >= thr:
                yield from bin_seg_bclr(bclrA, thr, start + mode*i, 
                                        min_seg, buff)
            else:
                pass


class MultiBayesCC(BayesCC):
    """
    This class implements binary segmentation for the Bayesian Changepoint via Logistic Regression (bclr) method 
    described in Thomas, Jauch, and Matteson (2024).
    """
    def __init__(self, X, prior_mean, prior_cov, n_iter, min_seg=20,
                 buff=0, scaled=True, burn_in=None, incl_last=True):
        super().__init__(X, prior_mean, prior_cov, n_iter,
                         scaled=scaled, burn_in=burn_in, incl_last=incl_last)
        self.min_seg = min_seg
        self.buff=buff
        
    def fit(self, thr):
        super().fit()
        super().transform(verbose=False)
        self.cps_ = [cp for cp in bin_seg_bclr(self, thr, start=0, min_seg=self.min_seg, buff=self.min_seg)]
        
    