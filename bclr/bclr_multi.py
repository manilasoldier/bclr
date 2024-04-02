import numpy as np
from sklearn.preprocessing import StandardScaler
import bclr

inv = np.linalg.inv
det = np.linalg.det

def bin_seg_bclr(bclr_obj, thr, start=0, 
                 min_seg=20, buff=0):
    
    incl_last = bclr_obj.incl_last
    mode = int(bclr_obj.post_k_mode)
    pm, psd = np.mean(bclr_obj.post_k), np.std(bclr_obj.post_k)
    yield (start + mode, bclr_obj.post_mode_prob, start+pm, psd, *bclr_obj.post_beta_mean)
    
    dat1 = bclr_obj.X[:mode, :]
    dat2 = bclr_obj.X[mode:, :]
    bclr1 = bclr.BayesCC(dat1, prior_mean=bclr_obj.prior_mean, 
                         prior_cov=bclr_obj.prior_cov, n_iter=bclr_obj.n_iter, 
                         scaled=False, incl_last=incl_last)
    
    bclr2 = bclr.BayesCC(dat2, prior_mean=bclr_obj.prior_mean, 
                         prior_cov=bclr_obj.prior_cov, n_iter=bclr_obj.n_iter, 
                         scaled=False, incl_last=incl_last)
    bclr1.fit()
    bclr2.fit()
    bclr1.transform(verbose=False)
    bclr2.transform(verbose=False)
    
    if incl_last:
        last = bclr_obj.n
    else:
        last = bclr_obj - 1
    
    for i, bclrA in enumerate([bclr1, bclr2]):
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


class MultiBayesCC:
    """
    Need to work on integrating the bclr class with this new class which fits 
    a multiple changepoint model. 
    
    Need to also figure out the best way to incorporate prior_kappa here...
    """
    def __init__(self, X, prior_mean, prior_cov, n_iter, 
                 min_size = 0, scaled=True, burn_in=None):
        
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
            
        self.n_iter = n_iter
        self.burn_in = int(burn_in)
        self.min_size = min_size