import numpy as np
from scipy.stats import binom
import warnings

class SegmentationWarning(Warning):
    pass

def _proc_cov(prior_cov, p):
    """
    Process covariance matrix to ensure it is valid and convert it to np.ndarray.

    Parameters
    ----------
    prior_cov : ndarray of shape (d,d)
        Symmetric positive (semi)definite prior covariance matrix. 
    p : int
        Integer corresponding to data dimension.

    Raises
    ------
    ValueError
        If prior_cov is invalid, raise error.

    Returns
    -------
    prior_cov : np.ndarray of shape (d,d)
        Symmetric positive (semi)definite prior covariance matrix. 

    """
    prior_cov = np.array(prior_cov)
    dim = prior_cov.shape[0]
    if dim != p:
        msg_dim = (f"Data is {p}-dimensional but prior covariance matrix is {dim}x{dim}."
                       f" Please specify a {p}x{p} prior covariance matrix")
        raise ValueError(msg_dim)
    
    msg_sym = "Prior covariance matrix should be symmetric"
    if dim != prior_cov.shape[1]:
        raise ValueError(msg_sym)
    elif not np.all(np.isclose(prior_cov, prior_cov.T)):
        raise ValueError(msg_sym)
    
    if np.any(np.linalg.eigvals(prior_cov) < 0):
        raise ValueError("Prior covariance matrix must be positive semi-definite")
    
    return prior_cov

def _proc_mean_cov(prior_mean, prior_cov, p):
    """
    Process mean and covariance to ensure they are valid and convert them to np.ndarray.

    Parameters
    ----------
    prior_mean : ndarray of length d
        Array containing prior mean of Normal distribution.
    prior_cov : ndarray of shape (d,d)
        Symmetric positive (semi)definite prior covariance matrix. 
    p : int
        Integer corresponding to dimension of data.

    Raises
    ------
    ValueError
        If prior_mean and prior_cov are invalid, raise error.

    Returns
    -------
    prior_mean : np.ndarray of length d
        Array containing prior mean of Normal distribution.
    prior_cov : np.ndarray of shape (d,d)
        Symmetric positive (semi)definite prior covariance matrix. 

    """
    prior_mean = np.array(prior_mean)
    prior_cov = _proc_cov(prior_cov, p)
    
    prior_len = len(prior_mean)
    if prior_len == p:
        prior_mean = np.array(prior_mean)
        if len(prior_mean.shape) > 1:
            raise ValueError("Prior mean vector should be one-dimensional")
    else:
        msg_len = (f"Prior mean has length {prior_len} but should have length {p} based on the dimension of the data")
        raise ValueError(msg_len)
        
    return prior_mean, prior_cov

def _fit_Class(fittable, rng, *args):
    """
    Return fitted object of class "fittable".
    
    Parameters
    ----------
    fittable : class
        An object of class fittable
    rng : np.random._generator.Generator, optional
        Random number generator to ensure reproducibility. The default is None.
    *args : extra arguments
        Arguments to be given to __init__ method of fittable object.

    Returns
    -------
    obj : fittable
        Fitted object of class "fittable".

    """
    obj = fittable(*args)
    obj.fit(rng=rng)
    return obj
        
    
def uni_binom(n, p, lam):
    """
    Interpolates between a discrete uniform and a unit-shifted binomial distribution .

    Parameters
    ----------
    n : int
        Number of trials parameter for binomial distribution.
    p : float
        Parameter for binomial distribution on [0, n-1].
    lam : float
        Interpolation parameter. Between 0 and 1.

    Returns
    -------
    ndarray
        Probability mass function on [0, n-1] (equivalently [1,n]).

    """
    if lam < 0 or lam > 1:
        raise ValueError("Lambda must be nonnegative and no greater than 1")
    k = np.arange(1,n+1)
    with warnings.catch_warnings():
        # suppress the RuntimeWarning that sometimes results...
        warnings.simplefilter("ignore")
        bin_p = binom.pmf(k-1, n-1, p)
    return bin_p**(lam) 

def prob_mode(arr):
    """
    Gets mode of given array. Gives error based on ``bclr_multi`` if
    min_size constraints yield an empty array which is fed to this function.

    Parameters
    ----------
    arr : ndarray of shape (n_obs,)
        Array of observed numbers (need not be integers).

    Returns
    -------
    ndarray
        Returns array of mode(s)..

    """
    arr_vals, arr_counts = np.unique(arr, return_counts=True)
    if len(arr_counts)==0:
        warnings.warn_explicit(message="""Argument min_size too large to find posterior mode... \n""", category=SegmentationWarning, 
                               filename="bclr_multi.py", lineno=116)
        return np.nan
    else:
        return arr_vals[np.argmax(arr_counts)]

def std_video(video, flip=False):
    """
    A function to standard a video to have frames with mean 0, variance 1 pixel intensity.

    Parameters
    ----------
    video : array-like of shape (n_frames, n_rows, n_cols)
        The greyscale video of interest
    flip : bool, optional
        If True, pixels in image will be inverted. The default is False.

    Returns
    -------
    ndarray
        Standardized video.

    """
    v_mean = np.mean(video, axis=(1,2))
    v_std = np.std(video, axis=(1,2))
    v_means=np.transpose(np.tile(v_mean, (video.shape[1], video.shape[2],1)), (2,0,1))
    v_stds=np.transpose(np.tile(v_std, (video.shape[1], video.shape[2],1)), (2,0,1))
    return (-1)**(flip)*(video-v_means)/(v_stds)

def gen_sim(n=1000, plus = -2, ind=25, seed=0):
    """
    The function used to generate the simulated videos in Thomas, Jauch, and Matteson (2023).

    Parameters
    ----------
    n : int, optional
        The number of simulated videos to generate. The default is 1000.
    plus : bool, optional
        Add this quantity to the random rectangular region within each image. The default is -2.
    ind : int, optional
        Where to locate the changepoint. The default is 25.
    seed : int, optional
        Seed for random number generation. The default is 0.

    Returns
    -------
    arr : array of shape (n, 50, 50, 50)
        Contains all simulated videos

    """
    np.random.seed(seed)
    k = 0
    arr = np.empty((n, 50, 50, 50))
    while k < n:
        noise = np.random.randn(50, 50, 50)
        for i in range(ind, 50):
            pt = np.random.randint(-20, 20, 2)
            size = np.random.randint(2, 5, 2)
            y1, y2, z1, z2 = (25-size[0]+pt[0]), (25+size[0]+pt[0]), (25-size[1]+pt[1]), (25+size[1]+pt[1])
            noise[i, y1:y2, z1:z2] = noise[i, y1:y2, z1:z2]+plus
    
        arr[k] = std_video(noise)
        k += 1
    return arr
