import numpy as np
from scipy.stats import binom
import warnings

class SegmentationWarning(Warning):
    pass

def sample_sep(n, k, delta=1):
    """
    Generates k+1 ordered random values with minimum distance delta between them. Note the last value is n.

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    delta : TYPE, optional
        DESCRIPTION. The default is 1.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if delta > n/(k+1):
        raise ValueError("Minimum distance spacing requirements cannot be satisfied.")
    rem = np.random.choice(np.arange(1, k+2), n-delta*(k+1))
    counts, _ = np.histogram(rem, np.arange(1, k+3))
    return np.cumsum(counts + delta)
        
    
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

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

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
