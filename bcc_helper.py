import numpy as np

def std_video(video, flip=False):
    """

    Parameters
    ----------
    video : TYPE
        DESCRIPTION.
    flip : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    v_mean = np.mean(video, axis=(1,2))
    v_std = np.std(video, axis=(1,2))
    v_means=np.transpose(np.tile(v_mean, (video.shape[1], video.shape[2],1)), (2,0,1))
    v_stds=np.transpose(np.tile(v_std, (video.shape[1], video.shape[2],1)), (2,0,1))
    return (-1)**(flip)*(video-v_means)/(v_stds)

def gen_sim(n=1000, plus = -2, ind=25, seed=0):
    """

    Parameters
    ----------
    n : TYPE, optional
        DESCRIPTION. The default is 1000.
    plus : TYPE, optional
        DESCRIPTION. The default is -2.
    ind : TYPE, optional
        DESCRIPTION. The default is 25.
    seed : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    arr : TYPE
        DESCRIPTION.

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
