from ..bclr_helper import std_video, gen_sim
import numpy as np

def test_vid():
    vid = np.squeeze(gen_sim(n=1, plus = -2, ind=25))
    std_vid = std_video(vid)
    vm = np.mean(std_vid, axis=(1,2))
    vsd = np.std(std_vid, axis=(1,2))
    
    np.testing.assert_almost_equal(vm, np.repeat(0, 50), 1e-10)
    np.testing.assert_almost_equal(vsd, np.repeat(1, 50), 1e-10)