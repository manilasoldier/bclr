from ..bclr_multi import MultiBayesCC
import pytest
import numpy as np

Xm = np.load("bclr_multi.npy")

prior_covBAD1 = np.array([[3, 1], [2,2]])
prior_covBAD2 = np.array([[1, 1], [1,0.999]])
prior_covBAD3 = np.array([[1,0,0], [0,1,0], [0,0,1]])
prior_covBAD4 = np.array([[1,0], [1,0], [0,1]])
prior_mean = [0,0]
prior_cov = np.array([[1,0], [0,1]])

@pytest.mark.parametrize('prior_covBAD', [prior_covBAD1, prior_covBAD2, prior_covBAD3, prior_covBAD4])
def test_bad_cov(prior_covBAD):
    with pytest.raises(ValueError) as e_bad_cov:
        MultiBayesCC(Xm, cps=2, prior_cov = prior_covBAD, n_iter=1000)
    
    assert e_bad_cov.type is ValueError

def test_fit_transform_cps():
    bclr_multi_good = MultiBayesCC(Xm, cps=2, prior_cov = prior_cov, n_iter=1000)
    bclr_multi_good.fit_transform(n_jobs=4) #check the n_jobs argument works...
    bclr_multi_good.transform() #check additional transform does nothing...
    est_cps = bclr_multi_good.cps_df()['Location'].to_numpy()
    np.testing.assert_almost_equal(est_cps, np.array([200, 500]), 2)
   
@pytest.mark.filterwarnings("ignore:Number of changepoints reduced due to nan values")
@pytest.mark.parametrize('warn,cps', [(True, 10), (False, [100, 250, 550])])
def test_fit_predict_cps(warn,cps):
    bclr_multi_good = MultiBayesCC(Xm, cps=cps, prior_cov = prior_cov, n_iter=1000, warnings=warn)
    est_cps = bclr_multi_good.fit_predict()['Location'].to_numpy()
    np.testing.assert_almost_equal(est_cps, np.array([200, 500]), 2)
  
@pytest.mark.filterwarnings("ignore:Number of changepoints reduced due to nan values")
@pytest.mark.parametrize('warn', [True, False])
def test_warm_up_cps(warn):
    bclr_multi_good = MultiBayesCC(Xm, cps=5, prior_cov = prior_cov, n_iter=1000, warnings=warn)
    bclr_multi_good.warm_up(n_iter_w = 400, thr=0.8)
    bclr_multi_good.fit_transform()
    est_cps = bclr_multi_good.cps_df(thr=0.8)['Location'].to_numpy()
    np.testing.assert_almost_equal(est_cps, np.array([200, 500]), 2)
    
@pytest.mark.parametrize("index", [1,2,3,4,5,6])
def test_init(index):
    if index==1:
        with pytest.raises(TypeError):
            MultiBayesCC(Xm, cps=2, prior_cov = prior_cov, n_iter=1000, warnings=False, rng=100)
        
    elif index==2:
        with pytest.raises(TypeError):
            MultiBayesCC(Xm, cps=3.5, prior_cov = prior_cov, n_iter=1000, warnings=False)
        
    elif index==3:
        with pytest.raises(ValueError):
            MultiBayesCC(Xm, cps=2, prior_cov = prior_cov, n_iter=1000, warnings=False, min_size=200)
        
    elif index==4:
        with pytest.raises(ValueError):
            MultiBayesCC(Xm, cps=2, prior_cov = prior_cov, n_iter=1000, warnings=False, min_size=1)
    
    elif index==5:
        with pytest.raises(ValueError):
            MultiBayesCC(Xm, cps=2, prior_cov = prior_cov, n_iter=0, warnings=False)
        
    else:
        with pytest.raises(ValueError):
            MultiBayesCC(Xm, cps=2, prior_cov = prior_cov, n_iter=1000, warnings=False, lam=2)

@pytest.mark.parametrize("iter_sch,thr_sch", [[[100, 250], [1.5, 0.5]], [[100, 250, 500],[0.75, 0.5]]])
def test_bad_fit_predict(iter_sch, thr_sch):
    with pytest.raises(ValueError):
        bclr_multi_bad = MultiBayesCC(Xm, cps=2, prior_cov = prior_cov, n_iter=1000, warnings=False)
        bclr_multi_bad.fit_predict(iter_sch=iter_sch, thr_sch=thr_sch)