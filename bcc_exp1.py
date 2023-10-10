import numpy as np
from bcc import BayesCC
#from bcc_helper import gen_sim
from changeforest import changeforest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
import detectda as dtda
import pickle

"""
This code runs the experiments for experiment 1, seen in our paper.
"""


#ims = gen_sim(seed=1677001)
#np.save("Experiment1_Minus2_Images1000.npy", ims)
ims = np.load("Experiment1_Minus2_Images1000.npy")
np.random.seed(8892030)
exp_data = {'bcc_tda': [], 'bcc_pca': [], 'cf_tda': [], 'cf_pca': []}
for i in range(10):
    noise_std = ims[i]
    
    noise_dt = dtda.ImageSeriesPlus(noise_std)
    noise_dt.fit(sigma=2)
    pers_stats = noise_dt.get_pers_stats()
    pers_stats_std0 = StandardScaler().fit_transform(pers_stats)
    
    noise_data = np.reshape(noise_std, (50, -1))
    noisePCA = PCA(n_components=36).fit_transform(noise_data)
    
    prior_cov = np.diag(np.repeat(3, 36))
    res_TDA_bcc = BayesCC(pers_stats_std0, prior_mean = np.repeat(0, 36), prior_cov = prior_cov, n_iter = 5000, scaled=True)
    res_TDA_bcc.fit(init_k = np.random.randint(1, 51))
    res_TDA_bcc.transform(verbose=False)
    
    res_PCA_bcc = BayesCC(noisePCA, prior_mean = np.repeat(0, 36), prior_cov = prior_cov, n_iter = 5000, scaled=False)     
    res_PCA_bcc.fit(init_k = np.random.randint(1, 51))
    res_PCA_bcc.transform(verbose=False)
    
    res_TDA_cf = changeforest(pers_stats_std0, "random_forest", "bs")
    res_PCA_cf = changeforest(noisePCA, "random_forest", "bs")
    
    res_TDA_kcp0 = rpt.Dynp(model='rbf', min_size=2).fit(pers_stats_std0)
    res_PCA_kcp0 = rpt.Dynp(model='rbf', min_size=2).fit(noisePCA)
    
    res_TDA_kcp = res_TDA_kcp0.predict(n_bkps=1)
    res_PCA_kcp = res_PCA_kcp0.predict(n_bkps=1)
    
    exp_data['bcc_tda'].append(res_TDA_bcc)
    exp_data['bcc_pca'].append(res_PCA_bcc)
    exp_data['cf_tda'].append({'best_split': res_TDA_cf.best_split, 'p_value': res_TDA_cf.p_value})
    exp_data['cf_pca'].append({'best_split': res_PCA_cf.best_split, 'p_value': res_PCA_cf.p_value})
    exp_data['kcp_tda'].append({'changepoint': res_TDA_kcp})
    exp_data['kcp_pca'].append({'changepoint': res_PCA_kcp})
    print("Iteration "+str(i+1))

with open('experiment1_data.pickle', 'wb') as handle:
    pickle.dump(exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)