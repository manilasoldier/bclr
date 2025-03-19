try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    print("Please ensure pandas, numpy, and matplotlib are installed...")

if __name__ == "__main__":
    names_exp3A_df = ['cp_bcc', 'bcc_x1', 'bcc_x2', 'bcc_x3', 'bcc_x4', 'bcc_x1sq', 'bcc_x1x2', 'bcc_x1x3',
                      'bcc_x1x4', 'bcc_x2sq', 'bcc_x2x3', 'bcc_x2x4', 'bcc_x3sq', 'bcc_x3x4', 'bccx4sq',
                      'rmse_bcc', 'cp_bccX', 'bcc_x1X', 'bcc_x2X', 'bcc_x3X', 'bcc_x4X', 'bcc_x1sqX', 'bcc_x1x2X', 'bcc_x1x3X',
                      'bcc_x1x4X', 'bcc_x2sqX', 'bcc_x2x3X', 'bcc_x2x4X', 'bcc_x3sqX', 'bcc_x3x4X', 'bccx4sqX',
                      'rmse_bccX', 'cp_cf', 'pval_cf', 'cp_cf_raw', 'pval_cf_raw', 'cp_kcp', 'cp_kcp_raw',
                      'cp_ecp', 'cp_ecp_raw', 'cp_gauss']
                      
    exp3A_df = pd.read_table('experimentCOV_data_prior.txt', sep=',', header=None, names=names_exp3A_df, index_col=False)
    
    print('P. Exact BCLR: %0.3f (%0.3f)' % (np.mean(exp3A_df['cp_bcc']==200), (np.std(exp3A_df['cp_bcc']==200)/np.sqrt(2500))))
    
    print('P. Exact CF: %0.3f (%0.3f)' % (np.mean(exp3A_df['cp_cf']==200), (np.std(exp3A_df['cp_cf']==200)/np.sqrt(2500))))
    
    print('P. Exact ECP: %0.3f (%0.3f)' % (np.mean(exp3A_df['cp_ecp']==200), (np.std(exp3A_df['cp_ecp']==200)/np.sqrt(2500))))
    
    print('P. Exact KCP: %0.3f (%0.3f)' % (np.mean(exp3A_df['cp_kcp']==200), (np.std(exp3A_df['cp_kcp']==200)/np.sqrt(2500))))
    
    print('P. Exact CF RAW: %0.3f (%0.3f)' % (np.mean(exp3A_df['cp_cf_raw']==200), (np.std(exp3A_df['cp_cf_raw']==200)/np.sqrt(2500))))
    
    print('P. Exact GAUSS: %0.3f (%0.3f)' % (np.mean(exp3A_df['cp_gauss']==200), (np.std(exp3A_df['cp_gauss']==200)/np.sqrt(2500))))
    
    print('----------------------------------------------------- \n\n')
    
    print('RMSE BCLR: %0.3f (%0.3f)' % (np.mean(exp3A_df['rmse_bcc']), np.std(exp3A_df['rmse_bcc'])))
    
    print('RMSE CF: %0.3f' % np.sqrt(np.mean((exp3A_df['cp_cf']-200)**2)))
    
    print('RMSE ECP: %0.3f' % np.sqrt(np.mean((exp3A_df['cp_ecp']-200)**2)))
    
    print('RMSE KCP: %0.3f' % np.sqrt(np.mean((exp3A_df['cp_kcp']-200)**2)))
    
    print('RMSE CF RAW: %0.3f' % np.sqrt(np.mean((exp3A_df['cp_cf_raw']-200)**2)))
    
    print('RMSE GAUSS: %0.3f' % np.sqrt(np.mean((exp3A_df['cp_gauss']-200)**2)))
    
    powers = np.array([[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [2, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 0, 1, 0],
           [1, 0, 0, 1],
           [0, 2, 0, 0],
           [0, 1, 1, 0],
           [0, 1, 0, 1],
           [0, 0, 2, 0],
           [0, 0, 1, 1],
           [0, 0, 0, 2]])
    
    names = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_1^2$', r'$x_1x_2$', r'$x_1x_3$', r'$x_1x_4$',
            r'$x_2^2$', r'$x_2x_3$', r'$x_2x_4$', r'$x_3^2$', r'$x_3x_4$', r'$x_4^2$']
    
    fig, ax = plt.subplots()
    ax.yaxis.grid(True)
    bp = ax.boxplot(x=exp3A_df.iloc[:, 1:15].to_numpy(), labels=names, sym='*', patch_artist=True)
    #17:31 in the second prior case...
    plt.setp(bp['fliers'], markersize=1.0)
    plt.xticks(rotation=30)
    plt.setp(bp["boxes"], facecolor="black")
    plt.setp(bp["medians"], color="white")
    ax.set_title("Covariance change\nBoxplot of posterior mean "+str(r'$\beta$')+" coefficients over 2500 simulations", loc='left')
    plt.show()