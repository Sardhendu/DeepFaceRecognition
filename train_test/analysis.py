import os
import numpy as np
import pandas as pd
from config import path_dict

def save_prediction_analysis(cv_act, cv_hat, cv_hat_prob, fold, epoch):
    cv_act = np.array(cv_act, dtype='float32').reshape(-1,1)
    cv_hat = np.array(cv_hat, dtype='float32').reshape(-1, 1)
    cv_hat_prob = np.array(cv_hat_prob, dtype='float32').reshape(-1, 1)
    fold_arr = np.tile(len(cv_hat_prob)).reshape(-1, 1)
    epoch_arr = np.tile(len(cv_hat_prob)).reshape(-1, 1)
    path = os.path.join(path_dict['cv_pred_analysis_path'], 'cv_prediction_analysis_fld_%s_epch_%s.csv'%(str(fold),
                                                                                                         str(epoch)))
    
    nw_data = np.column_stack((fold_arr, epoch_arr, cv_act, cv_hat, cv_hat_prob))
    nw_data = pd.DataFrame(nw_data, columns = ['fold', 'epoch', 'cv_actual', 'cv_predicted', 'cv_pred_prob'])
    nw_data.to_csv(path, index=None)
