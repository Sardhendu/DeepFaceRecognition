import os
import numpy as np
import pandas as pd
from config import path_dict

batch_num_image_idx_info_path = os.path.join(path_dict['batchFolderPath'], 'batch_num_image_idx_info.csv')
person_name_image_num_info_path = os.path.join(path_dict['data_model_path'], 'person_name_image_num_info.csv')

batch_num_image_idx_info = pd.read_csv(batch_num_image_idx_info_path)
person_name_image_num_info = pd.read_csv(person_name_image_num_info_path)


def save_prediction_analysis(cv_act, cv_hat, cv_hat_prob, fold, epoch):
    correct_arr = np.array(['y' if i == j else 'n' for i, j in zip(cv_act, cv_hat)], dtype=str).reshape(-1, 1)
    cv_act = np.array(cv_act, dtype='float32').reshape(-1, 1)
    cv_hat = np.array(cv_hat, dtype='float32').reshape(-1, 1)
    cv_hat_prob = np.array(cv_hat_prob, dtype='float32').reshape(-1, 1)
    fold_arr = np.tile(fold, len(cv_hat_prob)).reshape(-1, 1)
    epoch_arr = np.tile(epoch, len(cv_hat_prob)).reshape(-1, 1)
    
    if fold == 1:
        batch_num_image_idx_info_in = batch_num_image_idx_info[
            batch_num_image_idx_info['batch_num'] == 10].reset_index().drop('level_0', axis=1)
        p_name_batch_num_mrgd = person_name_image_num_info.merge(batch_num_image_idx_info_in,
                                                                 left_on=['index', 'image_label'],
                                                                 right_on=['index', 'image_label'], how='inner')
    else:
        raise ValueError('save_preiction_analysis only handled for FOld 1')
    
    path = os.path.join(path_dict['cv_pred_analysis_path'], 'cv_prediction_analysis_fld_%s_epch_%s.csv' % (str(fold),
                                                                                                           str(epoch)))
    
    nw_data = np.column_stack((fold_arr, epoch_arr, correct_arr, cv_act, cv_hat, cv_hat_prob))
    nw_data = pd.DataFrame(nw_data,
                           columns=['fold', 'epoch', 'is_correct_pred', 'cv_actual', 'cv_predicted', 'cv_pred_prob'])
    
    nw_data = pd.concat([p_name_batch_num_mrgd, nw_data], axis=1,
                        join_axes=[p_name_batch_num_mrgd.index])
    nw_data.to_csv(path, index=None)