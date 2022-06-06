from turtle import pd
import numpy as np
import pandas as pd

from aux_fnc import feat_extrac, feat_norm

def feat_extractor(idx_user, gestures, single_set_aux, best_centers, dataY_train_aux, sub_data_train):

    # if idx_user == 0:
    #     featX_train            = {}
    #     normalized_featX_train = {}
    #     dataY_train            = {}
    # #end

    feat_df        = pd.DataFrame()
    
    # featX_train[idx_user]            = {}
    # featX_train[idx_user]['Name']    = best_centers[idx_user]['Name']
    # featX_train[idx_user]['Feature'] = []

    # normalized_featX_train[idx_user]            = {}
    # normalized_featX_train[idx_user]['Name']    = best_centers[idx_user]['Name']
    # normalized_featX_train[idx_user]['Feature'] = []

    # dataY_train[idx_user]                 = {}
    # dataY_train[idx_user]['Name']         = best_centers[idx_user]['Name']
    # dataY_train[idx_user]['Actual Label'] = []

    sub_data_train['featX_train'][idx_user]            = {}
    sub_data_train['featX_train'][idx_user]['Name']    = best_centers[idx_user]['Name']
    sub_data_train['featX_train'][idx_user]['Feature'] = []

    sub_data_train['normalized_featX_train'][idx_user]            = {}
    sub_data_train['normalized_featX_train'][idx_user]['Name']    = best_centers[idx_user]['Name']
    sub_data_train['normalized_featX_train'][idx_user]['Feature'] = []

    sub_data_train['dataY_train'][idx_user]                 = {}
    sub_data_train['dataY_train'][idx_user]['Name']         = best_centers[idx_user]['Name']
    sub_data_train['dataY_train'][idx_user]['Actual Label'] = []

    for gesture in gestures:
        idx_gesture = np.where(gestures==gesture)[0].item()
        
        feat_train_temp = feat_extrac(single_set_aux,best_centers[idx_user]['Center'][idx_gesture]['Best Center'])
        feat_train_temp = np.array(feat_train_temp).T

        feat_df[str(idx_gesture)] = feat_train_temp
    #end

    # featX_train[idx_user]['Feature'] = feat_df.copy()                              #Feature vector
    sub_data_train['featX_train'][idx_user]['Feature'] = feat_df.copy()                              #Feature vector

    #Feature normalization
    # normalized_featX_train[idx_user]['Feature'] = feat_norm(featX_train[idx_user]['Feature'])
    # dataY_train[idx_user]['Actual Label']       = pd.DataFrame(data=dataY_train_aux[1:],columns=['Y'])

    sub_data_train['normalized_featX_train'][idx_user]['Feature'] = feat_norm(sub_data_train['featX_train'][idx_user]['Feature'])
    sub_data_train['dataY_train'][idx_user]['Actual Label']       = pd.DataFrame(data=dataY_train_aux[1:],columns=['Y'])

    # return normalized_featX_train, dataY_train
    return sub_data_train['normalized_featX_train'], sub_data_train['dataY_train']