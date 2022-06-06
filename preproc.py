from scipy import signal
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

from neural_net import *
from aux_fnc import filt_emg, muscle_activity, best_center_class

def preproc(subject_train, train_params, tau, idx_user, gestures, validation, sub_data_train):

    # if idx_user == 0:
    #     emgs_filt              = {}
    #     emgs_seg               = {}
    #     seg_idx                = {}
    #     best_centers           = {}
    # #end


    sub_data_train['emgs_filt'][idx_user]         = {}
    sub_data_train['emgs_filt'][idx_user]['Name'] = subject_train[idx_user]['Name']
    sub_data_train['emgs_filt'][idx_user]['Data'] = {}

    sub_data_train['emgs_seg'][idx_user]         = {}
    sub_data_train['emgs_seg'][idx_user]['Name'] = subject_train[idx_user]['Name']
    sub_data_train['emgs_seg'][idx_user]['Data'] = {}

    sub_data_train['seg_idx'][idx_user]          = {}
    sub_data_train['seg_idx'][idx_user]['Name']  = subject_train[idx_user]['Name']
    sub_data_train['seg_idx'][idx_user]['Start'] = {}
    sub_data_train['seg_idx'][idx_user]['End']   = {}

    sub_data_train['best_centers'][idx_user]           = {}
    sub_data_train['best_centers'][idx_user]['Name']   = subject_train[idx_user]['Name']
    sub_data_train['best_centers'][idx_user]['Center'] = {}

    # emgs_filt[idx_user]         = {}
    # emgs_filt[idx_user]['Name'] = subject_train[idx_user]['Name']
    # emgs_filt[idx_user]['Data'] = {}

    # emgs_seg[idx_user]         = {}
    # emgs_seg[idx_user]['Name'] = subject_train[idx_user]['Name']
    # emgs_seg[idx_user]['Data'] = {}

    # seg_idx[idx_user]          = {}
    # seg_idx[idx_user]['Name']  = subject_train[idx_user]['Name']
    # seg_idx[idx_user]['Start'] = {}
    # seg_idx[idx_user]['End']   = {}

    # best_centers[idx_user]            = {}
    # best_centers[idx_user]['Name']    = subject_train[idx_user]['Name']
    # best_centers[idx_user]['Center']  = {}

    a             = train_params['a']
    b             = train_params['b']
    fs            = train_params['fs']
    window        = train_params['window']
    numFreqOfSpec = train_params['numFreqOfSpec']
    nOver_win     = train_params['nOver_win']
    hamm_win      = train_params['hamm_win']
    min_seg       = train_params['min_seg']

    single_set_aux = np.empty((1,),dtype=object)

    dataY_train_aux = np.empty((1,))
    
    for gesture in gestures:
        idx_gesture = np.where(gestures==gesture)[0].item()

        subject_train_aux = subject_train[idx_user]['Data'][idx_gesture]['Signal']

        emgs_raw      = []
        emgs_raw_temp = subject_train_aux

        for key in emgs_raw_temp:
            if key != 'Gesture':
                if validation:
                    emgs_raw.append(emgs_raw_temp[key].drop(['Gesture','Validation'], axis=1).apply(pd.to_numeric).values)
                else:
                    emgs_raw.append(emgs_raw_temp[key].drop(['Gesture'], axis=1).apply(pd.to_numeric).values)
                #end
            #end
        #end
        
        #Filtered signal
        emgs_filt_temp = filt_emg(emgs_raw,b,a)

        #Detect muscle activity
        [emgs_seg_temp, null_a, null_b, idxS, idxE]  = muscle_activity(emgs_filt_temp,fs,window,numFreqOfSpec,nOver_win,hamm_win,tau,min_seg)
        
        emgs_seg_aux = np.array(emgs_seg_temp,dtype=object)

        if np.shape(emgs_seg_aux) != (np.shape(emgs_seg_aux)[0],):
            for i in range(np.shape(emgs_seg_aux)[0]): 
                single_set_aux     = np.append(single_set_aux,0)
                single_set_aux[-1] = np.array(emgs_seg_aux[i])
            #end
        else:
            single_set_aux = np.append(single_set_aux,emgs_seg_aux)
            single_set_aux = single_set_aux.astype(object)
        #end

        #Best center for each gesture
        best_centers_temp = best_center_class(emgs_seg_temp)                                                      

        #Saving data in the dictionaries
        # emgs_filt[idx_user]['Data'][idx_gesture] = {'Gesture': gesture, 'Filtered Signal': emgs_filt_temp}
        # emgs_seg[idx_user]['Data'][idx_gesture]  = {'Gesture': gesture, 'Segmented Signal': emgs_seg_temp}

        sub_data_train['emgs_filt'][idx_user]['Data'][idx_gesture] = {'Gesture': gesture, 'Filtered Signal': emgs_filt_temp}
        sub_data_train['emgs_seg'][idx_user]['Data'][idx_gesture]  = {'Gesture': gesture, 'Segmented Signal': emgs_seg_temp}

        # seg_idx[idx_user]['Start'][idx_gesture] = idxS
        # seg_idx[idx_user]['End'][idx_gesture]   = idxE

        sub_data_train['seg_idx'][idx_user]['Start'][idx_gesture] = idxS
        sub_data_train['seg_idx'][idx_user]['End'][idx_gesture]   = idxE

        # best_centers[idx_user]['Center'][idx_gesture] = {'Gesture': gesture, 'Best Center': best_centers_temp}

        sub_data_train['best_centers'][idx_user]['Center'][idx_gesture] = {'Gesture': gesture, 'Best Center': best_centers_temp}

        # dataY_train_temp = np.ones((len(emgs_filt[idx_user]['Data'][idx_gesture]['Filtered Signal']), 1))*(idx_gesture+1)
        dataY_train_temp = np.ones((len(sub_data_train['emgs_filt'][idx_user]['Data'][idx_gesture]['Filtered Signal']), 1))*(idx_gesture+1)
        dataY_train_aux  = np.vstack((dataY_train_aux,dataY_train_temp))
    #end

    # return single_set_aux[1:], best_centers, dataY_train_aux
    return single_set_aux[1:], sub_data_train['best_centers'], dataY_train_aux
