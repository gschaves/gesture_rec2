import numpy as np
import pandas as pd

from aux_fnc import feat_extrac, feat_norm, filt_emg, muscle_activity
from neural_net import L_model_forward


def predictor(idx_user, gestures, validation, subject_test, best_centers, clf, params, tau, ens,
              sub_data_train):

    # if idx_user == 0:
    #     emgs_filt_test        = {}
    #     emgs_seg_test         = {}
    #     seg_idx_test          = {}
    #     featX_test            = {}
    #     normalized_featX_test = {}
    #     acc_test              = {}
    #     predictions           = {}
    # #end

    a = params['a']
    b = params['b']
    fs = params['fs']
    window = params['window']
    numFreqOfSpec = params['numFreqOfSpec']
    nOver_win = params['nOver_win']
    hamm_win = params['hamm_win']
    min_seg = params['min_seg']
    stride = params['stride']
    win_len = params['win_len']

    feat_test_df = pd.DataFrame()

    #Defining the nested dictionaries
    # emgs_filt_test[idx_user]         = {}
    # emgs_filt_test[idx_user]['Name'] = subject_test[idx_user]['Name']
    # emgs_filt_test[idx_user]['Data'] = {}

    # emgs_seg_test[idx_user]         = {}
    # emgs_seg_test[idx_user]['Name'] = subject_test[idx_user]['Name']
    # emgs_seg_test[idx_user]['Data'] = {}

    # seg_idx_test[idx_user]          = {}
    # seg_idx_test[idx_user]['Name']  = subject_test[idx_user]['Name']
    # seg_idx_test[idx_user]['Start'] = {}
    # seg_idx_test[idx_user]['End']   = {}

    # featX_test[idx_user]            = {}
    # featX_test[idx_user]['Name']    = subject_test[idx_user]['Name']
    # featX_test[idx_user]['Feature'] = {}

    # normalized_featX_test[idx_user]            = {}
    # normalized_featX_test[idx_user]['Name']    = subject_test[idx_user]['Name']
    # normalized_featX_test[idx_user]['Feature'] = {}

    # predictions[idx_user]            = {}
    # predictions[idx_user]['Name']    = subject_test[idx_user]['Name']
    # predictions[idx_user]['Prediction'] = {}

    # acc_test[idx_user] = {}
    if ens == 0:
        sub_data_train['emgs_filt_test'][idx_user] = {}
        sub_data_train['emgs_filt_test'][idx_user]['Name'] = subject_test[idx_user]['Name']
        sub_data_train['emgs_filt_test'][idx_user]['Data'] = {}

        sub_data_train['emgs_seg_test'][idx_user] = {}
        sub_data_train['emgs_seg_test'][idx_user]['Name'] = subject_test[idx_user]['Name']
        sub_data_train['emgs_seg_test'][idx_user]['Data'] = {}

        sub_data_train['seg_idx_test'][idx_user] = {}
        sub_data_train['seg_idx_test'][idx_user]['Name'] = subject_test[idx_user]['Name']
        sub_data_train['seg_idx_test'][idx_user]['Start'] = {}
        sub_data_train['seg_idx_test'][idx_user]['End'] = {}

        sub_data_train['featX_test'][idx_user] = {}
        sub_data_train['featX_test'][idx_user]['Name'] = subject_test[idx_user]['Name']
        sub_data_train['featX_test'][idx_user]['Feature'] = {}

        sub_data_train['normalized_featX_test'][idx_user] = {}
        sub_data_train['normalized_featX_test'][idx_user]['Name'] = subject_test[idx_user]['Name']
        sub_data_train['normalized_featX_test'][idx_user]['Feature'] = {}

    sub_data_train['predictions'][idx_user] = {}
    sub_data_train['predictions'][idx_user]['Name'] = subject_test[idx_user]['Name']
    sub_data_train['predictions'][idx_user]['Prediction'] = {}

    sub_data_train['acc_test'][idx_user] = {}

    for gesture in gestures[1:]:
        idx_gesture = np.where(gestures == gesture)[0].item()

        emgs_raw_test = []
        emgs_raw_test_temp = subject_test[idx_user]['Data'][idx_gesture].copy()
        if validation:
            for key in emgs_raw_test_temp['Signal']:
                if key != 'Gesture':
                    emgs_raw_test.append(emgs_raw_test_temp['Signal'][key].drop(
                        ['Gesture', 'Validation'], axis=1).apply(pd.to_numeric).values)
                #end
            #end
        else:
            for key in emgs_raw_test_temp:
                if key != 'Gesture':
                    emgs_raw_test.append(emgs_raw_test_temp[key].drop(['Gesture'], axis=1).apply(
                        pd.to_numeric).values)
                #end
            #end
        #end

        if ens == 0:
            sub_data_train['featX_test'][idx_user]['Feature'][idx_gesture] = {}
            sub_data_train['normalized_featX_test'][idx_user]['Feature'][idx_gesture] = {}

        finalLabel = []

        trial_count = 0
        emg_count = 0

        for emg_raw_test in emgs_raw_test:
            if ens == 0:
                sub_data_train['normalized_featX_test'][idx_user]['Feature'][idx_gesture][
                    emg_count] = {}

            emg_len = len(emg_raw_test)

            start_point = 0
            win_count = 0

            predictions_temp = []

            while True:
                if not validation:
                    start_point = win_count * stride  #Window start point
                    end_point = start_point + win_len  #Window end point
                    if end_point > emg_len:
                        break
                    #end
                    windowed_emg = emg_raw_test[start_point:end_point]
                else:
                    windowed_emg = emg_raw_test
                #end

                #Filtered signal
                emg_filt_test_temp = filt_emg([windowed_emg], b, a)

                #Detect muscle activity
                [emg_seg_test_temp, null_a, null_b, idxS,
                 idxE] = muscle_activity(emg_filt_test_temp, fs, window, numFreqOfSpec, nOver_win,
                                         hamm_win, tau, min_seg)

                if idxS[0] != 0 and idxE[0] != len(windowed_emg):
                    windowed_emg = windowed_emg[idxS[0]:idxE[0]]
                    filt_window_emg = filt_emg([windowed_emg], b, a)

                    if ens == 0:
                        for gesture_feat in gestures:
                            idx_gesture_feat = np.where(gestures == gesture_feat)[0].item()

                            feat_test_temp = feat_extrac(
                                filt_window_emg,
                                best_centers[idx_user]['Center'][idx_gesture_feat]['Best Center'])
                            feat_test_temp = np.array(feat_test_temp).T

                            feat_test_df[str(idx_gesture_feat)] = feat_test_temp
                        #end

                        sub_data_train['featX_test'][idx_user]['Feature'][idx_gesture][
                            win_count] = feat_test_df.copy()  #Feature vector

                        #Feature normalization
                        sub_data_train['normalized_featX_test'][idx_user]['Feature'][idx_gesture][
                            emg_count][win_count] = feat_norm(
                                sub_data_train['featX_test'][idx_user]['Feature'][idx_gesture]
                                [win_count]).values
                    #end

                    X_test = np.transpose(sub_data_train['normalized_featX_test'][idx_user]
                                          ['Feature'][idx_gesture][emg_count][win_count])

                    predictions_temp_aux, caches = L_model_forward(
                        X_test, clf[idx_user]['Fit'], hidden_layers_activation_fn="tanh")

                    if max(predictions_temp_aux) > 0.5:
                        predictions_temp.append(np.argmax(predictions_temp_aux) + 1)
                    else:
                        predictions_temp.append(1)
                    #end

                else:
                    predictions_temp.append(1)
                    sub_data_train['featX_test'][idx_user]['Feature'][idx_gesture][
                        win_count] = pd.DataFrame(np.zeros((1, 6)))
                    if win_count == 0:
                        sub_data_train['normalized_featX_test'][idx_user]['Feature'][idx_gesture][
                            emg_count][win_count] = np.zeros((1, 6))

                    else:
                        sub_data_train['normalized_featX_test'][idx_user]['Feature'][idx_gesture][
                            emg_count][win_count] = sub_data_train['normalized_featX_test'][
                                idx_user]['Feature'][idx_gesture][emg_count][win_count - 1]
                    #end
                #end

                win_count += 1

                if validation:
                    break
            #end

            trial_count += 1

            pred = predictions_temp

            if not validation:
                pred[0] = 1
            #end

            postProcessedLabels = pred

            num_label = len(pred)

            for label_i in range(1, num_label):
                cond = pred[label_i] == pred[label_i - 1]
                postProcessedLabels[label_i] = 1 * cond + pred[label_i] * (1 - cond)
            #end

            uniqueLabels = list(set(postProcessedLabels))
            uniqueLabels = np.array(uniqueLabels)
            uniqueLabelsWithoutRest = uniqueLabels[np.where(uniqueLabels != 1)]
            if np.size(uniqueLabelsWithoutRest) == 0:
                finalLabel.append(1)
            elif np.size(uniqueLabelsWithoutRest) > 1:
                finalLabel.append(uniqueLabelsWithoutRest[0])
            else:
                finalLabel.append(uniqueLabelsWithoutRest)
            #end

            emg_count += 1
        #end

        sub_data_train['predictions'][idx_user]['Prediction'][idx_gesture] = finalLabel

        #print(idx_gesture)
    #end

    # return predictions
    return sub_data_train['predictions']
