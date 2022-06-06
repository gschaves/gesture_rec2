# IMPORTING THE LIBRARIES
import os
import pickle
import warnings
from operator import itemgetter
from test import test

import numpy as np
from scipy import signal

from accuracy import accuracy
from feat_extractor import feat_extractor
from feed_forward_nn import feed_forward_nn
from predictor import predictor
from preproc import preproc
from sub_test import sub_test
from sub_train import sub_train
from val_set import val_set

#/////////////////////////////

np.random.seed(100)

warnings.filterwarnings("ignore")

# Defining some constants
fs = 200  #Sample frequency
min_seg = 100  #Minimun segmentation length
numFreqOfSpec = 25  #Number of frequencies for the STFT
hamm_win = 25  #Size of the Hamming window
nOver_win = 10  #Number of samples to overlap in the Hamming window
tau_u = np.linspace(10, 20, 11)  #Constant to detect the muscle activity
filt_order = 5  #Filter of 5th order
filt_freq = 0.1  #Filter with cutoff frequency of 10Hz
validation = False
x_val_folds = 3
stride = 10
win_len = 500
layers_dims = [6, 6, 6]
reg = 1e-2
N_ens = 100
tau_val = np.array([10, 12.5, 15, 15, 17, 14.5, 19, 10, 12.5, 19])
val_count = 0

b, a = signal.butter(
    filt_order, filt_freq, 'low'
)  #Butterworth lowpass filter of "filt_order" order and cutoff frequency of "filt_freq" (normalized)

window = signal.windows.hamming(hamm_win)  #Hamming window

params = {
    'a': a,
    'b': b,
    'fs': fs,
    'min_seg': min_seg,
    'numFreqOfSpec': numFreqOfSpec,
    'hamm_win': hamm_win,
    'nOver_win': nOver_win,
    'window': window,
    'stride': stride,
    'win_len': win_len
}

users = np.array([
    'alejandroFlores', 'alexandraApellido', 'andresGuerra', 'andresJaramillo', 'cristhianMotoche',
    'dianitaCherrez', 'homeroApellido', 'jonathanZea', 'juanYuquilema', 'santiagoJaramillo'
])
gestures = np.array(['relax', 'fist', 'wave_in', 'wave_out', 'fingers_spread', 'double_tap'])

cwd = os.getcwd()
#pathName = cwd + '\Dataset'
pathName = os.path.join(cwd, "Dataset")
version = 'training'

sub_data_train_val = {}

sub_data_train_val['subject_train'] = {}
sub_data_train_val['subject_val'] = {}

sub_data_train_val['emgs_filt'] = {}
sub_data_train_val['emgs_seg'] = {}
sub_data_train_val['seg_idx'] = {}
sub_data_train_val['best_centers'] = {}

sub_data_train_val['featX_train'] = {}
sub_data_train_val['normalized_featX_train'] = {}
sub_data_train_val['dataY_train'] = {}

sub_data_train_val['clf'] = {}

sub_data_train_val['emgs_filt_test'] = {}
sub_data_train_val['emgs_seg_test'] = {}
sub_data_train_val['seg_idx_test'] = {}
sub_data_train_val['featX_test'] = {}
sub_data_train_val['normalized_featX_test'] = {}
sub_data_train_val['acc_test'] = {}
sub_data_train_val['predictions'] = {}

acc_val = {}

sub_data_train_test = {}

sub_data_train_test['subject_train'] = {}
sub_data_train_test['subject_val'] = {}

sub_data_train_test['emgs_filt'] = {}
sub_data_train_test['emgs_seg'] = {}
sub_data_train_test['seg_idx'] = {}
sub_data_train_test['best_centers'] = {}

sub_data_train_test['featX_train'] = {}
sub_data_train_test['normalized_featX_train'] = {}
sub_data_train_test['dataY_train'] = {}

sub_data_train_test['clf'] = {}

sub_data_train_test['emgs_filt_test'] = {}
sub_data_train_test['emgs_seg_test'] = {}
sub_data_train_test['seg_idx_test'] = {}
sub_data_train_test['featX_test'] = {}
sub_data_train_test['normalized_featX_test'] = {}
sub_data_train_test['acc_test'] = {}
sub_data_train_test['predictions'] = {}

acc_test = {}

# num_users   = len(users)
# num_gesture = len(gestures)

#Creating the subjects dictionaries
subject = sub_train(users, gestures, pathName, version)

if validation:

    for user in users:
        idx_user = np.where(users == user)[0].item()

        acc_val_aux = []

        for tau in tau_u:

            acc = 0

            for val_count in range(x_val_folds):

                # test_fnc = test()

                # print(test_fnc)
                # input("Press Enter to continue...")

                subject_train, subject_val = val_set(subject, idx_user, gestures, validation,
                                                     val_count, sub_data_train_val)

                # if idx_user == 0:
                #     break

                # if idx_user == 2:
                #     print(sub_data_train['subject_train'])
                #     input("Press Enter to continue...")

                single_set, best_centers, dataY_train_aux = preproc(subject_train, params, tau,
                                                                    idx_user, gestures, validation,
                                                                    sub_data_train_val)

                normalized_featX_train, dataY_train = feat_extractor(idx_user, gestures, single_set,
                                                                     best_centers, dataY_train_aux,
                                                                     sub_data_train_val)

                clf = feed_forward_nn(idx_user, normalized_featX_train, dataY_train, single_set,
                                      layers_dims, reg, sub_data_train_val)

                predictions = predictor(idx_user, gestures, validation, subject_val, best_centers,
                                        clf, params, tau, 0, sub_data_train_val)

                acc_aux = accuracy(idx_user, gestures, predictions)

                acc += acc_aux
                #end
            #end
            acc /= val_count + 1
            print((tau, acc))

            acc_val_aux.append((tau, acc))
        #end
        acc_val[idx_user] = {}
        acc_val[idx_user]['(tau_u,acc)'] = acc_val_aux

    #end

    tau_val = np.zeros((10, ))
    for i in range(0, 10):
        tau_val[i] = max(acc_val[i]['(tau_u,acc)'], key=itemgetter(1))[0]
    #end

    validation = False
#end

tau_u = tau_val

version = 'testing'

subject_test = sub_test(users, gestures, pathName, version)

acc_ens = []

for ens in range(N_ens):

    for user in users:
        idx_user = np.where(users == user)[0].item()

        if ens == 0:
            subject_train, subject_val = val_set(subject, idx_user, gestures, validation, val_count,
                                                 sub_data_train_test)
            #print("Aqui subject_train")

            single_set, best_centers, dataY_train_aux = preproc(subject_train, params,
                                                                tau_u[idx_user], idx_user, gestures,
                                                                validation, sub_data_train_test)
            #print("Aqui single_set")

            normalized_featX_train, dataY_train = feat_extractor(idx_user, gestures, single_set,
                                                                 best_centers, dataY_train_aux,
                                                                 sub_data_train_test)
            #print("Aqui norm feat")
        #end

        clf = feed_forward_nn(idx_user, normalized_featX_train, dataY_train, single_set,
                              layers_dims, reg, sub_data_train_test)
        #print("Aqui clf")

        predictions = predictor(idx_user, gestures, validation, subject_test, best_centers, clf,
                                params, tau_u[idx_user], ens, sub_data_train_test)
        #print("Aqui predictor")
        acc_aux = accuracy(idx_user, gestures, predictions)

        acc_test[idx_user] = acc_aux
        print(acc_aux)
    #end
    acc_ens.append(np.sum([acc_test[i] for i in range(10)]) / len(users))
    print(ens)
#end
with open("acc100_06_06.pkl", "wb") as f:
    pickle.dump(acc_ens, f)
