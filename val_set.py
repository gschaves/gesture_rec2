import numpy as np
import copy

def val_set(subject, idx_user, gestures, validation, val_count, sub_data_train):

    # if idx_user == 0:
    #     subject_val            = {}
    #     subject_train          = {}
    # #end

    # subject_train[idx_user]           = {}
    # subject_train[idx_user]['Name']   = subject[idx_user]['Name']
    # subject_train[idx_user]['Data']   = {}

    # subject_val[idx_user]           = {}
    # subject_val[idx_user]['Name']   = subject[idx_user]['Name']
    # subject_val[idx_user]['Data']   = {}

    sub_data_train['subject_train'][idx_user]           = {}
    sub_data_train['subject_train'][idx_user]['Name']   = subject[idx_user]['Name']
    sub_data_train['subject_train'][idx_user]['Data']   = {}

    sub_data_train['subject_val'][idx_user]           = {}
    sub_data_train['subject_val'][idx_user]['Name']   = subject[idx_user]['Name']
    sub_data_train['subject_val'][idx_user]['Data']   = {}

    for gesture in gestures:
        idx_gesture = np.where(gestures==gesture)[0].item()

        val_trial_index = []
        if validation == True:
            r1 = val_count % 5
            r2 = (1+val_count) % 5

            val_trial_index = np.array([r1, r2])+1

            subject_aux   = copy.deepcopy(subject)

            for t in range(1,6):
                if t in val_trial_index:
                    subject_aux[idx_user]['Data'][idx_gesture]['Trial'+str(t)]['Validation']=1
                else:
                    subject_aux[idx_user]['Data'][idx_gesture]['Trial'+str(t)]['Validation']=0
                #end
            #end

            subject_val_aux   = copy.deepcopy(subject_aux[idx_user]['Data'][idx_gesture])
            subject_train_aux = copy.deepcopy(subject_aux[idx_user]['Data'][idx_gesture])

            for t in range(1,6):
                if all(subject_val_aux['Trial'+str(t)]['Validation']) == 0:
                    del subject_val_aux['Trial'+str(t)]
                else:
                    del subject_train_aux['Trial'+str(t)]
                #end
            #end
        else:
            subject_train_aux = copy.deepcopy(subject[idx_user]['Data'][idx_gesture])
            subject_val_aux   = {}
        #end
        
        # subject_train[idx_user]['Data'][idx_gesture] = {'Gesture': gesture, 'Signal': subject_train_aux}
        # subject_val[idx_user]['Data'][idx_gesture]   = {'Gesture': gesture, 'Signal': subject_val_aux}

        sub_data_train['subject_train'][idx_user]['Data'][idx_gesture] = {'Gesture': gesture, 'Signal': subject_train_aux}
        sub_data_train['subject_val'][idx_user]['Data'][idx_gesture]   = {'Gesture': gesture, 'Signal': subject_val_aux}
    #end

    # return subject_train, subject_val
    return sub_data_train['subject_train'], sub_data_train['subject_val']