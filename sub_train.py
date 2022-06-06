import os

import numpy as np
import pandas as pd

from aux_fnc import trial


def sub_train(users, gestures, pathName, version):

    subject_gesture = {}
    subject = {}

    #Creating the subjects dictionaries
    for user in users:
        idx_user = np.where(users == user)[0].item()

        subject[idx_user] = {}
        subject[idx_user]['Name'] = user
        subject[idx_user]['Data'] = {}
        for gesture in gestures:
            #path = pathName + "\\" + user + "\\" + version + "\\" + user + "_" + gesture + ".csv"
            path = os.path.join(pathName, user, version, f"{user}_{gesture}.csv")
            df = pd.read_csv(path, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

            #Creating the gesture codes
            gesture_code = pd.DataFrame(gestures, columns=['Gestures'])
            code = gesture_code.loc[gesture_code['Gestures'] == gesture]

            code = np.array(code.index.values)

            N_samples = len(df.index)

            gestures_df = code * np.ones((N_samples, ), dtype=int)

            #Identifying the flags in the data. The flags marks the start of each trial

            flag = df.loc[df['ChannelOne'].isin(
                ['TrialOne', 'TrialTwo', 'TrialThree', 'TrialFour', 'TrialFive'])]
            flag_index = np.array(flag.index.values)
            flag_index = np.append(flag_index, len(df))

            df['Gesture'] = gestures_df  #Inserting the column "Gesture" to the DataFrame

            #Creating the dictionary with the data of each gesture
            subject_gesture = {
                'Name': user,
                'Data': {
                    'Gesture': gesture,
                    'Trial1': trial(df, flag_index, 0),
                    'Trial2': trial(df, flag_index, 1),
                    'Trial3': trial(df, flag_index, 2),
                    'Trial4': trial(df, flag_index, 3),
                    'Trial5': trial(df, flag_index, 4)
                }
            }

            idx_gesture = np.where(
                gestures == gesture)[0].item()  #how to use an np.array as index to another list
            subject[idx_user]['Data'][idx_gesture] = subject_gesture[
                'Data']  #Creating the final dictionary, with all of the subjects and gestures (F_i matrices)
        #end
    #end

    return subject
