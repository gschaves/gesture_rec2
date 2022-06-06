import os

import numpy as np
import pandas as pd

from aux_fnc import trial


def sub_test(users, gestures, pathName, version):

    subject_gesture = {}
    subject_test = {}

    #Creating the subjects dictionaries

    for user in users:
        idx_user = np.where(users == user)[0].item()

        subject_test[idx_user] = {}
        subject_test[idx_user]['Name'] = user
        subject_test[idx_user]['Data'] = {}

        for gesture in gestures[1:]:

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

            flag = df.loc[df['ChannelOne'].isin([
                'TrialOne', 'TrialTwo', 'TrialThree', 'TrialFour', 'TrialFive', 'TrialSix',
                'TrialSeven', 'TrialEight', 'TrialNine', 'TrialTen', 'TrialEleven', 'TrialTwelve',
                'TrialThirteen', 'TrialFourteen', 'TrialFifteen', 'TrialSixteen', 'TrialSeventeen',
                'TrialEighteen', 'TrialNineteen', 'TrialTwenty', 'TrialTwentyOne', 'TrialTwentyTwo',
                'TrialTwentyThree', 'TrialTwentyFour', 'TrialTwentyFive', 'TrialTwentySix',
                'TrialTwentySeven', 'TrialTwentyEight', 'TrialTwentyNine', 'TrialThirty'
            ])]
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
                    'Trial5': trial(df, flag_index, 4),
                    'Trial6': trial(df, flag_index, 5),
                    'Trial7': trial(df, flag_index, 6),
                    'Trial8': trial(df, flag_index, 7),
                    'Trial9': trial(df, flag_index, 8),
                    'Trial10': trial(df, flag_index, 9),
                    'Trial11': trial(df, flag_index, 10),
                    'Trial12': trial(df, flag_index, 11),
                    'Trial13': trial(df, flag_index, 12),
                    'Trial14': trial(df, flag_index, 13),
                    'Trial15': trial(df, flag_index, 14),
                    'Trial16': trial(df, flag_index, 15),
                    'Trial17': trial(df, flag_index, 16),
                    'Trial18': trial(df, flag_index, 17),
                    'Trial19': trial(df, flag_index, 18),
                    'Trial20': trial(df, flag_index, 19),
                    'Trial21': trial(df, flag_index, 20),
                    'Trial22': trial(df, flag_index, 21),
                    'Trial23': trial(df, flag_index, 22),
                    'Trial24': trial(df, flag_index, 23),
                    'Trial25': trial(df, flag_index, 24),
                    'Trial26': trial(df, flag_index, 25),
                    'Trial27': trial(df, flag_index, 26),
                    'Trial28': trial(df, flag_index, 27),
                    'Trial29': trial(df, flag_index, 28),
                    'Trial30': trial(df, flag_index, 29)
                }
            }

            idx_gesture = np.where(
                gestures == gesture)[0].item()  #how to use an np.array as index to another list
            subject_test[idx_user]['Data'][idx_gesture] = subject_gesture[
                'Data']  #Creating the final dictionary, with all of the subjects and gestures (F_i matrices)
        #end
    #end

    return subject_test
