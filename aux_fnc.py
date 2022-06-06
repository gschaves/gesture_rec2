from scipy import signal
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def trial(data,flag_index,i_trial):
  data_trial = data[flag_index[i_trial]+1:flag_index[i_trial+1]]
  return data_trial

#Function that rectifies and filters the signal
def filt_emg(emgs_raw,b,a):
  emg_filt = []
  for emg_raw in emgs_raw:

    #Signal normalized
    if np.max(abs(emg_raw))>1:
      emg_norm = emg_raw/128
    else:
      emg_norm = emg_raw
    #end

    #Signal rectified (R)
    emg_rect = abs(emg_norm)

    #Signal filtered (V)
    emg_filt.append(signal.filtfilt(b,a,emg_rect,axis=0))
  #end

  return emg_filt
################################################################################

#Function that detect the muscle activity
def muscle_activity(emgs_filt,fs,window,numFreqOfSpec,nOver_win,hamm_win,tau_u,min_seg):

  emgs_ts         = []
  gesture_index_s = []
  gesture_index_e = []

  tau_u_aux   = 0
  tau_u_count = 0
  for emg_filt in emgs_filt:
    #Sum along the rows (S)
    emg_sum = np.sum(emg_filt,axis=-1)
    
    #Spectrogram (P_C)
    f,t,spec = signal.stft(emg_sum,fs=fs,window=window,nperseg=numFreqOfSpec,noverlap=nOver_win,nfft=numFreqOfSpec*2,boundary=None, padded=False)

    #Modulus (P)
    spec_abs = ((hamm_win+1)/2)*abs(spec)                                        

    #Sum along the rows (U)
    spec_sum = np.sum(spec_abs,0)

    #Muscle activity detection
    great_than_tau = spec_sum >= tau_u
    great_than_tau = np.concatenate([[0],great_than_tau.astype(int),[0]])

    diff_great_than_tau = abs(np.diff(great_than_tau))

    if diff_great_than_tau[-1]==1:
      diff_great_than_tau[-2]=1

    diff_great_than_tau= diff_great_than_tau[:-1]

    index_nonzero = np.where(diff_great_than_tau==1)
    index_nonzero = np.array(index_nonzero[0])

    index_samp = np.floor(t*fs)-1

    nindex_nonzero = len(index_nonzero)

    #Where the muscle activity starts and ends
    #First condition: none activity is detected
    #Second condition: only the start is detected
    #Third condition: the start and end is detected
    if nindex_nonzero==0:
      index_s = 0
      index_e = len(emg_sum)
    elif nindex_nonzero==1:
      index_s = index_samp[index_nonzero].astype(int)
      index_e = len(emg_sum)
    else:
      index_s = index_samp[index_nonzero[0]].astype(int)
      index_e = index_samp[index_nonzero[-1]-1].astype(int)+1
    #end

    nxtra_samples = 25;
    index_s = np.maximum(0, index_s - nxtra_samples)
    index_e = np.minimum(len(emg_sum), index_e + nxtra_samples)

    #Check if the length of the activity is greater than "min_seg"
    #Else, all the signal is selected
    if (index_e - index_s) < min_seg:
      index_s = 0
      index_e = len(emg_sum)
    #end

    gesture_index_s = np.append(gesture_index_s,index_s)
    gesture_index_e = np.append(gesture_index_e,index_e)
    emgs_ts.append(emg_filt[index_s:index_e,:])
  #end

  return emgs_ts, spec, spec_abs, gesture_index_s.astype(int), gesture_index_e.astype(int)
################################################################################

#Compute the best centers of the class
def best_center_class(emgs_ts):
  mtxDistances = np.zeros((len(emgs_ts), len(emgs_ts)))

  for j in range(len(emgs_ts)):
    for k in range(len(emgs_ts)):
      if k>j:
        dist_dtw, path_dtw = fastdtw(emgs_ts[j],emgs_ts[k],radius = 100,dist=2)
        mtxDistances[j,k] = dist_dtw
        mtxDistances[k,j] = dist_dtw
      #end

    #end
    
  #end

  vectDistances = np.sum(mtxDistances,0)
  idx_min       = np.argmin(vectDistances)
  center_idx    = emgs_ts[idx_min]
  # print(mtxDistances,vectDistances,idx_min,center_idx, sep='\n')

  return center_idx
################################################################################

#Feature extraction
def feat_extrac(emgs_ts,center):
  dataX_temp   = [fastdtw(emg_ts,center,radius = 100,dist=2) for emg_ts in emgs_ts]
  dataX, pathX = map(list,zip(*dataX_temp))

  return dataX
################################################################################

#Feature normalization
def feat_norm(dataX):
  df_dataX      = dataX.copy()
  df_dataX_mean = pd.DataFrame(df_dataX.mean(axis=1),columns=['Mean'])
  df_dataX_std  = pd.DataFrame(df_dataX.std(axis=1),columns=['Std'])

  normalized_feat = df_dataX.sub(df_dataX_mean['Mean'],axis=0)
  normalized_feat = normalized_feat.div(df_dataX_std['Std'],axis=0)

  return normalized_feat
################################################################################