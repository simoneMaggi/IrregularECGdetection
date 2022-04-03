#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:54:54 2021

@author: simone

ONLY MLII DATASET
"""


#%%
import numpy as np
import Utils as u
import os
from biosppy.signals.tools import filter_signal
import pywt
from joblib import Parallel, delayed
from biosppy.signals import ecg
import pickle
path = "mit_bih/"



def calc_baseline(signal):
    """
     https://mitbal.wordpress.com/2014/07/08/baseline-wander-removal-dengan-wavelet/


    Calculate the baseline of signal.
    Args:
        signal (numpy 1d array): signal whose baseline should be calculated
    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]


def save_beat(path, beatdata):
    name = 'R'+beatdata[1]+'_'+str(beatdata[2])
    np.save(path+name+'.npy', beatdata[0])
    info = beatdata[3]
    with open(path+name+".pkl", "wb") as h:
        pickle.dump(info, h)


def preprocess_ecg_Wav(sig,fs = 360, threshold = 0.04):
    baseline = calc_baseline(sig)
    sig = sig - baseline
    
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(sig), w.dec_len)
    coeffs = pywt.wavedec(sig, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    filtered = pywt.waverec(coeffs, 'sym4')
    sig_out = ecg.ecg(signal=filtered, sampling_rate=fs, show=False)
    r_peaks=sig_out["rpeaks"]
    return filtered, r_peaks

def preprocess_ecg(sig, fs = 360):
    sig_out = ecg.ecg(signal=sig, sampling_rate=fs, show=False)
    sig=sig_out["filtered"]
    r_peaks=sig_out["rpeaks"]
    return sig, r_peaks

def preprocess_beat(seq, N):
    # Rescale
    M = np.max(np.abs(seq))
    seq = seq/M
    return seq

def bisearch(key, array):
    '''
    search value which is most closed to key
    :param key:
    :param array:
    :return:
    '''
    lo = 0
    hi = len(array)-1
    while lo <= hi:
        mid = lo + int((hi - lo) / 2)
        if key <array[mid]:
            hi = mid - 1
        elif key>array[mid] :
            lo = mid + 1
        else:
            return array[mid]
    if hi<0:
        return array[0]
    if lo>=len(array):
        return array[-1]
    return array[hi] if (key-array[hi])<(array[lo]-key) else array[lo]


# Beat lenght
N = 256
# Number of leads
L = 1
# Beat segmentation parameters
before,after = 130, 150
# data augmentation factor
data_aug = 0 # means that there is 40% probability of introducing in the dataset the beat inverted

# Record names
Records = ["100","101","103","105","106","108","109",
          "111","112","113","114","115","116","117","118","119",
          "121","122","123","124",
          "200","201","202","203","205","207","208","209",
          "210","212","213","214","215","219",
          "220","221","222","223","228","230","231","232","233","234"] #remove 102 104 107 217

# beat extraction from all patients -> shuffling -> divide in train/valid/test
Normal_beat_type = ["N", "L", "R", "e", "j"]
Abnormal_beat_type = ["a", "J", "A", "S",  "V", "E", "F", "/", "f", "Q", "!"]


# Start saving beats in two folders normal and abnormal
print('\n\t Intra-patient dataset creation')

# Function used for parallelization
def op(r, folder1, folder2):
    sig,ann = u.get_ecg(r)
    info = u.get_sig_info(r)
    if not ('MLII' in info['leads_names']):
        print("no MLII: "+r)
        return
    sig = sig[np.where(np.array(info['leads_names']) == 'MLII')][0]
    filtered, r_peaks = preprocess_ecg(sig)
    filtered = filtered[np.newaxis, :]
    
    for j,l in enumerate(ann.symbol):
        if l == '+':
            continue
        info['diagnosys'] = l
        info['label'] = 'normal' if (l in Normal_beat_type) else 'abnormal'
        r_peak = ann.sample[j]
        if int(r_peak + after) >= info['n_samples'] or int(r_peak - before) < 0:
            continue
        
        # Segment
        if l in Normal_beat_type:
            closed_rpeak_idx=bisearch(r_peak,r_peaks)
            if abs(closed_rpeak_idx-r_peak)<10:
                beat = filtered[:, int(r_peak-before):int(r_peak+after)]
                beat = preprocess_beat(beat, N)
                b = (beat, r, j, info)
                save_beat(folder1, b)
                if np.random.uniform(0,1) < data_aug:
                    b = (-1*beat, r, j, info)
                    save_beat(folder1, b)
        elif l in Abnormal_beat_type:
            beat = filtered[:, int(r_peak-before):int(r_peak+after)]
            beat = preprocess_beat(beat, N)
            b = (beat, r, j, info)
            save_beat(folder2, b)

path_normal = 'intra/normal/'
path_abnormal = 'intra/abnormal/'
# Make dirs for store temporarily normal and abnormal beats
if not os.path.exists(path_normal):
    os.makedirs(path_normal)
if not os.path.exists(path_abnormal):
    os.makedirs(path_abnormal)

print("\n\t Divide normal and abnormal beats in two temporary folder")

Parallel(n_jobs=-1, verbose=32, backend= "multiprocessing")(delayed(op)(r, path_normal, path_abnormal)
                           for r in Records)
print("\n\t END")






