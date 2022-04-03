#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:22:30 2020

@author: simone
"""
#%% libraries
import wfdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from biosppy.signals import ecg
import os
from tqdm import tqdm
import pickle
import multiprocessing
num_cores = multiprocessing.cpu_count()


# according to AAMI
normal_beat_ann = ['N', 'L', 'R', 'e', 'j', '.']
sinus_beat_codes = [426783006, 427084000, 426177001, 427393009, ]

def header_reader(header, rec):
    res = {}
    res['rec_name']=rec
    res['n_lead']=header.n_sig
    res['fs']=header.fs
    res['n_samples']=header.sig_len
    
    # Leads used
    res['leads_names'] = header.sig_name
    
    com = header.comments
    
    # Age
    res['age'] = com[0].split(' ')[0]
    
    # Sex
    res['sex']=com[0].split(' ')[1]
    
    # Medications
    res['medications'] = []
    med = com[1].split(',')
    for m in med:
        res['medications'].append(m)    
    return res

def header_reader_wfdb(header, rec):
    res = {}
    res['rec_name']=rec
    res['n_lead']=header.n_sig
    res['fs']=header.fs
    res['n_samples']=header.sig_len
    
    # Leads used
    res['leads_names'] = header.sig_name
    
    com = header.comments
    
    # Age
    res['age'] = com[0].split(' ')[1]
    
    # Sex
    res['sex']=com[1].split(' ')[1][0]
    
    # Diagnosys
    res['Dx'] = com[2].split(' ')[1]
    
    # Medications
    res['medications'] = []
    
    return res

def get_ecg(rec, path = 'mit_bih/', ann_file=True):
    sig = wfdb.rdsamp(path+rec)
    sig = np.transpose(sig[0])
    ann = None
    if ann_file:
        ann = wfdb.rdann(path+rec, 'atr')
    return (sig,ann)

def get_sig_info(rec, path='mit_bih/', type2=False):
    header = wfdb.rdheader(path+rec)
    if type2:
        h = header_reader_wfdb(header, rec)
    else:
        h = header_reader(header, rec)
    return h

def get_beat_info(beat_path):
    with open(beat_path+".pkl", "rb") as pkl_handle:
        res = pickle.load(pkl_handle)
    del res['medications']
    return res

def plot_multivariate(sig, title = ''):
    l = sig.shape[0]
    f, ax = plt.subplots(l,1,sharex=True)
    f.suptitle(title)
    for i,a in enumerate(ax):
        a.plot(sig[i])
        a.set_ylim(bottom = -1, top = 1)

def get_ecg_annotation(rec, path= 'mit_bih/'):
    return wfdb.rdann(path+rec, 'atr')


import pywt


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

    

def preprocess_ecg(sig,  threshold = 0.04):
    baseline = calc_baseline(sig)
    sig = sig - baseline
    
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(sig), w.dec_len)
    coeffs = pywt.wavedec(sig, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    filtered = pywt.waverec(coeffs, 'sym4')
    
    return filtered


def plot_ecg(rec, path = 'mit_bih/', 
             preprocess = False, start = 0, window = 1000, lead=None, figsize=None):
    sig, ann = get_ecg(rec)
    info = get_sig_info(rec)
    if preprocess:
        for i in range(sig.shape[0]):
            sig[i] = preprocess_ecg(sig[i])
            
    if window == 'all':
        end = info['n_samples']
    else:
        end = window+start if info['n_samples']>=(window+start) else info['n_samples']
    
    if not(lead is None):
        
        if not(lead is None):
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        plt.plot(sig[lead][start:end])
        plt.show()
        return
    f, ax = plt.subplots(info['n_lead'],1,sharex=True)
    for i,a in enumerate(ax):
        a.plot(sig[i][start:end])
    ax[0].set_title(info['sex']+" age: "+str(info['age']))

    plt.show()

def get_beat(path, rec_name):
    sig = np.load(path+rec_name+'.npy')
    info = get_beat_info(path+rec_name)
    return (sig, info)

def plot_beat(path, rec_name):
    sig, info = get_beat(path, rec_name)
    plot_multivariate(sig, title = 'label: '+info['label']+' leads: '+(' '.join(info['leads_names'])))

def plot_some_beat(beat_list, titles_list = None, suptitle = None, savefig = False, file= None):
    n = len(beat_list)
    nl = len(beat_list[0])
    f, ax = plt.subplots(nl, n, sharey=True)
    if not(suptitle is None):
        f.suptitle(suptitle)
    for i in range(n):
        if not (titles_list is None):
            if nl > 1:
                ax[0,i].set_title(titles_list[i])
            else:
                ax[i].set_title(titles_list[i])
        if nl == 1:
            ax[i].plot(beat_list[i][0])
            ax[i].set_ylim(bottom = -1, top = 1)
        else:
            for j in range(nl):
                ax[j,i].plot(beat_list[i][j])
                ax[j,i].set_ylim(bottom = -1, top = 1)
    if savefig:
        plt.savefig(file, dpi = 300)
    return ax
   


def data_explorer(path):
    print("EXPLORING THE ECG DATA IN PATH "+path)
    files = os.listdir(path)
    total = len(files)
    normal_tot_duration = 0
    normal = 0
    sexes = []
    ages = []
    normal_ages = []
    normal_male = 0
    normal_records = []
    abnormal_records = []
    anomalous_classes = {}
    leads_config = []
    pbar = tqdm(total = total)
    for i,f in enumerate(files):
        if i%50 == 0:
            pbar.update(n=50)
        # files of the tipe path/Rxxx_yyy.npy
        info = get_beat_info(path+'/'+f.split('.')[0])
        beat_pos = (f.split('_')[1]).split('.')[0]
        leads_config.append(' '.join(info['leads_names']))
        if info['label'] == 'normal':
            normal += 1
            normal_tot_duration += info['n_samples']/info['fs']
            normal_ages.append(int(info['age']))
            if info['sex']=='M':
                normal_male += 1
            normal_records.append(info['rec_name']+'_'+beat_pos)
        else:
            abnormal_records.append(info['rec_name']+'_'+beat_pos)
            try: 
                anomalous_classes[info['diagnosys']].append(info['rec_name']+'_'+beat_pos)
            except KeyError:
                anomalous_classes[info['diagnosys']] = [info['rec_name']+'_'+beat_pos]
                
            
        sexes.append(info['sex'])
        ages.append(int(info['age']))
        
    pbar.close()
    return {'n_beats': total, 'normal': normal, 'sexes':sexes, 'ages': ages,
            'tot_duration_normal': normal_tot_duration, 'normal_male':normal_male,
            'normal_ages':normal_ages, 'normal_records':normal_records, 'abnormal_records':abnormal_records,
            'anomalous_classes':anomalous_classes, 'leads_config':leads_config}
     



