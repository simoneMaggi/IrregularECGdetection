#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:12:38 2021

@author: simone
"""
import torch
from MyModel_20AAE import CBeatAAE
import Utils as u
from Ecgdataset import Ecgdataset
import Baselines as b
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from random import randint


aae = CBeatAAE(nz = 5)
aae.load_model()
# Load the dataset
dataset = Ecgdataset("intra/abnormal/")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle= True)


def save_ts_heatmap(input,output,title, save_path = None):
    x_points = np.arange(input.shape[1])
    fig, ax = plt.subplots(2, 1, sharex=True,figsize=(6, 6),gridspec_kw = {'height_ratios':[6,1]})
    sig_in = input[0, :]
    sig_out = output[0, :]
    ax[0].plot(x_points, sig_in,'k-',linewidth=2.5,label="input signal")
    ax[0].plot(x_points,sig_out,'k--',linewidth=2.5,label="output signal")
    ax[0].set_yticks([])
    ax[0].legend(loc="upper right")



    heat=(sig_out-sig_in)**2
    heat_norm=(heat-np.min(heat))/(np.max(heat)-np.min(heat))
    heat_norm=np.reshape(heat_norm,(1,-1))

    ax[1].imshow(heat_norm, cmap="jet", aspect="auto")
    ax[1].set_yticks([])
    fig.tight_layout()
    fig.show()
    if save_path is None:
        return
    fig.savefig(save_path)
    plt.clf()
    plt.close()




for i,load in enumerate(dataloader):
    beat = load[0][:1]
    Dx = load[2]['label'][0]
    typ = load[2]['diagnosys'][0]
    if typ != 'A':
        continue
    # AAE
    reconstructed = aae.netD(aae.netE(beat), load[1][:1]).detach()
    pred = aae.get_anomaly_score(beat, load[1][:1]) > aae.threeshold
    pred = "Abnormal" if pred else "Normal"
    save_ts_heatmap(beat[0].numpy(), reconstructed[0].numpy(), 'Predicted as Abnormal')
  
    print("diagnosys is "+typ)
    break