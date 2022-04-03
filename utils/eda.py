#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:48:00 2021

@author: simone
"""

#%%
import matplotlib.pyplot as plt
import Utils as u
import numpy as np
import os
import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()
from tqdm import tqdm
import pickle
import glob

def data_explorer_parallel(path):
    print("\n EXPLORING THE ECG DATA IN PATH "+path)
    files = glob.glob(path+"/**/*.*", recursive = True)
    total = len(files)
    normal_records = []
    abnormal_records = []
    
    outs = []
    def op(f):
        # files of the tipe path/Rxxx_yyy.npy
        info = u.get_beat_info(f.split('.')[0])
        beat_pos = f.split('R')[1].split('_')[1].split('.')[0]
        lead_conf = (' '.join(info['leads_names']))
        return (info['rec_name']+beat_pos, info['label'] == 'normal', info['diagnosys'], 
                lead_conf, info['sex'], info['age'])
    
    outs = Parallel(n_jobs=num_cores, backend="threading")(delayed(op)(files[i])
                           for i in tqdm(range(total)))    
    
    sexes = [o[4] for o in outs]
    ages = [o[5] for o in outs]
    normal = np.sum(np.array([o[1] for o in outs]))
    leads_config = [o[3] for o in outs]
    for o in outs:
        if o[1]:
            normal_records.append(o[0])
        else:
            abnormal_records.append(o[0])
    values = set(map(lambda x:x[2], outs))
    classes = {x:[y[0] for y in outs if y[2]==x] for x in values}
    return {'n_beats': total, 'normal': normal, 'sexes':sexes, 'ages': ages,
             'normal_records':normal_records, 'abnormal_records':abnormal_records,
            'classes':classes, 'leads_config':leads_config}
    
def plot_eda(eda, title):
    # Ages
    # plt.figure()
    # plt.title(title)
    # ages = [int(a) for a in eda['ages']]
    # plt.hist(ages, density = True, bins = 50)
    # plt.show()
    # Sex
    # plt.figure()
    # plt.title(title)
    # plt.hist(eda["sexes"], density = True)
    # plt.show()
    # Class balancement
    # plt.figure()
    # plt.title(title)
    # n = eda["n_beats"]
    # plt.bar(['normal', 'abnormal'], 
    #         [eda["normal"], n - eda["normal"]])
    # plt.show()
    # Leads configurations
    plt.figure()
    plt.yscale('log')
    plt.hist(eda['leads_config'])
    plt.grid(True)
    plt.show()
    
    # Class subdivision
    # plt.figure()
    # plt.title(title+"class subdivision")
    # plt.yscale('log')
    # plotted_v = [len(eda['classes'][k]) for k in eda['classes'].keys()]
    # plt.bar(eda['classes'].keys(), plotted_v)
    # plt.show()
    
def save_eda(path, plots = False):
    eda = data_explorer_parallel(path)
    name = path.replace('/', '_')
    with open('eda/'+name+".pkl", "wb") as h:
        pickle.dump(eda, h)
    if plots:
        plot_eda(eda, name)
    return eda
     
def load_eda(path):
    name = path.replace('/', '_')
    with open('eda/'+name+".pkl", "rb") as h:
        eda = pickle.load(h)
    return eda

def explore(path):
    name = path.replace('/', '_')
    if os.path.exists('eda/'+name+'.pkl'):
        eda = load_eda(path)
        plot_eda(eda, name)
    else:
        eda = save_eda(path, plots = True)
    
plt.style.use('Solarize_Light2')
explore("intra")

    
#%%
import matplotlib.gridspec as gridspec
def plot_batch(batch, titles_list = None, suptitle = None, column = 5):
    beat_list = batch[0]
    nr = int(beat_list.size(0)/column)
    nc = column
    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=nc, nrows=nr, figure=fig2)
    for i in range(nr):
        for j in range(nc):
            ax = fig2.add_subplot(spec2[i, j])
            ax.plot(beat_list[int(i*nr)+j])
            ax.set_title(titles_list[int(i*nr) + j])

        return ax

#%% plot some train beats
from Ecgdataset import Ecgdataset
import torch.utils.data


data = Ecgdataset('intra/normal/')

dl = torch.utils.data.DataLoader(data, batch_size=5,
                                        shuffle=True)

some_beat = next(iter(dl))

u.plot_some_beat(some_beat[0], titles_list=[' '.join([some_beat[2]['label'][i],
                                                     some_beat[2]['diagnosys'][i]
                                                     ]) for i in range(len(some_beat[0]))] )

                                                    
#%%

    
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     