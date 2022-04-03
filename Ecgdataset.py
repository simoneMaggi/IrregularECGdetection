#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:27:54 2021

@author: simone
"""

from torch.utils.data.dataset import Dataset
import glob
import numpy as np
import torch
import Utils as u
from joblib import Parallel, delayed
import random

class Ecgdataset(Dataset):
    def __init__(self, folder_path, db = None, upload_on_ram = False, batch_size = 500, shuffle = True):
        if not (db is None):
            self.upload_on_ram = True
            self.db = db
            if shuffle:
                random.shuffle(self.db)
            self.data_len = len(db)
            return
        
        self.folder_path = folder_path
        # Get ecg list
        self.beat_list = glob.glob(folder_path + '*.npy')
        
        # Calculate len
        self.data_len = len(self.beat_list)
        
        # Upload on Ram to speed up (use only if your hardware enough)
        self.upload_on_ram = upload_on_ram
        
        if self.upload_on_ram:
            print("\n uploading "+folder_path+" on RAM")
            self.db = Parallel(n_jobs=-1, verbose = 1, backend="multiprocessing",
                               batch_size=batch_size)(delayed(self.load_index)(i)
                           for i in range(self.data_len)) 
            if shuffle:
               random.shuffle(self.db)                                         
                                                      
                                                      
                                                      
    def get_db(self):
        return self.db
    
    def melting(self, i):
        item = self.__getitem__(i)
        line = item[0].squeeze().numpy().tolist()
        line.append(int(item[2]['label']=='abnormal'))
        return line
    
    def melt(self):
        print("\n melting ")
        if self.upload_on_ram:
            melted = [[*(item[0].squeeze().numpy().tolist()), int(item[2]['label']=='abnormal')] for item in self.db]
        else:
            melted = Parallel(n_jobs=-1,verbose = 2, backend="multiprocessing")(delayed(self.melting)(i)
                                                                            for i in range(self.data_len))  
        return np.array(melted)
     
    def encode_sex(self, sex):
        res = [0, 0, 1]
        if sex == 'F':
            res = [1, 0, 0]
        elif sex == 'M':
            res = [0, 1, 0]
        return torch.tensor(res, dtype = torch.float32)
    
    def load_index(self, index):
        beat_path = self.beat_list[index]
        
        # Get ecg data
        beat = torch.from_numpy(np.load(beat_path)).float()
        
        # Get ecg labels. The file name is of the type Rxxx_xxx.npy
        info = u.get_beat_info(beat_path.split('.')[0])
        
        sex = info['sex']
        sex = self.encode_sex(sex)
        label = sex
        
        return (beat, label, info)
    
    def __getitem__(self, index):
        if self.upload_on_ram:
            return self.db[index]
        else:
            return self.load_index(index)
    
    def __len__(self):
        return self.data_len

        
        
        
        
        
        