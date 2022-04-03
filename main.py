#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:07:04 2021

@author: simone
"""
#%%
import os
import random
import multiprocessing
import torch
import torch.nn.parallel
import torch.utils.data
from Ecgdataset import Ecgdataset
from AAECG_2 import CBeatAAE
import Baselines as b


# Root directory 
root = "intra/"
path_normal = root+'normal/'
path_abnormal = root+'abnormal/'

# Datasets
normal_ecg = Ecgdataset(path_normal, upload_on_ram = True)
normal_ecg = normal_ecg.get_db()

abnormal_ecg = Ecgdataset(path_abnormal, upload_on_ram = True)
abnormal_ecg = abnormal_ecg.get_db()

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Batch size during training
batch_size = 256

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of training epochs
num_epochs = 20

# Decide which device we want to run on
if ngpu >0:
    device = torch.device("cuda:0" if ( ngpu > 0 and torch.cuda.is_available()) else "cpu")
else:
    device = torch.device("cpu")
print(device)

#%%
"""
per fare cross validation:
    metto tutti i normali nel train
    metto tutti gli anormali nel test
    prendo i-esimo 1/n_fold normali dal train e li metto nel test
    prendo i primi tot del train e li metto nel valid
    prendo i primi tot dal test e li metto nel valid(sono abnormali perche nel test ho messo prima anormali e poi normali)
"""
n_fold = 5
normal_valid_perc = 0.05
abnormal_valid_perc = 0.1

model = CBeatAAE()
model_name = 'AAE'
melt = True if model_name=='PCA' else False
# If the first i fold has already been done
start = 0

model.verbose()

for i in range(start, n_fold):
    train_ecg = normal_ecg.copy()
    
    # Test ECG
    test_ecg = abnormal_ecg.copy()
    # select the normal to put in the test
    test_ecg.extend(train_ecg[int(i/n_fold*len(normal_ecg)): int((i+1)/n_fold*len(normal_ecg))])
    train_ecg[int(i/n_fold*len(normal_ecg)): int((i+1)/n_fold*len(normal_ecg))] = []
    
    # Validation ECG 
    valid_ecg = train_ecg[:int(normal_valid_perc*len(train_ecg))]
    train_ecg[:int(normal_valid_perc*len(train_ecg))] = []
    
    valid_ecg.extend(test_ecg[:int(abnormal_valid_perc*len(abnormal_ecg))])
    test_ecg[:int(abnormal_valid_perc*len(abnormal_ecg))] = []
    
    # Create the datasets
    train_set = Ecgdataset(None, db = train_ecg) 
    validation_set = Ecgdataset(None, db = valid_ecg)
    test_set = Ecgdataset(None, db = test_ecg)
    
    if melt:
        train_dl = train_set.melt()
        valid_dl = validation_set.melt()
        test_dl = test_set.melt()
    else:
        # Create the dataloader
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True, drop_last=True)
        valid_dl = torch.utils.data.DataLoader(validation_set, batch_size=batch_size*10)
        test_dl = torch.utils.data.DataLoader(test_set, batch_size = batch_size*10)
        
    model.initialize()
    model.train(train_dl, valid_dl, num_epochs, lambda_rec = 1, lambda_adv = 1,lambda_tv = 0.001,beta1 = 0)
    
    if not os.path.exists("models/"+model_name+"/fold_"+str(i)+"/"):
        os.makedirs("models/"+model_name+"/fold_"+str(i)+"/")
    
    model.test(test_dl, result_folder="models/"+model_name+"/fold_"+str(i)+"/")
    
    del train_ecg
    del valid_ecg
    del test_ecg




          

