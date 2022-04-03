#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:34:19 2021

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
root = "inter/"
path_train = root+'train/'
path_valid = root+'valid/'
path_test = root+'test/'

# Datasets
train = Ecgdataset(path_train, upload_on_ram = True)
train = train.get_db()

valid = Ecgdataset(path_valid, upload_on_ram = True)
valid = valid.get_db()

test = Ecgdataset(path_test, upload_on_ram = True)
test = test.get_db()

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Batch size during training
batch_size = 256

# Number of training epochs
num_epochs = 20

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
if ngpu >0:
    device = torch.device("cuda:0" if ( ngpu > 0 and torch.cuda.is_available()) else "cpu")
else:
    device = torch.device("cpu")
print(device)

#%%

model = CBeatAAE()
model_name = 'AAE'
melt = True if model_name=='PCA' else False
# If the first i fold has already been done
start = 0

model.verbose()


if melt:
    train_dl = train.melt()
    valid_dl = valid.melt()
    test_dl = test.melt()
else:
    # Create the dataloader
    train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=True, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(valid, batch_size=batch_size*10)
    test_dl = torch.utils.data.DataLoader(test, batch_size = batch_size*10)
    
model.initialize()
model.train(train_dl, valid_dl, num_epochs, lambda_rec = 1, lambda_adv = 1,lambda_tv = 0.001,beta1 = 0)

if not os.path.exists("models/"+model_name+"/"):
    os.makedirs("models/"+model_name+"/")

model.test(test_dl, result_folder="models/"+model_name+"/")






          

